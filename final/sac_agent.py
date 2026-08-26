"""
Minimal self-contained SAC agent (continuous actions, squashed Gaussian policy,
twin Q critics, learnable temperature).

Why we don't reuse mbrl's vendored pytorch_sac or stable-baselines3 SAC:
  - mbrl's SACAgent is driven through Hydra DictConfigs + a custom Logger, which
    inflates a training script that otherwise has no Hydra usage.
  - SB3 SAC's `.learn()` owns its env-interaction loop and replay sampling, so
    injecting mbrl's synthetic model-rollout transitions at the right cadence
    would require monkey-patching its internals.
  - Writing SAC directly is ~200 lines, has a clean state_dict / load_state_dict
    surface for checkpointing, and samples from any replay buffer we hand it,
    which is exactly what MBPO needs (real + synthetic mixed batches).

This agent only needs a `.sample_batch(batch_size) -> (obs, act, rew, next_obs,
not_done)` method on whatever buffers are provided, so it pairs with mbrl's
ReplayBuffer via a tiny adapter in train_mbpo.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# Clamp learnable temperature.
#   MAX=log(1.0): avoids the α → ∞ feedback loop from run #1 (α hit 103, entropy
#     bonus dwarfed Q, policy became maximally random).
#   MIN=log(0.25): run #5 used 0.15 and still fell into a persistent idle
#     attractor after step 200k; α auto-tune then began climbing (hit 0.63 by
#     step 250k) as the update rule tried to push entropy back up. Raising the
#     floor forces sustained exploration pressure before the collapse happens.
LOG_ALPHA_MIN = math.log(0.25)
LOG_ALPHA_MAX = math.log(1.0)


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.trunk = _mlp(obs_dim, hidden, 2 * act_dim, depth)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pre_tanh = mu + std * noise
        action = torch.tanh(pre_tanh)
        # log-prob with tanh correction; numerically stable form
        normal_logp = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        tanh_corr = 2 * (math.log(2.0) - pre_tanh - F.softplus(-2 * pre_tanh))
        log_prob = (normal_logp - tanh_corr).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        batched = t.ndim == 2
        if not batched:
            t = t.unsqueeze(0)
        mu, log_std = self.forward(t)
        if deterministic:
            action = torch.tanh(mu)
        else:
            action = torch.tanh(mu + log_std.exp() * torch.randn_like(mu))
        action = action.cpu().numpy()
        return action if batched else action[0]


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1, depth)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1, depth)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class SACUpdateInfo:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float


class SACAgent:
    """Twin-Q SAC with learnable temperature. Operates on a batch dict with keys
    obs, act, rew, next_obs, not_done (all numpy float32)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 0.15,
        target_entropy: float = None,
        hidden: int = 256,
        depth: int = 2,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = GaussianActor(obs_dim, act_dim, hidden, depth).to(self.device)
        self.critic = TwinQ(obs_dim, act_dim, hidden, depth).to(self.device)
        self.critic_target = TwinQ(obs_dim, act_dim, hidden, depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(math.log(init_temperature), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: dict) -> SACUpdateInfo:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        not_done = torch.as_tensor(batch["not_done"], dtype=torch.float32, device=self.device).view(-1, 1)

        # critic target
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * next_logp
            y = rew + self.gamma * not_done * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # actor
        new_act, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # temperature
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)

        # target soft update
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return SACUpdateInfo(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "target_entropy": self.target_entropy,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
