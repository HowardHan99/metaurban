"""
Train MBPO (Model-Based Policy Optimization, Janner et al. 2019) on the MetaUrban
SidewalkStaticMetaUrbanEnv (PointNav).

MBPO = learn a probabilistic ensemble dynamics model from real transitions, then
augment a SAC replay buffer with short k-step rollouts unrolled inside the learned
model. SAC updates sample batches mixed from real + synthetic buffers (real_ratio ~ 50%).
See: Janner, Fu, Zhang, Levine, "When to Trust Your Model: Model-Based Policy
Optimization", NeurIPS 2019.

Resumable from any checkpoint via --resume_from <path>. A checkpoint is a single
.pt file holding dynamics model weights + normalizer, SAC networks/optimizers,
both replay buffers, RNG state, and the real-env step counter, so resuming picks
back up with zero drift.

Why we're not using mbrl.algorithms.mbpo.train():
  - It's hard-wired to old-gym's step() 4-tuple; MetaUrban returns gymnasium's
    5-tuple (terminated, truncated).
  - Its checkpointing is Hydra-run-directory based, not point-in-time.
We reuse mbrl's dynamics model (GaussianMLP ensemble + OneDTransitionRewardModel)
and ModelEnv for imagined rollouts — the expensive/hairy pieces — and write the
outer MBPO loop ourselves so we own serialization.

Default budget is 500k real env steps (MBPO is far more sample-efficient than
the midterm's PPO/SAC, which used 3M).

Usage:
    python train_mbpo.py --total_timesteps 500000 --seed 0
    python train_mbpo.py --resume_from ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_step_200000.pt \
                        --total_timesteps 500000
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import gymnasium as gym

import mbrl.models
import mbrl.util.common as mbrl_common
import mbrl.util.replay_buffer as mbrl_rb

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from metaurban import SidewalkStaticMetaUrbanEnv

from env_config import ENV_CONFIG, MBPO_ENV_OVERRIDES, EVAL_VEC_ENV_SEEDS
from sac_agent import SACAgent


# ---------- env wrappers ----------

class IdlePenaltyWrapper(gym.Wrapper):
    """Penalize standing still — same as midterm's SAC wrapper. MBPO's inner
    policy inherits SAC's brake-local-minimum failure without it."""

    def __init__(self, env, penalty: float = 0.1, speed_threshold: float = 0.5):
        super().__init__(env)
        self.penalty = penalty
        self.speed_threshold = speed_threshold

    def reset(self, *, seed=None, options=None, **kwargs):
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            speed = self.env.unwrapped.vehicle.speed_km_h
        except AttributeError:
            speed = None
        if speed is not None and speed < self.speed_threshold:
            reward -= self.penalty
        return obs, reward, terminated, truncated, info


class GoalProximityRewardWrapper(gym.Wrapper):
    """Potential-based shaping on route_completion: reward += coeff * Δrc.

    Background: run #4 used `coeff * rc^2` but that *backfired* — rewarding
    the state of being-at-middle-rc gave the agent a strong incentive to
    cruise safely at rc≈0.5 and accumulate shaped reward without committing
    to the risky final approach. Run #4 peaked lower than run #3.

    Potential-based shaping (Ng & Russell 1999) only rewards *progress* —
    never holding position. Per episode the total added reward is
    `coeff * (rc_final - rc_initial)`, so:
      - crashes mid-route: small bonus (up to rc at crash)
      - reaches goal: `coeff * 1.0` bonus (in addition to the env's +15
        arrive_dest reward)
    This preserves the optimal policy theoretically and can't be gamed
    by idling.

    Why coeff=20: pure Δrc over an episode is just 1.0, which is small next
    to crash_penalty=1.0 and success_reward=15. We need a non-trivial
    gradient all along the path. With coeff=20:
      - Completing the full route adds +20 to return.
      - Per-step reward for typical Δrc ≈ 1/path_length is small enough
        not to swamp other terms, but the cumulative signal is strong.
      - Agent sees "reach rc=1.0" as roughly doubling the terminal payoff
        vs. "reach rc=0.5 and crash".
    """

    def __init__(self, env, coeff: float = 20.0):
        super().__init__(env)
        self.coeff = float(coeff)
        self.last_rc = 0.0

    def reset(self, *, seed=None, options=None, **kwargs):
        ret = self.env.reset(seed=seed, **kwargs)
        self.last_rc = 0.0
        return ret

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rc = float(info.get("route_completion", 0.0))
        reward += self.coeff * (rc - self.last_rc)
        self.last_rc = rc
        return obs, reward, terminated, truncated, info


def make_env(
    env_cfg: dict,
    seed: int,
    use_idle_penalty: bool = True,
    goal_proximity_coeff: float = 0.0,
):
    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **env_cfg))
    if use_idle_penalty:
        env = IdlePenaltyWrapper(env)
    if goal_proximity_coeff > 0.0:
        env = GoalProximityRewardWrapper(env, coeff=goal_proximity_coeff)
    env = Monitor(env)
    return env


# ---------- termination fn for MetaUrban ----------

def metaurban_termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """MetaUrban's termination depends on the underlying simulator state (crash,
    out-of-road), which we cannot recover from obs alone. Returning 'never
    terminal' is the standard MBPO workaround for envs without closed-form
    termination — it is safe as long as rollout_length stays short (~1–5 steps)
    so predicted trajectories don't drift into clearly-terminal regions."""
    return torch.zeros(next_obs.shape[0], 1, dtype=torch.bool, device=next_obs.device)


# ---------- dynamics model factory ----------

def build_dynamics_model(
    obs_dim: int,
    act_dim: int,
    ensemble_size: int,
    num_elites: int,
    hid_size: int,
    num_layers: int,
    device: str,
) -> mbrl.models.OneDTransitionRewardModel:
    base = mbrl.models.GaussianMLP(
        in_size=obs_dim + act_dim,
        out_size=obs_dim + 1,  # delta-obs + reward
        device=device,
        num_layers=num_layers,
        ensemble_size=ensemble_size,
        hid_size=hid_size,
        deterministic=False,
        propagation_method="random_model",
        learn_logvar_bounds=True,
    )
    wrapped = mbrl.models.OneDTransitionRewardModel(
        base,
        target_is_delta=True,
        normalize=True,
        learned_rewards=True,
        num_elites=num_elites,
    )
    return wrapped


# ---------- model-env rollout populates synthetic buffer ----------

def populate_synthetic_buffer(
    model_env: mbrl.models.ModelEnv,
    real_buf: mbrl_rb.ReplayBuffer,
    syn_buf: mbrl_rb.ReplayBuffer,
    agent: SACAgent,
    rollout_length: int,
    rollout_batch_size: int,
) -> int:
    """Sample starting states from the real buffer, roll rollout_length steps
    through the learned model using current policy, and append transitions to
    the synthetic buffer. Mirrors mbrl/algorithms/mbpo.py but uses our SAC."""
    if len(real_buf) == 0:
        return 0
    batch_size = min(rollout_batch_size, len(real_buf))
    batch = real_buf.sample(batch_size)
    initial_obs = batch.obs
    model_state = model_env.reset(initial_obs_batch=initial_obs, return_as_np=True)
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    n_added = 0
    for _ in range(rollout_length):
        action = agent.actor.act(obs, deterministic=False)
        next_obs, rewards, pred_dones, model_state = model_env.step(action, model_state, sample=True)
        live = ~accum_dones
        if live.sum() == 0:
            break
        # mbrl ReplayBuffer has no add_batch → iterate live rows.
        # rewards/pred_dones from ModelEnv are (batch, 1) — flatten before indexing.
        rewards_flat = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones_flat = np.asarray(pred_dones).reshape(-1).astype(bool)
        for i in np.nonzero(live)[0]:
            syn_buf.add(
                obs[i].astype(np.float32),
                action[i].astype(np.float32),
                next_obs[i].astype(np.float32),
                float(rewards_flat[i]),
                bool(dones_flat[i]),
            )
            n_added += 1
        obs = next_obs
        accum_dones |= dones_flat
    return n_added


# ---------- SAC batch mixing ----------

def sample_mixed_batch(
    real_buf: mbrl_rb.ReplayBuffer,
    syn_buf: mbrl_rb.ReplayBuffer,
    batch_size: int,
    real_ratio: float,
) -> Dict[str, np.ndarray]:
    n_real = int(round(batch_size * real_ratio))
    n_real = max(1, min(n_real, len(real_buf)))
    n_syn = batch_size - n_real
    real = real_buf.sample(n_real)
    if n_syn > 0 and len(syn_buf) > 0:
        syn = syn_buf.sample(min(n_syn, len(syn_buf)))
        obs = np.concatenate([real.obs, syn.obs], axis=0)
        act = np.concatenate([real.act, syn.act], axis=0)
        rew = np.concatenate([real.rewards, syn.rewards], axis=0)
        nxt = np.concatenate([real.next_obs, syn.next_obs], axis=0)
        done = np.concatenate([real.dones, syn.dones], axis=0)
    else:
        obs, act, rew, nxt, done = real.obs, real.act, real.rewards, real.next_obs, real.dones
    return {
        "obs": obs.astype(np.float32),
        "act": act.astype(np.float32),
        "rew": rew.astype(np.float32),
        "next_obs": nxt.astype(np.float32),
        "not_done": (1.0 - done.astype(np.float32)),
    }


# ---------- evaluation ----------

def evaluate_policy(
    vec_env: SubprocVecEnv,
    agent: SACAgent,
    n_eval_episodes: int,
) -> Dict[str, float]:
    """Roll `n_eval_episodes` episodes across the vec env, deterministic policy.
    Returns mean/median/IQR of return, mean length, success rate."""
    n_envs = vec_env.num_envs
    ep_returns = []
    ep_lengths = []
    ep_successes = []
    obs = vec_env.reset()
    cur_ret = np.zeros(n_envs, dtype=np.float64)
    cur_len = np.zeros(n_envs, dtype=np.int64)
    while len(ep_returns) < n_eval_episodes:
        action = agent.actor.act(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)
        cur_ret += reward
        cur_len += 1
        for i, d in enumerate(done):
            if d:
                if len(ep_returns) < n_eval_episodes:
                    ep_returns.append(float(cur_ret[i]))
                    ep_lengths.append(int(cur_len[i]))
                    # SB3 VecEnv convention: terminal obs lives in infos[i] and
                    # original info under "terminal_info" or raw info; gymnasium-
                    # monitored envs put is_success inside the info dict.
                    info = infos[i]
                    succ = bool(info.get("arrive_dest") or info.get("is_success", False))
                    ep_successes.append(succ)
                cur_ret[i] = 0.0
                cur_len[i] = 0
    arr = np.asarray(ep_returns[:n_eval_episodes])
    lens = np.asarray(ep_lengths[:n_eval_episodes])
    succ = np.asarray(ep_successes[:n_eval_episodes], dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "q25_return": float(np.percentile(arr, 25)),
        "q75_return": float(np.percentile(arr, 75)),
        "mean_length": float(lens.mean()),
        "success_rate": float(succ.mean()),
        "n_episodes": int(len(arr)),
    }


# ---------- checkpoint ----------

def save_checkpoint(
    path: Path,
    step: int,
    agent: SACAgent,
    dynamics: mbrl.models.OneDTransitionRewardModel,
    real_buf: mbrl_rb.ReplayBuffer,
    syn_buf: mbrl_rb.ReplayBuffer,
    args_dict: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf_dir = path.parent / f"buffers_step_{step}"
    real_dir = buf_dir / "real"
    syn_dir = buf_dir / "syn"
    real_dir.mkdir(parents=True, exist_ok=True)
    syn_dir.mkdir(parents=True, exist_ok=True)
    real_buf.save(real_dir)
    syn_buf.save(syn_dir)

    torch.save(
        {
            "step": step,
            "sac": agent.state_dict(),
            "dynamics": dynamics.state_dict(),
            "real_buf_dir": str(buf_dir / "real"),
            "syn_buf_dir": str(buf_dir / "syn"),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    agent: SACAgent,
    dynamics: mbrl.models.OneDTransitionRewardModel,
    real_buf: mbrl_rb.ReplayBuffer,
    syn_buf: mbrl_rb.ReplayBuffer,
) -> int:
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(ckpt["sac"])
    dynamics.load_state_dict(ckpt["dynamics"])
    real_buf.load(ckpt["real_buf_dir"])
    syn_buf.load(ckpt["syn_buf_dir"])
    torch.set_rng_state(ckpt["rng_torch"].cpu().to(torch.uint8))
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    return int(ckpt["step"])


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description="MBPO on MetaUrban (final project)")
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", type=str, default="./final_logs/MBPO")
    p.add_argument("--device", type=str, default="cuda")

    # MBPO hyperparameters (Janner et al. defaults, scaled for social nav low-dim obs)
    p.add_argument("--init_steps", type=int, default=5_000, help="Random-policy env steps before training begins")
    p.add_argument("--real_buffer_size", type=int, default=1_000_000)
    p.add_argument("--ensemble_size", type=int, default=7)
    p.add_argument("--num_elites", type=int, default=5)
    p.add_argument("--model_hid_size", type=int, default=200)
    p.add_argument("--model_num_layers", type=int, default=4)
    p.add_argument("--model_lr", type=float, default=1e-3)
    p.add_argument("--model_weight_decay", type=float, default=1e-5)
    p.add_argument("--model_batch_size", type=int, default=256)
    p.add_argument("--model_val_ratio", type=float, default=0.1)
    p.add_argument("--model_train_freq", type=int, default=500, help="Retrain dynamics every N env steps")
    p.add_argument("--model_train_max_epochs", type=int, default=40)
    p.add_argument("--model_train_patience", type=int, default=5)
    p.add_argument("--rollout_length", type=int, default=1)
    p.add_argument("--rollout_batch_size", type=int, default=10_000)
    p.add_argument("--model_retain_epochs", type=int, default=1)
    p.add_argument("--real_ratio", type=float, default=0.5,
                   help="Fraction of real transitions in each SAC batch. Janner's paper uses "
                        "0.05; on our noisy lidar-state task 0.05 let the learned model's "
                        "errors dominate SAC updates (500k-step run #1 had α runaway to 103).")
    p.add_argument("--sac_updates_per_step", type=int, default=10)
    p.add_argument("--sac_batch_size", type=int, default=256)
    p.add_argument("--sac_actor_lr", type=float, default=3e-4)
    p.add_argument("--sac_critic_lr", type=float, default=3e-4)
    p.add_argument("--sac_alpha_lr", type=float, default=3e-4)
    p.add_argument("--sac_gamma", type=float, default=0.99)
    p.add_argument("--sac_tau", type=float, default=0.005)
    p.add_argument("--sac_init_temp", type=float, default=0.15)
    p.add_argument("--goal_proximity_coeff", type=float, default=0.0,
                   help="Potential-based shaping: train-env reward += coeff * Δroute_completion. "
                        "Only rewards progress, never standing still (Ng & Russell 1999 result "
                        "that this preserves optimal policy). coeff=20 makes full-route "
                        "completion add +20 to return. Run #4's coeff*rc^2 shaping backfired "
                        "(agent learned to cruise safely at mid-rc). Eval env unshaped.")
    p.add_argument("--sac_hidden", type=int, default=256)
    p.add_argument("--sac_depth", type=int, default=2)

    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--n_eval_envs", type=int, default=10)
    p.add_argument("--checkpoint_freq", type=int, default=50_000)
    p.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint.pt to resume from")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"mbpo_seed{args.seed}"
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = log_dir / "checkpoints"
    eval_log_path = log_dir / "eval_log.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- training env (single) ---
    env_cfg = copy.deepcopy(ENV_CONFIG)
    env_cfg["training"] = True
    env_cfg.update(MBPO_ENV_OVERRIDES)
    env = make_env(env_cfg, seed=args.seed, use_idle_penalty=True,
                   goal_proximity_coeff=args.goal_proximity_coeff)

    # --- eval env (vec) ---
    eval_cfg = copy.deepcopy(ENV_CONFIG)
    eval_cfg["training"] = False
    eval_cfg.update(MBPO_ENV_OVERRIDES)
    eval_env = SubprocVecEnv(
        [partial(make_env, eval_cfg, seed, False) for seed in EVAL_VEC_ENV_SEEDS[: args.n_eval_envs]]
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  act_range=[{env.action_space.low.min()}, {env.action_space.high.max()}]")

    # --- dynamics ---
    dynamics = build_dynamics_model(
        obs_dim, act_dim,
        ensemble_size=args.ensemble_size,
        num_elites=args.num_elites,
        hid_size=args.model_hid_size,
        num_layers=args.model_num_layers,
        device=args.device,
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics, optim_lr=args.model_lr, weight_decay=args.model_weight_decay,
    )
    model_env = mbrl.models.ModelEnv(
        env, dynamics, metaurban_termination_fn, reward_fn=None,
        generator=torch.Generator(device=args.device),
    )

    # --- replay buffers ---
    real_buf = mbrl_rb.ReplayBuffer(
        args.real_buffer_size, (obs_dim,), (act_dim,), rng=np.random.default_rng(args.seed),
    )
    # Synthetic buffer holds model_retain_epochs passes worth of rollouts:
    # rollout_batch_size * rollout_length transitions per refill × retained epochs.
    syn_capacity = max(
        args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        args.sac_batch_size * 10,
    )
    syn_buf = mbrl_rb.ReplayBuffer(
        syn_capacity, (obs_dim,), (act_dim,), rng=np.random.default_rng(args.seed + 1),
    )

    # --- SAC agent ---
    agent = SACAgent(
        obs_dim=obs_dim, act_dim=act_dim, device=args.device,
        actor_lr=args.sac_actor_lr, critic_lr=args.sac_critic_lr, alpha_lr=args.sac_alpha_lr,
        gamma=args.sac_gamma, tau=args.sac_tau, init_temperature=args.sac_init_temp,
        hidden=args.sac_hidden, depth=args.sac_depth,
    )

    # --- resume ---
    start_step = 0
    if args.resume_from:
        p = Path(args.resume_from)
        if not p.is_file():
            raise FileNotFoundError(f"--resume_from not found: {p}")
        start_step = load_checkpoint(p, agent, dynamics, real_buf, syn_buf)
        print(f"[resume] step={start_step:,}  real_buf={len(real_buf):,}  syn_buf={len(syn_buf):,}")
        if start_step >= args.total_timesteps:
            print(f"[resume] checkpoint already at {start_step:,} ≥ target {args.total_timesteps:,} — nothing to do.")
            return

    # --- init: random-policy rollout to seed real buffer ---
    # Only pass `seed` on the very first reset — MetaUrban only accepts seeds in
    # [start_seed, start_seed + num_scenarios) = [0, 1000) with our config, and
    # its internal scenario rotation handles variety across subsequent resets.
    obs, _ = env.reset(seed=args.seed)
    if start_step < args.init_steps:
        print(f"[init] collecting {args.init_steps - start_step:,} random transitions...")
        while start_step < args.init_steps:
            a = env.action_space.sample()
            nxt, r, term, trunc, _info = env.step(a)
            real_buf.add(obs.astype(np.float32), a.astype(np.float32),
                         nxt.astype(np.float32), float(r), bool(term))
            obs = nxt if not (term or trunc) else env.reset()[0]
            start_step += 1

    # --- main loop ---
    print(f"[train] MBPO {start_step:,} → {args.total_timesteps:,} steps")
    print(f"[train] ensemble={args.ensemble_size}({args.num_elites} elites)  "
          f"rollout_len={args.rollout_length}  rollout_batch={args.rollout_batch_size}  "
          f"real_ratio={args.real_ratio}  sac_updates/step={args.sac_updates_per_step}  "
          f"goal_proximity_coeff={args.goal_proximity_coeff} (train env only)")
    t_start = time.time()
    last_model_train_step = start_step - (start_step % args.model_train_freq)
    step = start_step

    args_dict = vars(args)

    while step < args.total_timesteps:
        # 1. interact
        a = agent.actor.act(obs, deterministic=False)
        a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
        nxt, r, term, trunc, info = env.step(a)
        real_buf.add(obs.astype(np.float32), a, nxt.astype(np.float32), float(r), bool(term))
        obs = nxt if not (term or trunc) else env.reset()[0]
        step += 1

        # 2. retrain dynamics + refill synthetic buffer
        if step - last_model_train_step >= args.model_train_freq:
            last_model_train_step = step
            dynamics.update_normalizer(real_buf.get_all())
            train_it, val_it = mbrl_common.get_basic_buffer_iterators(
                real_buf,
                batch_size=args.model_batch_size,
                val_ratio=args.model_val_ratio,
                ensemble_size=args.ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,
            )
            _train_losses, _val_losses = model_trainer.train(
                train_it, val_it,
                num_epochs=args.model_train_max_epochs,
                patience=args.model_train_patience,
            )
            val_loss_str = f"{_val_losses[-1]:.4g}" if _val_losses else "n/a"
            # Clear + refill synthetic buffer. mbrl's ReplayBuffer doesn't expose
            # clear(); simplest is to reset the write pointer via re-instantiation.
            syn_buf = mbrl_rb.ReplayBuffer(
                syn_capacity, (obs_dim,), (act_dim,),
                rng=np.random.default_rng(args.seed + step),
            )
            n_added = populate_synthetic_buffer(
                model_env, real_buf, syn_buf, agent,
                rollout_length=args.rollout_length,
                rollout_batch_size=args.rollout_batch_size,
            )
            elapsed = time.time() - t_start
            print(
                f"[step {step:>7,}] dynamics retrained  "
                f"val_loss={val_loss_str}  "
                f"syn_added={n_added:,}  elapsed={elapsed/60:.1f}min"
            )

        # 3. SAC updates
        for _ in range(args.sac_updates_per_step):
            if len(real_buf) < args.sac_batch_size:
                break
            batch = sample_mixed_batch(real_buf, syn_buf, args.sac_batch_size, args.real_ratio)
            agent.update(batch)

        # 4. eval
        if step % args.eval_freq == 0:
            eval_start = time.time()
            stats = evaluate_policy(eval_env, agent, n_eval_episodes=args.n_eval_envs)
            stats["step"] = step
            stats["wall_time_s"] = time.time() - t_start
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(stats) + "\n")
            print(
                f"[eval  {step:>7,}] return={stats['mean_return']:.2f} "
                f"[{stats['q25_return']:.1f},{stats['q75_return']:.1f}]  "
                f"success={stats['success_rate']:.0%}  "
                f"len={stats['mean_length']:.0f}  "
                f"(eval took {time.time()-eval_start:.1f}s)"
            )

        # 5. checkpoint
        if step % args.checkpoint_freq == 0:
            ckpt_path = ckpt_dir / f"ckpt_step_{step}.pt"
            save_checkpoint(ckpt_path, step, agent, dynamics, real_buf, syn_buf, args_dict)
            latest = ckpt_dir / "ckpt_latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
            print(f"[ckpt  {step:>7,}] saved → {ckpt_path}")

    # final save
    ckpt_path = ckpt_dir / f"ckpt_step_{step}.pt"
    save_checkpoint(ckpt_path, step, agent, dynamics, real_buf, syn_buf, args_dict)
    print(f"[done] MBPO training finished at step {step:,}. Final ckpt → {ckpt_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
