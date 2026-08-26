"""
Run a trained MBPO SAC policy in the simulator with a live 3D window.
Mirrors midterm/run_agent_live.py but loads our custom SACAgent checkpoint
(the .pt file written by final/train_mbpo.py) instead of an SB3 .zip.

Usage:
    python run_agent_live.py --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt
    python run_agent_live.py --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_step_500000.pt \
                             --episodes 5 --seed 42
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE

from env_config import ENV_CONFIG, MBPO_ENV_OVERRIDES
from sac_agent import SACAgent


def parse_args():
    p = argparse.ArgumentParser(description="Run MBPO policy live in the simulator")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt_*.pt from train_mbpo.py")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Starting scenario seed (must be in [0, 1000))")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from the policy instead of the deterministic mean")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg["use_render"] = True
    cfg["window_size"] = (960, 960)
    cfg["vehicle_config"]["show_dest_mark"] = True
    cfg["vehicle_config"]["show_line_to_navi_mark"] = True
    cfg.update(MBPO_ENV_OVERRIDES)

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=args.seed, log_level=50, **cfg))

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sac_sd = ckpt["sac"]
    meta = sac_sd.get("meta", {})
    obs_dim = meta.get("obs_dim") or int(np.prod(env.observation_space.shape))
    act_dim = meta.get("act_dim") or int(np.prod(env.action_space.shape))
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, device=args.device)
    agent.load_state_dict(sac_sd)

    print(HELP_MESSAGE)
    step = ckpt.get("step", "?")
    print(f"\nRunning MBPO checkpoint (step={step}) from {args.checkpoint} for {args.episodes} episode(s).")
    print("Press H for help, R to reset scenario, Esc to quit.\n")

    for ep in range(args.episodes):
        # First reset may pass a seed; subsequent ones must not (MetaUrban only
        # accepts seeds in [start_seed, start_seed+num_scenarios)).
        if ep == 0:
            obs, _ = env.reset(seed=args.seed)
        else:
            obs, _ = env.reset()
        total, steps, done, info = 0.0, 0, False, {}
        while not done:
            a = agent.actor.act(obs, deterministic=not args.stochastic)
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            steps += 1
            done = term or trunc
            env.render()
        print(f"Episode {ep+1}/{args.episodes}: return={total:.2f}  "
              f"length={steps}  "
              f"success={info.get('arrive_dest') or info.get('is_success', False)}")

    env.close()


if __name__ == "__main__":
    main()
