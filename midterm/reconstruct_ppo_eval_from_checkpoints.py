"""
Rebuild eval curve points from PPO checkpoint .zip files (manual / advanced).

For the usual “resume overwrote evaluations.npz” case, run:

  python restore_ppo_eval_logs.py --run_dir ./midterm_logs/PPO/ppo_seed2

This file is for custom --max_step / --output without rewriting evaluations.npz.
"""
import argparse
import copy
import os
import re
import warnings
from functools import partial

import numpy as np

warnings.filterwarnings("ignore", message=r".*weights_only.*", category=FutureWarning)
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG

CKPT_RE = re.compile(r"ppo_(\d+)_steps\.zip$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Reconstruct eval curve from PPO checkpoints")
    p.add_argument(
        "--checkpoints_dir", type=str,
        default="./midterm_logs/PPO/ppo_seed2/checkpoints",
    )
    p.add_argument(
        "--output", type=str,
        default="./midterm_logs/PPO/ppo_seed2/eval_logs/from_checkpoints.npz",
    )
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument(
        "--max_step", type=int, default=None,
        help="Only use checkpoints with timestep <= this (e.g. 1200000 to match resume start).",
    )
    p.add_argument(
        "--n_envs", type=int, default=10,
        help="Eval parallel envs (seeds 950..950+n-1); match train_ppo eval setup.",
    )
    return p.parse_args()


def make_env(env_cfg, seed):
    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **env_cfg))
    env = Monitor(env)
    return env


def list_checkpoints(checkpoints_dir):
    out = []
    for name in os.listdir(checkpoints_dir):
        m = CKPT_RE.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(checkpoints_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def main():
    args = parse_args()
    pairs = list_checkpoints(args.checkpoints_dir)
    if args.max_step is not None:
        pairs = [(s, p) for s, p in pairs if s <= args.max_step]
    if not pairs:
        print(f"No matching checkpoints in {args.checkpoints_dir}")
        return

    eval_cfg = copy.deepcopy(ENV_CONFIG)
    eval_cfg["training"] = False
    eval_env = SubprocVecEnv(
        [partial(make_env, eval_cfg, seed) for seed in range(950, 950 + args.n_envs)]
    )

    timesteps = []
    results = []
    ep_lengths = []

    try:
        for step, path in pairs:
            print(f"Evaluating {path} ({step:,} steps)...", flush=True)
            model = PPO.load(path, env=eval_env)
            eval_env._reset_seeds()
            eval_env._reset_options()
            print(
                "  → MetaUrban eval in progress (can take several minutes; no bar)...",
                flush=True,
            )
            rew, lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
            )
            print(
                f"  → mean return={float(np.mean(rew)):.2f} ({args.n_eval_episodes} eps)",
                flush=True,
            )
            timesteps.append(step)
            results.append(list(rew))
            ep_lengths.append(list(lengths))
    finally:
        try:
            eval_env.close()
        except (BrokenPipeError, EOFError, OSError):
            pass

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(
        args.output,
        timesteps=np.array(timesteps, dtype=np.int64),
        results=np.array(results, dtype=np.float64),
        ep_lengths=np.array(ep_lengths, dtype=np.float64),
    )
    print(f"Wrote {len(timesteps)} eval points to {args.output}")


if __name__ == "__main__":
    main()
