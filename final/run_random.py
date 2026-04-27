"""
Random-agent baseline under the *same* reward config MBPO uses
(MBPO_ENV_OVERRIDES). This matters: the midterm random baseline used the base
ENV_CONFIG with no_negative_reward=True, so its returns aren't directly
comparable to MBPO's returns. Running this script gives a baseline that IS
comparable to MBPO, so the eval curve in plot_results.py has a meaningful floor.

Usage:
    python run_random.py --num_runs 3 --episodes_per_run 50
"""
import argparse
import copy
import json
import os

import numpy as np

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG, MBPO_ENV_OVERRIDES, EVAL_VEC_ENV_SEEDS


def parse_args():
    p = argparse.ArgumentParser(description="Random-agent baseline under MBPO reward config")
    p.add_argument("--num_runs", type=int, default=3)
    p.add_argument("--episodes_per_run", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./final_logs/Random")
    return p.parse_args()


def run_once(env_cfg: dict, num_episodes: int, seed: int) -> dict:
    cfg = copy.deepcopy(env_cfg)
    cfg["training"] = False
    cfg["use_render"] = False
    cfg.update(MBPO_ENV_OVERRIDES)

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **cfg))
    returns, lengths = [], []
    successes = 0

    for ep in range(num_episodes):
        _, _ = env.reset(seed=seed + ep)
        total, steps, done = 0.0, 0, False
        info = {}
        while not done:
            a = env.action_space.sample()
            _, r, term, trunc, info = env.step(a)
            total += float(r)
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
        if info.get("arrive_dest", False) or info.get("is_success", False):
            successes += 1

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": successes / num_episodes,
        "all_returns": [float(x) for x in returns],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    for i in range(args.num_runs):
        seed = i * 1000
        print(f"\n=== Random run {i+1}/{args.num_runs} (seed={seed}) ===")
        res = run_once(ENV_CONFIG, args.episodes_per_run, seed)
        all_results.append(res)
        print(f"  mean return = {res['mean_return']:.2f} ± {res['std_return']:.2f}  "
              f"success = {res['success_rate']:.2%}")

    overall_mean = float(np.mean([r["mean_return"] for r in all_results]))
    overall_std = float(np.std([r["mean_return"] for r in all_results]))
    n_total = sum(len(r["all_returns"]) for r in all_results)
    overall_sr = (sum(r["success_rate"] * len(r["all_returns"]) for r in all_results) / n_total
                  if n_total else 0.0)

    print(f"\n{'='*60}\n"
          f"Overall mean return across {args.num_runs} runs: {overall_mean:.2f} ± {overall_std:.2f}\n"
          f"Pooled success rate: {overall_sr:.2%}\n"
          f"{'='*60}")

    summary = {
        "num_runs": args.num_runs,
        "episodes_per_run": args.episodes_per_run,
        "reward_overrides": MBPO_ENV_OVERRIDES,
        "overall_mean_return": overall_mean,
        "overall_std_return": overall_std,
        "overall_success_rate": overall_sr,
        "per_run_results": all_results,
    }
    out = os.path.join(args.output_dir, "random_agent_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
