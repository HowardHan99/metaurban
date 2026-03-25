"""
Run a random agent in the MetaUrban environment and report mean performance.
The midterm requires running a random agent 3 times and reporting mean return.

Usage:
    python run_random.py --num_runs 3 --episodes_per_run 50
"""
import argparse
import copy
import json
import os

import numpy as np

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG, EVAL_VEC_ENV_SEEDS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate random agent")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--episodes_per_run", type=int, default=50, help="Episodes per run")
    parser.add_argument("--output_dir", type=str, default="./midterm_logs/Random")
    return parser.parse_args()


def run_random_agent(env_cfg, num_episodes, seed):
    cfg = copy.deepcopy(env_cfg)
    cfg["training"] = False
    cfg["use_render"] = False

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **cfg))

    episode_returns = []
    episode_lengths = []
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        # Match Stable-Baselines3 EvalCallback: it only records info["is_success"].
        if info.get("is_success", False):
            success_count += 1

    env.close()

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": success_count / num_episodes,
        "all_returns": [float(r) for r in episode_returns],
    }


def random_agent_eval_protocol_success_rate(env_cfg):
    """
    Fraction of successes over one random-policy episode per EvalCallback worker seed.
    Mirrors train_ppo / train_sac: SubprocVecEnv with start_seed in EVAL_VEC_ENV_SEEDS.
    Uses info['is_success'] only, like SB3 EvalCallback.
    """
    cfg = copy.deepcopy(env_cfg)
    cfg["training"] = False
    cfg["use_render"] = False
    ok = 0
    for start_seed in EVAL_VEC_ENV_SEEDS:
        env = SidewalkStaticMetaUrbanEnv(dict(start_seed=start_seed, log_level=50, **cfg))
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            ok += 1
        env.close()
    n = len(EVAL_VEC_ENV_SEEDS)
    return ok / n if n else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for run_idx in range(args.num_runs):
        seed = run_idx * 1000
        print(f"\n=== Random Agent Run {run_idx + 1}/{args.num_runs} (seed={seed}) ===")
        result = run_random_agent(ENV_CONFIG, args.episodes_per_run, seed)
        all_results.append(result)
        print(f"  Mean return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(f"  Mean episode length: {result['mean_length']:.1f}")

    run_means = [r["mean_return"] for r in all_results]
    overall_mean = float(np.mean(run_means))
    overall_std = float(np.std(run_means))

    print(f"\n{'='*60}")
    print(f"Overall mean return across {args.num_runs} runs: {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"Per-run means: {[f'{m:.2f}' for m in run_means]}")
    total_n = sum(len(r["all_returns"]) for r in all_results)
    overall_sr = (
        sum(r["success_rate"] * len(r["all_returns"]) for r in all_results) / total_n
        if total_n
        else 0.0
    )
    print(f"Pooled success rate (all episodes): {overall_sr:.2%}")
    print(f"{'='*60}")

    print(
        f"\n=== Eval-matched random baseline (start_seeds {EVAL_VEC_ENV_SEEDS[0]}.."
        f"{EVAL_VEC_ENV_SEEDS[-1]}, same as training EvalCallback) ==="
    )
    eval_matched_sr = random_agent_eval_protocol_success_rate(ENV_CONFIG)
    print(f"  Success rate: {eval_matched_sr:.2%}")

    summary = {
        "num_runs": args.num_runs,
        "episodes_per_run": args.episodes_per_run,
        "overall_mean_return": overall_mean,
        "overall_std_return": overall_std,
        "overall_success_rate": overall_sr,
        "eval_matched_random_success_rate": eval_matched_sr,
        "eval_vec_env_seeds": list(EVAL_VEC_ENV_SEEDS),
        "per_run_results": all_results,
    }
    output_path = os.path.join(args.output_dir, "random_agent_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
