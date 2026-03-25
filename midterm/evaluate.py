"""
Evaluate a trained RL agent (PPO or SAC) and report metrics.

Usage:
    python evaluate.py --algo ppo --model_path ./midterm_logs/PPO/ppo_seed0/best_model/best_model.zip --episodes 50
    python evaluate.py --algo sac --model_path ./midterm_logs/SAC/sac_seed0/best_model/best_model.zip --episodes 50
"""
import argparse
import copy
import json
import os

import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG

ALGO_MAP = {"ppo": PPO, "sac": SAC}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./midterm_logs/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg["use_render"] = False

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=args.seed, log_level=50, **cfg))

    algo_cls = ALGO_MAP[args.algo]
    model = algo_cls.load(args.model_path, env=env)

    episode_returns = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    out_of_road_count = 0
    route_completions = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        if info.get("arrive_dest", False) or info.get("is_success", False):
            success_count += 1
        if info.get("crash", False):
            crash_count += 1
        if info.get("out_of_road", False):
            out_of_road_count += 1
        if "route_completion" in info:
            route_completions.append(info["route_completion"])

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.episodes} | "
                  f"Return: {total_reward:.2f} | "
                  f"Running mean: {np.mean(episode_returns):.2f}")

    env.close()

    results = {
        "algo": args.algo,
        "model_path": args.model_path,
        "episodes": args.episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": success_count / args.episodes,
        "crash_rate": crash_count / args.episodes,
        "out_of_road_rate": out_of_road_count / args.episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_route_completion": float(np.mean(route_completions)) if route_completions else None,
    }

    print(f"\n{'='*60}")
    print(f"  Algorithm:        {args.algo.upper()}")
    print(f"  Mean return:      {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Success rate:     {results['success_rate']:.2%}")
    print(f"  Crash rate:       {results['crash_rate']:.2%}")
    print(f"  Out-of-road rate: {results['out_of_road_rate']:.2%}")
    print(f"  Mean ep length:   {results['mean_episode_length']:.1f}")
    if results["mean_route_completion"] is not None:
        print(f"  Route completion: {results['mean_route_completion']:.2%}")
    print(f"{'='*60}")

    out_file = os.path.join(args.output_dir, f"{args.algo}_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
