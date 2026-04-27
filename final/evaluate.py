"""
Evaluate an MBPO checkpoint and report metrics (mirrors midterm/evaluate.py
output format so the final report can compare side-by-side).

Usage:
    python evaluate.py --checkpoint ./final_logs/MBPO/mbpo_seed0/checkpoints/ckpt_latest.pt \
                       --episodes 50
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from metaurban import SidewalkStaticMetaUrbanEnv

from env_config import ENV_CONFIG, MBPO_ENV_OVERRIDES
from sac_agent import SACAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an MBPO checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt_*.pt from train_mbpo.py")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--output_dir", type=str, default="./final_logs/eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg["use_render"] = False
    cfg.update(MBPO_ENV_OVERRIDES)

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=args.seed, log_level=50, **cfg))
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    # Load only the SAC piece of the checkpoint (we don't need the dynamics model
    # for evaluation — the saved SAC actor is a fully-trained policy on its own).
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sac_sd = ckpt["sac"]
    meta = sac_sd.get("meta", {})
    agent = SACAgent(
        obs_dim=meta.get("obs_dim", obs_dim),
        act_dim=meta.get("act_dim", act_dim),
        device=args.device,
        gamma=meta.get("gamma", 0.99),
        tau=meta.get("tau", 0.005),
    )
    agent.load_state_dict(sac_sd)

    returns, lengths = [], []
    success = crash = out_of_road = 0
    route_completions = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r, steps, done = 0.0, 0, False
        info = {}
        while not done:
            a = agent.actor.act(obs, deterministic=args.deterministic)
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            total_r += float(r)
            steps += 1
            done = term or trunc
        returns.append(total_r)
        lengths.append(steps)
        if info.get("arrive_dest", False) or info.get("is_success", False):
            success += 1
        if info.get("crash", False):
            crash += 1
        if info.get("out_of_road", False):
            out_of_road += 1
        if "route_completion" in info:
            route_completions.append(info["route_completion"])
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}/{args.episodes} | R={total_r:.2f} | running mean={np.mean(returns):.2f}")

    env.close()

    results = {
        "algo": "mbpo",
        "checkpoint": args.checkpoint,
        "step": int(ckpt.get("step", -1)),
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": success / args.episodes,
        "crash_rate": crash / args.episodes,
        "out_of_road_rate": out_of_road / args.episodes,
        "mean_episode_length": float(np.mean(lengths)),
        "mean_route_completion": float(np.mean(route_completions)) if route_completions else None,
    }

    print(f"\n{'='*60}")
    print(f"  Algorithm:        MBPO")
    print(f"  Checkpoint step:  {results['step']:,}")
    print(f"  Mean return:      {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Success rate:     {results['success_rate']:.2%}")
    print(f"  Crash rate:       {results['crash_rate']:.2%}")
    print(f"  Out-of-road rate: {results['out_of_road_rate']:.2%}")
    print(f"  Mean ep length:   {results['mean_episode_length']:.1f}")
    if results["mean_route_completion"] is not None:
        print(f"  Route completion: {results['mean_route_completion']:.2%}")
    print(f"{'='*60}")

    out_file = Path(args.output_dir) / f"mbpo_eval_step_{results['step']}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
