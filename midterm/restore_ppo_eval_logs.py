#!/usr/bin/env python3
"""
Restore a full PPO eval curve after resume overwrote evaluations.npz.

Finds checkpoints with training step strictly before a *cutoff*, re-evaluates
those policies, merges with the existing log, and writes eval_logs/evaluations.npz.

Cutoff = min(first timestep in current evaluations.npz, --only_before_timestep) when
both apply. Use --only_before_timestep 1200000 if your npz starts at 3M but you only
want the pre-1.2M curve filled from checkpoints (avoids evaluating dozens of zips).

Usage (from midterm/):

    python restore_ppo_eval_logs.py --run_dir ./midterm_logs/PPO/ppo_seed2
    python restore_ppo_eval_logs.py --run_dir ./midterm_logs/PPO/ppo_seed2 \\
        --only_before_timestep 1200000

If evaluations.npz is missing, all checkpoints below --only_before_timestep (or all
if unset) are evaluated.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import warnings

# One Gym import warning per process is enough; SubprocVecEnv spawns many workers.
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
from datetime import datetime
from functools import partial

import numpy as np

# PyTorch/SB3 FutureWarning on torch.load(weights_only=...) — not actionable for local checkpoints.
warnings.filterwarnings(
    "ignore",
    message=r".*weights_only.*",
    category=FutureWarning,
)
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG

CKPT_RE = re.compile(r"ppo_(\d+)_steps\.zip$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Restore merged evaluations.npz from checkpoints + current eval log"
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default="./midterm_logs/PPO/ppo_seed2",
        help="PPO run folder (contains checkpoints/ and eval_logs/)",
    )
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument(
        "--n_envs",
        type=int,
        default=10,
        help="Eval VecEnv size; seeds 950..950+n-1 (same as train_ppo.py)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which checkpoints would be evaluated; do not write files",
    )
    p.add_argument(
        "--force_all_checkpoints",
        action="store_true",
        help="Evaluate every checkpoint (still capped by --only_before_timestep if set), merge with current log",
    )
    p.add_argument(
        "--only_before_timestep",
        type=int,
        default=None,
        help="Only evaluate checkpoints with training step < this (e.g. 1200000). "
        "If set with an existing npz, cutoff = min(first npz timestep, this). "
        "Use when npz starts very late (e.g. 3M) but you only need early segment restored.",
    )
    return p.parse_args()


def make_env(env_cfg, seed):
    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **env_cfg))
    env = Monitor(env)
    return env


def list_checkpoints(checkpoints_dir: str) -> list[tuple[int, str]]:
    out = []
    if not os.path.isdir(checkpoints_dir):
        return out
    for name in os.listdir(checkpoints_dir):
        m = CKPT_RE.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(checkpoints_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def load_eval_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)
    ts = np.asarray(data["timesteps"], dtype=np.int64).reshape(-1)
    res = np.asarray(data["results"], dtype=np.float64)
    if res.ndim == 1:
        res = res.reshape(1, -1)
    el = np.asarray(data["ep_lengths"], dtype=np.float64)
    if el.ndim == 1:
        el = el.reshape(1, -1)
    if len(ts) != len(res) or len(ts) != len(el):
        return None
    return ts, res, el


def merge_eval_arrays(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (ts, res, el), sort by ts, drop duplicate timesteps keeping the last row."""
    ts_cat = np.concatenate([p[0] for p in parts])
    res_cat = np.vstack([p[1] for p in parts])
    el_cat = np.vstack([p[2] for p in parts])
    order = np.argsort(ts_cat)
    ts_s, res_s, el_s = ts_cat[order], res_cat[order], el_cat[order]
    keep = np.zeros(len(ts_s), dtype=bool)
    seen: set[int] = set()
    for i in range(len(ts_s) - 1, -1, -1):
        t = int(ts_s[i])
        if t in seen:
            continue
        seen.add(t)
        keep[i] = True
    return ts_s[keep], res_s[keep], el_s[keep]


def evaluate_checkpoints(
    pairs: list[tuple[int, str]],
    n_eval_episodes: int,
    n_envs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_cfg = copy.deepcopy(ENV_CONFIG)
    eval_cfg["training"] = False
    eval_env = SubprocVecEnv(
        [partial(make_env, eval_cfg, seed) for seed in range(950, 950 + n_envs)]
    )
    timesteps: list[int] = []
    results: list[list[float]] = []
    ep_lengths: list[list[float]] = []
    try:
        for step, path in pairs:
            print(f"Evaluating {os.path.basename(path)} ({step:,} env steps)...", flush=True)
            model = PPO.load(path, env=eval_env)
            # PPO._setup_model calls set_random_seed → env.seed(0) → next reset uses
            # seeds 0,1,2,... MetaUrban uses reset(seed) as *scenario index* (must be in
            # [start_index, start_index+num_scenarios)), so clear pending SB3 seeds.
            eval_env._reset_seeds()
            eval_env._reset_options()
            print(
                "  → running eval in MetaUrban (no progress bar; often several minutes per checkpoint)...",
                flush=True,
            )
            rew, lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
            )
            mean_r = float(np.mean(rew))
            print(f"  → done. mean return={mean_r:.2f} over {n_eval_episodes} episodes", flush=True)
            timesteps.append(step)
            results.append(list(map(float, rew)))
            ep_lengths.append(list(map(float, lengths)))
    finally:
        try:
            eval_env.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
    return (
        np.array(timesteps, dtype=np.int64),
        np.array(results, dtype=np.float64),
        np.array(ep_lengths, dtype=np.float64),
    )


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    eval_dir = os.path.join(run_dir, "eval_logs")
    target_npz = os.path.join(eval_dir, "evaluations.npz")

    all_ckpt = list_checkpoints(ckpt_dir)
    if not all_ckpt:
        print(f"No ppo_*_steps.zip checkpoints under {ckpt_dir}")
        return 1

    current = load_eval_npz(target_npz)
    first_t = int(np.min(current[0])) if current is not None else None
    if args.only_before_timestep is not None and first_t is not None:
        cutoff = min(first_t, args.only_before_timestep)
        print(
            f"Cutoff timestep: min(npz first={first_t:,}, --only_before_timestep={args.only_before_timestep:,}) "
            f"= {cutoff:,}"
        )
    elif args.only_before_timestep is not None:
        cutoff = args.only_before_timestep
        print(f"Cutoff timestep: --only_before_timestep = {cutoff:,}")
    elif first_t is not None:
        cutoff = first_t
        print(f"Cutoff timestep: first row in evaluations.npz = {cutoff:,}")
    else:
        cutoff = None
        print("Cutoff: none (all checkpoints)")

    if args.force_all_checkpoints:
        to_eval = [(s, p) for s, p in all_ckpt if cutoff is None or s < cutoff]
        print(
            f"Mode: --force_all_checkpoints — {len(to_eval)} checkpoint(s) with step < cutoff "
            f"(or all if no cutoff)"
        )
    elif cutoff is None:
        to_eval = all_ckpt
        print("No existing evaluations.npz and no --only_before_timestep — evaluating all checkpoints")
    else:
        to_eval = [(s, p) for s, p in all_ckpt if s < cutoff]
        print(f"Reconstructing {len(to_eval)} checkpoint(s) with step < {cutoff:,}")

    if not to_eval:
        print("Nothing to restore (no checkpoints before your current eval range).")
        return 0

    if args.dry_run:
        for s, p in to_eval:
            print(f"  would eval: {s:,}  {p}")
        return 0

    os.makedirs(eval_dir, exist_ok=True)
    if current is not None and os.path.isfile(target_npz):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(eval_dir, f"evaluations_backup_{stamp}.npz")
        shutil.copy2(target_npz, backup)
        print(f"Backed up current log to {backup}")

    early_ts, early_res, early_el = evaluate_checkpoints(
        to_eval, args.n_eval_episodes, args.n_envs
    )

    # Sidecar file: never written by SB3 during training — plot_results merges this so the
    # pre-resume segment survives even if evaluations.npz is overwritten later.
    reco_path = os.path.join(eval_dir, "checkpoint_reconstruction.npz")
    np.savez(
        reco_path,
        timesteps=early_ts,
        results=early_res,
        ep_lengths=early_el,
    )
    print(
        f"Saved checkpoint-only segment for plotting ({len(early_ts)} points) → {reco_path}",
        flush=True,
    )

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [(early_ts, early_res, early_el)]
    if current is not None:
        parts.append(current)

    ts_m, res_m, el_m = merge_eval_arrays(parts)
    print(
        f"Merged eval timestep range: {int(ts_m.min()):,} … {int(ts_m.max()):,} ({len(ts_m)} points)",
        flush=True,
    )
    if len(early_ts) > 0 and int(ts_m.min()) > int(early_ts.min()):
        print(
            "WARNING: merged evaluations.npz dropped the checkpoint segment (unexpected). "
            "Use checkpoint_reconstruction.npz via plot_results (auto-merged) for the full curve.",
            flush=True,
        )
    np.savez(
        target_npz,
        timesteps=ts_m,
        results=res_m,
        ep_lengths=el_m,
    )
    print(f"Wrote merged eval log: {len(ts_m)} points → {target_npz}")
    print(f"Next: cd midterm && python plot_results.py   (includes {reco_path} if present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
