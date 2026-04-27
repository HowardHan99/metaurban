"""
Plot paper-style comparison figures for PPO, SAC, and MBPO.

The learning-curve figure uses one subplot per method and includes the random
baseline on each panel. The success-rate figure mirrors that layout.

Usage:
    python plot_results.py --include_midterm
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mbpo_jsonl(path: str):
    if not os.path.isfile(path):
        print(f"[warn] MBPO log not found: {path}")
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    rows.sort(key=lambda r: r["step"])
    ts = np.asarray([r["step"] for r in rows], dtype=np.int64)
    mean = np.asarray([r["mean_return"] for r in rows], dtype=np.float64)
    q25 = np.asarray([r.get("q25_return", r["mean_return"]) for r in rows], dtype=np.float64)
    q75 = np.asarray([r.get("q75_return", r["mean_return"]) for r in rows], dtype=np.float64)
    succ = np.asarray([r.get("success_rate", np.nan) for r in rows], dtype=np.float64)
    return ts, mean, (q25, q75), succ


def load_sb3_eval_npz(path: str):
    """Load midterm SB3 evaluations.npz → (ts, mean, (q25,q75), succ_rate)."""
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    ts = np.asarray(d["timesteps"], dtype=np.int64).reshape(-1)
    results = np.asarray(d["results"], dtype=np.float64)
    if results.ndim == 1:
        results = results.reshape(1, -1)
    mean = np.mean(results, axis=1)
    q25 = np.percentile(results, 25, axis=1)
    q75 = np.percentile(results, 75, axis=1)
    succ_rate = None
    if "successes" in d.files:
        s = np.asarray(d["successes"])
        if s.dtype == object:
            s = np.stack([np.atleast_1d(x).astype(bool) for x in s], axis=0)
        else:
            s = s.astype(bool)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        succ_rate = s.astype(np.float64).mean(axis=1)
    return ts, mean, (q25, q75), succ_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mbpo_log", type=str, default="./final_logs/MBPO/mbpo_seed0/eval_log.jsonl")
    p.add_argument("--random_log", type=str, default="./final_logs/Random/random_agent_results.json")
    p.add_argument("--include_midterm", action="store_true",
                   help="Overlay midterm PPO and SAC eval curves for comparison")
    p.add_argument("--ppo_log", type=str,
                   default="./midterm/midterm_logs/PPO/ppo_seed1/eval_logs/evaluations.npz")
    p.add_argument("--sac_log", type=str,
                   default="./midterm/midterm_logs/SAC/sac_seed0/eval_logs/evaluations.npz")
    p.add_argument("--output", type=str, default="./final_logs/learning_curves.png")
    p.add_argument("--output_success", type=str, default="./final_logs/success_rates.png")
    return p.parse_args()


def _plot_band(ax, ts, mean, band, color, label):
    ax.plot(ts, mean, color=color, linewidth=2, label=label)
    if band is not None:
        q25, q75 = band
        ax.fill_between(ts, q25, q75, color=color, alpha=0.15)


def _style_ax(ax, title, ylabel):
    ax.set_xlabel("Training Environment Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)


def _annotate_peak(ax, ts, mean, color):
    peak_idx = int(np.argmax(mean))
    peak_x = ts[peak_idx]
    peak_y = mean[peak_idx]
    ax.axvline(x=peak_x, color=color, linestyle=":", alpha=0.5)
    ax.annotate(
        f"peak: {peak_y:.1f}\n@ {peak_x / 1e6:.2f}M steps",
        xy=(peak_x, peak_y),
        xytext=(15, -10),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
        color=color,
    )


def main():
    args = parse_args()
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    random_mean = None
    if os.path.exists(args.random_log):
        with open(args.random_log) as f:
            random_mean = json.load(f).get("overall_mean_return")

    mbpo = load_mbpo_jsonl(args.mbpo_log)
    ppo = load_sb3_eval_npz(args.ppo_log) if args.include_midterm else None
    sac = load_sb3_eval_npz(args.sac_log) if args.include_midterm else None

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    if ppo is not None:
        ts, mean, band, _ = ppo
        _plot_band(axes[0], ts, mean, band, "tab:blue", "PPO (On-Policy)")
        _annotate_peak(axes[0], ts, mean, "tab:blue")
        print(f"PPO: {len(ts)} eval points, peak={mean.max():.2f}, final={mean[-1]:.2f}")
    if random_mean is not None:
        axes[0].axhline(y=random_mean, color="gray", linestyle="--", linewidth=2, label="Random Agent")
    _style_ax(axes[0], "PPO (On-Policy) Learning Curve", "Mean Return")

    if sac is not None:
        ts, mean, band, _ = sac
        _plot_band(axes[1], ts, mean, band, "tab:orange", "SAC (Off-Policy)")
        _annotate_peak(axes[1], ts, mean, "tab:orange")
        print(f"SAC: {len(ts)} eval points, peak={mean.max():.2f}, final={mean[-1]:.2f}")
    if random_mean is not None:
        axes[1].axhline(y=random_mean, color="gray", linestyle="--", linewidth=2, label="Random Agent")
    _style_ax(axes[1], "SAC (Off-Policy) Learning Curve", "Mean Return")

    if mbpo is not None:
        ts, mean, band, _ = mbpo
        _plot_band(axes[2], ts, mean, band, "tab:green", "MBPO (Model-Based)")
        _annotate_peak(axes[2], ts, mean, "tab:green")
        print(f"MBPO: {len(ts)} eval points, peak={mean.max():.2f}, final={mean[-1]:.2f}")
    if random_mean is not None:
        axes[2].axhline(y=random_mean, color="gray", linestyle="--", linewidth=2, label="Random Agent")
    _style_ax(axes[2], "MBPO (Model-Based) Learning Curve", "Mean Return")

    fig.suptitle("MetaUrban PointNav — Learning Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")

    # --- success-rate fig ---
    any_succ = (mbpo and mbpo[3] is not None and not np.all(np.isnan(mbpo[3]))) or \
               (ppo and ppo[3] is not None) or (sac and sac[3] is not None)
    if any_succ:
        fig2, sxs = plt.subplots(1, 3, figsize=(24, 6))

        if ppo is not None and ppo[3] is not None:
            sxs[0].plot(ppo[0], ppo[3], color="tab:blue", linewidth=2, label="PPO (On-Policy)")
        _style_ax(sxs[0], "PPO (On-Policy) Eval Success Rate", "Success rate")
        sxs[0].set_ylim(-0.02, 1.02)

        if sac is not None and sac[3] is not None:
            sxs[1].plot(sac[0], sac[3], color="tab:orange", linewidth=2, label="SAC (Off-Policy)")
        _style_ax(sxs[1], "SAC (Off-Policy) Eval Success Rate", "Success rate")
        sxs[1].set_ylim(-0.02, 1.02)

        if mbpo is not None and mbpo[3] is not None and not np.all(np.isnan(mbpo[3])):
            sxs[2].plot(mbpo[0], mbpo[3], color="tab:green", linewidth=2, label="MBPO (Model-Based)")
        _style_ax(sxs[2], "MBPO (Model-Based) Eval Success Rate", "Success rate")
        sxs[2].set_ylim(-0.02, 1.02)

        fig2.suptitle(
            "MetaUrban PointNav — Eval Success Rate (per Eval checkpoint)",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(args.output_success, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.output_success}")


if __name__ == "__main__":
    main()
