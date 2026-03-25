"""
Plot learning curves: PPO and SAC on separate subplots, with random mean-return
baseline on the return figure only. Reads SB3 EvalCallback logs (evaluations.npz)
and random agent results.

If evaluations.npz contains a ``successes`` array (SB3 EvalCallback default when
info includes ``is_success``), also writes mean eval success rate vs steps
(no random baseline on that figure).

Usage:
    python plot_results.py
    python plot_results.py --ppo_log ./midterm_logs/PPO/ppo_seed1/eval_logs \
                           --sac_log ./midterm_logs/SAC/sac_seed0/eval_logs \
                           --output ./midterm_logs/learning_curves.png \
                           --output_success ./midterm_logs/success_rates.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_evaluations_npz(path: str) -> str:
    if path.endswith(".npz"):
        return path
    return os.path.join(path, "evaluations.npz")


def load_sb3_eval_npz(npz_path: str):
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64).reshape(-1)
    results = np.asarray(data["results"], dtype=np.float64)
    if results.ndim == 1:
        results = results.reshape(1, -1)
    successes = None
    if "successes" in data.files:
        successes = np.asarray(data["successes"])
        if successes.dtype == object:
            successes = np.stack([np.atleast_1d(x).astype(bool) for x in successes], axis=0)
        else:
            successes = successes.astype(bool)
        if successes.ndim == 1:
            successes = successes.reshape(1, -1)
    return timesteps, results, successes


def load_sb3_eval_log(log_dir_or_npz):
    npz_path = resolve_evaluations_npz(log_dir_or_npz)
    loaded = load_sb3_eval_npz(npz_path)
    if loaded is None:
        print(f"WARNING: {npz_path} not found")
        return None, None, None
    timesteps, results, _successes = loaded
    mean_returns = np.mean(results, axis=1)
    q25 = np.percentile(results, 25, axis=1)
    q75 = np.percentile(results, 75, axis=1)
    return timesteps, mean_returns, (q25, q75)


def load_success_eval_log(log_dir_or_npz):
    """Mean success rate per eval (fraction of successful eval episodes)."""
    npz_path = resolve_evaluations_npz(log_dir_or_npz)
    loaded = load_sb3_eval_npz(npz_path)
    if loaded is None:
        print(f"WARNING: {npz_path} not found")
        return None, None, None
    timesteps, _results, successes = loaded
    if successes is None:
        print(f"WARNING: no 'successes' key in {npz_path} — skipping success plot for this run")
        return None, None, None
    s = successes.astype(np.float64)
    mean_sr = np.mean(s, axis=1)
    n = s.shape[1]
    # Normal approx. SE for binomial proportion (eval uses n parallel eval episodes).
    var = np.clip(mean_sr * (1.0 - mean_sr), 1e-8, None)
    se = np.sqrt(var / n)
    lo = np.clip(mean_sr - 1.96 * se, 0.0, 1.0)
    hi = np.clip(mean_sr + 1.96 * se, 0.0, 1.0)
    return timesteps, mean_sr, (lo, hi)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot midterm learning curves")
    parser.add_argument("--ppo_log", type=str, default="./midterm_logs/PPO/ppo_seed1/eval_logs")
    parser.add_argument("--sac_log", type=str, default="./midterm_logs/SAC/sac_seed0/eval_logs")
    parser.add_argument("--random_log", type=str, default="./midterm_logs/Random/random_agent_results.json")
    parser.add_argument("--output", type=str, default="./midterm_logs/learning_curves.png")
    parser.add_argument(
        "--output_success",
        type=str,
        default="./midterm_logs/success_rates.png",
        help="Path for PPO/SAC eval success-rate figure (skipped if logs lack 'successes').",
    )
    return parser.parse_args()


def plot_curve(ax, ts, mean, band, color, label):
    ax.plot(ts, mean, color=color, linewidth=2, label=label)
    if band is not None:
        q25, q75 = band
        ax.fill_between(ts, q25, q75, color=color, alpha=0.15)


def add_random_baseline(ax, random_mean):
    if random_mean is not None:
        ax.axhline(y=random_mean, color="gray", linestyle="--", linewidth=2, label="Random Agent")


def style_ax(ax, title, ylabel="Mean Return"):
    ax.set_xlabel("Training Environment Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    random_mean = None
    if os.path.exists(args.random_log):
        with open(args.random_log) as f:
            random_data = json.load(f)
        random_mean = random_data["overall_mean_return"]
        print(f"Random agent mean return: {random_mean:.2f}")
    else:
        print(f"WARNING: Random agent log not found at {args.random_log}")

    ppo_result = load_sb3_eval_log(args.ppo_log)
    sac_result = load_sb3_eval_log(args.sac_log)
    ppo_succ = load_success_eval_log(args.ppo_log)
    sac_succ = load_success_eval_log(args.sac_log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left subplot: PPO ---
    if ppo_result[0] is not None:
        ts, mean, band = ppo_result
        plot_curve(ax1, ts, mean, band, "tab:blue", "PPO (On-Policy)")
        peak_idx = np.argmax(mean)
        ax1.axvline(x=ts[peak_idx], color="tab:blue", linestyle=":", alpha=0.5)
        ax1.annotate(f"peak: {mean[peak_idx]:.1f}\n@ {ts[peak_idx]/1e6:.2f}M steps",
                     xy=(ts[peak_idx], mean[peak_idx]),
                     xytext=(15, -10), textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.7),
                     color="tab:blue")
        print(f"PPO: {len(ts)} eval points, peak={mean[peak_idx]:.2f} at {ts[peak_idx]}, final={mean[-1]:.2f}")
    else:
        print("PPO eval log not found — skipping")
    add_random_baseline(ax1, random_mean)
    style_ax(ax1, "PPO (On-Policy) Learning Curve", ylabel="Mean Return")

    # --- Right subplot: SAC ---
    if sac_result[0] is not None:
        ts, mean, band = sac_result
        plot_curve(ax2, ts, mean, band, "tab:orange", "SAC (Off-Policy)")
        peak_idx = np.argmax(mean)
        ax2.axvline(x=ts[peak_idx], color="tab:orange", linestyle=":", alpha=0.5)
        ax2.annotate(f"peak: {mean[peak_idx]:.1f}\n@ {ts[peak_idx]/1e6:.2f}M steps",
                     xy=(ts[peak_idx], mean[peak_idx]),
                     xytext=(15, -10), textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="tab:orange", alpha=0.7),
                     color="tab:orange")
        print(f"SAC: {len(ts)} eval points, peak={mean[peak_idx]:.2f} at {ts[peak_idx]}, final={mean[-1]:.2f}")
    else:
        print("SAC eval log not found — skipping")
    add_random_baseline(ax2, random_mean)
    style_ax(ax2, "SAC (Off-Policy) Learning Curve", ylabel="Mean Return")

    fig.suptitle("MetaUrban PointNav — Learning Curves", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")

    # --- Success rate figure (eval episodes per checkpoint) ---
    if ppo_succ[0] is not None or sac_succ[0] is not None:
        fig2, (sx1, sx2) = plt.subplots(1, 2, figsize=(16, 6))
        if ppo_succ[0] is not None:
            ts, mean, band = ppo_succ
            plot_curve(sx1, ts, mean, band, "tab:blue", "PPO (On-Policy)")
            peak_idx = np.argmax(mean)
            print(
                f"PPO success: {len(ts)} eval points, peak={mean[peak_idx]:.2%} at {ts[peak_idx]}, "
                f"final={mean[-1]:.2%}"
            )
        else:
            print("PPO success curve skipped (no data)")
            sx1.text(0.5, 0.5, "No success data", ha="center", va="center", transform=sx1.transAxes)
        style_ax(sx1, "PPO (On-Policy) Eval Success Rate", ylabel="Success rate")
        sx1.set_ylim(-0.02, 1.02)

        if sac_succ[0] is not None:
            ts, mean, band = sac_succ
            plot_curve(sx2, ts, mean, band, "tab:orange", "SAC (Off-Policy)")
            peak_idx = np.argmax(mean)
            print(
                f"SAC success: {len(ts)} eval points, peak={mean[peak_idx]:.2%} at {ts[peak_idx]}, "
                f"final={mean[-1]:.2%}"
            )
        else:
            print("SAC success curve skipped (no data)")
            sx2.text(0.5, 0.5, "No success data", ha="center", va="center", transform=sx2.transAxes)
        style_ax(sx2, "SAC (Off-Policy) Eval Success Rate", ylabel="Success rate")
        sx2.set_ylim(-0.02, 1.02)

        fig2.suptitle(
            "MetaUrban PointNav — Eval Success Rate (per EvalCallback checkpoint)",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        out_s = args.output_success
        sdir = os.path.dirname(out_s)
        if sdir:
            os.makedirs(sdir, exist_ok=True)
        plt.savefig(out_s, dpi=150, bbox_inches="tight")
        print(f"Success-rate plot saved to {out_s}")
    else:
        print("No 'successes' in PPO/SAC eval logs — skipping success-rate figure.")


if __name__ == "__main__":
    main()
