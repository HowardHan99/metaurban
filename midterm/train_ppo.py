"""
Train PPO (on-policy policy gradient) in MetaUrban PointNav environment.

PPO is on-policy: it collects fresh rollouts with the current policy, performs
n_epochs gradient updates on that batch, then discards the data.  Each PPO
update uses n_steps * n_envs = 200 * 20 = 4000 fresh transitions (matches reference).

Timestep budget rationale:
  - The original RL/PointNav/train_ppo.py targets 1e8 steps (~7h on 16 CPUs).
  - For the midterm, 3M steps (~25 min on 16 CPUs) gives enough learning signal
    to show a curve and compare against SAC and the random baseline.

Usage:
    python train_ppo.py --total_timesteps 3000000 --n_envs 20 --seed 0
"""
import argparse
import copy
import os
from functools import partial

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.ppo import PPO

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG, EVAL_VEC_ENV_SEEDS


class ResumingEvalCallback(EvalCallback):
    """
    SB3's EvalCallback starts with empty logs each run. On resume, the first eval
    overwrites evaluations.npz and drops all pre-resume points (broken learning curves).
    This subclass reloads existing evaluations.npz in _init_callback so history is kept.
    """

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.log_path is None:
            return
        npz_path = self.log_path + ".npz"
        if not os.path.isfile(npz_path):
            return
        data = np.load(npz_path, allow_pickle=True)
        ts = np.atleast_1d(data["timesteps"])
        self.evaluations_timesteps = [int(x) for x in ts]
        r = np.atleast_2d(data["results"])
        self.evaluations_results = [list(np.asarray(row).flatten()) for row in r]
        if "ep_lengths" in data:
            el = np.atleast_2d(data["ep_lengths"])
            self.evaluations_length = [list(np.asarray(row).flatten()) for row in el]
        else:
            self.evaluations_length = []
        n_ts = len(self.evaluations_timesteps)
        if len(self.evaluations_results) != n_ts or len(self.evaluations_length) != n_ts:
            print(f"WARNING: inconsistent evaluations.npz (timesteps vs results/lengths) — not merging")
            self.evaluations_timesteps = []
            self.evaluations_results = []
            self.evaluations_length = []
            self.evaluations_successes = []
            self.best_mean_reward = -np.inf
            self.last_mean_reward = -np.inf
            return
        if "successes" in data.files:
            suc = data["successes"]
            if suc.dtype == object:
                self.evaluations_successes = [list(np.atleast_1d(x)) for x in suc]
            else:
                s2 = np.atleast_2d(suc)
                self.evaluations_successes = [list(np.asarray(row).flatten()) for row in s2]
            if len(self.evaluations_successes) != n_ts:
                self.evaluations_successes = []
        if self.evaluations_results:
            means = [float(np.mean(ep)) for ep in self.evaluations_results]
            self.best_mean_reward = max(means)
            self.last_mean_reward = means[-1]
            print(f"    Loaded {len(self.evaluations_timesteps)} prior eval points from {npz_path} "
                  f"(best_mean_reward={self.best_mean_reward:.2f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on MetaUrban")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--log_dir", type=str, default="./midterm_logs/PPO")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to a .zip checkpoint to resume from. "
             "Training continues from that checkpoint's step count up to --total_timesteps."
    )
    parser.add_argument(
        "--resume_lr", type=float, default=None,
        help="Learning rate to use when resuming. Lower (e.g. 1e-4) reduces risk of "
             "catastrophic forgetting. If not set, keeps the checkpoint's learning rate."
    )
    return parser.parse_args()


def make_env(env_cfg, seed):
    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **env_cfg))
    env = Monitor(env)
    return env


def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_name = f"ppo_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = copy.deepcopy(ENV_CONFIG)
    env_cfg["training"] = True

    eval_cfg = copy.deepcopy(ENV_CONFIG)
    eval_cfg["training"] = False

    env = SubprocVecEnv(
        [partial(make_env, env_cfg, seed) for seed in range(args.n_envs)]
    )
    eval_env = SubprocVecEnv(
        [partial(make_env, eval_cfg, seed) for seed in EVAL_VEC_ENV_SEEDS]
    )

    if args.resume_from:
        print(f"=== Resuming PPO from checkpoint: {args.resume_from} ===")
        model = PPO.load(
            args.resume_from,
            env=env,
            tensorboard_log=os.path.join(log_dir, "tb_logs"),
            verbose=1,
        )
        if args.resume_lr is not None:
            model.learning_rate = args.resume_lr
            print(f"    Using resume_lr={args.resume_lr} (lower LR to reduce forgetting)")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            n_steps=200,
            batch_size=256,
            n_epochs=10,
            vf_coef=1.0,
            max_grad_norm=10.0,
            ent_coef=0.0,
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(log_dir, "tb_logs"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    eval_cb = ResumingEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # When resuming, reset_num_timesteps=False continues the step counter from
    # where the checkpoint left off, so the x-axis in logs stays continuous.
    resuming = args.resume_from is not None
    already_done = model.num_timesteps if resuming else 0
    remaining = args.total_timesteps - already_done
    if remaining <= 0:
        print(f"Checkpoint is already at {already_done:,} steps — nothing to do.")
        env.close(); eval_env.close(); return

    print(f"=== Training PPO: {already_done:,} → {args.total_timesteps:,} steps "
          f"({remaining:,} remaining) ===")
    print(f"Logs: {log_dir}")

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=not resuming,   # False = continue counter, True = start fresh
    )
    model.save(os.path.join(log_dir, "final_model"))

    env.close()
    eval_env.close()
    print(f"=== PPO training complete. Model saved to {log_dir}/final_model.zip ===")


if __name__ == "__main__":
    main()
