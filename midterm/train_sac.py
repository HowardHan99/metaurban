"""
Train SAC (off-policy Q-function-based RL) in MetaUrban PointNav environment.

SAC is off-policy because it learns from a replay buffer of past transitions collected
by any version of the policy, not just the current one.  This makes it more sample-
efficient than PPO but requires a larger buffer and careful tuning of gradient_steps.

Key hyperparameter rationale (v2 — fixes "always-brake" local minimum):
  - no_negative_reward=False: required for SAC to distinguish good vs bad states.
  - Reward rebalancing: the original config had crash penalties (2.0) >> per-step
    driving reward (~0.01–0.1), so the Q-function learned that standing still
    (total ≈ -10 over 1000 steps) beats moving (risk of -200 from penalties).
    Fix: reduce penalties, boost driving/success reward, add idle penalty.
  - IdlePenaltyWrapper: adds -0.1 per step when speed < 0.5 km/h.  This makes
    braking cost ~100 over a full episode, breaking the "freeze" local minimum.
  - gradient_steps=1: the v1 setting of n_envs caused aggressive early overfitting
    on a small buffer.  1:1 ratio is standard for single-env SAC; with vec-envs
    the buffer fills fast enough that 1 gradient step per env.step() works well.
  - learning_starts=10_000: 10× more random transitions before training starts,
    giving the Q-function diverse (s,a,r,s') tuples to bootstrap from.
  - buffer_size=1_000_000: large enough to hold the full training history.

Usage:
    python train_sac.py --total_timesteps 3000000 --n_envs 10 --seed 0
"""
import argparse
import copy
import os
from functools import partial

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.sac import SAC

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG, EVAL_VEC_ENV_SEEDS


class IdlePenaltyWrapper(gym.Wrapper):
    """Penalize the agent for standing still / braking to zero speed."""

    def __init__(self, env, penalty: float = 0.1, speed_threshold: float = 0.5):
        super().__init__(env)
        self.penalty = penalty
        self.speed_threshold = speed_threshold

    def reset(self, *, seed=None, options=None, **kwargs):
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            speed = self.env.unwrapped.vehicle.speed_km_h
        except AttributeError:
            speed = None
        if speed is not None and speed < self.speed_threshold:
            reward -= self.penalty
        return obs, reward, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on MetaUrban")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--log_dir", type=str, default="./midterm_logs/SAC")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .zip model to resume training from")
    return parser.parse_args()


def make_env(env_cfg, seed, use_idle_penalty=True):
    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=seed, log_level=50, **env_cfg))
    if use_idle_penalty:
        env = IdlePenaltyWrapper(env, penalty=0.1, speed_threshold=0.5)
    env = Monitor(env)
    return env


def _get_sac_env_overrides():
    """Reward-balance overrides that fix the 'always-brake' local minimum."""
    return dict(
        no_negative_reward=False,
        driving_reward=3.0,
        success_reward=15.0,
        speed_reward=1.0,
        lateral_penalty=0.5,
        crash_vehicle_penalty=1.0,
        crash_object_penalty=1.0,
        crash_human_penalty=1.0,
        crash_building_penalty=1.0,
        out_of_road_penalty=2.0,
        steering_range_penalty=0.5,
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_name = f"sac_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    sac_overrides = _get_sac_env_overrides()

    env_cfg = copy.deepcopy(ENV_CONFIG)
    env_cfg["training"] = True
    env_cfg.update(sac_overrides)

    eval_cfg = copy.deepcopy(ENV_CONFIG)
    eval_cfg["training"] = False
    eval_cfg.update(sac_overrides)

    env = SubprocVecEnv(
        [partial(make_env, env_cfg, seed, True) for seed in range(args.n_envs)]
    )
    eval_env = SubprocVecEnv(
        [partial(make_env, eval_cfg, seed, False) for seed in EVAL_VEC_ENV_SEEDS]
    )

    if args.resume_from:
        model = SAC.load(args.resume_from, env=env)
        model.tensorboard_log = os.path.join(log_dir, "tb_logs")
        print(f"Resumed SAC from {args.resume_from}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(log_dir, "tb_logs"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    print(f"=== Training SAC for {args.total_timesteps} steps with {args.n_envs} envs ===")
    print(f"    Reward overrides: {sac_overrides}")
    print(f"    IdlePenaltyWrapper: -0.1/step when speed < 0.5 km/h")
    print(f"    gradient_steps=1 | buffer=1M | learning_starts=10k | batch=256")
    print(f"Logs: {log_dir}")

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(os.path.join(log_dir, "final_model"))

    env.close()
    eval_env.close()
    print(f"=== SAC training complete. Model saved to {log_dir}/final_model.zip ===")


if __name__ == "__main__":
    main()
