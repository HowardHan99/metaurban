import argparse
import copy
import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation

from env_config import ENV_CONFIG, EVAL_VEC_ENV_SEEDS


class IdlePenaltyWrapper(gym.Wrapper):
    """Penalize the agent for standing still / braking to zero speed."""

    def __init__(self, env: gym.Env, penalty: float = 0.1, speed_threshold: float = 0.5):
        super().__init__(env)
        self.penalty = float(penalty)
        self.speed_threshold = float(speed_threshold)

    def reset(self, *, seed=None, options=None, **kwargs):
        # MetaUrban BaseEnv.reset() does not support Gymnasium's `options`
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


class CleanDictObsWrapper(gym.Wrapper):
    """
    Convert MetaUrban raw dict obs into a clean Dict observation for SB3:
      - image: uint8 RGB, shape (H, W, 3)
      - state: float32 vector
    Drops depth/semantic from the policy input.
    """

    def __init__(self, env: gym.Env, image_width: int, image_height: int):
        super().__init__(env)
        raw_space = self.env.observation_space
        if not isinstance(raw_space, spaces.Dict):
            raise TypeError(f"Expected Dict observation_space, got {type(raw_space)}")
        if "state" not in raw_space.spaces:
            raise KeyError("Raw observation_space does not contain 'state'")

        self.image_width = int(image_width)
        self.image_height = int(image_height)

        state_space = raw_space.spaces["state"]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=state_space.shape,
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = self.env.action_space

    @staticmethod
    def _clean_sensor_array(arr: Any) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[..., 0]
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr

    @staticmethod
    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)

        if img.ndim == 4:
            img = img[..., 0]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if img.dtype != np.uint8:
            if img.size > 0 and np.max(img) <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        if img.ndim != 3:
            raise ValueError(f"Unexpected cleaned image ndim: {img.ndim}, shape={img.shape}")
        if img.shape[2] < 3:
            raise ValueError(f"Unexpected cleaned image channels: {img.shape}")
        if img.shape[2] > 3:
            img = img[:, :, :3]

        return img

    def _extract_clean_obs(self, raw_obs: dict) -> dict:
        if not isinstance(raw_obs, dict):
            raise TypeError(f"Expected dict raw_obs, got {type(raw_obs)}")
        if "image" not in raw_obs or "state" not in raw_obs:
            raise KeyError(f"raw_obs keys missing image/state: {list(raw_obs.keys())}")

        image = self._clean_sensor_array(raw_obs["image"])
        image = self._to_uint8_rgb(image)

        expected_shape = self.observation_space.spaces["image"].shape
        if image.shape != expected_shape:
            raise ValueError(f"Image shape mismatch. Got {image.shape}, expected {expected_shape}.")

        state = np.asarray(raw_obs["state"], dtype=np.float32)
        expected_state_shape = self.observation_space.spaces["state"].shape
        if state.shape != expected_state_shape:
            raise ValueError(f"State shape mismatch. Got {state.shape}, expected {expected_state_shape}.")

        return {"image": image, "state": state}

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        return self._extract_clean_obs(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extract_clean_obs(raw_obs), reward, terminated, truncated, info


def _to_xy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def patch_orca_planning_fallback() -> None:
    """
    Patch MetaUrban ORCA planning so reset() does not crash when bind.demo is missing.
    Must run inside each worker process before env.reset().
    """
    try:
        import metaurban.policy.get_planning as gp
    except Exception as e:
        print(f"[WARN] Failed to import metaurban.policy.get_planning: {e}")
        return

    try:
        import metaurban.component.navigation_module.orca_navigation as onav
    except Exception as e:
        print(f"[WARN] Failed to import orca_navigation: {e}")
        onav = None

    bind_obj = getattr(gp, "bind", None)
    has_demo = hasattr(bind_obj, "demo") if bind_obj is not None else False
    if has_demo:
        return

    def fallback_get_planning(*args, **kwargs):
        if len(args) < 3:
            raise ValueError("fallback_get_planning expects at least 3 positional args")

        start_positions_list = args[0]
        goals_list = args[2]
        n_agents = min(len(start_positions_list), len(goals_list))

        time_length_all = []
        points_all = []
        speed_all = []
        early_stop_all = []

        for i in range(n_agents):
            start_xy = _to_xy(start_positions_list[i])
            goal_xy = _to_xy(goals_list[i])

            n_points = 60
            xs = np.linspace(start_xy[0], goal_xy[0], n_points, dtype=np.float32)
            ys = np.linspace(start_xy[1], goal_xy[1], n_points, dtype=np.float32)

            points = [np.array([x, y], dtype=np.float32) for x, y in zip(xs, ys)]
            seg_len = np.linalg.norm(np.diff(np.stack([xs, ys], axis=1), axis=0), axis=1)
            total_len = float(seg_len.sum()) if len(seg_len) > 0 else 0.0

            time_length_all.append([[total_len]])
            points_all.append(points)
            speed_all.append([[1.0]])
            early_stop_all.append([[]])

        return time_length_all, points_all, speed_all, early_stop_all

    gp.get_planning = fallback_get_planning
    if onav is not None:
        onav.get_planning = fallback_get_planning


def parse_args():
    parser = argparse.ArgumentParser(description="Train image+state SAC on MetaUrban")
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--checkpoint_freq", type=int, default=20_000)
    parser.add_argument("--log_dir", type=str, default="./midterm_logs/SAC_image_state")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a .zip model to resume")
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=120)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--idle_penalty", type=float, default=0.1)
    parser.add_argument("--speed_threshold", type=float, default=0.5)
    parser.add_argument("--debug_reset", action="store_true", help="Run a reset() on train/eval env before training.")
    parser.add_argument("--disable_eval", action="store_true", help="Disable EvalCallback to avoid eval_env reset crashes.")
    return parser.parse_args()


def _get_sac_env_overrides() -> dict:
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


def build_image_state_env_config(image_width: int, image_height: int, training: bool) -> dict:
    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = training
    cfg.update(_get_sac_env_overrides())
    cfg.update(
        dict(
            use_render=False,
            image_observation=True,
            agent_observation=ThreeSourceMixObservation,
            interface_panel=[],
            sensors=dict(
                rgb_camera=(RGBCamera, image_width, image_height),
                depth_camera=(DepthCamera, 84, 84),
                semantic_camera=(SemanticCamera, 84, 84),
            ),
        )
    )

    if "vehicle_config" in cfg:
        cfg["vehicle_config"] = copy.deepcopy(cfg["vehicle_config"])
        cfg["vehicle_config"].update(
            dict(
                show_lidar=False,
                show_navi_mark=False,
                show_line_to_navi_mark=False,
                show_dest_mark=False,
            )
        )
    return cfg


def make_env(
    env_cfg: dict,
    seed: int,
    image_width: int,
    image_height: int,
    use_idle_penalty: bool = True,
    idle_penalty: float = 0.1,
    speed_threshold: float = 0.5,
):
    def _init():
        patch_orca_planning_fallback()

        cfg = copy.deepcopy(env_cfg)
        cfg["start_seed"] = int(seed)
        cfg["log_level"] = 50

        env = SidewalkStaticMetaUrbanEnv(cfg)
        env = CleanDictObsWrapper(env, image_width=image_width, image_height=image_height)

        if use_idle_penalty:
            env = IdlePenaltyWrapper(env, penalty=idle_penalty, speed_threshold=speed_threshold)

        env = Monitor(env)
        return env

    return _init


def inspect_vec_obs(obs, prefix="obs"):
    if isinstance(obs, dict):
        print(f"[DEBUG] {prefix} keys: {list(obs.keys())}")
        for k, v in obs.items():
            arr = np.asarray(v)
            print(f"[DEBUG] {prefix}[{k}] shape={arr.shape}, dtype={arr.dtype}")
    else:
        arr = np.asarray(obs)
        print(f"[DEBUG] {prefix} shape={arr.shape}, dtype={arr.dtype}")


def build_train_and_eval_envs(args):
    train_cfg = build_image_state_env_config(args.image_width, args.image_height, training=True)

    # 先和 train 保持一致，避免 training=False 的 reset 路径出问题
    eval_cfg = build_image_state_env_config(args.image_width, args.image_height, training=True)

    train_seeds = [args.seed + i for i in range(args.n_envs)]

    if len(EVAL_VEC_ENV_SEEDS) > 0:
        eval_seeds = [
            int(EVAL_VEC_ENV_SEEDS[i % len(EVAL_VEC_ENV_SEEDS)]) + args.seed
            for i in range(max(1, min(len(EVAL_VEC_ENV_SEEDS), args.n_envs)))
        ]
    else:
        eval_seeds = [args.seed + 1000]

    env_fns = [
        make_env(
            train_cfg,
            seed=s,
            image_width=args.image_width,
            image_height=args.image_height,
            use_idle_penalty=True,
            idle_penalty=args.idle_penalty,
            speed_threshold=args.speed_threshold,
        )
        for s in train_seeds
    ]

    eval_env_fns = [
        make_env(
            eval_cfg,
            seed=s,
            image_width=args.image_width,
            image_height=args.image_height,
            use_idle_penalty=False,
            idle_penalty=args.idle_penalty,
            speed_threshold=args.speed_threshold,
        )
        for s in eval_seeds
    ]

    if args.debug_reset:
        print("[DEBUG] debug_reset=True, using DummyVecEnv for BOTH train and eval envs.")
        env = DummyVecEnv(env_fns)
        eval_env = DummyVecEnv(eval_env_fns)
    else:
        if args.n_envs == 1:
            print("[DEBUG] n_envs=1, using DummyVecEnv for train env.")
            env = DummyVecEnv(env_fns)
        else:
            env = SubprocVecEnv(env_fns)

        eval_env = DummyVecEnv(eval_env_fns)

    return env, eval_env, train_seeds, eval_seeds

class SafeEvalCallback(EvalCallback):
    """
    EvalCallback that never crashes training.
    If evaluation fails (e.g. env.reset() error), skip this eval round and continue.
    """

    def _on_step(self) -> bool:
        try:
            return super()._on_step()
        except Exception as e:
            print(f"[WARN] Eval failed, skipping this round: {repr(e)}")
            return True

def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_name = f"sac_imgstate_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = None
    eval_env = None

    try:
        env, eval_env, train_seeds, eval_seeds = build_train_and_eval_envs(args)

        print(f"=== Training image+state SAC for {args.total_timesteps} steps with {args.n_envs} envs ===")
        print(f"    image size: {args.image_width}x{args.image_height}")
        print(f"    reward overrides: {_get_sac_env_overrides()}")
        print(f"    idle penalty: -{args.idle_penalty}/step when speed < {args.speed_threshold} km/h")
        print(
            f"    buffer={args.buffer_size} | learning_starts={args.learning_starts} | "
            f"batch={args.batch_size} | gradient_steps=1"
        )
        print(f"    train_seeds: {train_seeds}")
        print(f"    eval_seeds: {eval_seeds}")
        print(f"    logs: {log_dir}")

        print(f"[DEBUG] train obs space: {env.observation_space}")
        print(f"[DEBUG] eval obs space:  {eval_env.observation_space}")
        print(f"[DEBUG] action space:    {env.action_space}")

        if args.debug_reset:
            print("[DEBUG] Running one reset() on train env...")
            train_obs = env.reset()
            inspect_vec_obs(train_obs, prefix="train_reset_obs")

            print("[DEBUG] Running one reset() on eval env...")
            eval_obs = eval_env.reset()
            inspect_vec_obs(eval_obs, prefix="eval_reset_obs")

        if args.resume_from:
            model = SAC.load(args.resume_from, env=env, device="auto", seed=args.seed)
            print(f"Resumed SAC from {args.resume_from}")
        else:
            model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1,
                seed=args.seed,
                tensorboard_log=os.path.join(log_dir, "tb_logs"),
                device="auto",
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="sac_imgstate",
        )

        if args.disable_eval:
            print("[INFO] EvalCallback disabled. Training will run without periodic evaluation.")
            callbacks = CallbackList([checkpoint_cb])
        else:
            eval_cb = SafeEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_dir, "best_model"),
                log_path=os.path.join(log_dir, "eval_logs"),
                eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                n_eval_episodes=max(1, len(eval_seeds)),
                deterministic=True,
                render=False,
            )
            callbacks = CallbackList([checkpoint_cb, eval_cb])

        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        model.save(os.path.join(log_dir, "final_model"))
        print(f"=== SAC training complete. Model saved to {log_dir}/final_model.zip ===")

    finally:
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception as e:
                print(f"[WARN] eval_env.close() failed: {e}")

        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"[WARN] env.close() failed: {e}")

        print("[INFO] Environments closed.")


if __name__ == "__main__":
    main()