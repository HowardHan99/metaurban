import argparse
import copy
import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

from env_config import ENV_CONFIG


class CleanDictObsWrapper(gym.Wrapper):
    """
    Same logic as training:
      - image: uint8 RGB, shape (H, W, 3)
      - state: float32 vector
    Drops depth/semantic from policy input.
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


def _get_sac_env_overrides() -> dict:
    """
    Keep the same reward-related overrides as training script.
    """
    return dict(
        no_negative_reward=False,
        driving_reward=1.5,
        success_reward=30.0,
        speed_reward=1.0,
        lateral_penalty=0.5,
        crash_vehicle_penalty=1.0,
        crash_object_penalty=3.0,
        crash_human_penalty=3.0,
        crash_building_penalty=1.0,
        out_of_road_penalty=4.0,
        steering_range_penalty=0.5,
        crash_object_done=True,
    )


def build_social_image_state_env_config(
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> dict:
    """
    Follow train_sac_image_state_test.py closely:
    - start from ENV_CONFIG
    - use ThreeSourceMixObservation
    - register rgb/depth/semantic sensors
    - keep use_render=False for checkpoint compatibility
    Then add social env parameters.
    """
    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg.update(_get_sac_env_overrides())

    cfg.update(
        dict(
            use_render=True,  # IMPORTANT: keep same as training path
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

    # ---- social env additions ----
    cfg.update(
        dict(
            map=args.map,
            horizon=args.horizon,
            num_scenarios=args.num_scenarios,
            traffic_density=args.traffic_density,
            object_density=args.object_density,
            accident_prob=args.accident_prob,
            scene_type=args.scene_type,
            scene_building_source=args.scene_building_source,
            crossing_ped_num=args.crossing_ped_num,
            vulnerable_ped_num=args.vulnerable_ped_num,
            group_cluster_num=args.group_cluster_num,
            group_cluster_size_min=args.group_cluster_size_min,
            group_cluster_size_max=args.group_cluster_size_max,
            group_spawn_near_ego=args.group_spawn_near_ego,
            group_spawn_min_radius=args.group_spawn_min_radius,
            group_spawn_max_radius=args.group_spawn_max_radius,
            group_route_min_ego_distance=args.group_route_min_ego_distance,
            group_route_min_separation=args.group_route_min_separation,
            group_route_start_exclusion_points=args.group_route_start_exclusion_points,
            group_route_start_exclusion_radius=args.group_route_start_exclusion_radius,
            group_member_radius=args.group_member_radius,
            group_member_ring_step=args.group_member_ring_step,
            group_member_radius_jitter=args.group_member_radius_jitter,
            group_member_ring_step_jitter=args.group_member_ring_step_jitter,
            group_member_idle_shift_prob=args.group_member_idle_shift_prob,
            group_member_idle_shift_steps_mean=args.group_member_idle_shift_steps_mean,
            group_member_idle_shift_radius=args.group_member_idle_shift_radius,
            group_release_enable=args.group_release_enable,
            group_release_steps_mean=args.group_release_steps_mean,
            group_release_steps_std=args.group_release_steps_std,
            group_release_steps_min=args.group_release_steps_min,
            vulnerable_elderly_ratio=args.vulnerable_elderly_ratio,
            vulnerable_distracted_ratio=args.vulnerable_distracted_ratio,
            vulnerable_pause_prob=args.vulnerable_pause_prob,
            vulnerable_pause_steps_mean=args.vulnerable_pause_steps_mean,
            spawn_human_num=args.spawn_human_num,
            spawn_elderly_num=args.spawn_elderly_num,
            spawn_wheelchairman_num=args.spawn_wheelchairman_num,
            spawn_increase_per_episode=args.spawn_increase_per_episode,
            ignore_success_done=args.ignore_success_done,
        )
    )

    return cfg


def make_social_env(env_cfg: dict, seed: int, image_width: int, image_height: int):
    def _init():
        cfg = copy.deepcopy(env_cfg)
        cfg["start_seed"] = int(seed)
        cfg["log_level"] = 50

        env = SocialDynamicMetaUrbanEnv(cfg)
        env = CleanDictObsWrapper(env, image_width=image_width, image_height=image_height)
        env = Monitor(env)
        return env

    return _init


def inspect_obs(obs, prefix="obs"):
    if isinstance(obs, dict):
        print(f"[DEBUG] {prefix} keys: {list(obs.keys())}")
        for k, v in obs.items():
            arr = np.asarray(v)
            print(f"[DEBUG] {prefix}[{k}] shape={arr.shape}, dtype={arr.dtype}")
    else:
        arr = np.asarray(obs)
        print(f"[DEBUG] {prefix} shape={arr.shape}, dtype={arr.dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAC image+state checkpoint on SocialDynamicMetaUrbanEnv")

    parser.add_argument("--checkpoint", type=str, default="~/metaurban/sac_imgstate_260000_steps.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--image_width", type=int, default=80)
    parser.add_argument("--image_height", type=int, default=60)

    parser.add_argument("--map", type=str, default="C")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--num_scenarios", type=int, default=100)

    parser.add_argument("--traffic_density", type=float, default=0.0)
    parser.add_argument("--object_density", type=float, default=0.01)
    parser.add_argument("--accident_prob", type=float, default=0.0)

    parser.add_argument(
        "--scene_type",
        type=str,
        default="default",
        choices=["default", "commercial", "commute", "leisure", "constrained"],
    )
    parser.add_argument(
        "--scene_building_source",
        type=str,
        default="default",
        choices=["scene", "default"],
    )

    parser.add_argument("--crossing_ped_num", type=int, default=6)
    parser.add_argument("--vulnerable_ped_num", type=int, default=8)
    parser.add_argument("--group_cluster_num", type=int, default=4)
    parser.add_argument("--group_cluster_size_min", type=int, default=3)
    parser.add_argument("--group_cluster_size_max", type=int, default=5)

    parser.add_argument("--group_spawn_near_ego", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group_spawn_min_radius", type=float, default=12.0)
    parser.add_argument("--group_spawn_max_radius", type=float, default=20.0)
    parser.add_argument("--group_route_min_ego_distance", type=float, default=12.0)
    parser.add_argument("--group_route_min_separation", type=float, default=5.5)
    parser.add_argument("--group_route_start_exclusion_points", type=int, default=4)
    parser.add_argument("--group_route_start_exclusion_radius", type=float, default=10.0)

    parser.add_argument("--group_member_radius", type=float, default=1.45)
    parser.add_argument("--group_member_ring_step", type=float, default=0.62)
    parser.add_argument("--group_member_radius_jitter", type=float, default=0.16)
    parser.add_argument("--group_member_ring_step_jitter", type=float, default=0.12)
    parser.add_argument("--group_member_idle_shift_prob", type=float, default=0.015)
    parser.add_argument("--group_member_idle_shift_steps_mean", type=int, default=18)
    parser.add_argument("--group_member_idle_shift_radius", type=float, default=0.22)

    parser.add_argument("--group_release_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group_release_steps_mean", type=int, default=180)
    parser.add_argument("--group_release_steps_std", type=int, default=40)
    parser.add_argument("--group_release_steps_min", type=int, default=60)

    parser.add_argument("--vulnerable_elderly_ratio", type=float, default=0.6)
    parser.add_argument("--vulnerable_distracted_ratio", type=float, default=0.4)
    parser.add_argument("--vulnerable_pause_prob", type=float, default=0.02)
    parser.add_argument("--vulnerable_pause_steps_mean", type=int, default=16)

    parser.add_argument("--spawn_human_num", type=int, default=40)
    parser.add_argument("--spawn_elderly_num", type=int, default=0)
    parser.add_argument("--spawn_wheelchairman_num", type=int, default=1)
    parser.add_argument("--spawn_increase_per_episode", type=int, default=0)
    parser.add_argument("--ignore_success_done", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = os.path.expanduser(args.checkpoint)

    env_cfg = build_social_image_state_env_config(
        image_width=args.image_width,
        image_height=args.image_height,
        args=args,
    )

    vec_env = DummyVecEnv([
        make_social_env(
            env_cfg,
            seed=args.seed,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    ])

    print("[DEBUG] observation_space:", vec_env.observation_space)
    print("[DEBUG] action_space:", vec_env.action_space)

    obs = vec_env.reset()
    inspect_obs(obs, prefix="reset_obs")

    print(f"\nLoading SAC checkpoint from: {args.checkpoint}")
    model = SAC.load(args.checkpoint, env=vec_env, device="auto")

    print("\nStart rollout...")
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        if step % 50 == 0:
            print(
                f"[step {step:05d}] "
                f"reward={float(reward[0]): .4f} "
                f"done={bool(done[0])}"
            )

        if done[0]:
            print(f"Episode done at step {step}")
            obs = vec_env.reset()

    vec_env.close()


if __name__ == "__main__":
    main()