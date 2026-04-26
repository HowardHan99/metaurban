import argparse
import os
from typing import Any, List, Optional

import gymnasium as gym
import imageio.v2 as imageio
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


_PLANNING_PATCHED = False


def _to_xy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def patch_planning_fallback() -> None:
    global _PLANNING_PATCHED
    if _PLANNING_PATCHED:
        return

    try:
        import metaurban.policy.get_planning as gp
    except Exception as e:
        print(f"[WARN] cannot import metaurban.policy.get_planning: {e}")
        return

    try:
        import metaurban.manager.humanoid_manager as hm
    except Exception as e:
        print(f"[WARN] cannot import metaurban.manager.humanoid_manager: {e}")
        hm = None

    try:
        import metaurban.component.navigation_module.orca_navigation as onav
    except Exception as e:
        print(f"[WARN] cannot import orca_navigation: {e}")
        onav = None

    old_get_planning = gp.get_planning

    def safe_get_planning(*args, **kwargs):
        try:
            return old_get_planning(*args, **kwargs)
        except ValueError as e:
            if "need at least one array to stack" not in str(e):
                raise

            print("[WARN] get_planning failed with empty nexts, using fallback straight-line planning")

            if len(args) < 3:
                raise

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

                n_points = 20
                xs = np.linspace(start_xy[0], goal_xy[0], n_points, dtype=np.float32)
                ys = np.linspace(start_xy[1], goal_xy[1], n_points, dtype=np.float32)

                points = [np.array([x, y], dtype=np.float32) for x, y in zip(xs, ys)]

                seg = np.stack([xs, ys], axis=1)
                seg_len = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                total_len = float(seg_len.sum()) if len(seg_len) > 0 else 0.0

                time_length_all.append([[total_len]])
                points_all.append(points)
                speed_all.append([[1.0]])
                early_stop_all.append([[]])

            return time_length_all, points_all, speed_all, early_stop_all

    gp.get_planning = safe_get_planning

    if hm is not None:
        hm.get_planning = safe_get_planning

    if onav is not None:
        onav.get_planning = safe_get_planning

    _PLANNING_PATCHED = True
    print("[INFO] planning fallback patch installed")


class CleanDictObsWrapper(gym.Wrapper):
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


class SeededResetWrapper(gym.Wrapper):
    def __init__(self, env, base_seed: int, num_scenarios: int):
        super().__init__(env)
        self.base_seed = int(base_seed)
        self.num_scenarios = int(num_scenarios)
        self.reset_count = 0

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is None:
            seed = (self.base_seed + self.reset_count) % self.num_scenarios
        self.reset_count += 1
        return self.env.reset(seed=seed, options=options, **kwargs)


def build_config(args):
    den_scale = 1
    return dict(
        crswalk_density=1,
        object_density=args.object_density,
        use_render=args.use_render,
        walk_on_all_regions=False,
        map=args.map,
        manual_control=False,
        drivable_area_extension=55,
        height_scale=1,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=args.horizon,
        on_continuous_line_done=False,
        out_of_route_done=True,
        relax_out_of_road_done=True,
        max_lateral_dist=15.0,
        show_sidewalk=True,
        show_crosswalk=True,
        random_spawn_lane_index=False,
        num_scenarios=args.num_scenarios,
        accident_prob=0,
        window_size=(1200, 900),
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
            policy_reverse=False,
        ),
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
        spawn_human_num=int(args.spawn_human_num * den_scale),
        spawn_wheelchairman_num=max(1, int(args.spawn_human_num // 20)),
        spawn_elderly_num=args.spawn_elderly_num,
        spawn_increase_per_episode=args.spawn_increase_per_episode,
        spawn_edog_num=0,
        spawn_erobot_num=0,
        spawn_drobot_num=0,
        max_actor_num=30,
        ignore_success_done=args.ignore_success_done,
        image_observation=True,
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[],
        sensors=dict(
            rgb_camera=(RGBCamera, args.image_width, args.image_height),
            depth_camera=(DepthCamera, 84, 84),
            semantic_camera=(SemanticCamera, 84, 84),
        ),
    )


def make_env(
    env_cfg: dict,
    seed: int,
    image_width: int,
    image_height: int,
):
    def _init():
        patch_planning_fallback()

        env = SocialDynamicMetaUrbanEnv(env_cfg)
        env = SeededResetWrapper(env, base_seed=seed, num_scenarios=env_cfg["num_scenarios"])
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


def _extract_rgb_frame(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None

    arr = np.asarray(frame)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    if arr.ndim != 3 or arr.shape[2] != 3:
        return None

    if arr.dtype != np.uint8:
        if arr.size > 0 and np.max(arr) <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def capture_frame_from_vec_env(vec_env: DummyVecEnv) -> Optional[np.ndarray]:
    try:
        base_env = vec_env.envs[0]
        frame = base_env.render()
        rgb = _extract_rgb_frame(frame)
        if rgb is not None:
            return rgb
    except Exception:
        pass

    try:
        unwrapped = vec_env.envs[0].unwrapped
        frame = unwrapped.render()
        rgb = _extract_rgb_frame(frame)
        if rgb is not None:
            return rgb
    except Exception:
        pass

    return None


def save_video(frames: List[np.ndarray], output_path: str, fps: int) -> None:
    if len(frames) == 0:
        print("[WARN] No frames captured, video not saved.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[INFO] Video saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAC checkpoint on the same social env config as training")
    # python ~/metaurban/metaurban/examples/drive_sac_imgstate_social.py --no-record_video
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/howardhan/metaurban/sac_imgstate_260000_steps.zip",
    )
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--image_width", type=int, default=80)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--use_render", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--record_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_path", type=str, default="./midterm_logs/SAC_image_state_drive_social/eval_videos/eval_rollout_440000.mp4")
    parser.add_argument("--video_fps", type=int, default=20)

    parser.add_argument("--map", type=str, default="C")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--object_density", type=float, default=0.01)

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

    parser.add_argument("--crossing_ped_num", type=int, default=2)
    parser.add_argument("--vulnerable_ped_num", type=int, default=2)

    parser.add_argument("--group_cluster_num", type=int, default=4)
    parser.add_argument("--group_cluster_size_min", type=int, default=3)
    parser.add_argument("--group_cluster_size_max", type=int, default=5)
    parser.add_argument("--group_spawn_near_ego", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group_spawn_min_radius", type=float, default=12.0)
    parser.add_argument("--group_spawn_max_radius", type=float, default=18.0)
    parser.add_argument("--group_route_min_ego_distance", type=float, default=8.0)
    parser.add_argument("--group_route_min_separation", type=float, default=5.5)
    parser.add_argument("--group_route_start_exclusion_points", type=int, default=2)
    parser.add_argument("--group_route_start_exclusion_radius", type=float, default=6.0)
    parser.add_argument("--group_member_radius", type=float, default=1.6)
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

    parser.add_argument("--spawn_human_num", type=int, default=50)
    parser.add_argument("--spawn_increase_per_episode", type=int, default=0)
    parser.add_argument("--spawn_elderly_num", type=int, default=0)
    parser.add_argument("--ignore_success_done", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.video_path = os.path.expanduser(args.video_path)

    env_cfg = build_config(args)

    vec_env = DummyVecEnv([
        make_env(
            env_cfg,
            seed=args.seed,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    ])

    frames: List[np.ndarray] = []

    try:
        print("[DEBUG] observation_space:", vec_env.observation_space)
        print("[DEBUG] action_space:", vec_env.action_space)

        obs = vec_env.reset()
        inspect_obs(obs, prefix="reset_obs")

        if args.record_video:
            first_frame = capture_frame_from_vec_env(vec_env)
            if first_frame is not None:
                frames.append(first_frame)
            else:
                print("[WARN] Failed to capture initial frame.")

        print(f"\nLoading SAC checkpoint from: {args.checkpoint}")
        model = SAC.load(args.checkpoint, env=vec_env, device="auto")

        print("\nStart rollout...")
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            if args.record_video:
                frame = capture_frame_from_vec_env(vec_env)
                if frame is not None:
                    frames.append(frame)

            if step % 50 == 0:
                print(
                    f"[step {step:05d}] "
                    f"reward={float(reward[0]): .4f} "
                    f"done={bool(done[0])}"
                )

            if done[0]:
                print(f"Episode done at step {step}")
                obs = vec_env.reset()

                if args.record_video:
                    frame = capture_frame_from_vec_env(vec_env)
                    if frame is not None:
                        frames.append(frame)

    finally:
        try:
            vec_env.close()
        except Exception as e:
            print(f"[WARN] vec_env.close() failed: {e}")

        if args.record_video:
            save_video(frames, args.video_path, args.video_fps)


if __name__ == "__main__":
    main()