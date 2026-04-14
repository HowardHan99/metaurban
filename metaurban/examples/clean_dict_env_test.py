import argparse
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.engine.logger import get_logger


# =========================================================
# ORCA fallback patch
# =========================================================
def _to_xy(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def patch_orca_planning_fallback():
    """
    Phase-1/2 debug fallback:
    bypass broken ORCA bind/demo planning and return a simple straight-line plan.
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
        print("[INFO] ORCA bind.demo found. No fallback patch needed.")
        return

    print("[WARN] bind.demo not found. Applying flexible get_planning() fallback.")

    def fallback_get_planning(*args, **kwargs):
        if len(args) < 3:
            raise ValueError("fallback_get_planning expects at least 3 positional args.")

        # Based on your branch debug:
        # args[0] looks like start positions
        # args[2] looks like goal positions
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

            # nested to satisfy current branch indexing
            time_length_all.append([[total_len]])
            points_all.append(points)
            speed_all.append([[1.0]])
            early_stop_all.append([[]])

        return time_length_all, points_all, speed_all, early_stop_all

    gp.get_planning = fallback_get_planning
    if onav is not None:
        onav.get_planning = fallback_get_planning

    print("[INFO] Fallback patch applied to get_planning().")


# =========================================================
# observation cleaning utils
# =========================================================
def clean_sensor_array(arr):
    arr = np.asarray(arr)

    # e.g. (H, W, C, K) -> (H, W, C)
    if arr.ndim == 4:
        arr = arr[..., 0]

    if arr.ndim == 2:
        arr = arr[..., None]

    return arr


def to_uint8_rgb(img):
    img = np.asarray(img)

    if img.ndim == 4:
        img = img[..., 0]

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def extract_clean_obs(raw_obs):
    assert isinstance(raw_obs, dict), f"Expected dict obs, got {type(raw_obs)}"

    image = raw_obs.get("image", None)
    state = raw_obs.get("state", None)

    if image is None:
        raise KeyError("raw_obs does not contain key 'image'")
    if state is None:
        raise KeyError("raw_obs does not contain key 'state'")

    image = clean_sensor_array(image)
    image = to_uint8_rgb(image)
    state = np.asarray(state, dtype=np.float32)

    return {
        "image": image,   # (H, W, 3), uint8
        "state": state,   # (271,), float32
    }


# =========================================================
# clean dict wrapper
# =========================================================
class CleanDictObsWrapper(gym.Wrapper):
    """
    Wrap MetaUrban env and expose only:
        obs = {
            "image": uint8 RGB image, shape (H, W, 3)
            "state": float32 state vector, shape (271,)
        }

    This is the version you should use with SB3 MultiInputPolicy.
    """

    def __init__(self, env):
        super().__init__(env)

        # Infer clean obs shape from one reset sample
        raw_obs, _ = self.env.reset()
        clean_obs = extract_clean_obs(raw_obs)

        image = clean_obs["image"]
        state = clean_obs["state"]

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=image.shape,
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=state.shape,
                dtype=np.float32,
            ),
        })

        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        clean_obs = extract_clean_obs(raw_obs)
        return clean_obs, info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        clean_obs = extract_clean_obs(raw_obs)
        return clean_obs, reward, terminated, truncated, info


# =========================================================
# debug utils
# =========================================================
def describe_array(name, arr):
    arr = np.asarray(arr)
    arr_min = np.nanmin(arr) if arr.size > 0 else np.nan
    arr_max = np.nanmax(arr) if arr.size > 0 else np.nan
    has_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr_min:.4f}, max={arr_max:.4f}, has_nan={has_nan}"
    )


def print_obs(obs, tag="obs"):
    print(f"\n[{tag}]")
    print(f"type={type(obs)}")
    if isinstance(obs, dict):
        print(f"keys={list(obs.keys())}")
        for k, v in obs.items():
            describe_array(f"obs['{k}']", v)


def save_obs_image(obs, save_path):
    img = obs["image"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# =========================================================
# env config
# =========================================================
def build_env_config(density_obj=0.3):
    return dict(
        crswalk_density=1,
        object_density=density_obj,
        walk_on_all_regions=False,
        use_render=True,
        map="X",
        manual_control=False,
        default_expert=False,
        drivable_area_extension=55,
        height_scale=1,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        window_size=(1200, 900),
        agent_type="coco",
        tiny=True,
        image_observation=True,
        sensors=dict(
            rgb_camera=(RGBCamera, 320, 240),
            depth_camera=(DepthCamera, 128, 128),
            semantic_camera=(SemanticCamera, 128, 128),
        ),
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[],
    )


# =========================================================
# main test
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density_obj", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="clean_wrapper_test_out")
    parser.add_argument("--test_sb3", action="store_true")
    args = parser.parse_args()

    patch_orca_planning_fallback()

    logger = get_logger()
    logger.info("Creating base MetaUrban env...")

    base_env = SidewalkStaticMetaUrbanEnv(build_env_config(args.density_obj))
    env = CleanDictObsWrapper(base_env)

    try:
        print(HELP_MESSAGE)
        print("\n[wrapped observation_space]")
        print(env.observation_space)
        print("\n[action_space]")
        print(env.action_space)

        obs, info = env.reset(seed=args.seed)
        print_obs(obs, tag="reset_clean_obs")
        save_obs_image(obs, os.path.join(args.save_dir, "reset.png"))

        ep_return = 0.0
        for step in range(1, args.max_steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            print(f"\n[step {step}] reward={reward:.5f} term={terminated} trunc={truncated} ep_return={ep_return:.5f}")

            if step <= 3:
                print_obs(obs, tag=f"step_{step}_clean_obs")
                save_obs_image(obs, os.path.join(args.save_dir, f"step_{step:06d}.png"))

            if terminated or truncated:
                print(f"[episode done] at step={step}")
                obs, info = env.reset()
                ep_return = 0.0

        if args.test_sb3:
            print("\n[SB3 test] building PPO(MultiInputPolicy)...")
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                n_steps=32,
                batch_size=32,
            )
            print("[SB3 test] model created successfully.")

            action, _ = model.predict(obs, deterministic=False)
            print(f"[SB3 test] predicted action = {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            print(f"[SB3 test] one step succeeded, reward={reward:.5f}")

        print("\n[DONE] CleanDict wrapper test finished.")

    finally:
        env.close()