"""
Phase 1: env + observation debug for MetaUrban
- bypass ORCA planning dependency for reset()
- print raw observation shapes
- clean image from weird stacked shape -> standard RGB
- save cleaned image/state pairs for later MultiInputPolicy testing
"""

import argparse
import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.engine.logger import get_logger


def _to_xy(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def patch_orca_planning_fallback():
    """
    Patch get_planning() directly so env.reset() can work even if bind.demo is missing
    or the ORCA C++ binding is incompatible.

    We intentionally accept *args, **kwargs because different branches may call
    get_planning with slightly different signatures.
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
        """
        Flexible fallback for Phase 1 only.
        We only need reset() to pass, so we generate a simple straight-line path
        and mimic the nested return structure expected by orca_navigation.py.
        """
        print(f"[DEBUG] fallback_get_planning called with {len(args)} positional args.")
        for idx, a in enumerate(args):
            try:
                arr = np.asarray(a, dtype=object)
                shape = arr.shape
            except Exception:
                shape = "N/A"
            print(f"  arg[{idx}] type={type(a)} shape={shape}")

        if len(args) < 3:
            raise ValueError("fallback_get_planning expects at least 3 positional args here.")

        # Based on your debug print:
        # arg[0] ~ start positions, shape (1,1,2)
        # arg[2] ~ goal positions,  shape (1,1,2)
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

            # Make them indexable as [0][0]
            time_length_all.append([[total_len]])

            # points must stay iterable for: for p in positions
            points_all.append(points)

            # Make speeds indexable as [0][0]
            speed_all.append([[1.0]])

            # keep iterable
            early_stop_all.append([[]])

        return time_length_all, points_all, speed_all, early_stop_all

    gp.get_planning = fallback_get_planning

    if onav is not None:
        onav.get_planning = fallback_get_planning

    print("[INFO] Fallback patch applied to get_planning().")


def print_space(space, prefix="observation_space"):
    print(f"\n[{prefix}] {space}")
    if isinstance(space, gym.spaces.Dict):
        for k, v in space.spaces.items():
            print(f"  - key='{k}': {v}")
    elif hasattr(space, "shape"):
        print(f"  - shape={space.shape}, dtype={getattr(space, 'dtype', None)}")


def describe_array(name, arr):
    arr = np.asarray(arr)
    arr_min = np.nanmin(arr) if arr.size > 0 else np.nan
    arr_max = np.nanmax(arr) if arr.size > 0 else np.nan
    has_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr_min:.4f}, max={arr_max:.4f}, has_nan={has_nan}"
    )


def print_obs_summary(obs, tag="obs"):
    print(f"\n[{tag}] type={type(obs)}")
    if isinstance(obs, dict):
        print(f"keys={list(obs.keys())}")
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                describe_array(f"obs['{k}']", v)
            else:
                print(f"obs['{k}']: type={type(v)}")
    elif isinstance(obs, np.ndarray):
        describe_array("obs", obs)
    else:
        print(obs)


def clean_sensor_array(arr):
    arr = np.asarray(arr)

    # (H, W, C, K) -> (H, W, C)
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


def save_debug_sample(save_dir, step_idx, raw_obs, clean_obs, reward, tm, tc, info):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img = clean_obs["image"]
    state = clean_obs["state"]

    payload = {
        "state": state,
        "reward": float(reward),
        "terminal": bool(tm),
        "trunc": bool(tc),
        "info": info,
        "raw_obs_keys": list(raw_obs.keys()) if isinstance(raw_obs, dict) else None,
        "raw_image_shape": tuple(np.asarray(raw_obs["image"]).shape)
        if isinstance(raw_obs, dict) and "image" in raw_obs else None,
        "clean_image_shape": tuple(img.shape),
    }

    np.save(save_dir / f"step_{step_idx:06d}.npy", payload, allow_pickle=True)
    cv2.imwrite(str(save_dir / f"step_{step_idx:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build_env_config(density_obj):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density_obj", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="phase1_debug_out")
    args = parser.parse_args()

    patch_orca_planning_fallback()

    config = build_env_config(args.density_obj)
    logger = get_logger()
    logger.info("Phase 1 env/observation debug script starting.")

    env = SidewalkStaticMetaUrbanEnv(config)

    try:
        print(HELP_MESSAGE)
        print_space(env.observation_space)
        print(f"\naction_space: {env.action_space}")
        try:
            print(f"action low={env.action_space.low}, high={env.action_space.high}")
        except Exception:
            pass

        raw_obs, info = env.reset(seed=args.seed)
        print_obs_summary(raw_obs, tag="reset_raw")

        clean_obs = extract_clean_obs(raw_obs)
        print("\n[clean reset obs]")
        describe_array("clean_obs['image']", clean_obs["image"])
        describe_array("clean_obs['state']", clean_obs["state"])

        save_debug_sample(args.save_dir, 0, raw_obs, clean_obs, 0.0, False, False, info)
        print(f"[saved] {args.save_dir}/step_000000.*")

        repeat = 5
        action = None
        ep_return = 0.0
        ep_len = 0

        for step in range(1, args.max_steps + 1):
            if action is None or step % repeat == 1:
                action = env.action_space.sample()
                action = np.clip(action * 1.5, -1.0, 1.0)

            raw_obs, reward, tm, tc, info = env.step(action)
            clean_obs = extract_clean_obs(raw_obs)

            ep_return += float(reward)
            ep_len += 1

            print(
                f"\n[step {step}] reward={float(reward):.5f} "
                f"tm={tm} tc={tc} ep_len={ep_len} ep_return={ep_return:.5f}"
            )

            if step <= 3 or step % 10 == 0 or tm or tc:
                print_obs_summary(raw_obs, tag=f"step_{step}_raw")
                print("[clean step obs]")
                describe_array("clean_obs['image']", clean_obs["image"])
                describe_array("clean_obs['state']", clean_obs["state"])
                if isinstance(info, dict):
                    print(f"[info keys] {list(info.keys())}")

            if step <= 3 or step % args.save_every == 0 or tm or tc:
                save_debug_sample(args.save_dir, step, raw_obs, clean_obs, reward, tm, tc, info)
                print(f"[saved] {args.save_dir}/step_{step:06d}.*")

            if tm or tc:
                print(f"\n[episode done] step={step}, return={ep_return:.5f}")
                raw_obs, info = env.reset(
                    ((env.current_seed + 1) % config["num_scenarios"])
                    + env.engine.global_config["start_seed"]
                )
                clean_obs = extract_clean_obs(raw_obs)
                ep_return = 0.0
                ep_len = 0
                print_obs_summary(raw_obs, tag="reset_after_done_raw")
                print("[clean reset_after_done obs]")
                describe_array("clean_obs['image']", clean_obs["image"])
                describe_array("clean_obs['state']", clean_obs["state"])

        print("\n[DONE] Phase 1 debug finished.")
        print(f"Saved files in: {os.path.abspath(args.save_dir)}")

    finally:
        env.close()