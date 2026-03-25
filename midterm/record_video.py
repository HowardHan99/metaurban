"""
Record videos of trained agents (or random) for demos / reports.

- Default ``--view main``: captures the **on-screen 3D simulation window** (chase / main camera),
  not the headless top-down bird's-eye. Requires a working display (Panda3D onscreen).
- All episodes are concatenated into **one** MP4 by default.
- ``--view top_down``: previous behavior (TopDownRenderer, headless-friendly).

Usage:
    python record_video.py --algo random --episodes 3
    python record_video.py --algo ppo --model_path ./midterm_logs/PPO/ppo_seed0/best_model/best_model.zip --episodes 3
    python record_video.py --algo sac --model_path ./midterm_logs/SAC/sac_seed1/best_model/best_model.zip --episodes 3
    python record_video.py --algo ppo --model_path ... --view top_down --split
"""
from __future__ import annotations

import argparse
import copy
import os

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC

from metaurban import SidewalkStaticMetaUrbanEnv
from env_config import ENV_CONFIG

ALGO_MAP = {"ppo": PPO, "sac": SAC}


def parse_args():
    parser = argparse.ArgumentParser(description="Record agent videos (main 3D window or top-down)")
    parser.add_argument("--algo", type=str, required=True, choices=["random", "ppo", "sac"])
    parser.add_argument("--model_path", type=str, default=None, help="Path to .zip model (required for ppo/sac)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./midterm_logs/videos")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Combined video path (default: {output_dir}/{algo}_combined.mp4)",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["main", "top_down"],
        default="main",
        help="main = 3D simulation window (needs display); top_down = BEV headless renderer",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        nargs=2,
        default=[1200, 900],
        metavar=("W", "H"),
        help="Main window size when --view main (must match MetaUrban expectations)",
    )
    parser.add_argument(
        "--screen_size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("W", "H"),
        help="Top-down canvas size when --view top_down",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also write one MP4 per episode ({algo}_episode_0.mp4, ...)",
    )
    parser.add_argument(
        "--show_window",
        action="store_true",
        help="With top_down: pop up BEV window while recording (main view always uses a real window)",
    )
    return parser.parse_args()


def _sac_reward_overrides(cfg: dict) -> None:
    cfg["no_negative_reward"] = False
    cfg["driving_reward"] = 3.0
    cfg["success_reward"] = 15.0
    cfg["speed_reward"] = 1.0
    cfg["lateral_penalty"] = 0.5
    cfg["crash_vehicle_penalty"] = 1.0
    cfg["crash_object_penalty"] = 1.0
    cfg["crash_human_penalty"] = 1.0
    cfg["crash_building_penalty"] = 1.0
    cfg["out_of_road_penalty"] = 2.0
    cfg["steering_range_penalty"] = 0.5


def frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    if len(f.shape) == 2:
        return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    if f.shape[2] == 4:
        return cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
    if f.shape[2] == 3:
        return cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    return f


def grab_main_frame(env) -> np.ndarray | None:
    """
    Main Panda3D window via engine._get_window_image().

    That helper returns **RGB** (it reverses framebuffer channels). OpenCV VideoWriter
    expects **BGR**; writing RGB as-is swaps R/B vs the on-screen window (orange↔blue sky).
    """
    try:
        img = env.engine._get_window_image()
    except Exception as e:
        print(f"WARNING: main-window capture failed ({e}). Try --view top_down on headless machines.")
        return None
    if img is None or img.size == 0:
        return None
    return frame_to_bgr(img)


def grab_top_down_frame(env, show_window: bool, screen_size: tuple[int, int]):
    frame = env.render(
        mode="top_down",
        window=show_window,
        screen_record=True,
        screen_size=screen_size,
    )
    if frame is None:
        return None
    return frame_to_bgr(np.asarray(frame))


def write_mp4(path: str, frames: list, fps: int) -> None:
    if not frames:
        print(f"WARNING: no frames for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for fr in frames:
        f = np.asarray(fr)
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_RGBA2BGR)
        elif f.shape[2] == 3:
            # grab_main_frame / grab_top_down_frame already supply BGR
            pass
        writer.write(f)
    writer.release()
    print(f"Saved {path} ({len(frames)} frames)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg["vehicle_config"] = copy.deepcopy(cfg["vehicle_config"])
    cfg["vehicle_config"]["show_dest_mark"] = True
    cfg["vehicle_config"]["show_line_to_navi_mark"] = True

    if args.view == "main":
        cfg["use_render"] = True
        cfg["window_size"] = tuple(args.window_size)
    else:
        cfg["use_render"] = False

    if args.algo == "sac":
        _sac_reward_overrides(cfg)

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=args.seed, log_level=50, **cfg))

    model = None
    if args.algo != "random":
        if args.model_path is None:
            raise ValueError(f"--model_path required for algo={args.algo}")
        model = ALGO_MAP[args.algo].load(args.model_path)

    combined_frames: list = []
    screen_size = (int(args.screen_size[0]), int(args.screen_size[1]))

    for ep in range(args.episodes):
        episode_frames: list = []
        obs, _ = env.reset(seed=args.seed + ep)

        if args.view == "main":
            env.render()
            f0 = grab_main_frame(env)
            if f0 is not None:
                episode_frames.append(f0)
        else:
            f0 = grab_top_down_frame(env, args.show_window, screen_size)
            if f0 is not None:
                episode_frames.append(f0)

        total_reward = 0.0
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if args.view == "main":
                env.render()
                frame = grab_main_frame(env)
            else:
                frame = grab_top_down_frame(env, args.show_window, screen_size)
            if frame is not None:
                episode_frames.append(frame)

        combined_frames.extend(episode_frames)
        print(f"Episode {ep}: {len(episode_frames)} frames, return={total_reward:.2f}")

        if args.split:
            split_path = os.path.join(args.output_dir, f"{args.algo}_episode_{ep}.mp4")
            write_mp4(split_path, episode_frames, args.fps)

    env.close()

    out_combined = args.output or os.path.join(args.output_dir, f"{args.algo}_combined.mp4")
    write_mp4(out_combined, combined_frames, args.fps)
    print(f"\nCombined video ({args.episodes} episodes): {out_combined}")


if __name__ == "__main__":
    main()
