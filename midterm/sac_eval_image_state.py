
import argparse
import copy
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from torch.utils.tensorboard import SummaryWriter
from env_config import ENV_CONFIG


class IdlePenaltyWrapper(gym.Wrapper):
    """Same wrapper style as training, but disabled by default for eval."""

    def __init__(self, env: gym.Env, penalty: float = 0.1, speed_threshold: float = 0.5):
        super().__init__(env)
        self.penalty = float(penalty)
        self.speed_threshold = float(speed_threshold)

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


class CleanDictObsWrapper(gym.Wrapper):
    """
    Convert MetaUrban raw dict obs into a clean Dict observation for SB3:
      - image: uint8 RGB, shape (H, W, 3)
      - state: float32 vector
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


def build_image_state_env_config(image_width: int, image_height: int) -> dict:
    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = True
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


def make_eval_env(
    seed: int,
    image_width: int,
    image_height: int,
    use_idle_penalty: bool = False,
    idle_penalty: float = 0.1,
    speed_threshold: float = 0.5,
):
    patch_orca_planning_fallback()

    cfg = build_image_state_env_config(image_width, image_height)
    cfg["start_seed"] = int(seed)
    cfg["log_level"] = 50

    env = SidewalkStaticMetaUrbanEnv(cfg)
    env = CleanDictObsWrapper(env, image_width=image_width, image_height=image_height)
    if use_idle_penalty:
        env = IdlePenaltyWrapper(env, penalty=idle_penalty, speed_threshold=speed_threshold)
    env = Monitor(env)
    return env


def parse_checkpoint_step(path: Path) -> int:
    m = re.search(r"_(\d+)_steps\.zip$", path.name)
    if m:
        return int(m.group(1))
    if path.stem == "final_model":
        return 10**18
    return -1


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    candidates = []
    for p in checkpoint_dir.glob("*.zip"):
        if p.is_file():
            candidates.append(p)
    candidates.sort(key=parse_checkpoint_step)
    return candidates


def get_done_reason(info: Dict[str, Any]) -> str:
    reason_keys = [
        "done_reason",
        "termination_reason",
        "episode_result",
        "result",
    ]
    for k in reason_keys:
        if k in info and info[k] is not None:
            return str(info[k])

    if info.get("arrive_dest", False):
        return "arrive_dest"
    if info.get("crash_vehicle", False):
        return "crash_vehicle"
    if info.get("crash_object", False):
        return "crash_object"
    if info.get("crash_human", False):
        return "crash_human"
    if info.get("out_of_road", False):
        return "out_of_road"
    if info.get("max_step", False):
        return "max_step"
    return "unknown"


def episode_success(info: Dict[str, Any]) -> int:
    if "success" in info:
        return int(bool(info["success"]))
    return int(bool(info.get("arrive_dest", False)))


def evaluate_one_checkpoint(
    model_path: Path,
    seeds: List[int],
    image_width: int,
    image_height: int,
    deterministic: bool,
    max_steps_per_episode: Optional[int],
) -> List[Dict[str, Any]]:
    env = make_eval_env(
        seed=seeds[0] if len(seeds) > 0 else 0,
        image_width=image_width,
        image_height=image_height,
        use_idle_penalty=False,
    )
    model = SAC.load(str(model_path), env=env, device="auto")

    episode_rows: List[Dict[str, Any]] = []
    try:
        for episode_idx, seed in enumerate(seeds):
            obs, info = env.reset(seed=int(seed))
            terminated = False
            truncated = False
            done = False
            ep_reward = 0.0
            ep_len = 0
            final_info = info if isinstance(info, dict) else {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                final_info = info if isinstance(info, dict) else {}

                if max_steps_per_episode is not None and ep_len >= max_steps_per_episode:
                    truncated = True
                    done = True
                else:
                    done = bool(terminated or truncated)

            row = {
                "checkpoint_name": model_path.name,
                "checkpoint_path": str(model_path),
                "step": parse_checkpoint_step(model_path),
                "episode_idx": episode_idx,
                "seed": int(seed),
                "success": episode_success(final_info),
                "episode_reward": float(ep_reward),
                "episode_length": int(ep_len),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "done_reason": get_done_reason(final_info),
                "arrive_dest": int(bool(final_info.get("arrive_dest", False))),
                "crash_vehicle": int(bool(final_info.get("crash_vehicle", False))),
                "crash_object": int(bool(final_info.get("crash_object", False))),
                "crash_human": int(bool(final_info.get("crash_human", False))),
                "out_of_road": int(bool(final_info.get("out_of_road", False))),
                "max_step": int(bool(final_info.get("max_step", False))),
            }
            episode_rows.append(row)
    finally:
        env.close()

    return episode_rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = np.array([r["success"] for r in rows], dtype=np.float32)
    rewards = np.array([r["episode_reward"] for r in rows], dtype=np.float32)
    lengths = np.array([r["episode_length"] for r in rows], dtype=np.float32)

    reason_hist: Dict[str, int] = {}
    for r in rows:
        reason_hist[r["done_reason"]] = reason_hist.get(r["done_reason"], 0) + 1

    return {
        "checkpoint_name": rows[0]["checkpoint_name"],
        "checkpoint_path": rows[0]["checkpoint_path"],
        "step": int(rows[0]["step"]),
        "num_episodes": int(len(rows)),
        "success_rate": float(success.mean()) if len(success) else 0.0,
        "success_count": int(success.sum()) if len(success) else 0,
        "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "length_std": float(lengths.std()) if len(lengths) else 0.0,
        "done_reason_hist": reason_hist,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(seed_str: str) -> List[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all SAC checkpoints and save results.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./midterm_logs/SAC_image_state/sac_imgstate_seed0_0415_1149/checkpoints",
        help="Directory containing checkpoint zip files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./midterm_logs/SAC_image_state/sac_imgstate_seed0_0415_1149/eval_after_train",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
    )
    parser.add_argument("--image_width", type=int, default=80)
    parser.add_argument("--image_height", type=int, default=60)
    # parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N checkpoints after sorting. Useful for quick debugging.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Evaluate from largest step to smallest step.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "eval_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tb_eval"))

    seeds = parse_seeds(args.seeds)
    checkpoints = find_checkpoints(checkpoint_dir)
    if args.reverse:
        checkpoints = list(reversed(checkpoints))
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]

    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No .zip checkpoints found in: {checkpoint_dir}")

    print(f"[INFO] Found {len(checkpoints)} checkpoints")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Output dir: {output_dir}")

    all_episode_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, ckpt_path in enumerate(checkpoints, start=1):
        print(f"\n=== [{idx}/{len(checkpoints)}] Evaluating {ckpt_path.name} ===")
        episode_rows = evaluate_one_checkpoint(
            model_path=ckpt_path,
            seeds=seeds,
            image_width=args.image_width,
            image_height=args.image_height,
            deterministic=args.deterministic,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        summary = summarize_rows(episode_rows)
        all_episode_rows.extend(episode_rows)
        summary_rows.append(summary)

        step = summary["step"]

        writer.add_scalar("eval/success_rate", summary["success_rate"], step)
        writer.add_scalar("eval/reward_mean", summary["reward_mean"], step)
        writer.add_scalar("eval/episode_length", summary["length_mean"], step)

        print(
            f"[RESULT] step={summary['step']} | "
            f"success_rate={summary['success_rate']:.3f} | "
            f"reward_mean={summary['reward_mean']:.3f} | "
            f"length_mean={summary['length_mean']:.1f}"
        )

    summary_rows.sort(key=lambda x: x["step"])
    ranked_rows = sorted(
        summary_rows,
        key=lambda x: (x["success_rate"], x["reward_mean"]),
        reverse=True,
    )

    write_csv(output_dir / "checkpoint_eval_summary.csv", summary_rows)
    write_csv(output_dir / "checkpoint_eval_episodes.csv", all_episode_rows)

    with open(output_dir / "checkpoint_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    with open(output_dir / "checkpoint_eval_ranking.json", "w", encoding="utf-8") as f:
        json.dump(ranked_rows, f, indent=2, ensure_ascii=False)

    best = ranked_rows[0]
    print("\n=== BEST CHECKPOINT ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\nSaved files to: {output_dir}")
    writer.close()


if __name__ == "__main__":
    main()
