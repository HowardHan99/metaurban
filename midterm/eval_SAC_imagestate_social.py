import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv


LABEL_NAMES = ["NEGATIVE_SOCIAL", "NEUTRAL", "POSITIVE_SOCIAL"]


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


class FusionStudentNet(nn.Module):
    def __init__(
        self,
        ego_dim: int,
        action_dim: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
        image_feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.state_action_encoder = nn.Sequential(
            nn.Linear(ego_dim + action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, ego_state, action):
        img_feat = self.image_encoder(image)
        state_action = torch.cat([ego_state, action], dim=1)
        sa_feat = self.state_action_encoder(state_action)
        fused = torch.cat([img_feat, sa_feat], dim=1)
        logits = self.classifier(fused)
        return logits


class StudentRewardWrapper(gym.Wrapper):
    """
    Use trained student model:
      image + ego state dims [7, 8] + action -> label -> reward
    and add it to the base env reward.
    """

    def __init__(
        self,
        env: gym.Env,
        student_model_path: str,
        reward_scale: float = 1.0,
        negative_reward: float = -1.0,
        neutral_reward: float = 0.0,
        positive_reward: float = 1.0,
        state_indices=(7, 8),
        device: str = "auto",
    ):
        super().__init__(env)

        self.student_model_path = str(student_model_path)
        self.reward_scale = float(reward_scale)
        self.state_indices = tuple(state_indices)

        self.label_to_reward = {
            0: float(negative_reward),
            1: float(neutral_reward),
            2: float(positive_reward),
        }

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(self.student_model_path, map_location=self.device)

        self.ego_dim = int(ckpt.get("ego_dim", 9))
        self.action_dim = int(ckpt.get("action_dim", 2))
        self.num_classes = int(ckpt.get("num_classes", 3))
        self.dropout = float(ckpt.get("dropout", 0.2))
        self.image_width = int(ckpt.get("image_width", 128))
        self.image_height = int(ckpt.get("image_height", 72))
        self.pretrained_backbone = bool(ckpt.get("pretrained_backbone", False))
        self.freeze_backbone = bool(ckpt.get("freeze_backbone", False))

        self.model = FusionStudentNet(
            ego_dim=self.ego_dim,
            action_dim=self.action_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
            pretrained_backbone=self.pretrained_backbone,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
        ])

    def _build_ego_input(self, full_state: np.ndarray) -> np.ndarray:
        full_state = np.asarray(full_state, dtype=np.float32).reshape(-1)

        if self.ego_dim == len(self.state_indices):
            vals = []
            for idx in self.state_indices:
                vals.append(full_state[idx] if idx < len(full_state) else 0.0)
            return np.asarray(vals, dtype=np.float32)

        ego = np.zeros((self.ego_dim,), dtype=np.float32)
        for idx in self.state_indices:
            if idx < len(full_state) and idx < self.ego_dim:
                ego[idx] = full_state[idx]
        return ego

    def _prepare_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        out = np.zeros((self.action_dim,), dtype=np.float32)
        n = min(len(action), self.action_dim)
        out[:n] = action[:n]
        return out

    @torch.no_grad()
    def _infer_label(self, obs: dict, action: np.ndarray):
        image = obs["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = self.transform(image).unsqueeze(0).to(self.device)

        full_state = np.asarray(obs["state"], dtype=np.float32)
        ego = self._build_ego_input(full_state)
        ego = torch.tensor(ego, dtype=torch.float32, device=self.device).unsqueeze(0)

        action = self._prepare_action(action)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.model(image, ego, action)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))
        return pred, probs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        pred_label, probs = self._infer_label(obs, action)
        student_reward = self.label_to_reward[pred_label] * self.reward_scale
        total_reward = float(reward + student_reward)

        info = dict(info)
        info["base_env_reward"] = float(reward)
        info["student_label"] = int(pred_label)
        info["student_label_name"] = LABEL_NAMES[pred_label]
        info["student_reward"] = float(student_reward)
        info["student_probs"] = probs.tolist()
        info["total_reward_after_student"] = float(total_reward)

        return obs, total_reward, terminated, truncated, info


def _to_xy(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def patch_orca_planning_fallback() -> None:
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


def build_config(args):
    den_scale = 1
    return dict(
        crswalk_density=1,
        object_density=args.object_density,
        use_render=False,
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
    )


def build_image_state_env_config(image_width: int, image_height: int, training: bool, args: argparse.Namespace) -> dict:
    cfg = build_config(args)

    cfg["image_observation"] = True
    cfg["agent_observation"] = ThreeSourceMixObservation
    cfg["sensors"] = dict(
        rgb_camera=(RGBCamera, image_width, image_height),
        depth_camera=(DepthCamera, 84, 84),
        semantic_camera=(SemanticCamera, 84, 84),
    )

    return cfg


def make_eval_env(
    seed: int,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
):
    patch_orca_planning_fallback()

    cfg = build_image_state_env_config(
        image_width=image_width,
        image_height=image_height,
        training=True,
        args=args,
    )
    cfg["start_seed"] = int(seed)
    cfg["log_level"] = 50

    env = SocialDynamicMetaUrbanEnv(cfg)
    env = CleanDictObsWrapper(env, image_width=image_width, image_height=image_height)
    env = StudentRewardWrapper(
        env,
        student_model_path=args.student_model_path,
        reward_scale=args.student_reward_scale,
        negative_reward=args.student_negative_reward,
        neutral_reward=args.student_neutral_reward,
        positive_reward=args.student_positive_reward,
        state_indices=(7, 8),
        device=args.student_device,
    )
    if args.use_idle_penalty:
        env = IdlePenaltyWrapper(env, penalty=args.idle_penalty, speed_threshold=args.speed_threshold)
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
    reason_keys = ["done_reason", "termination_reason", "episode_result", "result"]
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
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    env = make_eval_env(
        seed=seeds[0] if len(seeds) > 0 else 0,
        image_width=image_width,
        image_height=image_height,
        args=args,
    )
    model = SAC.load(str(model_path), env=env, device="auto")

    episode_rows: List[Dict[str, Any]] = []
    try:
        for episode_idx, seed in enumerate(seeds):
            obs, info = env.reset(seed=int(seed))
            terminated = False
            truncated = False
            done = False
            ep_total_reward = 0.0
            ep_base_env_reward = 0.0
            ep_student_reward = 0.0
            ep_len = 0
            label_counts = {name: 0 for name in LABEL_NAMES}
            final_info = info if isinstance(info, dict) else {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                step_total_reward = float(reward)
                step_base_env_reward = float(info.get("base_env_reward", reward))
                step_student_reward = float(info.get("student_reward", 0.0))
                step_label_name = str(info.get("student_label_name", "NEUTRAL"))

                ep_total_reward += step_total_reward
                ep_base_env_reward += step_base_env_reward
                ep_student_reward += step_student_reward
                ep_len += 1
                if step_label_name in label_counts:
                    label_counts[step_label_name] += 1
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
                "episode_total_reward": float(ep_total_reward),
                "episode_base_env_reward": float(ep_base_env_reward),
                "episode_student_reward": float(ep_student_reward),
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
                "negative_social_steps": int(label_counts["NEGATIVE_SOCIAL"]),
                "neutral_steps": int(label_counts["NEUTRAL"]),
                "positive_social_steps": int(label_counts["POSITIVE_SOCIAL"]),
            }
            episode_rows.append(row)
    finally:
        env.close()

    return episode_rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = np.array([r["success"] for r in rows], dtype=np.float32)
    total_rewards = np.array([r["episode_total_reward"] for r in rows], dtype=np.float32)
    base_rewards = np.array([r["episode_base_env_reward"] for r in rows], dtype=np.float32)
    student_rewards = np.array([r["episode_student_reward"] for r in rows], dtype=np.float32)
    lengths = np.array([r["episode_length"] for r in rows], dtype=np.float32)
    neg_steps = np.array([r["negative_social_steps"] for r in rows], dtype=np.float32)
    neu_steps = np.array([r["neutral_steps"] for r in rows], dtype=np.float32)
    pos_steps = np.array([r["positive_social_steps"] for r in rows], dtype=np.float32)

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
        "total_reward_mean": float(total_rewards.mean()) if len(total_rewards) else 0.0,
        "total_reward_std": float(total_rewards.std()) if len(total_rewards) else 0.0,
        "base_env_reward_mean": float(base_rewards.mean()) if len(base_rewards) else 0.0,
        "base_env_reward_std": float(base_rewards.std()) if len(base_rewards) else 0.0,
        "student_reward_mean": float(student_rewards.mean()) if len(student_rewards) else 0.0,
        "student_reward_std": float(student_rewards.std()) if len(student_rewards) else 0.0,
        "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "length_std": float(lengths.std()) if len(lengths) else 0.0,
        "negative_social_steps_mean": float(neg_steps.mean()) if len(neg_steps) else 0.0,
        "neutral_steps_mean": float(neu_steps.mean()) if len(neu_steps) else 0.0,
        "positive_social_steps_mean": float(pos_steps.mean()) if len(pos_steps) else 0.0,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAC checkpoints on SocialDynamicMetaUrbanEnv with student reward logging."
    )

    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/howardhan/metaurban/midterm_logs/SAC_image_state_drive_social/sac_drive_social_seed20_0422_0930/checkpoints")
    parser.add_argument("--output_dir", type=str,
                        default="/home/howardhan/metaurban/midterm_logs/SAC_image_state_drive_social/sac_drive_social_seed20_0422_0930/checkpoints/eval_social_student")
    parser.add_argument("--seeds", type=str, default="20,30,40,50,60,70,80,90,100",
                        help="Comma-separated list of seeds to evaluate")
    parser.add_argument("--image_width", type=int, default=80)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reverse", action="store_true")

    parser.add_argument("--student_model_path", type=str,
                        default="/home/howardhan/metaurban/recorded_dataset/student_runs/20260420_170419/best_student_image_ego_action.pt")
    parser.add_argument("--student_reward_scale", type=float, default=0.05)
    parser.add_argument("--student_negative_reward", type=float, default=-1.0)
    parser.add_argument("--student_neutral_reward", type=float, default=0.0)
    parser.add_argument("--student_positive_reward", type=float, default=1.0)
    parser.add_argument("--student_device", type=str, default="auto")

    parser.add_argument("--use_idle_penalty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--idle_penalty", type=float, default=0.1)
    parser.add_argument("--speed_threshold", type=float, default=0.5)

    parser.add_argument("--map", type=str, default="C")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--object_density", type=float, default=0.01)

    parser.add_argument("--scene_type", type=str, default="default",
                        choices=["default", "commercial", "commute", "leisure", "constrained"])
    parser.add_argument("--scene_building_source", type=str, default="default",
                        choices=["scene", "default"])

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
    parser.add_argument("--spawn_increase_per_episode", type=int, default=0,
                        help="Increase spawn_human_num by this amount after each reset (0=no increase)")
    parser.add_argument("--spawn_elderly_num", type=int, default=0)
    parser.add_argument("--ignore_success_done", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "eval_social_student"
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
    print(f"[INFO] Student model: {args.student_model_path}")

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
            args=args,
        )
        summary = summarize_rows(episode_rows)
        all_episode_rows.extend(episode_rows)
        summary_rows.append(summary)

        step = summary["step"]

        writer.add_scalar("eval/success_rate", summary["success_rate"], step)
        writer.add_scalar("eval/total_reward_mean", summary["total_reward_mean"], step)
        writer.add_scalar("eval/base_env_reward_mean", summary["base_env_reward_mean"], step)
        writer.add_scalar("eval/student_reward_mean", summary["student_reward_mean"], step)
        writer.add_scalar("eval/episode_length", summary["length_mean"], step)
        writer.add_scalar("eval/negative_social_steps_mean", summary["negative_social_steps_mean"], step)
        writer.add_scalar("eval/neutral_steps_mean", summary["neutral_steps_mean"], step)
        writer.add_scalar("eval/positive_social_steps_mean", summary["positive_social_steps_mean"], step)

        print(
            f"[RESULT] step={summary['step']} | "
            f"success_rate={summary['success_rate']:.3f} | "
            f"base_env_reward_mean={summary['base_env_reward_mean']:.3f} | "
            f"student_reward_mean={summary['student_reward_mean']:.3f} | "
            f"total_reward_mean={summary['total_reward_mean']:.3f}"
        )

    summary_rows.sort(key=lambda x: x["step"])
    ranked_rows = sorted(
        summary_rows,
        key=lambda x: (x["success_rate"], x["total_reward_mean"], x["student_reward_mean"]),
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