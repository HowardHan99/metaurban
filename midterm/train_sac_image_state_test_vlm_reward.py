import argparse
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

from env_config import EVAL_VEC_ENV_SEEDS


LABEL_NAMES = ["NEGATIVE_SOCIAL", "NEUTRAL", "POSITIVE_SOCIAL"]
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


class VLMRewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.vlm_rewards = []
        self.base_rewards = []
        self.total_rewards = []

        self.ep_vlm = 0.0
        self.ep_base = 0.0
        self.ep_total = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        for i, info in enumerate(infos):
            if "student_reward" not in info:
                continue

            vlm_r = float(info["student_reward"])
            base_r = float(info.get("base_env_reward", 0.0))

            if rewards is not None:
                total_r = float(rewards[i])
            else:
                total_r = vlm_r + base_r

            self.logger.record("reward/vlm_step", vlm_r)
            self.logger.record("reward/base_step", base_r)
            self.logger.record("reward/total_step", total_r)

            self.vlm_rewards.append(vlm_r)
            self.base_rewards.append(base_r)
            self.total_rewards.append(total_r)

            self.ep_vlm += vlm_r
            self.ep_base += base_r
            self.ep_total += total_r

            if dones is not None and dones[i]:
                self.logger.record("reward/vlm_episode", self.ep_vlm)
                self.logger.record("reward/base_episode", self.ep_base)
                self.logger.record("reward/total_episode", self.ep_total)

                self.ep_vlm = 0.0
                self.ep_base = 0.0
                self.ep_total = 0.0

        return True

    def _on_rollout_end(self):
        if len(self.vlm_rewards) > 0:
            self.logger.record("reward/vlm_reward_mean", float(np.mean(self.vlm_rewards)))
            self.logger.record("reward/base_reward_mean", float(np.mean(self.base_rewards)))
            self.logger.record("reward/total_reward_mean", float(np.mean(self.total_rewards)))

            self.logger.record("reward/vlm_reward_min", float(np.min(self.vlm_rewards)))
            self.logger.record("reward/vlm_reward_max", float(np.max(self.vlm_rewards)))
            self.logger.record("reward/base_reward_min", float(np.min(self.base_rewards)))
            self.logger.record("reward/base_reward_max", float(np.max(self.base_rewards)))
            self.logger.record("reward/total_reward_min", float(np.min(self.total_rewards)))
            self.logger.record("reward/total_reward_max", float(np.max(self.total_rewards)))

            self.vlm_rewards = []
            self.base_rewards = []
            self.total_rewards = []


class IdlePenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = 0.1, speed_threshold: float = 0.5):
        super().__init__(env)
        self.penalty = float(penalty)
        self.speed_threshold = float(speed_threshold)

    def reset(self, *, seed=None, options=None, **kwargs):
        return self.env.reset(seed=seed, options=options, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        speed = None
        try:
            speed = self.env.unwrapped.agent.speed_km_h
        except AttributeError:
            try:
                speed = self.env.unwrapped.vehicle.speed_km_h
            except AttributeError:
                speed = None

        if speed is not None and speed < self.speed_threshold:
            reward -= self.penalty

        return obs, reward, terminated, truncated, info


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
            raise ValueError(f"Unexpected image ndim: {img.ndim}, shape={img.shape}")
        if img.shape[2] < 3:
            raise ValueError(f"Unexpected image channels: {img.shape}")
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
        for src_i, idx in enumerate(self.state_indices):
            if idx < len(full_state) and src_i < self.ego_dim:
                ego[src_i] = full_state[idx]
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
    student_model_path: str,
    student_reward_scale: float,
    student_negative_reward: float,
    student_neutral_reward: float,
    student_positive_reward: float,
    student_device: str,
    use_idle_penalty: bool = True,
    idle_penalty: float = 0.1,
    speed_threshold: float = 0.5,
):
    def _init():
        patch_planning_fallback()

        env = SocialDynamicMetaUrbanEnv(env_cfg)
        env = SeededResetWrapper(env, base_seed=seed, num_scenarios=env_cfg["num_scenarios"])
        env = CleanDictObsWrapper(env, image_width=image_width, image_height=image_height)

        env = StudentRewardWrapper(
            env,
            student_model_path=student_model_path,
            reward_scale=student_reward_scale,
            negative_reward=student_negative_reward,
            neutral_reward=student_neutral_reward,
            positive_reward=student_positive_reward,
            state_indices=(7, 8),
            device=student_device,
        )

        if use_idle_penalty:
            env = IdlePenaltyWrapper(env, penalty=idle_penalty, speed_threshold=speed_threshold)

        env = Monitor(env)
        return env

    return _init


class SafeEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        try:
            return super()._on_step()
        except Exception as e:
            print(f"[WARN] Eval failed, skipping this round: {repr(e)}")
            return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on SocialDynamicMetaUrbanEnv in drive-social style")

    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=20000)
    parser.add_argument("--checkpoint_freq", type=int, default=20000)
    parser.add_argument("--log_dir", type=str, default="./midterm_logs/SAC_image_state_drive_social")
    parser.add_argument("--resume_from", type=str, default="./sac_imgstate_260000_steps.zip")

    parser.add_argument("--image_width", type=int, default=80)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--idle_penalty", type=float, default=0.1)
    parser.add_argument("--speed_threshold", type=float, default=0.3)
    parser.add_argument("--disable_eval", action="store_true")

    parser.add_argument(
        "--student_model_path",
        type=str,
        default="/home/howardhan/metaurban/recorded_dataset/student_runs/20260420_170419/best_student_image_ego_action.pt",
    )
    parser.add_argument("--student_reward_scale", type=float, default=0.05)
    parser.add_argument("--student_negative_reward", type=float, default=-1.0)
    parser.add_argument("--student_neutral_reward", type=float, default=0.0)
    parser.add_argument("--student_positive_reward", type=float, default=1.0)
    parser.add_argument("--student_device", type=str, default="auto")

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
    parser.add_argument("--spawn_increase_per_episode", type=int, default=0)
    parser.add_argument("--spawn_elderly_num", type=int, default=0)
    parser.add_argument("--ignore_success_done", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_name = f"sac_drive_social_seed{args.seed}_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join(args.log_dir, run_name)
    tb_log_dir = os.path.join(log_dir, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    env = None
    eval_env = None

    try:
        train_cfg = build_config(args)
        env = DummyVecEnv([
            make_env(
                train_cfg,
                seed=args.seed,
                image_width=args.image_width,
                image_height=args.image_height,
                student_model_path=args.student_model_path,
                student_reward_scale=args.student_reward_scale,
                student_negative_reward=args.student_negative_reward,
                student_neutral_reward=args.student_neutral_reward,
                student_positive_reward=args.student_positive_reward,
                student_device=args.student_device,
                use_idle_penalty=True,
                idle_penalty=args.idle_penalty,
                speed_threshold=args.speed_threshold,
            )
        ])

        if args.disable_eval:
            eval_env = None
            eval_seeds = []
        else:
            eval_seed = int(EVAL_VEC_ENV_SEEDS[0]) + args.seed if len(EVAL_VEC_ENV_SEEDS) > 0 else args.seed + 1000
            eval_seed = eval_seed % args.num_scenarios
            eval_cfg = build_config(args)
            eval_env = DummyVecEnv([
                make_env(
                    eval_cfg,
                    seed=eval_seed,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    student_model_path=args.student_model_path,
                    student_reward_scale=args.student_reward_scale,
                    student_negative_reward=args.student_negative_reward,
                    student_neutral_reward=args.student_neutral_reward,
                    student_positive_reward=args.student_positive_reward,
                    student_device=args.student_device,
                    use_idle_penalty=False,
                    idle_penalty=args.idle_penalty,
                    speed_threshold=args.speed_threshold,
                )
            ])
            eval_seeds = [eval_seed]

        print(f"=== Training SAC for {args.total_timesteps} steps ===")
        print(f"seed: {args.seed}")
        print(f"image size: {args.image_width}x{args.image_height}")
        print(f"log dir: {log_dir}")
        print(f"tensorboard: {tb_log_dir}")
        print(f"train obs space: {env.observation_space}")
        print(f"action space: {env.action_space}")
        if len(eval_seeds) > 0:
            print(f"eval seeds: {eval_seeds}")

        if args.resume_from and os.path.exists(args.resume_from):
            model = SAC.load(args.resume_from, env=env, device="auto", seed=args.seed)
            model.tensorboard_log = tb_log_dir
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
                tensorboard_log=tb_log_dir,
                device="auto",
            )
            print("Created new SAC model.")

        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq, 1),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="sac_imgstate",
        )

        vlm_cb = VLMRewardLoggerCallback()
        callback_list = [checkpoint_cb, vlm_cb]

        if not args.disable_eval and eval_env is not None:
            eval_cb = SafeEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_dir, "best_model"),
                log_path=os.path.join(log_dir, "eval_logs"),
                eval_freq=max(args.eval_freq, 1),
                n_eval_episodes=1,
                deterministic=True,
                render=False,
            )
            callback_list.insert(1, eval_cb)
        else:
            print("[INFO] Eval disabled.")

        callbacks = CallbackList(callback_list)

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="SAC_image_state",
            reset_num_timesteps=not (args.resume_from and os.path.exists(args.resume_from)),
        )

        model.save(os.path.join(log_dir, "final_model"))
        print(f"=== Training complete. Saved to {log_dir}/final_model.zip ===")

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