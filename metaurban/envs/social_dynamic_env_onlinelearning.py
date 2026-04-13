"""
SocialDynamicMetaUrbanEnv
=========================
A drop-in replacement for ``SidewalkDynamicMetaUrbanEnv`` that uses
``SocialScenarioManager`` and role-oriented social behavior controls.
"""

from __future__ import annotations

import logging
import numpy as np
from metaurban.envs.sidewalk_dynamic_env_onlinelearning import SidewalkDynamicMetaUrbanEnv
from metaurban.component.map.pg_map import MapGenerateMethod
from metaurban.component.map.base_map import BaseMap
from metaurban.manager.scene_builder import SceneBuilder
from metaurban.utils import Config
from metaurban.constants import TerminationState


logger = logging.getLogger(__name__)

SOCIAL_EXTRA_CONFIG = dict(
    # Scene type configuration
    scene_type="default",  # default, commercial, commute, leisure, constrained
    scene_building_source="scene",  # scene or default

    # Robot spawn location
    spawn_robot_on_sidewalk=False,

    # Core social role counts
    crossing_ped_num=8,
    signaling_ped_num=0,
    vulnerable_ped_num=4,
    spawn_elderly_num=2,
    group_ped_pair_num=3,
    spawn_wheelchairman_num=1,
    spawn_edog_num=0,
    spawn_erobot_num=0,
    spawn_drobot_num=0,
    max_actor_num=1,

    # Behavior controls
    ped_ego_yield_radius=3.0,
    crossing_assertive_radius=4.0,
    crossing_speed_scale=1.10,
    signaling_prob=0.0,
    signaling_steps_mean=0,
    vulnerable_yield_radius=5.0,
    vulnerable_speed_scale=0.70,
    vulnerable_elderly_ratio=0.6,
    vulnerable_distracted_ratio=0.4,
    vulnerable_pause_prob=0.02,
    vulnerable_pause_steps_mean=16,
    pedestrian_sidewalk_only=False,
    pedestrian_allow_crosswalk=False,

    # Group placement controls
    group_spawn_near_ego=False,
    group_spawn_min_radius=5.0,
    group_spawn_max_radius=10.0,
    group_route_min_ego_distance=8.0,
    group_route_min_separation=5.5,
    group_cluster_num=4,
    group_cluster_size_min=3,
    group_cluster_size_max=5,
    group_member_radius=1.45,
    group_member_ring_step=0.62,
    group_member_radius_jitter=0.16,
    group_member_ring_step_jitter=0.12,
    group_member_idle_shift_prob=0.015,
    group_member_idle_shift_steps_mean=18,
    group_member_idle_shift_radius=0.22,
    group_cluster_min_separation=3.8,
    group_release_enable=True,
    group_release_steps_mean=180,
    group_release_steps_std=40,
    group_release_steps_min=60,
    ignore_success_done=False,

    # Required by sidewalk asset manager in this branch
    object_density=1.0,
    show_ego_navigation=False,
)


class SocialDynamicMetaUrbanEnv(SidewalkDynamicMetaUrbanEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self._init_reward_trackers()
        self._logged_reward_split_fallback = False

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(SOCIAL_EXTRA_CONFIG)
        return config

    def _init_reward_trackers(self):
        self._episode_env_reward = 0.0
        self._episode_vlm_reward = 0.0
        self._episode_total_reward = 0.0
        self._last_env_reward = 0.0
        self._last_vlm_reward = 0.0
        self._last_total_reward = 0.0

    def _post_process_config(self, config):
        config = super()._post_process_config(config)

        scene_type = str(config.get("scene_type", "default")).lower()
        if scene_type not in ("default", "commercial", "commute", "leisure", "constrained"):
            logger.warning("Unknown scene_type '%s'. Fallback to 'default'.", scene_type)
            scene_type = "default"
        config["scene_type"] = scene_type

        if config.get("scene_building_source", "scene") == "default":
            config["scene_building_source"] = "default"
        else:
            config["scene_building_source"] = "scene"

        if scene_type == "default":
            config["scene_building_source"] = "default"
            return config

        if config.get("map") == self.default_config_copy.get("map"):
            scene_builder = SceneBuilder(scene_type=scene_type)
            map_pattern = scene_builder.get_map_pattern()
            config["map"] = map_pattern
            config["map_config"][BaseMap.GENERATE_TYPE] = MapGenerateMethod.BIG_BLOCK_SEQUENCE
            config["map_config"][BaseMap.GENERATE_CONFIG] = map_pattern

        return config

    def reset(self, seed=None):
        self._init_reward_trackers()
        obs, info = super().reset(seed=seed)

        if self.config.get("spawn_robot_on_sidewalk", False):
            try:
                self._place_agent_on_sidewalk()
            except Exception:
                logger.exception("Failed to place ego on sidewalk; fallback to default spawn")

        return obs, info

    def _coerce_step_result(self, step_result):
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = bool(done)
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
        return obs, reward, terminated, truncated, info

    def _extract_reward_split(self, total_reward: float, info: dict):
        """
        Best-effort reward split.

        Priority:
        1. Reuse explicit keys if parent env already exposes them.
        2. Reconstruct from known weighted/raw VLM keys.
        3. Fall back to treating the returned reward as env reward and 0 as VLM reward.
        """
        # Total reward may already be exposed explicitly.
        total_reward = float(info.get("total_reward", total_reward))

        # Most useful explicit env keys to try.
        explicit_env_keys = (
            "env_reward",
            "native_reward",
            "base_reward",
            "reward_env",
            "step_env_reward",
            "raw_env_reward",
        )
        explicit_vlm_raw_keys = (
            "vlm_reward_raw",
            "raw_vlm_reward",
            "step_vlm_reward_raw",
        )
        explicit_vlm_weighted_keys = (
            "vlm_reward",
            "reward_vlm",
            "step_vlm_reward",
            "weighted_vlm_reward",
        )

        env_reward = None
        for key in explicit_env_keys:
            if key in info:
                env_reward = float(info[key])
                break

        vlm_reward_raw = None
        for key in explicit_vlm_raw_keys:
            if key in info:
                vlm_reward_raw = float(info[key])
                break

        vlm_reward_weighted = None
        for key in explicit_vlm_weighted_keys:
            if key in info:
                vlm_reward_weighted = float(info[key])
                break

        vlm_weight = float(self.config.get("vlm_reward_weight", 1.0))

        if env_reward is not None and vlm_reward_raw is not None:
            return env_reward, vlm_reward_raw, total_reward

        if env_reward is not None and vlm_reward_weighted is not None:
            if abs(vlm_weight) > 1e-8:
                vlm_reward_raw = vlm_reward_weighted / vlm_weight
            else:
                vlm_reward_raw = 0.0
            return env_reward, float(vlm_reward_raw), total_reward

        if env_reward is None and vlm_reward_raw is not None:
            env_reward = total_reward - vlm_weight * vlm_reward_raw
            return float(env_reward), float(vlm_reward_raw), total_reward

        if env_reward is None and vlm_reward_weighted is not None:
            env_reward = total_reward - vlm_reward_weighted
            if abs(vlm_weight) > 1e-8:
                vlm_reward_raw = vlm_reward_weighted / vlm_weight
            else:
                vlm_reward_raw = 0.0
            return float(env_reward), float(vlm_reward_raw), total_reward

        if not self._logged_reward_split_fallback:
            logger.warning(
                "Reward split keys were not found in step info. Falling back to env_reward=returned_reward and "
                "vlm_reward=0.0. If you want exact split logging, expose explicit env/VLM reward keys in the parent env."
            )
            self._logged_reward_split_fallback = True

        return total_reward, 0.0, total_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self._coerce_step_result(super().step(action))
        info = dict(info)

        env_reward, vlm_reward, total_reward = self._extract_reward_split(reward, info)

        self._last_env_reward = float(env_reward)
        self._last_vlm_reward = float(vlm_reward)
        self._last_total_reward = float(total_reward)

        self._episode_env_reward += self._last_env_reward
        self._episode_vlm_reward += self._last_vlm_reward
        self._episode_total_reward += self._last_total_reward

        # Step-level logging fields.
        info["env_reward"] = self._last_env_reward
        info["vlm_reward"] = self._last_vlm_reward
        info["total_reward"] = self._last_total_reward

        # Episode-level logging fields for Monitor(info_keywords=...).
        info["episode_env_reward"] = self._episode_env_reward
        info["episode_vlm_reward"] = self._episode_vlm_reward
        info["episode_total_reward"] = self._episode_total_reward

        return obs, float(total_reward), terminated, truncated, info

    def _place_agent_on_sidewalk(self):
        if not hasattr(self, "agent") or self.agent is None:
            return

        agent = self.agent
        if not hasattr(agent, "position"):
            return

        try:
            current_pos = agent.position
            if len(current_pos) < 2:
                return

            hm = getattr(self.engine, "humanoid_manager", None)
            start_points = list(getattr(hm, "start_points", [])) if hm is not None else []
            if hm is not None and start_points:
                pick_idx = int(self.np_random.integers(len(start_points)))
                anchor_world = hm._to_block_coordinate(start_points[pick_idx])
                anchor_xy = np.array([float(anchor_world[0]), float(anchor_world[1])], dtype=float)
                jitter = self.np_random.normal(loc=0.0, scale=0.40, size=2)
                jitter_norm = float(np.linalg.norm(jitter))
                if jitter_norm > 1.0:
                    jitter = jitter / jitter_norm * 1.0
                new_xy = anchor_xy + jitter
                agent.set_position((float(new_xy[0]), float(new_xy[1])))
                return

            peds = list(getattr(hm, "_traffic_humanoids", [])) if hm is not None else []
            if peds:
                pick_idx = int(self.np_random.integers(len(peds)))
                ped_pos = peds[pick_idx].position
                if len(ped_pos) >= 2:
                    ped_xy = np.array([float(ped_pos[0]), float(ped_pos[1])], dtype=float)
                    jitter = self.np_random.normal(loc=0.0, scale=0.45, size=2)
                    jitter_norm = float(np.linalg.norm(jitter))
                    if jitter_norm > 1.2:
                        jitter = jitter / jitter_norm * 1.2
                    new_xy = ped_xy + jitter
                    agent.set_position((float(new_xy[0]), float(new_xy[1])))
                    return

            sidewalk_offset = float(self.np_random.uniform(6.5, 9.0))
            side = int(self.np_random.choice([-1, 1]))
            heading = float(agent.heading_theta)
            perp_x = side * sidewalk_offset * np.sin(heading)
            perp_y = side * sidewalk_offset * np.cos(heading)

            new_x = float(current_pos[0]) + perp_x
            new_y = float(current_pos[1]) + perp_y
            agent.set_position((new_x, new_y))

            if side > 0:
                agent.set_heading_theta(heading - np.pi / 6)
            else:
                agent.set_heading_theta(heading + np.pi / 6)

        except Exception:
            logger.exception("Error while positioning ego on sidewalk")

    def setup_engine(self) -> None:
        super().setup_engine()
        from metaurban.manager.social_scenario_manager import SocialScenarioManager
        scene_type = self.config.get("scene_type", "commercial")
        manager = SocialScenarioManager(scene_type=scene_type)
        self.engine.update_manager("humanoid_manager", manager)

    def done_function(self, vehicle_id: str):
        done, done_info = super().done_function(vehicle_id)

        if self.config.get("ignore_success_done", False) and done_info.get(TerminationState.SUCCESS, False):
            fatal = (
                done_info.get(TerminationState.OUT_OF_ROAD, False)
                or done_info.get(TerminationState.CRASH_VEHICLE, False)
                or done_info.get(TerminationState.CRASH_OBJECT, False)
                or done_info.get(TerminationState.CRASH_BUILDING, False)
                or done_info.get(TerminationState.CRASH_HUMAN, False)
                or done_info.get(TerminationState.MAX_STEP, False)
            )
            if not fatal:
                done = False

        return done, done_info
