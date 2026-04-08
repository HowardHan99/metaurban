"""
SocialDynamicMetaUrbanEnv
=========================
A drop-in replacement for ``SidewalkDynamicMetaUrbanEnv`` that uses
``SocialScenarioManager`` and role-oriented social behavior controls.
"""

from __future__ import annotations

import logging
import numpy as np
from metaurban.envs.sidewalk_dynamic_env import SidewalkDynamicMetaUrbanEnv
from metaurban.component.map.pg_map import MapGenerateMethod
from metaurban.component.map.base_map import BaseMap
from metaurban.manager.scene_builder import SceneBuilder
from metaurban.utils import Config
from metaurban.constants import TerminationState


logger = logging.getLogger(__name__)

SOCIAL_EXTRA_CONFIG = dict(
    # Scene type configuration
    scene_type="commercial",  # One of: commercial, commute, leisure, constrained
    scene_building_source="scene",  # One of: scene, default

    # Robot spawn location configuration
    # If True, robot spawns on sidewalk; if False, robot spawns on road (default)
    spawn_robot_on_sidewalk=False,

    # Core four social role counts
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

    # Group placement controls (preview-oriented)
    group_spawn_near_ego=False,
    group_spawn_min_radius=5.0,
    group_spawn_max_radius=10.0,
    group_cluster_num=3,
    group_cluster_size_min=5,
    group_cluster_size_max=8,
    group_member_radius=1.35,
    group_member_ring_step=0.55,
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
    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(SOCIAL_EXTRA_CONFIG)
        return config

    def _post_process_config(self, config):
        config = super()._post_process_config(config)

        if config.get("scene_building_source", "scene") == "default":
            config["scene_building_source"] = "default"
        else:
            config["scene_building_source"] = "scene"

        # If user does not explicitly choose a map sequence, use scene_type pattern.
        if config.get("map") == self.default_config_copy.get("map"):
            scene_type = config.get("scene_type", "commercial")
            scene_builder = SceneBuilder(scene_type=scene_type)
            map_pattern = scene_builder.get_map_pattern()
            config["map"] = map_pattern
            config["map_config"][BaseMap.GENERATE_TYPE] = MapGenerateMethod.BIG_BLOCK_SEQUENCE
            config["map_config"][BaseMap.GENERATE_CONFIG] = map_pattern

        return config

    def reset(self, seed=None):
        """Override reset to optionally place agent on sidewalk after standard reset."""
        obs, info = super().reset(seed=seed)

        # If enabled, move agent to sidewalk after reset
        if self.config.get("spawn_robot_on_sidewalk", False):
            try:
                self._place_agent_on_sidewalk()
            except Exception:
                # If sidewalk placement fails, keep agent at default spawn
                logger.exception("Failed to place ego on sidewalk; fallback to default spawn")

        return obs, info

    def _place_agent_on_sidewalk(self):
        """Place the ego near pedestrian anchors, which are already on sidewalk regions."""
        if not hasattr(self, "agent") or self.agent is None:
            return

        agent = self.agent
        if not hasattr(agent, "position"):
            return

        try:
            # Current ego state.
            current_pos = agent.position
            if len(current_pos) < 2:
                return

            # Preferred path: use humanoid start_points (always prepared at reset)
            # since they are sampled from sidewalk walkable regions.
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

            # Secondary path: place ego close to an active pedestrian.
            peds = list(getattr(hm, "_traffic_humanoids", [])) if hm is not None else []
            if peds:
                pick_idx = int(self.np_random.integers(len(peds)))
                ped_pos = peds[pick_idx].position
                if len(ped_pos) >= 2:
                    ped_xy = np.array([float(ped_pos[0]), float(ped_pos[1])], dtype=float)
                    # Small offset to avoid exact overlap at spawn.
                    jitter = self.np_random.normal(loc=0.0, scale=0.45, size=2)
                    jitter_norm = float(np.linalg.norm(jitter))
                    if jitter_norm > 1.2:
                        jitter = jitter / jitter_norm * 1.2
                    new_xy = ped_xy + jitter
                    agent.set_position((float(new_xy[0]), float(new_xy[1])))
                    return

            # Fallback path: move ego far enough from lane center to reach sidewalk.
            sidewalk_offset = float(self.np_random.uniform(6.5, 9.0))
            side = int(self.np_random.choice([-1, 1]))
            heading = float(agent.heading_theta)
            perp_x = side * sidewalk_offset * np.sin(heading)
            perp_y = side * sidewalk_offset * np.cos(heading)

            new_x = float(current_pos[0]) + perp_x
            new_y = float(current_pos[1]) + perp_y
            agent.set_position((new_x, new_y))

            # Rotate slightly inward towards the road.
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