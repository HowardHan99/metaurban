"""
SocialDynamicMetaUrbanEnv
=========================
A drop-in replacement for ``SidewalkDynamicMetaUrbanEnv`` that uses
``SocialScenarioManager`` and role-oriented social behavior controls.
"""

from __future__ import annotations

from metaurban.envs.sidewalk_dynamic_env import SidewalkDynamicMetaUrbanEnv
from metaurban.utils import Config

SOCIAL_EXTRA_CONFIG = dict(
    # Scene type configuration
    scene_type="commercial",  # One of: commercial, commute, leisure, constrained

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

    def setup_engine(self) -> None:
        super().setup_engine()
        from metaurban.manager.social_scenario_manager import SocialScenarioManager
        scene_type = self.config.get("scene_type", "commercial")
        manager = SocialScenarioManager(scene_type=scene_type)
        self.engine.update_manager("humanoid_manager", manager)
