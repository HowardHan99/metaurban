"""
SocialDynamicMetaUrbanEnv
=========================
A drop-in replacement for ``SidewalkDynamicMetaUrbanEnv`` that uses
``SocialScenarioManager`` to populate the scene with socially-rich agents.

New config keys (all default to the same values as the base env)
----------------------------------------------------------------
spawn_static_human_num : int   = 5
    Pedestrians that stand around as bystanders (waiting, chatting, looking at
    their phone).  They are real physics objects that the ego can collide with.

ped_linger_prob : float = 0.002
    Per-step, per-pedestrian probability of entering a "linger" pause.
    At 10 Hz this is roughly one spontaneous stop per ~100 steps (~10 s) per
    pedestrian on average.

ped_linger_steps_mean : int = 50
    Mean linger duration in simulation steps (~5 s at 10 Hz).  Actual duration
    is sampled from Normal(mean, 0.4*mean).

ped_group_num : int = 3
    Number of side-by-side walking pairs.  Each pair consists of a leader
    following the ORCA path and a follower that mirrors the leader with a
    0.8 m lateral offset, creating a natural "two people walking together"
    appearance.

ped_ego_yield_radius : float = 3.0
    Pedestrians within this many metres of the ego vehicle freeze for one
    simulation step.  Set to 0.0 to disable.

Quick start
-----------
>>> from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv
>>> env = SocialDynamicMetaUrbanEnv(dict(
...     spawn_human_num=40,
...     spawn_wheelchairman_num=3,
...     spawn_static_human_num=8,
...     ped_linger_prob=0.003,
...     ped_group_num=5,
...     ped_ego_yield_radius=4.0,
...     num_scenarios=100,
... ))
>>> obs, info = env.reset()
"""

from __future__ import annotations

from typing import Union

from metaurban.envs.sidewalk_dynamic_env import (
    SidewalkDynamicMetaUrbanEnv,
    METAURBAN_DEFAULT_CONFIG,
)
from metaurban.utils import Config

# ──────────────────────────────────────────────────────────────────────────────
# Additional default config values (social behavior knobs)
# ──────────────────────────────────────────────────────────────────────────────

SOCIAL_EXTRA_CONFIG = dict(
    # Static bystanders — pedestrians that stand in place
    spawn_static_human_num=5,

    # Linger behavior — spontaneous pauses while walking
    ped_linger_prob=0.002,
    ped_linger_steps_mean=50,

    # Group walking — N side-by-side pairs
    ped_group_num=3,

    # Ego-yield zone — pedestrians freeze when ego is close
    ped_ego_yield_radius=3.0,
)


class SocialDynamicMetaUrbanEnv(SidewalkDynamicMetaUrbanEnv):
    """
    MetaUrban dynamic environment enriched with four social agent behaviors.

    Use in place of ``SidewalkDynamicMetaUrbanEnv`` for offline dataset
    collection or RL training that targets social navigation.
    """

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(SOCIAL_EXTRA_CONFIG)
        return config

    def setup_engine(self) -> None:
        # Run the parent setup (registers map_manager, asset_manager,
        # traffic_manager, and the stock humanoid_manager).
        super().setup_engine()

        # Swap out the stock humanoid manager for our social-aware version.
        from metaurban.manager.social_scenario_manager import SocialScenarioManager

        self.engine.update_manager("humanoid_manager", SocialScenarioManager())
