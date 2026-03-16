"""
SocialScenarioManager
=====================
Drop-in replacement for ``PGBackgroundSidewalkAssetsManager`` that injects
four socially-rich agent behaviors on top of the existing ORCA-planned paths.

Behavior catalogue
------------------
1. **Static bystanders** (``spawn_static_human_num``)
   Pedestrians that stand in place — waiting, chatting, reading a phone.
   They occupy ORCA planning slots but their position is never updated.

2. **Lingering walkers** (``ped_linger_prob``, ``ped_linger_steps_mean``)
   Any dynamic pedestrian randomly pauses for ``~ped_linger_steps_mean`` steps
   with probability ``ped_linger_prob`` per simulation step.  This creates
   stop-and-wait, window-shopping, and phone-checking moments.

3. **Group walking pairs** (``ped_group_num``)
   ``ped_group_num`` pairs of pedestrians walk side-by-side.  The "follower"
   mirrors the leader's ORCA position with a fixed lateral offset of
   ``GROUP_OFFSET`` metres, visually forming a two-person group.

4. **Ego-yield zone** (``ped_ego_yield_radius``)
   Pedestrians that come within ``ped_ego_yield_radius`` metres of the ego
   vehicle freeze for one step.  This triggers the ``failure_to_yield`` and
   ``unsafe_proximity`` social scenarios more consistently.

New config keys (all have safe defaults so they are backwards-compatible)
-------------------------------------------------------------------------
spawn_static_human_num : int   = 0
ped_linger_prob        : float = 0.002   (per-step per-pedestrian probability)
ped_linger_steps_mean  : int   = 50      (steps ≈ 5 s at 10 Hz)
ped_group_num          : int   = 0       (number of side-by-side pairs)
ped_ego_yield_radius   : float = 0.0     (metres; 0 = disabled)
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np

from metaurban.manager.humanoid_manager import (
    PGBackgroundSidewalkAssetsManager,
    get_dest_heading,
    get_planning,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Behavior tags
# ──────────────────────────────────────────────────────────────────────────────
_NORMAL          = "normal"
_STATIC          = "static"
_GROUP_LEADER    = "group_leader"
_GROUP_FOLLOWER  = "group_follower"

GROUP_OFFSET = 0.8   # lateral separation between group members (metres)


class SocialScenarioManager(PGBackgroundSidewalkAssetsManager):
    """Extends the default humanoid manager with four social behavior modes."""

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        # Per-pedestrian state — populated lazily on first after_step() call.
        self._behavior_initialized: bool = False
        self._per_ped_behavior:  List[str]            = []
        self._linger_counters:   List[int]            = []
        self._group_partners:    Dict[int, int]       = {}  # follower_idx -> leader_idx
        self._group_offsets:     Dict[int, float]     = {}  # follower_idx -> angle offset (rad)
        self._rng = np.random.default_rng()

    # ──────────────────────────────────────────────────────────────────────
    # Config helpers
    # ──────────────────────────────────────────────────────────────────────

    @property
    def _static_num(self) -> int:
        return int(self.engine.global_config.get("spawn_static_human_num", 0))

    @property
    def _linger_prob(self) -> float:
        return float(self.engine.global_config.get("ped_linger_prob", 0.002))

    @property
    def _linger_steps_mean(self) -> int:
        return int(self.engine.global_config.get("ped_linger_steps_mean", 50))

    @property
    def _group_num(self) -> int:
        return int(self.engine.global_config.get("ped_group_num", 0))

    @property
    def _yield_radius(self) -> float:
        return float(self.engine.global_config.get("ped_ego_yield_radius", 0.0))

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        # Patch spawn_num so ORCA reserves slots for static pedestrians.
        # We temporarily inflate the stored spawn_num, run the parent reset,
        # then restore the correct total so after_step indexing is consistent.
        extra = self._static_num
        if extra > 0:
            self.spawn_num += extra

        super().reset()

        # Undo the temporary inflation (parent reset() already used spawn_num).
        if extra > 0:
            self.spawn_num -= extra

        # Flag that behavior tables need rebuilding.
        self._behavior_initialized = False
        self._rng = np.random.default_rng(self.engine.global_random_seed)

    # ──────────────────────────────────────────────────────────────────────
    # Lazy behavior table initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_behavior_tables(self) -> None:
        """Called once per episode on the first after_step() with live agents."""
        n = len(self._traffic_humanoids)
        if n == 0:
            return

        # Default: everyone is a normal walker.
        self._per_ped_behavior = [_NORMAL] * n
        self._linger_counters  = [0]      * n
        self._group_partners   = {}
        self._group_offsets    = {}

        # Mark the LAST _static_num pedestrians as static bystanders.
        static_num = min(self._static_num, n)
        for i in range(n - static_num, n):
            self._per_ped_behavior[i] = _STATIC

        # Assign group pairs from the remaining dynamic pedestrians.
        dynamic_indices = [i for i in range(n) if self._per_ped_behavior[i] == _NORMAL]
        group_pairs = min(self._group_num, len(dynamic_indices) // 2)
        shuffled = list(self._rng.permuted(dynamic_indices))
        for pair_idx in range(group_pairs):
            leader_i   = shuffled[pair_idx * 2]
            follower_i = shuffled[pair_idx * 2 + 1]
            self._per_ped_behavior[leader_i]   = _GROUP_LEADER
            self._per_ped_behavior[follower_i] = _GROUP_FOLLOWER
            self._group_partners[follower_i]   = leader_i
            # Perpendicular offset angle: ±90° from the direction of travel;
            # randomly pick left or right so pairs don't always clump on one side.
            side = self._rng.choice([-1, 1])
            self._group_offsets[follower_i] = side * np.pi / 2.0

        self._behavior_initialized = True
        logger.debug(
            "SocialScenarioManager: %d agents — %d static, %d group pairs, "
            "linger_prob=%.4f linger_mean=%d yield_radius=%.1f",
            n, static_num, group_pairs,
            self._linger_prob, self._linger_steps_mean, self._yield_radius,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Extend _create_humanoids_once to include StaticPedestrian slots
    # ──────────────────────────────────────────────────────────────────────

    def _create_humanoids_once(self, map, spawn_num, max_actor_num) -> None:
        # The parent builds an agent_types list of length:
        #   spawn_human_num + spawn_wheelchairman_num + spawn_edog_num + spawn_erobot_num
        # We need spawn_num to also include _static_num extra slots.
        # Temporarily override the parent list by monkey-patching via super() then
        # adding static pedestrians to the trigger list ourselves.

        static_num = self._static_num
        dynamic_num = spawn_num - static_num

        # Let parent handle the dynamic pedestrians using its own agent_types.
        super()._create_humanoids_once(map, dynamic_num, max_actor_num)

        if static_num == 0:
            return

        # Spawn static pedestrians in the remaining ORCA slots.
        from metaurban.component.agents.pedestrian.pedestrian_type import StaticPedestrian

        heading = 0
        static_names: List[str] = []
        block = map.blocks[1]

        for i in range(dynamic_num, dynamic_num + static_num):
            spawn_point = self._to_block_coordinate(self.start_points[i])
            v_config = {"spawn_position_heading": [spawn_point, heading]}
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            ped = self.spawn_object(StaticPedestrian, vehicle_config=v_config)
            static_names.append(ped.name)

        # Append static pedestrians to the existing trigger bucket so they
        # are activated alongside the dynamic pedestrians.
        from metaurban.manager.humanoid_manager import BlockHumanoids
        if self.block_triggered_humanoids:
            # Merge into the last (actually first, due to .reverse()) trigger bucket.
            first_bucket = self.block_triggered_humanoids[-1]
            merged_names = list(first_bucket.humanoids) + static_names
            self.block_triggered_humanoids[-1] = BlockHumanoids(
                trigger_road=first_bucket.trigger_road,
                humanoids=merged_names,
            )
        else:
            # Fallback: create a new trigger bucket using block's road.
            trigger_road = block.pre_block_socket.positive_road
            self.block_triggered_humanoids.append(
                BlockHumanoids(trigger_road=trigger_road, humanoids=static_names)
            )

    # ──────────────────────────────────────────────────────────────────────
    # after_step — main social behavior injection point
    # ──────────────────────────────────────────────────────────────────────

    def after_step(self, *args, **kwargs):
        if len(self._traffic_humanoids) == 0:
            return dict()

        # Lazy init on first call after reset.
        if not self._behavior_initialized:
            self._init_behavior_tables()

        # Advance (or re-plan) ORCA positions — mirrors parent logic exactly.
        try:
            positions, speeds = next(self.points), next(self.speeds)
        except StopIteration:
            self._replan()
            positions, speeds = next(self.points), next(self.speeds)

        # Ego vehicle position (for yield zone).
        ego_pos: Optional[np.ndarray] = None
        if self._yield_radius > 0:
            try:
                ego_pos = np.array(self.engine.agent.position[:2])
            except Exception:
                pass

        for idx, (v, raw_pos, spd) in enumerate(
            zip(self._traffic_humanoids, positions, speeds)
        ):
            pos = self._to_block_coordinate(raw_pos)
            behavior = (
                self._per_ped_behavior[idx]
                if idx < len(self._per_ped_behavior)
                else _NORMAL
            )

            # ── Static bystanders: never move ─────────────────────────────
            if behavior == _STATIC:
                continue

            # ── Ego-yield zone ─────────────────────────────────────────────
            if ego_pos is not None:
                ped_pos = np.array(v.position[:2])
                if np.linalg.norm(ego_pos - ped_pos) < self._yield_radius:
                    continue  # pause this step

            # ── Lingering ──────────────────────────────────────────────────
            if idx < len(self._linger_counters) and self._linger_counters[idx] > 0:
                self._linger_counters[idx] -= 1
                continue

            # Randomly enter linger state.
            if (
                self._linger_prob > 0
                and behavior in (_NORMAL, _GROUP_LEADER, _GROUP_FOLLOWER)
                and self._rng.random() < self._linger_prob
            ):
                duration = max(
                    1,
                    int(self._rng.normal(
                        self._linger_steps_mean,
                        self._linger_steps_mean * 0.4,
                    )),
                )
                if idx < len(self._linger_counters):
                    self._linger_counters[idx] = duration
                continue

            # ── Group follower: override position to walk beside leader ─────
            if behavior == _GROUP_FOLLOWER and idx in self._group_partners:
                leader_idx = self._group_partners[idx]
                if leader_idx < len(positions):
                    leader_pos = self._to_block_coordinate(positions[leader_idx])
                    offset_angle = self._group_offsets.get(idx, 0.0)
                    # Compute perpendicular direction from leader's movement.
                    leader_prev = np.array(v.position[:2])  # use follower's pos as proxy
                    direction = np.array(leader_pos) - np.array(leader_prev)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-4:
                        perp = np.array([-direction[1], direction[0]]) / norm
                    else:
                        perp = np.array([np.cos(offset_angle), np.sin(offset_angle)])
                    pos = (
                        float(leader_pos[0]) + float(perp[0]) * GROUP_OFFSET,
                        float(leader_pos[1]) + float(perp[1]) * GROUP_OFFSET,
                        0.0 if len(leader_pos) > 2 else 0.0,
                    )

            # ── Normal movement (mirrors parent exactly) ───────────────────
            prev_pos = v.position
            heading_angle = get_dest_heading(v, pos)
            speed_val = spd / self.engine.global_config["physics_world_step_size"]

            from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
            if isinstance(v, BasePedestrian) and v.render:
                v.set_anim_by_speed(speed_val)

            v.set_position(pos)
            try:
                v._body.setAngularMovement(heading_angle * 3)
            except Exception:
                heading_angle = np.arctan2(
                    pos[1] - prev_pos[1], pos[0] - prev_pos[0]
                )
                v.set_heading_theta(heading_angle)

        return dict()

    # ──────────────────────────────────────────────────────────────────────
    # Internal: re-plan ORCA when path is exhausted (copied from parent)
    # ──────────────────────────────────────────────────────────────────────

    def _replan(self) -> None:
        import copy as _copy
        self.start_points = _copy.deepcopy(self.end_points)
        _, self.end_points = self.random_start_and_end_points(
            self.walkable_regions_mask[:, :, 0],
            self.spawn_num + self.d_robot_num + self._static_num,
        )
        time_length, points, speed, early_stop_points = get_planning(
            [self.start_points],
            [self.walkable_regions_mask],
            [self.end_points],
            [len(self.start_points)],
            1,
        )
        self.points      = iter(points[0])
        self.time_length = time_length[0]
        self.speeds      = iter(speed[0])
        self.es_points   = early_stop_points[0]
