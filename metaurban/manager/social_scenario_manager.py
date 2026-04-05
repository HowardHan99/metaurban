"""
SocialScenarioManager
=====================
A role-oriented social behavior manager for dynamic sidewalk scenes.

This manager extends the default ``PGBackgroundSidewalkAssetsManager`` with
three active social roles:
1. crossing pedestrian
2. vulnerable pedestrian
3. group pedestrian

Signaling is intentionally disabled for now because its visual evidence is not
strong enough for reliable VLM labeling in the current setup.

The role set is intentionally small to maximize interpretability and training
signal quality for offline social reward learning. The vulnerable role is
further diversified into subtypes (wheelchair / elderly / distracted) to make
social vulnerability behavior richer without introducing many new top-level
roles.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from metaurban.manager.humanoid_manager import (
    PGBackgroundSidewalkAssetsManager,
    get_dest_heading,
)

logger = logging.getLogger(__name__)

ROLE_NORMAL = "normal"
ROLE_CROSSING = "crossing"
ROLE_VULNERABLE = "vulnerable"
ROLE_GROUP_LEADER = "group_leader"
ROLE_GROUP_FOLLOWER = "group_follower"

VUL_SUB_WHEELCHAIR = "wheelchair"
VUL_SUB_ELDERLY = "elderly"
VUL_SUB_DISTRACTED = "distracted"

GROUP_OFFSET = 0.8


class SocialScenarioManager(PGBackgroundSidewalkAssetsManager):
    """Drop-in manager with 4 social role behaviors."""

    def __init__(self, scene_type: str = "commercial"):
        super().__init__()
        self.scene_type = scene_type
        self._roles_initialized: bool = False
        self._ped_role: List[str] = []
        self._group_partners: Dict[int, int] = {}
        self._group_offsets: Dict[int, float] = {}
        self._vulnerable_subtype: Dict[int, str] = {}
        self._vulnerable_speed_scale_map: Dict[int, float] = {}
        self._vulnerable_yield_scale_map: Dict[int, float] = {}
        self._vulnerable_pause_counters: Dict[int, int] = {}
        self._rng = np.random.default_rng()

    @property
    def _crossing_num(self) -> int:
        return int(self.engine.global_config.get("crossing_ped_num", 8))

    @property
    def _signaling_num(self) -> int:
        # Keep key for backward compatibility, but hard-disable signaling.
        return 0

    @property
    def _vulnerable_num(self) -> int:
        return int(self.engine.global_config.get("vulnerable_ped_num", 4))

    @property
    def _group_pair_num(self) -> int:
        return int(self.engine.global_config.get("group_ped_pair_num", 3))

    @property
    def _yield_radius(self) -> float:
        return float(self.engine.global_config.get("ped_ego_yield_radius", 3.0))

    @property
    def _crossing_assertive_radius(self) -> float:
        return float(self.engine.global_config.get("crossing_assertive_radius", 4.0))

    @property
    def _crossing_speed_scale(self) -> float:
        return float(self.engine.global_config.get("crossing_speed_scale", 1.10))

    @property
    def _vulnerable_yield_radius(self) -> float:
        return float(self.engine.global_config.get("vulnerable_yield_radius", 5.0))

    @property
    def _vulnerable_speed_scale(self) -> float:
        return float(self.engine.global_config.get("vulnerable_speed_scale", 0.70))

    @property
    def _vulnerable_elderly_ratio(self) -> float:
        return float(self.engine.global_config.get("vulnerable_elderly_ratio", 0.6))

    @property
    def _vulnerable_distracted_ratio(self) -> float:
        return float(self.engine.global_config.get("vulnerable_distracted_ratio", 0.4))

    @property
    def _vulnerable_pause_prob(self) -> float:
        return float(self.engine.global_config.get("vulnerable_pause_prob", 0.02))

    @property
    def _vulnerable_pause_steps_mean(self) -> int:
        return int(self.engine.global_config.get("vulnerable_pause_steps_mean", 16))

    def reset(self):
        super().reset()
        logger.info(f"SocialScenarioManager set to scene_type: {self.scene_type}")
        self._roles_initialized = False
        self._rng = np.random.default_rng(self.engine.global_random_seed)

    def _assign_role(self, indices: List[int], role: str, count: int) -> List[int]:
        if count <= 0 or len(indices) == 0:
            return indices
        pick = min(count, len(indices))
        chosen = list(self._rng.permutation(indices)[:pick])
        for idx in chosen:
            self._ped_role[idx] = role
        chosen_set = set(chosen)
        return [idx for idx in indices if idx not in chosen_set]

    def _init_roles(self) -> None:
        n = len(self._traffic_humanoids)
        if n == 0:
            return

        self._ped_role = [ROLE_NORMAL] * n
        self._group_partners = {}
        self._group_offsets = {}
        self._vulnerable_subtype = {}
        self._vulnerable_speed_scale_map = {}
        self._vulnerable_yield_scale_map = {}
        self._vulnerable_pause_counters = {}

        remaining = list(range(n))

        # Prefer assigning pre-existing wheelchair/elderly actors to vulnerable role.
        vulnerable_candidates: List[int] = []
        for idx, ped in enumerate(self._traffic_humanoids):
            cls_name = ped.__class__.__name__.lower()
            if "wheelchair" in cls_name or "elderly" in cls_name:
                vulnerable_candidates.append(idx)

        preferred_vulnerable = vulnerable_candidates[: self._vulnerable_num]
        for idx in preferred_vulnerable:
            self._ped_role[idx] = ROLE_VULNERABLE
            setattr(self._traffic_humanoids[idx], "social_role", ROLE_VULNERABLE)
        remaining = [idx for idx in remaining if idx not in set(preferred_vulnerable)]

        need_more_vulnerable = max(0, self._vulnerable_num - len(preferred_vulnerable))
        remaining = self._assign_role(remaining, ROLE_VULNERABLE, need_more_vulnerable)

        pair_num = min(self._group_pair_num, len(remaining) // 2)
        shuffled = list(self._rng.permutation(remaining))
        used_group: List[int] = []
        for i in range(pair_num):
            leader = shuffled[2 * i]
            follower = shuffled[2 * i + 1]
            self._ped_role[leader] = ROLE_GROUP_LEADER
            self._ped_role[follower] = ROLE_GROUP_FOLLOWER
            self._group_partners[follower] = leader
            side = self._rng.choice([-1, 1])
            self._group_offsets[follower] = side * np.pi / 2.0
            used_group.extend([leader, follower])

        remaining = [idx for idx in remaining if idx not in set(used_group)]
        remaining = self._assign_role(remaining, ROLE_CROSSING, self._crossing_num)

        for idx, role in enumerate(self._ped_role):
            setattr(self._traffic_humanoids[idx], "social_role", role)

        self._init_vulnerable_profiles()

        self._roles_initialized = True
        logger.info(
            "Social role histogram: crossing=%d vulnerable=%d group=%d normal=%d",
            self._ped_role.count(ROLE_CROSSING),
            self._ped_role.count(ROLE_VULNERABLE),
            self._ped_role.count(ROLE_GROUP_LEADER) + self._ped_role.count(ROLE_GROUP_FOLLOWER),
            self._ped_role.count(ROLE_NORMAL),
        )

        if self._vulnerable_subtype:
            logger.info(
                "Vulnerable subtype histogram: wheelchair=%d elderly=%d distracted=%d",
                sum(1 for s in self._vulnerable_subtype.values() if s == VUL_SUB_WHEELCHAIR),
                sum(1 for s in self._vulnerable_subtype.values() if s == VUL_SUB_ELDERLY),
                sum(1 for s in self._vulnerable_subtype.values() if s == VUL_SUB_DISTRACTED),
            )

    def _sample_non_wheelchair_vulnerable_subtype(self) -> str:
        elderly_w = max(0.0, self._vulnerable_elderly_ratio)
        distracted_w = max(0.0, self._vulnerable_distracted_ratio)
        total = elderly_w + distracted_w
        if total <= 1e-9:
            return VUL_SUB_ELDERLY
        p_elderly = elderly_w / total
        return VUL_SUB_ELDERLY if self._rng.random() < p_elderly else VUL_SUB_DISTRACTED

    def _init_vulnerable_profiles(self) -> None:
        """Assign per-agent vulnerable subtypes and per-subtype behavior scales."""
        for idx, role in enumerate(self._ped_role):
            if role != ROLE_VULNERABLE:
                continue

            ped = self._traffic_humanoids[idx]
            cls_name = ped.__class__.__name__.lower()

            if "wheelchair" in cls_name:
                subtype = VUL_SUB_WHEELCHAIR
                speed_scale = float(self._rng.uniform(0.45, 0.70))
                yield_scale = float(self._rng.uniform(1.30, 1.70))
            elif "elderly" in cls_name:
                subtype = VUL_SUB_ELDERLY
                speed_scale = float(self._rng.uniform(0.55, 0.80))
                yield_scale = float(self._rng.uniform(1.15, 1.45))
            else:
                subtype = self._sample_non_wheelchair_vulnerable_subtype()
                if subtype == VUL_SUB_ELDERLY:
                    speed_scale = float(self._rng.uniform(0.55, 0.80))
                    yield_scale = float(self._rng.uniform(1.10, 1.40))
                else:
                    speed_scale = float(self._rng.uniform(0.70, 0.95))
                    yield_scale = float(self._rng.uniform(0.90, 1.10))

            self._vulnerable_subtype[idx] = subtype
            self._vulnerable_speed_scale_map[idx] = speed_scale
            self._vulnerable_yield_scale_map[idx] = yield_scale
            self._vulnerable_pause_counters[idx] = 0
            setattr(ped, "vulnerable_subtype", subtype)

    def after_step(self, *args, **kwargs):
        if len(self._traffic_humanoids) == 0:
            return dict()

        if not self._roles_initialized:
            self._init_roles()

        try:
            positions, speeds = next(self.points), next(self.speeds)
        except StopIteration:
            # Reuse parent logic to refresh trajectories when current one ends.
            self.start_points = self.end_points.copy()
            _, self.end_points = self.random_start_and_end_points(
                self.walkable_regions_mask[:, :, 0], self.spawn_num + self.d_robot_num
            )
            from metaurban.manager.humanoid_manager import get_planning
            time_length, points, speed, early_stop_points = get_planning(
                [self.start_points],
                [self.walkable_regions_mask],
                [self.end_points],
                [len(self.start_points)],
                1,
            )
            self.points = iter(points[0])
            self.time_length = time_length[0]
            self.speeds = iter(speed[0])
            self.es_points = early_stop_points[0]
            positions, speeds = next(self.points), next(self.speeds)

        ego_pos: Optional[np.ndarray] = None
        try:
            ego_pos = np.array(self.engine.agent.position[:2])
        except Exception:
            pass

        for idx, (ped, raw_pos, orca_speed) in enumerate(zip(self._traffic_humanoids, positions, speeds)):
            role = self._ped_role[idx] if idx < len(self._ped_role) else ROLE_NORMAL
            target_pos = self._to_block_coordinate(raw_pos)

            if role == ROLE_GROUP_FOLLOWER and idx in self._group_partners:
                leader_idx = self._group_partners[idx]
                if leader_idx < len(positions):
                    leader_pos = self._to_block_coordinate(positions[leader_idx])
                    ped_prev = np.array(ped.position[:2])
                    move = np.array(leader_pos[:2]) - ped_prev
                    norm = np.linalg.norm(move)
                    if norm > 1e-4:
                        perp = np.array([-move[1], move[0]]) / norm
                    else:
                        angle = self._group_offsets.get(idx, np.pi / 2.0)
                        perp = np.array([np.cos(angle), np.sin(angle)])
                    target_pos = (
                        float(leader_pos[0]) + float(perp[0]) * GROUP_OFFSET,
                        float(leader_pos[1]) + float(perp[1]) * GROUP_OFFSET,
                        0.0,
                    )

            ped_pos = np.array(ped.position[:2])
            dist_to_ego = float(np.linalg.norm(ego_pos - ped_pos)) if ego_pos is not None else float("inf")

            if role == ROLE_VULNERABLE:
                subtype = self._vulnerable_subtype.get(idx, VUL_SUB_ELDERLY)

                # Distracted vulnerable agents occasionally pause even without ego pressure.
                if subtype == VUL_SUB_DISTRACTED:
                    counter = self._vulnerable_pause_counters.get(idx, 0)
                    if counter > 0:
                        self._vulnerable_pause_counters[idx] = counter - 1
                        continue
                    if self._vulnerable_pause_prob > 0 and self._rng.random() < self._vulnerable_pause_prob:
                        duration = max(
                            1,
                            int(self._rng.normal(self._vulnerable_pause_steps_mean, self._vulnerable_pause_steps_mean * 0.35)),
                        )
                        self._vulnerable_pause_counters[idx] = duration
                        continue

                personal_yield_radius = self._vulnerable_yield_radius * self._vulnerable_yield_scale_map.get(idx, 1.0)
                if dist_to_ego < personal_yield_radius:
                    continue
            elif role != ROLE_CROSSING:
                if dist_to_ego < self._yield_radius:
                    continue

            speed_scale = 1.0
            if role == ROLE_VULNERABLE:
                speed_scale = self._vulnerable_speed_scale * self._vulnerable_speed_scale_map.get(idx, 1.0)
            elif role == ROLE_CROSSING:
                speed_scale = self._crossing_speed_scale
                if dist_to_ego < self._crossing_assertive_radius:
                    speed_scale *= 1.1

            prev_pos = ped.position
            heading = get_dest_heading(ped, target_pos)
            speed_val = (
                orca_speed / self.engine.global_config["physics_world_step_size"]
            ) * speed_scale

            from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
            if isinstance(ped, BasePedestrian) and ped.render:
                ped.set_anim_by_speed(speed_val)

            ped.set_position(target_pos)
            try:
                ped._body.setAngularMovement(heading * 3)
            except Exception:
                fallback_heading = np.arctan2(
                    target_pos[1] - prev_pos[1],
                    target_pos[0] - prev_pos[0],
                )
                ped.set_heading_theta(fallback_heading)

        return dict()
