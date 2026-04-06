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
from typing import Dict, List, Optional, Set

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
        self._group_cluster_map: Dict[int, int] = {}
        self._group_members: Dict[int, List[int]] = {}
        self._group_member_slot: Dict[int, int] = {}
        self._group_cluster_positions: Dict[int, np.ndarray] = {}
        self._group_cluster_drifts: Dict[int, np.ndarray] = {}
        self._group_release_counter: Dict[int, int] = {}
        self._group_released: Set[int] = set()
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
    def _group_cluster_num(self) -> int:
        configured = int(self.engine.global_config.get("group_cluster_num", 0))
        return configured if configured > 0 else self._group_pair_num

    @property
    def _group_size_min(self) -> int:
        return int(self.engine.global_config.get("group_cluster_size_min", 5))

    @property
    def _group_size_max(self) -> int:
        return int(self.engine.global_config.get("group_cluster_size_max", 8))

    @property
    def _group_spawn_near_ego(self) -> bool:
        return bool(self.engine.global_config.get("group_spawn_near_ego", False))

    @property
    def _group_spawn_min_radius(self) -> float:
        return float(self.engine.global_config.get("group_spawn_min_radius", 5.0))

    @property
    def _group_spawn_max_radius(self) -> float:
        return float(self.engine.global_config.get("group_spawn_max_radius", 10.0))

    @property
    def _group_release_enable(self) -> bool:
        return bool(self.engine.global_config.get("group_release_enable", True))

    @property
    def _group_release_steps_mean(self) -> int:
        return int(self.engine.global_config.get("group_release_steps_mean", 180))

    @property
    def _group_release_steps_std(self) -> int:
        return int(self.engine.global_config.get("group_release_steps_std", 40))

    @property
    def _group_release_steps_min(self) -> int:
        return int(self.engine.global_config.get("group_release_steps_min", 60))

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
        self._group_cluster_positions.clear()
        self._group_cluster_drifts.clear()
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
        self._group_cluster_map = {}
        self._group_members = {}
        self._group_member_slot = {}
        self._group_cluster_positions = {}
        self._group_cluster_drifts = {}
        self._group_release_counter = {}
        self._group_released = set()
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

        # Create group clusters: each cluster has 5-8 members by default.
        size_min = max(2, self._group_size_min)
        size_max = max(size_min, self._group_size_max)
        target_cluster_num = min(self._group_cluster_num, len(remaining) // size_min)
        shuffled = list(self._rng.permutation(remaining))
        used_group: List[int] = []

        cursor = 0
        for cluster_id in range(target_cluster_num):
            left = len(shuffled) - cursor
            clusters_left_after = target_cluster_num - cluster_id - 1
            max_size_this = min(size_max, left - clusters_left_after * size_min)
            if max_size_this < size_min:
                break
            cluster_size = int(self._rng.integers(size_min, max_size_this + 1))
            members = shuffled[cursor:cursor + cluster_size]
            cursor += cluster_size

            if not members:
                continue

            leader = members[0]
            self._group_members[cluster_id] = members
            self._ped_role[leader] = ROLE_GROUP_LEADER
            self._group_cluster_map[leader] = cluster_id
            self._group_member_slot[leader] = 0

            for slot, member_idx in enumerate(members[1:], start=1):
                self._ped_role[member_idx] = ROLE_GROUP_FOLLOWER
                self._group_cluster_map[member_idx] = cluster_id
                self._group_member_slot[member_idx] = slot
                self._group_partners[member_idx] = leader

            leader_pos = self._traffic_humanoids[leader].position
            cluster_center = np.array([float(leader_pos[0]), float(leader_pos[1])], dtype=float)
            self._group_cluster_positions[cluster_id] = cluster_center.copy()
            self._group_cluster_drifts[cluster_id] = np.array([0.0, 0.0])
            release_steps = max(
                self._group_release_steps_min,
                int(self._rng.normal(self._group_release_steps_mean, self._group_release_steps_std)),
            )
            self._group_release_counter[cluster_id] = release_steps
            used_group.extend(members)

        # Optional preview-friendly behavior: place group clusters around ego
        # so they are visible in the camera at episode start.
        if len(self._group_members) > 0 and self._group_spawn_near_ego:
            try:
                ego_obj = None
                active_agents = getattr(self.engine.agent_manager, "active_agents", {})
                if active_agents:
                    ego_obj = list(active_agents.values())[0]
                if ego_obj is None:
                    raise RuntimeError("No active ego agent found")
                ego_pos = np.array(ego_obj.position[:2], dtype=float)
                # Prefer valid pedestrian anchors near ego so clusters stay on walkable areas.
                candidates: List[np.ndarray] = []
                for ped in self._traffic_humanoids:
                    p = ped.position
                    candidates.append(np.array([float(p[0]), float(p[1])], dtype=float))
                candidates.sort(key=lambda p: float(np.linalg.norm(p - ego_pos)))

                cluster_ids = sorted(self._group_members.keys())
                chosen: List[np.ndarray] = []
                for cluster_id in cluster_ids:
                    picked = None
                    for cand in candidates:
                        d_ego = float(np.linalg.norm(cand - ego_pos))
                        if d_ego < self._group_spawn_min_radius or d_ego > self._group_spawn_max_radius:
                            continue
                        if any(float(np.linalg.norm(cand - c)) < 2.5 for c in chosen):
                            continue
                        picked = cand
                        break
                    if picked is None:
                        # Fallback to current center if no candidate in range.
                        picked = self._group_cluster_positions.get(cluster_id, ego_pos)
                    chosen.append(np.array(picked, dtype=float))
                    self._group_cluster_positions[cluster_id] = np.array(picked, dtype=float)
            except Exception:
                logger.exception("Failed to place group clusters near ego; using default cluster positions")

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

    def _group_slot_offset(self, cluster_id: int, slot: int) -> np.ndarray:
        members = self._group_members.get(cluster_id, [])
        group_size = max(1, len(members))
        if slot <= 0 or group_size <= 1:
            return np.array([0.25, 0.0], dtype=float)

        follower_count = group_size - 1
        idx = slot - 1
        angle = 2.0 * np.pi * (idx / max(1, follower_count))
        ring = idx // 6
        radius = 0.9 + 0.35 * ring
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=float)

    def _release_cluster(self, cluster_id: int) -> None:
        if cluster_id in self._group_released:
            return
        members = self._group_members.get(cluster_id, [])
        released_count = 0
        for idx in members:
            if 0 <= idx < len(self._ped_role):
                self._ped_role[idx] = ROLE_NORMAL
                setattr(self._traffic_humanoids[idx], "social_role", ROLE_NORMAL)
                released_count += 1
            self._group_cluster_map.pop(idx, None)
            self._group_member_slot.pop(idx, None)
            self._group_partners.pop(idx, None)
        self._group_released.add(cluster_id)
        self._group_cluster_drifts.pop(cluster_id, None)
        self._group_cluster_positions.pop(cluster_id, None)
        self._group_release_counter.pop(cluster_id, None)
        logger.info("Released group cluster %d with %d pedestrians", cluster_id, released_count)

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

        if self._group_release_enable and self._group_release_counter:
            to_release: List[int] = []
            for cluster_id in list(self._group_release_counter.keys()):
                self._group_release_counter[cluster_id] -= 1
                if self._group_release_counter[cluster_id] <= 0:
                    to_release.append(cluster_id)
            for cluster_id in to_release:
                self._release_cluster(cluster_id)

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

            # Handle group members: they stand together in small clusters, not following global trajectory
            if role in (ROLE_GROUP_LEADER, ROLE_GROUP_FOLLOWER):
                cluster_id = self._group_cluster_map.get(idx, None)
                if cluster_id is not None and cluster_id not in self._group_released:
                    # Clustered pedestrians hold a fixed chatting formation.
                    cluster_center = self._group_cluster_positions.get(cluster_id)
                    if cluster_center is not None:
                        slot = self._group_member_slot.get(idx, 0)
                        offset = self._group_slot_offset(cluster_id, slot)
                        actual_pos = cluster_center + offset
                        target_pos = (float(actual_pos[0]), float(actual_pos[1]), 0.0)

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
            elif role in (ROLE_GROUP_LEADER, ROLE_GROUP_FOLLOWER):
                # Group members are static during clustering phase.
                speed_scale = 0.0
            elif role == ROLE_CROSSING:
                speed_scale = self._crossing_speed_scale
                if dist_to_ego < self._crossing_assertive_radius:
                    speed_scale *= 1.1

            prev_pos = ped.position
            if role in (ROLE_GROUP_LEADER, ROLE_GROUP_FOLLOWER):
                cluster_id = self._group_cluster_map.get(idx, None)
                center = self._group_cluster_positions.get(cluster_id) if cluster_id is not None else None
                if center is not None:
                    heading = np.arctan2(center[1] - prev_pos[1], center[0] - prev_pos[0])
                else:
                    heading = get_dest_heading(ped, target_pos)
            else:
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
