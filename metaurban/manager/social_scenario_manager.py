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
GROUP_MODE_STANDING = "standing_group"

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
        self._group_cluster_phase: Dict[int, float] = {}
        self._group_cluster_member_radius: Dict[int, float] = {}
        self._group_cluster_ring_step: Dict[int, float] = {}
        self._group_cluster_mode: Dict[int, str] = {}
        self._group_cluster_compression: Dict[int, float] = {}
        self._group_member_dynamic_offset: Dict[int, np.ndarray] = {}
        self._group_member_dynamic_timer: Dict[int, int] = {}
        self._group_member_heading: Dict[int, float] = {}
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
    def _group_route_min_ego_distance(self) -> float:
        return float(self.engine.global_config.get("group_route_min_ego_distance", 8.0))

    @property
    def _group_route_min_separation(self) -> float:
        return float(self.engine.global_config.get("group_route_min_separation", 5.5))

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
    def _group_member_radius(self) -> float:
        return float(self.engine.global_config.get("group_member_radius", 1.60))

    @property
    def _group_member_ring_step(self) -> float:
        return float(self.engine.global_config.get("group_member_ring_step", 0.75))

    @property
    def _group_member_radius_jitter(self) -> float:
        return float(self.engine.global_config.get("group_member_radius_jitter", 0.22))

    @property
    def _group_member_ring_step_jitter(self) -> float:
        return float(self.engine.global_config.get("group_member_ring_step_jitter", 0.18))

    @property
    def _group_member_idle_shift_prob(self) -> float:
        return float(self.engine.global_config.get("group_member_idle_shift_prob", 0.015))

    @property
    def _group_member_idle_shift_steps_mean(self) -> int:
        return int(self.engine.global_config.get("group_member_idle_shift_steps_mean", 18))

    @property
    def _group_member_idle_shift_radius(self) -> float:
        return float(self.engine.global_config.get("group_member_idle_shift_radius", 0.22))

    @property
    def _group_cluster_min_separation(self) -> float:
        return float(self.engine.global_config.get("group_cluster_min_separation", 3.8))

    @property
    def _pedestrian_sidewalk_only(self) -> bool:
        return bool(self.engine.global_config.get("pedestrian_sidewalk_only", True))

    @property
    def _pedestrian_allow_crosswalk(self) -> bool:
        return bool(self.engine.global_config.get("pedestrian_allow_crosswalk", False))

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

    def _get_walkable_regions(self, current_map):
        base_mask = super()._get_walkable_regions(current_map)
        if not self._pedestrian_sidewalk_only:
            return base_mask

        import cv2

        sidewalk_mask = np.zeros_like(base_mask)
        region_groups = [
            self.sidewalks,
            self.sidewalks_near_road,
            self.sidewalks_farfrom_road,
            self.sidewalks_near_road_buffer,
            self.sidewalks_farfrom_road_buffer,
        ]
        if self._pedestrian_allow_crosswalk:
            region_groups.append(self.crosswalks)

        for regions in region_groups:
            for region in regions.values():
                polygon_array = np.array(region["polygon"])
                polygon_array += self.mask_translate
                polygon_array = np.floor(polygon_array).astype(int)
                polygon_array = polygon_array.reshape((-1, 1, 2))
                cv2.fillPoly(sidewalk_mask, [polygon_array], [255, 255, 255])

        sidewalk_mask = cv2.flip(sidewalk_mask, 0)
        walkable_pixels = int(np.count_nonzero(sidewalk_mask[:, :, 0]))
        if walkable_pixels <= 0:
            logger.warning("Sidewalk-only mask is empty. Falling back to default walkable regions.")
            return base_mask

        logger.info(
            "SocialScenarioManager uses sidewalk-only walkable mask (allow_crosswalk=%s, pixels=%d)",
            self._pedestrian_allow_crosswalk,
            walkable_pixels,
        )
        return sidewalk_mask

    def reset(self):
        super().reset()
        logger.info(f"SocialScenarioManager set to scene_type: {self.scene_type}")
        self._roles_initialized = False
        self._group_cluster_positions.clear()
        self._group_cluster_drifts.clear()
        self._group_cluster_phase.clear()
        self._group_cluster_member_radius.clear()
        self._group_cluster_ring_step.clear()
        self._group_cluster_mode.clear()
        self._group_cluster_compression.clear()
        self._group_member_dynamic_offset.clear()
        self._group_member_dynamic_timer.clear()
        self._group_member_heading.clear()
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
        self._group_cluster_phase = {}
        self._group_cluster_member_radius = {}
        self._group_cluster_ring_step = {}
        self._group_cluster_mode = {}
        self._group_cluster_compression = {}
        self._group_member_dynamic_offset = {}
        self._group_member_dynamic_timer = {}
        self._group_member_heading = {}
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

            for member_idx in members:
                self._group_member_dynamic_offset[member_idx] = np.array([0.0, 0.0], dtype=float)
                self._group_member_dynamic_timer[member_idx] = 0
                self._group_member_heading[member_idx] = 0.0

            leader_pos = self._traffic_humanoids[leader].position
            cluster_center = np.array([float(leader_pos[0]), float(leader_pos[1])], dtype=float)
            self._group_cluster_positions[cluster_id] = cluster_center.copy()
            self._group_cluster_drifts[cluster_id] = np.array([0.0, 0.0])
            self._group_cluster_phase[cluster_id] = float(self._rng.uniform(0.0, 2.0 * np.pi))
            self._group_cluster_member_radius[cluster_id] = max(
                1.1,
                float(self._group_member_radius + self._rng.uniform(-self._group_member_radius_jitter, self._group_member_radius_jitter)),
            )
            self._group_cluster_ring_step[cluster_id] = max(
                0.4,
                float(self._group_member_ring_step + self._rng.uniform(-self._group_member_ring_step_jitter, self._group_member_ring_step_jitter)),
            )
            self._group_cluster_mode[cluster_id] = GROUP_MODE_STANDING
            self._group_cluster_compression[cluster_id] = 1.0
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
                ego_path_points = self._extract_ego_global_path_points(ego_obj)
                if ego_path_points:
                    logger.info("Route-aware group placement using %d ego path points", len(ego_path_points))
                cluster_ids = sorted(self._group_members.keys())
                placed = self._pick_cluster_center_candidates(ego_pos, cluster_ids, ego_path_points)
                for cluster_id in cluster_ids:
                    if cluster_id in placed:
                        self._group_cluster_positions[cluster_id] = placed[cluster_id]
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
        if group_size <= 1:
            return np.array([0.0, 0.0], dtype=float)

        idx = max(0, slot)
        inner_cap = min(6, group_size)
        ring = idx // inner_cap
        ring_index = idx % inner_cap
        phase = self._group_cluster_phase.get(cluster_id, 0.0)

        if ring == 0:
            angle = 2.0 * np.pi * (ring_index / max(1, inner_cap)) + phase
        else:
            outer_count = max(1, group_size - inner_cap)
            angle = 2.0 * np.pi * (ring_index / outer_count) + phase + np.pi / outer_count

        member_radius = self._group_cluster_member_radius.get(cluster_id, self._group_member_radius)
        ring_step = self._group_cluster_ring_step.get(cluster_id, self._group_member_ring_step)
        compression = self._group_cluster_compression.get(cluster_id, 1.0)
        member_radius *= compression
        ring_step *= compression
        radius = member_radius + ring_step * ring
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=float)

    def _estimate_local_group_compression(self, center: np.ndarray, sidewalk_mask: Optional[np.ndarray]) -> float:
        if sidewalk_mask is None:
            return 1.0

        dirs = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, -1.0]),
            np.array([0.707, 0.707]),
            np.array([-0.707, 0.707]),
            np.array([-0.707, -0.707]),
            np.array([0.707, -0.707]),
        ]
        probe_steps = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6]
        min_clearance = 3.0
        for d in dirs:
            clearance = 0.0
            for step in probe_steps:
                p = center + d * step
                if not self._is_point_on_mask(p, sidewalk_mask):
                    break
                clearance = step
            min_clearance = min(min_clearance, clearance)

        if min_clearance < 1.0:
            return 0.62
        if min_clearance < 1.4:
            return 0.72
        if min_clearance < 1.9:
            return 0.82
        if min_clearance < 2.4:
            return 0.92
        return 1.0

    def _update_group_clusters_pre_step(self, ego_pos: Optional[np.ndarray]) -> None:
        sidewalk_mask = self._build_group_sidewalk_mask()
        for cluster_id, center in list(self._group_cluster_positions.items()):
            if cluster_id in self._group_released:
                continue
            self._group_cluster_compression[cluster_id] = self._estimate_local_group_compression(center, sidewalk_mask)

    def _extract_ego_global_path_points(self, ego_obj) -> List[np.ndarray]:
        """Extract checkpoint samples from ego's initial global route for group placement."""
        nav = getattr(ego_obj, "navigation", None)
        checkpoints = getattr(nav, "checkpoints", None) if nav is not None else None
        if checkpoints is None or len(checkpoints) == 0:
            return []

        ego_pos = np.array(ego_obj.position[:2], dtype=float)
        try:
            heading_vec = np.array(ego_obj.heading[:2], dtype=float)
            heading_norm = float(np.linalg.norm(heading_vec))
            if heading_norm > 1e-6:
                heading_vec = heading_vec / heading_norm
            else:
                heading_vec = None
        except Exception:
            heading_vec = None

        min_r = max(self._group_spawn_min_radius, self._group_route_min_ego_distance)
        max_r = self._group_spawn_max_radius
        min_sep = max(self._group_cluster_min_separation, self._group_route_min_separation)
        picked: List[np.ndarray] = []
        for ckpt in checkpoints:
            p = np.array([float(ckpt[0]), float(ckpt[1])], dtype=float)
            vec = p - ego_pos
            dist = float(np.linalg.norm(vec))
            if dist < min_r or dist > max_r:
                continue
            if heading_vec is not None:
                cos_forward = float(np.dot(vec / max(dist, 1e-6), heading_vec))
                if cos_forward < -0.15:
                    continue
            if picked and any(float(np.linalg.norm(p - q)) < min_sep for q in picked):
                continue
            picked.append(p)
        return picked

    def _scene_anchor_templates(self, cluster_count: int) -> List[tuple]:
        """Scene-aware (angle_deg, radius_ratio) templates for realistic social hotspots."""
        templates = {
            "commercial": [(-35, 0.45), (35, 0.45), (-80, 0.65), (80, 0.65), (150, 0.55), (-150, 0.55)],
            "commute": [(-20, 0.50), (20, 0.50), (-45, 0.70), (45, 0.70), (165, 0.60), (-165, 0.60)],
            "leisure": [(-60, 0.55), (60, 0.55), (0, 0.75), (180, 0.75), (-120, 0.45), (120, 0.45)],
            "constrained": [(-70, 0.40), (70, 0.40), (-110, 0.55), (110, 0.55), (170, 0.50), (-170, 0.50)],
        }
        base = templates.get(self.scene_type, templates["commercial"])
        out: List[tuple] = []
        for i in range(cluster_count):
            a_deg, r_ratio = base[i % len(base)]
            ring = i // len(base)
            out.append((a_deg + (8 * ring), min(0.90, r_ratio + 0.12 * ring)))
        return out

    def _build_group_sidewalk_mask(self) -> Optional[np.ndarray]:
        """Build a sidewalk-only mask for selecting group cluster centers."""
        try:
            import cv2

            sidewalk_mask = np.zeros_like(self.walkable_regions_mask)
            region_groups = [
                self.sidewalks,
                self.sidewalks_near_road,
                self.sidewalks_farfrom_road,
                self.sidewalks_near_road_buffer,
                self.sidewalks_farfrom_road_buffer,
            ]

            for regions in region_groups:
                for region in regions.values():
                    polygon_array = np.array(region["polygon"])
                    polygon_array += self.mask_translate
                    polygon_array = np.floor(polygon_array).astype(int)
                    polygon_array = polygon_array.reshape((-1, 1, 2))
                    cv2.fillPoly(sidewalk_mask, [polygon_array], [255, 255, 255])

            sidewalk_mask = cv2.flip(sidewalk_mask, 0)
            if int(np.count_nonzero(sidewalk_mask[:, :, 0])) <= 0:
                return None
            return sidewalk_mask
        except Exception:
            logger.exception("Failed to build sidewalk-only mask for group placement")
            return None

    def _is_point_on_mask(self, xy: np.ndarray, mask: np.ndarray) -> bool:
        """Check whether a world-coordinate point lies on a binary walkable mask."""
        x = int(np.floor(float(xy[0]) + self.mask_translate[0]))
        y = int(np.floor(float(xy[1]) + self.mask_translate[1]))
        y = mask.shape[0] - 1 - y
        if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            return False
        return bool(mask[y, x, 0] > 0)

    def _pick_cluster_center_candidates(
        self,
        ego_pos: np.ndarray,
        cluster_ids: List[int],
        ego_path_points: Optional[List[np.ndarray]] = None,
    ) -> Dict[int, np.ndarray]:
        """Snap template anchors to legal pedestrian positions so clusters appear in realistic places."""
        candidates: List[np.ndarray] = []
        for ped in self._traffic_humanoids:
            p = ped.position
            candidates.append(np.array([float(p[0]), float(p[1])], dtype=float))

        if ego_path_points:
            candidates.extend([np.array(p, dtype=float) for p in ego_path_points])

        sidewalk_mask = self._build_group_sidewalk_mask()
        if sidewalk_mask is not None:
            sidewalk_candidates = [p for p in candidates if self._is_point_on_mask(p, sidewalk_mask)]
            if len(sidewalk_candidates) > 0:
                candidates = sidewalk_candidates
            else:
                logger.warning("No sidewalk candidate found for group placement. Fallback to default candidates.")

        min_r = self._group_spawn_min_radius
        max_r = self._group_spawn_max_radius
        span = max(1e-3, max_r - min_r)
        templates = self._scene_anchor_templates(len(cluster_ids))
        min_sep = max(self._group_cluster_min_separation, self._group_route_min_separation)

        available = list(candidates)
        placed: Dict[int, np.ndarray] = {}
        chosen: List[np.ndarray] = []

        for i, cluster_id in enumerate(cluster_ids):
            if ego_path_points and len(ego_path_points) > 0:
                path_idx = min(len(ego_path_points) - 1, int(i * max(1, len(ego_path_points)) / max(1, len(cluster_ids))))
                desired = np.array(ego_path_points[path_idx], dtype=float)
            else:
                angle_deg, ratio = templates[i]
                angle = np.deg2rad(angle_deg)
                radius = min_r + span * float(ratio)
                desired = ego_pos + np.array([np.cos(angle), np.sin(angle)], dtype=float) * radius

            best_idx = -1
            best_score = float("inf")
            for j, cand in enumerate(available):
                d_ego = float(np.linalg.norm(cand - ego_pos))
                if d_ego < min_r or d_ego > max_r:
                    continue
                if any(float(np.linalg.norm(cand - c)) < min_sep for c in chosen):
                    continue
                score = float(np.linalg.norm(cand - desired))
                if score < best_score:
                    best_score = score
                    best_idx = j

            if best_idx >= 0:
                picked = available.pop(best_idx)
            else:
                picked = desired

            # Even in fallback mode, keep cluster centers separated.
            for existing in chosen:
                diff = np.array(picked, dtype=float) - existing
                dist = float(np.linalg.norm(diff))
                if dist >= min_sep:
                    continue
                if dist < 1e-5:
                    theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
                    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                else:
                    direction = diff / dist
                picked = existing + direction * min_sep

            chosen.append(np.array(picked, dtype=float))
            placed[cluster_id] = np.array(picked, dtype=float)

        return placed

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
            self._group_member_dynamic_offset.pop(idx, None)
            self._group_member_dynamic_timer.pop(idx, None)
        self._group_released.add(cluster_id)
        self._group_cluster_drifts.pop(cluster_id, None)
        self._group_cluster_positions.pop(cluster_id, None)
        self._group_cluster_member_radius.pop(cluster_id, None)
        self._group_cluster_ring_step.pop(cluster_id, None)
        self._group_cluster_mode.pop(cluster_id, None)
        self._group_cluster_compression.pop(cluster_id, None)
        self._group_release_counter.pop(cluster_id, None)
        for idx in members:
            self._group_member_heading.pop(idx, None)
        logger.info("Released group cluster %d with %d pedestrians", cluster_id, released_count)

    def _update_group_member_dynamic_offset(self, idx: int) -> np.ndarray:
        """Occasionally nudge grouped members for subtle realism without breaking formation."""
        timer = self._group_member_dynamic_timer.get(idx, 0)
        offset = self._group_member_dynamic_offset.get(idx, np.array([0.0, 0.0], dtype=float))
        if timer > 0:
            self._group_member_dynamic_timer[idx] = timer - 1
            # Smoothly decay displacement near the end of a small move window.
            if timer < 6:
                offset = offset * 0.72
                self._group_member_dynamic_offset[idx] = offset
            return offset

        if self._rng.random() < self._group_member_idle_shift_prob:
            theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
            radius = float(self._rng.uniform(0.08, self._group_member_idle_shift_radius))
            offset = np.array([np.cos(theta) * radius, np.sin(theta) * radius], dtype=float)
            duration = max(
                6,
                int(self._rng.normal(self._group_member_idle_shift_steps_mean, self._group_member_idle_shift_steps_mean * 0.35)),
            )
            self._group_member_dynamic_offset[idx] = offset
            self._group_member_dynamic_timer[idx] = duration
            return offset

        self._group_member_dynamic_offset[idx] = offset * 0.0
        return self._group_member_dynamic_offset[idx]

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

        ego_pos: Optional[np.ndarray] = None
        try:
            ego_pos = np.array(self.engine.agent.position[:2])
        except Exception:
            pass

        if self._group_release_enable and self._group_release_counter:
            to_release: List[int] = []
            for cluster_id in list(self._group_release_counter.keys()):
                self._group_release_counter[cluster_id] -= 1
                if self._group_release_counter[cluster_id] <= 0:
                    to_release.append(cluster_id)
            for cluster_id in to_release:
                self._release_cluster(cluster_id)

        self._update_group_clusters_pre_step(ego_pos)

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
                        offset = offset + self._update_group_member_dynamic_offset(idx)
                        actual_pos = cluster_center + offset
                        # Keep 2D target like default pedestrian pipeline so z/ground height
                        # is resolved consistently by the engine.
                        target_pos = (float(actual_pos[0]), float(actual_pos[1]))

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

            if role in (ROLE_GROUP_LEADER, ROLE_GROUP_FOLLOWER):
                cluster_id = self._group_cluster_map.get(idx, None)
                center = self._group_cluster_positions.get(cluster_id) if cluster_id is not None else None
                if center is not None and cluster_id not in self._group_released:
                    member_pos = np.array([float(target_pos[0]), float(target_pos[1])], dtype=float)
                    slot = self._group_member_slot.get(idx, 0)
                    static_pos = center + self._group_slot_offset(cluster_id, slot)
                    inward = center - static_pos
                    inward_norm = float(np.linalg.norm(inward))
                    if inward_norm > 1e-6:
                        target_heading = float(np.arctan2(inward[1], inward[0]))
                    else:
                        target_heading = float(ped.heading_theta)

                    prev_heading = float(self._group_member_heading.get(idx, target_heading))
                    delta = float(np.arctan2(np.sin(target_heading - prev_heading), np.cos(target_heading - prev_heading)))
                    heading_rad = prev_heading + 0.25 * delta
                    self._group_member_heading[idx] = heading_rad

                    from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
                    if isinstance(ped, BasePedestrian) and ped.render:
                        dynamic_mag = float(np.linalg.norm(self._group_member_dynamic_offset.get(idx, np.zeros(2))))
                        ped.set_anim_by_speed(0.20 if dynamic_mag > 1e-3 else 0.0)

                    ped.set_position(target_pos)
                    ped.set_heading_theta(heading_rad)
                    try:
                        ped._body.setAngularVelocity((0.0, 0.0, 0.0))
                    except Exception:
                        pass
                    continue
                heading = get_dest_heading(ped, target_pos)
            else:
                heading = get_dest_heading(ped, target_pos)
            prev_pos = ped.position
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
