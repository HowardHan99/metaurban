"""
Offline Dataset Collector for Social Reward Learning

Runs the MetaUrban simulator under a given policy and records every step into an
in-memory episode buffer.  When the episode ends (or on demand) the buffer is flushed
to disk as a compressed NumPy archive (.npz), one file per episode.

Collected fields per step
--------------------------
obs          : np.ndarray  -- raw lidar/state observation returned by env.step()
action       : np.ndarray  -- action taken at this step
reward       : float       -- original task reward
rgb_frame    : np.ndarray  -- (H, W, 3) uint8 from the main camera, or None if headless
ego_pos      : (x, y)      -- ego world position
ego_heading  : float       -- ego heading in radians
ego_speed    : float       -- ego speed in m/s
route_completion : float   -- navigation progress [0, 1]
lateral_dist : float       -- lateral distance from reference line (m)
min_agent_dist : float     -- minimum euclidean distance to any other agent (m)
crash_vehicle  : bool
crash_human    : bool
crash_object   : bool
out_of_road    : bool
step_index     : int        -- global step counter within the episode
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _nearest_agent_distance(env) -> float:
    """
    Return the minimum Euclidean distance from the ego vehicle to any other
    spawned agent (vehicle or pedestrian) in the scene.

    Falls back to float('inf') when no other agents are present.
    """
    try:
        ego = env.agent
        ego_pos = np.array(ego.position[:2])
        min_dist = float("inf")
        for name, obj in env.engine.agent_manager.active_agents.items():
            if obj is ego:
                continue
            try:
                other_pos = np.array(obj.position[:2])
                d = float(np.linalg.norm(ego_pos - other_pos))
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue
        # Also check traffic participants registered separately
        if hasattr(env.engine, "object_manager"):
            for name, obj in env.engine.object_manager.spawned_objects.items():
                try:
                    other_pos = np.array(obj.position[:2])
                    d = float(np.linalg.norm(ego_pos - other_pos))
                    if d < min_dist:
                        min_dist = d
                except Exception:
                    continue
    except Exception:
        min_dist = float("inf")
    return min_dist


def _get_rgb_frame(env) -> Optional[np.ndarray]:
    """
    Retrieve the current RGB frame from the main_camera sensor.
    Returns None when the camera is not available (headless mode).
    """
    try:
        sensor = env.engine.get_sensor("main_camera")
        frame = sensor.perceive(env.agent, clip=False)
        if frame is not None:
            arr = np.array(frame, dtype=np.uint8)
            return arr
    except Exception:
        pass
    return None


def _extract_ego_state(env) -> Dict[str, Any]:
    """Return a dict of scalar state variables for the ego vehicle."""
    ego = env.agent
    pos = ego.position
    return {
        "ego_pos_x":    float(pos[0]),
        "ego_pos_y":    float(pos[1]),
        "ego_heading":  float(ego.heading_theta),
        "ego_speed":    float(ego.speed),
    }


# ---------------------------------------------------------------------------
# Episode buffer
# ---------------------------------------------------------------------------

class EpisodeBuffer:
    """
    Accumulates per-step data for a single episode in lists, then serialises
    to a compressed .npz file on flush.

    The RGB frames are stored in a separate list and saved as a uint8 array
    only if at least one non-None frame was captured.
    """

    def __init__(self, scenario_index: int, seed: int):
        self.scenario_index = scenario_index
        self.seed = seed
        self._steps: List[Dict[str, Any]] = []
        self._rgb_frames: List[Optional[np.ndarray]] = []
        self._has_frames = False

    # ------------------------------------------------------------------
    def record(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: dict,
        env,
    ) -> None:
        """Append one step of data to the buffer."""
        step_idx = len(self._steps)
        ego_state = _extract_ego_state(env)
        min_dist  = _nearest_agent_distance(env)
        rgb       = _get_rgb_frame(env)

        if rgb is not None:
            self._has_frames = True
        self._rgb_frames.append(rgb)

        self._steps.append(
            {
                "step_index":       step_idx,
                "obs":              np.asarray(obs, dtype=np.float32),
                "action":           np.asarray(action, dtype=np.float32),
                "reward":           float(reward),
                "route_completion": float(info.get("route_completion", 0.0)),
                "lateral_dist":     float(info.get("lateral_dist", 0.0)),
                "min_agent_dist":   min_dist,
                "crash_vehicle":    bool(info.get("crash", False) or info.get("crash_vehicle", False)),
                "crash_human":      bool(info.get("crash_human", False)),
                "crash_object":     bool(info.get("crash_object", False)),
                "out_of_road":      bool(info.get("out_of_road", False)),
                **ego_state,
            }
        )

    # ------------------------------------------------------------------
    def flush(self, out_dir: Path) -> Path:
        """
        Write all buffered steps to a compressed .npz file and return its path.

        File name format:  episode_s{scenario_index:06d}_seed{seed}_T{timestamp}.npz
        """
        if not self._steps:
            raise RuntimeError("EpisodeBuffer is empty; nothing to flush.")

        out_dir.mkdir(parents=True, exist_ok=True)
        ts  = int(time.time() * 1000) % 10_000_000
        fname = f"episode_s{self.scenario_index:06d}_seed{self.seed}_T{ts}.npz"
        out_path = out_dir / fname

        # Stack scalar arrays
        n = len(self._steps)
        arrays: Dict[str, np.ndarray] = {
            "step_index":       np.array([s["step_index"]       for s in self._steps], dtype=np.int32),
            "obs":              np.stack([s["obs"]              for s in self._steps]),
            "action":           np.stack([s["action"]           for s in self._steps]),
            "reward":           np.array([s["reward"]           for s in self._steps], dtype=np.float32),
            "route_completion": np.array([s["route_completion"] for s in self._steps], dtype=np.float32),
            "lateral_dist":     np.array([s["lateral_dist"]     for s in self._steps], dtype=np.float32),
            "min_agent_dist":   np.array([s["min_agent_dist"]   for s in self._steps], dtype=np.float32),
            "crash_vehicle":    np.array([s["crash_vehicle"]    for s in self._steps], dtype=bool),
            "crash_human":      np.array([s["crash_human"]      for s in self._steps], dtype=bool),
            "crash_object":     np.array([s["crash_object"]     for s in self._steps], dtype=bool),
            "out_of_road":      np.array([s["out_of_road"]      for s in self._steps], dtype=bool),
            "ego_pos_x":        np.array([s["ego_pos_x"]        for s in self._steps], dtype=np.float32),
            "ego_pos_y":        np.array([s["ego_pos_y"]        for s in self._steps], dtype=np.float32),
            "ego_heading":      np.array([s["ego_heading"]      for s in self._steps], dtype=np.float32),
            "ego_speed":        np.array([s["ego_speed"]        for s in self._steps], dtype=np.float32),
            # metadata
            "scenario_index":   np.array([self.scenario_index], dtype=np.int32),
            "seed":             np.array([self.seed],           dtype=np.int32),
            "n_steps":          np.array([n],                   dtype=np.int32),
        }

        if self._has_frames:
            # Replace None frames with a black placeholder matching the first
            # valid frame's shape.
            first_valid = next(f for f in self._rgb_frames if f is not None)
            placeholder = np.zeros_like(first_valid)
            frames = np.stack(
                [f if f is not None else placeholder for f in self._rgb_frames],
                axis=0,
            )  # (T, H, W, 3)
            arrays["rgb_frames"] = frames

        np.savez_compressed(str(out_path), **arrays)
        logger.info("Flushed %d steps -> %s", n, out_path)
        return out_path

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()
        self._rgb_frames.clear()
        self._has_frames = False


# ---------------------------------------------------------------------------
# Sliding-window clip extractor
# ---------------------------------------------------------------------------

class ClipExtractor:
    """
    Reads a saved episode .npz file and slices it into overlapping clips using
    a sliding window.

    Each clip is saved as a separate .npz file under ``clips_dir``.

    Clip file name format:
        clip_{episode_stem}_w{window_start:05d}.npz

    Parameters
    ----------
    window_steps : int
        Number of simulation steps per clip (default 40 @ 10Hz => 4 s).
    stride_steps : int
        Stride between clip start positions (default equals window_steps for
        non-overlapping clips; use a smaller value for overlapping clips).
    sim_hz : float
        Simulation step frequency in Hz, stored as metadata.
    """

    def __init__(
        self,
        window_steps: int = 40,
        stride_steps: Optional[int] = None,
        sim_hz: float = 10.0,
    ):
        self.window_steps = window_steps
        self.stride_steps = stride_steps if stride_steps is not None else window_steps
        self.sim_hz = sim_hz

    # ------------------------------------------------------------------
    def extract(self, episode_path: Path, clips_dir: Path) -> List[Path]:
        """
        Slice *episode_path* into clips and write them to *clips_dir*.
        Returns list of written clip paths.
        """
        clips_dir.mkdir(parents=True, exist_ok=True)
        data = np.load(str(episode_path), allow_pickle=False)
        n = int(data["n_steps"][0])

        if n < self.window_steps:
            logger.warning(
                "Episode %s has only %d steps (< window %d); skipping.",
                episode_path.name, n, self.window_steps,
            )
            return []

        stem = episode_path.stem
        written: List[Path] = []

        for start in range(0, n - self.window_steps + 1, self.stride_steps):
            end = start + self.window_steps
            clip_arrays: Dict[str, np.ndarray] = {
                "clip_start":    np.array([start],           dtype=np.int32),
                "clip_end":      np.array([end],             dtype=np.int32),
                "window_steps":  np.array([self.window_steps], dtype=np.int32),
                "sim_hz":        np.array([self.sim_hz],     dtype=np.float32),
                "scenario_index": data["scenario_index"],
                "seed":           data["seed"],
            }

            # Slice all per-step arrays
            for key in data.files:
                arr = data[key]
                if arr.ndim >= 1 and arr.shape[0] == n:
                    clip_arrays[key] = arr[start:end]
                elif key not in clip_arrays:
                    clip_arrays[key] = arr  # scalar metadata

            fname   = f"clip_{stem}_w{start:05d}.npz"
            out_path = clips_dir / fname
            np.savez_compressed(str(out_path), **clip_arrays)
            written.append(out_path)

        logger.info(
            "Extracted %d clips from %s -> %s",
            len(written), episode_path.name, clips_dir,
        )
        return written

    # ------------------------------------------------------------------
    def extract_batch(
        self,
        episodes_dir: Path,
        clips_dir: Path,
        pattern: str = "episode_*.npz",
    ) -> List[Path]:
        """Process all episodes matching *pattern* in *episodes_dir*."""
        all_clips: List[Path] = []
        for ep_path in sorted(episodes_dir.glob(pattern)):
            all_clips.extend(self.extract(ep_path, clips_dir))
        logger.info("Batch extraction done: %d total clips.", len(all_clips))
        return all_clips


# ---------------------------------------------------------------------------
# Rolling collector (wraps env.step loop)
# ---------------------------------------------------------------------------

class RollingCollector:
    """
    Wraps an environment and a policy, drives the rollout, and writes episode
    files.  Call ``collect(n_episodes)`` to run.

    Parameters
    ----------
    env          : MetaUrban gym environment (already instantiated).
    policy_fn    : Callable[[np.ndarray], np.ndarray]
                   A function that takes an observation and returns an action.
                   Pass ``lambda obs: env.action_space.sample()`` for random.
    out_dir      : Root directory for dataset output.
    capture_rgb  : Whether to save RGB frames (requires render mode).
    seeds        : Explicit list of seeds.  If None, uses sequential seeding.
    """

    def __init__(
        self,
        env,
        policy_fn,
        out_dir: str | Path = "dataset/episodes",
        capture_rgb: bool = False,
        seeds: Optional[List[int]] = None,
    ):
        self.env        = env
        self.policy_fn  = policy_fn
        self.out_dir    = Path(out_dir)
        self.capture_rgb = capture_rgb
        self.seeds      = seeds

    # ------------------------------------------------------------------
    def collect(self, n_episodes: int) -> List[Path]:
        """Run *n_episodes* rollouts and return paths to written .npz files."""
        written: List[Path] = []

        for ep_idx in range(n_episodes):
            seed = self.seeds[ep_idx] if self.seeds else None
            obs, info = self.env.reset(seed=seed)
            scenario_idx = info.get("scenario_index", ep_idx)
            actual_seed  = info.get("seed", ep_idx) if seed is None else seed

            buf = EpisodeBuffer(scenario_index=scenario_idx, seed=actual_seed)
            terminated = truncated = False

            while not (terminated or truncated):
                action = self.policy_fn(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                buf.record(obs, action, reward, info, self.env)

            path = buf.flush(self.out_dir)
            written.append(path)
            logger.info(
                "Episode %d/%d done: %d steps, seed=%s -> %s",
                ep_idx + 1, n_episodes, len(buf), actual_seed, path.name,
            )

        return written
