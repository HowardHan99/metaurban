"""
Minimal dataset collector utilities for social offline data generation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


RAW_DATASET_VERSION = "metaurban-vlm-raw-v1"
COLLECTOR_VERSION = "1.0.0"


def _get_rgb_frame(env, obs=None) -> Optional[np.ndarray]:
    """Return RGB for the current, already-completed simulator step."""
    if isinstance(obs, dict) and "image" in obs:
        frame = np.asarray(obs["image"])
        if frame.ndim == 4:
            frame = frame[..., -1]
        if frame.ndim == 3:
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                scale = 255.0 if frame.size and float(np.nanmax(frame)) <= 1.0 else 1.0
                frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(frame)

    try:
        sensor = env.engine.get_sensor("main_camera")
        frame = sensor.perceive(to_float=False)
        if frame is not None:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            return np.ascontiguousarray(frame)
    except Exception:
        pass
    return None


def _nearest_agent_distance(env) -> float:
    try:
        ego = env.agent
        ego_pos = np.array(ego.position[:2])
        min_dist = float("inf")

        candidates = list(env.engine.agent_manager.active_agents.values())
        humanoid_manager = getattr(env.engine, "humanoid_manager", None)
        if humanoid_manager is not None:
            # Social pedestrians are managed separately from controllable ego
            # agents.  ``_traffic_humanoids`` is the active, scene-visible set;
            # the previous implementation only examined the ego agent itself.
            candidates.extend(getattr(humanoid_manager, "_traffic_humanoids", []))

        seen = set()
        for obj in candidates:
            if obj is ego:
                continue
            identity = id(obj)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                d = float(np.linalg.norm(ego_pos - np.array(obj.position[:2])))
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue

        return min_dist
    except Exception:
        return float("inf")


def _finite_or_none(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class StructuredEpisodeWriter:
    """Write one episode as relative RGB files plus a JSONL transition log.

    Every record is a sampled post-transition state. The action is deliberately
    named ``action_from_previous_state`` because it produced the stored state;
    it is metadata and is never placed under ``model_input``.
    """

    def __init__(
        self,
        dataset_root: Path,
        episode_id: str,
        scenario_index: int,
        seed: int,
        sampling_interval: int = 1,
    ):
        if sampling_interval < 1:
            raise ValueError("sampling_interval must be at least 1")

        self.dataset_root = Path(dataset_root)
        self.episode_id = str(episode_id)
        self.scenario_index = int(scenario_index)
        self.seed = int(seed)
        self.sampling_interval = int(sampling_interval)
        self.episode_dir = self.dataset_root / "episodes" / self.episode_id
        self.frames_dir = self.episode_dir / "frames"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.frames_dir.mkdir()
        self.records_path = self.episode_dir / "records.jsonl"
        self._records_file = self.records_path.open("x", encoding="utf-8")
        self.record_count = 0
        self.first_step_index: Optional[int] = None
        self.last_step_index: Optional[int] = None
        self._closed = False

    def should_record(self, step_index: int, terminated: bool, truncated: bool) -> bool:
        return step_index % self.sampling_interval == 0 or terminated or truncated

    def record(
        self,
        *,
        step_index: int,
        post_obs,
        action_from_previous_state,
        environment_reward: float,
        info: dict,
        env,
        terminated: bool,
        truncated: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_record(step_index, terminated, truncated):
            return None
        if self._closed:
            raise RuntimeError("Cannot record into a closed episode")
        if self.last_step_index is not None and step_index <= self.last_step_index:
            raise ValueError("step_index must be strictly increasing")

        frame = _get_rgb_frame(env, post_obs)
        if frame is None:
            raise RuntimeError(
                "RGB capture failed for a sampled transition; refusing to write an invalid record"
            )
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB frame, got shape {frame.shape}")

        ego = env.agent
        ego_speed = _finite_or_none(ego.speed)
        ego_heading = _finite_or_none(ego.heading_theta)
        if ego_speed is None or ego_heading is None:
            raise ValueError("ego_speed and ego_heading must be finite")

        action = np.asarray(action_from_previous_state, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(action)):
            raise ValueError("action_from_previous_state must contain only finite values")

        frame_name = f"{int(step_index):06d}.png"
        relative_image_path = Path("episodes") / self.episode_id / "frames" / frame_name
        Image.fromarray(frame, mode="RGB").save(self.dataset_root / relative_image_path)

        record = {
            "episode_id": self.episode_id,
            "scenario_index": self.scenario_index,
            "seed": self.seed,
            "step_index": int(step_index),
            "model_input": {
                "image_path": relative_image_path.as_posix(),
                "ego_speed": ego_speed,
                "ego_heading": ego_heading,
            },
            "transition_metadata": {
                "action_from_previous_state": action.tolist(),
            },
            "evaluation_only": {
                "min_agent_dist": _finite_or_none(_nearest_agent_distance(env)),
                "crash_human": bool(info.get("crash_human", False)),
                "crash_vehicle": bool(info.get("crash_vehicle", False) or info.get("crash", False)),
                "crash_object": bool(info.get("crash_object", False)),
                "out_of_road": bool(info.get("out_of_road", False)),
                "route_completion": _finite_or_none(info.get("route_completion", 0.0)),
                "environment_reward": _finite_or_none(environment_reward),
                "terminal": bool(terminated),
                "truncation": bool(truncated),
            },
        }
        self._records_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self._records_file.flush()

        self.record_count += 1
        if self.first_step_index is None:
            self.first_step_index = int(step_index)
        self.last_step_index = int(step_index)
        return record

    def close(self, *, simulator_steps: int, terminated: bool, truncated: bool) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("Episode is already closed")
        self._records_file.close()
        self._closed = True
        summary = {
            "episode_id": self.episode_id,
            "scenario_index": self.scenario_index,
            "seed": self.seed,
            "sampling_interval": self.sampling_interval,
            "simulator_steps": int(simulator_steps),
            "record_count": self.record_count,
            "first_step_index": self.first_step_index,
            "last_step_index": self.last_step_index,
            "terminal": bool(terminated),
            "truncation": bool(truncated),
        }
        (self.episode_dir / "episode.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return summary


class EpisodeBuffer:
    def __init__(self, scenario_index: int, seed: int):
        self.scenario_index = scenario_index
        self.seed = seed
        self._steps: List[Dict[str, Any]] = []
        self._frames: List[Optional[np.ndarray]] = []
        self._has_frames = False

    def record(self, obs: np.ndarray, action: np.ndarray, reward: float, info: dict, env) -> None:
        ego = env.agent
        pos = ego.position
        frame = _get_rgb_frame(env, obs)
        if frame is not None:
            self._has_frames = True
        self._frames.append(frame)

        self._steps.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "route_completion": float(info.get("route_completion", 0.0)),
                "lateral_dist": float(info.get("lateral_dist", 0.0)),
                "min_agent_dist": float(_nearest_agent_distance(env)),
                "crash_vehicle": bool(info.get("crash_vehicle", False) or info.get("crash", False)),
                "crash_human": bool(info.get("crash_human", False)),
                "crash_object": bool(info.get("crash_object", False)),
                "out_of_road": bool(info.get("out_of_road", False)),
                "ego_pos_x": float(pos[0]),
                "ego_pos_y": float(pos[1]),
                "ego_heading": float(ego.heading_theta),
                "ego_speed": float(ego.speed),
            }
        )

    def flush(self, out_dir: Path) -> Path:
        if not self._steps:
            raise RuntimeError("EpisodeBuffer is empty")

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000) % 10_000_000
        path = out_dir / f"episode_s{self.scenario_index:06d}_seed{self.seed}_T{ts}.npz"

        n = len(self._steps)
        arrays: Dict[str, np.ndarray] = {
            "step_index": np.arange(n, dtype=np.int32),
            "obs": np.stack([s["obs"] for s in self._steps]),
            "action": np.stack([s["action"] for s in self._steps]),
            "reward": np.array([s["reward"] for s in self._steps], dtype=np.float32),
            "route_completion": np.array([s["route_completion"] for s in self._steps], dtype=np.float32),
            "lateral_dist": np.array([s["lateral_dist"] for s in self._steps], dtype=np.float32),
            "min_agent_dist": np.array([s["min_agent_dist"] for s in self._steps], dtype=np.float32),
            "crash_vehicle": np.array([s["crash_vehicle"] for s in self._steps], dtype=bool),
            "crash_human": np.array([s["crash_human"] for s in self._steps], dtype=bool),
            "crash_object": np.array([s["crash_object"] for s in self._steps], dtype=bool),
            "out_of_road": np.array([s["out_of_road"] for s in self._steps], dtype=bool),
            "ego_pos_x": np.array([s["ego_pos_x"] for s in self._steps], dtype=np.float32),
            "ego_pos_y": np.array([s["ego_pos_y"] for s in self._steps], dtype=np.float32),
            "ego_heading": np.array([s["ego_heading"] for s in self._steps], dtype=np.float32),
            "ego_speed": np.array([s["ego_speed"] for s in self._steps], dtype=np.float32),
            "scenario_index": np.array([self.scenario_index], dtype=np.int32),
            "seed": np.array([self.seed], dtype=np.int32),
            "n_steps": np.array([n], dtype=np.int32),
        }

        if self._has_frames:
            first = next(f for f in self._frames if f is not None)
            zeros = np.zeros_like(first)
            arrays["rgb_frames"] = np.stack([(f if f is not None else zeros) for f in self._frames], axis=0)

        np.savez_compressed(str(path), **arrays)
        return path

    def __len__(self) -> int:
        return len(self._steps)


class ClipExtractor:
    def __init__(self, window_steps: int = 40, stride_steps: Optional[int] = None, sim_hz: float = 10.0):
        self.window_steps = window_steps
        self.stride_steps = stride_steps if stride_steps is not None else window_steps
        self.sim_hz = sim_hz

    def extract(self, episode_path: Path, clips_dir: Path):
        clips_dir.mkdir(parents=True, exist_ok=True)
        data = np.load(str(episode_path), allow_pickle=False)
        n = int(data["n_steps"][0])
        if n < self.window_steps:
            return []

        written = []
        for start in range(0, n - self.window_steps + 1, self.stride_steps):
            end = start + self.window_steps
            arrs: Dict[str, np.ndarray] = {
                "clip_start": np.array([start], dtype=np.int32),
                "clip_end": np.array([end], dtype=np.int32),
                "window_steps": np.array([self.window_steps], dtype=np.int32),
                "sim_hz": np.array([self.sim_hz], dtype=np.float32),
            }
            for k in data.files:
                v = data[k]
                if v.ndim >= 1 and v.shape[0] == n:
                    arrs[k] = v[start:end]
                else:
                    arrs[k] = v
            out = clips_dir / f"clip_{episode_path.stem}_w{start:05d}.npz"
            np.savez_compressed(str(out), **arrs)
            written.append(out)
        return written

    def extract_batch(self, episodes_dir: Path, clips_dir: Path, pattern: str = "episode_*.npz"):
        out = []
        for ep in sorted(episodes_dir.glob(pattern)):
            out.extend(self.extract(ep, clips_dir))
        return out
