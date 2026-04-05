"""
Minimal dataset collector utilities for social offline data generation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _get_rgb_frame(env) -> Optional[np.ndarray]:
    try:
        sensor = env.engine.get_sensor("main_camera")
        frame = sensor.perceive(env.agent, clip=False)
        if frame is not None:
            return np.asarray(frame, dtype=np.uint8)
    except Exception:
        pass
    return None


def _nearest_agent_distance(env) -> float:
    try:
        ego = env.agent
        ego_pos = np.array(ego.position[:2])
        min_dist = float("inf")

        for obj in env.engine.agent_manager.active_agents.values():
            if obj is ego:
                continue
            try:
                d = float(np.linalg.norm(ego_pos - np.array(obj.position[:2])))
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue

        if hasattr(env.engine, "object_manager"):
            for obj in env.engine.object_manager.spawned_objects.values():
                try:
                    d = float(np.linalg.norm(ego_pos - np.array(obj.position[:2])))
                    if d < min_dist:
                        min_dist = d
                except Exception:
                    continue

        return min_dist
    except Exception:
        return float("inf")


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
        frame = _get_rgb_frame(env)
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
