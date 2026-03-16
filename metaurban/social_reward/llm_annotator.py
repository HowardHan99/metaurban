"""
LLM Annotation Pipeline
=======================
Takes sliding-window clip .npz files produced by ``ClipExtractor``, extracts a
representative set of frames, sends them to a vision LLM, and writes one
annotation JSON file per clip.

Supported backends
------------------
* ``openai``   — GPT-4o / GPT-4-vision (requires ``OPENAI_API_KEY``)
* ``google``   — Gemini 1.5 Pro (requires ``GOOGLE_API_KEY``)
* ``mock``     — deterministic stub for unit tests (no API key needed)

Output format
-------------
Each clip produces a ``<clip_stem>.json`` file:
{
  "clip_id":   "clip_episode_s000000_seed42_T0_w00000",
  "clip_file": "clips/clip_episode_s000000_seed42_T0_w00000.npz",
  "has_pedestrian": true,
  "ego_was_moving": true,
  "annotations": [
    {
      "clip_id": "...",
      "label": "failure_to_yield",
      "present": true,
      "severity": 2,
      "confidence": 2,
      "start_frame": 5,
      "end_frame": 18,
      "evidence": "Ego continues forward while pedestrian in crosswalk yields path"
    },
    ...   (one entry per label)
  ],
  "social_penalty": 0.43,
  "penalty_breakdown": { "failure_to_yield": 0.43, ... }
}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metaurban.social_reward.taxonomy import (
    SOCIAL_ISSUE_DEFINITIONS,
    LLM_SYSTEM_PROMPT_FORMATTED,
    validate_annotation,
    compute_social_penalty,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame sampling helpers
# ---------------------------------------------------------------------------

def _frames_from_clip(
    data: np.lib.npyio.NpzFile,
    max_frames: int = 8,
) -> Optional[List[np.ndarray]]:
    """
    Return up to *max_frames* RGB frames uniformly sampled from the clip.
    Returns None if the clip has no ``rgb_frames`` array.
    """
    if "rgb_frames" not in data.files:
        return None
    frames: np.ndarray = data["rgb_frames"]  # (T, H, W, 3) uint8
    T = frames.shape[0]
    indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)
    return [frames[i] for i in indices]


def _frame_to_base64(frame: np.ndarray) -> str:
    """Encode a uint8 HxWx3 numpy array as a base64 JPEG string."""
    try:
        from PIL import Image
        img = Image.fromarray(frame.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Fallback: raw PNG via stdlib only (slow but dependency-free)
        import struct, zlib
        h, w, c = frame.shape
        assert c == 3
        raw = b"".join(
            b"\x00" + frame[row].tobytes() for row in range(h)
        )
        def chunk(ctype, data):
            c_len = struct.pack(">I", len(data))
            c_crc = struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
            return c_len + ctype + data + c_crc
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        idat = zlib.compress(raw)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        return base64.b64encode(png).decode("utf-8")


# ---------------------------------------------------------------------------
# Clip metadata helpers
# ---------------------------------------------------------------------------

def _has_pedestrian(data: np.lib.npyio.NpzFile) -> bool:
    """True if any step in the clip had a pedestrian closer than 20 m."""
    if "min_agent_dist" in data.files:
        return bool(np.any(data["min_agent_dist"] < 20.0))
    return True  # conservative default


def _ego_was_moving(data: np.lib.npyio.NpzFile) -> bool:
    """True if mean ego speed > 0.2 m/s."""
    if "ego_speed" in data.files:
        return bool(np.mean(data["ego_speed"]) > 0.2)
    return True


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_openai(
    clip_id: str,
    frames_b64: Optional[List[str]],
    clip_text_summary: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    """Call OpenAI vision API and return parsed annotation list."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_content: List[Dict] = []

    # Attach frames if available
    if frames_b64:
        for i, b64 in enumerate(frames_b64):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })

    user_content.append({
        "type": "text",
        "text": (
            f"clip_id: {clip_id}\n\n"
            f"Trajectory summary:\n{clip_text_summary}\n\n"
            "Annotate all 8 social issue labels as instructed."
        ),
    })

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT_FORMATTED},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("OpenAI API error (attempt %d): %s", attempt + 1, exc)
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    raise RuntimeError(f"OpenAI annotation failed for clip {clip_id} after {max_retries} retries.")


def _call_google(
    clip_id: str,
    frames_b64: Optional[List[str]],
    clip_text_summary: str,
    model: str = "gemini-1.5-pro-latest",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    """Call Google Gemini vision API and return parsed annotation list."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = genai.GenerativeModel(model)

    parts: List[Any] = []
    if frames_b64:
        for b64 in frames_b64:
            parts.append({"mime_type": "image/jpeg", "data": b64})
    parts.append(
        f"{LLM_SYSTEM_PROMPT_FORMATTED}\n\n"
        f"clip_id: {clip_id}\n\nTrajectory summary:\n{clip_text_summary}\n\n"
        "Annotate all 8 social issue labels as instructed."
    )

    for attempt in range(max_retries):
        try:
            response = gemini.generate_content(parts)
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rstrip("`").strip()
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("Google API error (attempt %d): %s", attempt + 1, exc)
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    raise RuntimeError(f"Google annotation failed for clip {clip_id} after {max_retries} retries.")


def _call_mock(
    clip_id: str,
    frames_b64: Optional[List[str]],
    clip_text_summary: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Deterministic mock backend.

    Returns annotations with ``present=False`` for all labels except
    ``unsafe_proximity`` if any crash flag was set in the text summary.
    Useful for unit tests and dry-runs.
    """
    has_crash = "crash" in clip_text_summary.lower()
    result = []
    for label in SOCIAL_ISSUE_DEFINITIONS:
        present = has_crash and label == "unsafe_proximity"
        result.append({
            "clip_id":     clip_id,
            "label":       label,
            "present":     present,
            "severity":    2 if present else 0,
            "confidence":  3 if present else 3,
            "start_frame": 0,
            "end_frame":   0,
            "evidence":    "Mock: crash detected" if present else "N/A",
        })
    return result


_BACKENDS = {
    "openai": _call_openai,
    "google": _call_google,
    "mock":   _call_mock,
}


# ---------------------------------------------------------------------------
# Trajectory text summary (replaces frames when rgb is unavailable)
# ---------------------------------------------------------------------------

def _build_trajectory_summary(data: np.lib.npyio.NpzFile) -> str:
    """
    Produce a compact textual description of the clip's kinematics and events
    for text-only LLM calls (or as supplemental context alongside frames).
    """
    lines: List[str] = []
    n = int(data["n_steps"][0]) if "n_steps" in data.files else len(data["ego_speed"])

    spd  = data["ego_speed"]    if "ego_speed"    in data.files else None
    dist = data["min_agent_dist"] if "min_agent_dist" in data.files else None
    rc   = data["route_completion"] if "route_completion" in data.files else None

    lines.append(f"Clip length: {n} steps")
    if spd is not None:
        lines.append(
            f"Ego speed: mean={float(np.mean(spd)):.2f} max={float(np.max(spd)):.2f} m/s"
        )
    if dist is not None:
        lines.append(
            f"Min agent distance: mean={float(np.mean(dist)):.2f} "
            f"min={float(np.min(dist)):.2f} m"
        )
    if rc is not None:
        lines.append(
            f"Route completion: {float(rc[0]):.2f} -> {float(rc[-1]):.2f}"
        )

    # Events
    events: List[str] = []
    for flag in ("crash_vehicle", "crash_human", "crash_object", "out_of_road"):
        if flag in data.files and np.any(data[flag]):
            step_idx = int(np.argmax(data[flag]))
            events.append(f"{flag} at step {step_idx}")
    if events:
        lines.append("Events: " + "; ".join(events))
    else:
        lines.append("Events: none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main annotator class
# ---------------------------------------------------------------------------

class ClipAnnotator:
    """
    Annotates one or more clip .npz files using a vision LLM.

    Parameters
    ----------
    backend : str
        One of ``"openai"``, ``"google"``, ``"mock"``.
    model : str
        Model name passed to the backend (backend-specific default if empty).
    max_frames : int
        Number of RGB frames to sample per clip for the vision request.
    out_dir : str | Path
        Directory where annotation .json files are written.
    skip_existing : bool
        If True, skip clips whose annotation JSON already exists.
    global_lambda : float
        Lambda passed to ``compute_social_penalty`` for the reward summary.
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "",
        max_frames: int = 8,
        out_dir: str | Path = "dataset/annotations",
        skip_existing: bool = True,
        global_lambda: float = 0.5,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_BACKENDS)}")
        self.backend        = backend
        self.model          = model
        self.max_frames     = max_frames
        self.out_dir        = Path(out_dir)
        self.skip_existing  = skip_existing
        self.global_lambda  = global_lambda
        self._call_fn       = _BACKENDS[backend]

    # ------------------------------------------------------------------
    def annotate_clip(self, clip_path: Path) -> Optional[Dict[str, Any]]:
        """
        Annotate a single clip and return the annotation dict (also saved).
        Returns None if the clip was skipped or failed validation.
        """
        clip_id   = clip_path.stem
        out_path  = self.out_dir / f"{clip_id}.json"

        if self.skip_existing and out_path.exists():
            logger.debug("Skipping existing annotation: %s", out_path.name)
            return json.loads(out_path.read_text())

        data = np.load(str(clip_path), allow_pickle=False)

        has_ped     = _has_pedestrian(data)
        ego_moving  = _ego_was_moving(data)
        frames_b64  = None

        raw_frames = _frames_from_clip(data, max_frames=self.max_frames)
        if raw_frames is not None:
            frames_b64 = [_frame_to_base64(f) for f in raw_frames]

        summary = _build_trajectory_summary(data)

        # Call LLM backend
        call_kwargs: Dict[str, Any] = {}
        if self.model:
            call_kwargs["model"] = self.model

        raw_annotations: List[Dict] = self._call_fn(
            clip_id=clip_id,
            frames_b64=frames_b64,
            clip_text_summary=summary,
            **call_kwargs,
        )

        # Validate and inject clip_id if missing
        validated: List[Dict] = []
        for ann in raw_annotations:
            ann["clip_id"] = clip_id  # ensure it's set
            ok, errs = validate_annotation(ann, has_ped, ego_moving)
            if not ok:
                logger.warning("Annotation validation warnings for %s/%s: %s",
                               clip_id, ann.get("label", "?"), errs)
            validated.append(ann)

        # Compute reward label
        penalty, breakdown = compute_social_penalty(validated, global_lambda=self.global_lambda)

        result: Dict[str, Any] = {
            "clip_id":          clip_id,
            "clip_file":        str(clip_path),
            "has_pedestrian":   has_ped,
            "ego_was_moving":   ego_moving,
            "annotations":      validated,
            "social_penalty":   round(float(penalty), 6),
            "penalty_breakdown": {k: round(float(v), 6) for k, v in breakdown.items()},
        }

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Annotated %s -> penalty=%.4f  (%s)", clip_id, penalty, out_path.name)
        return result

    # ------------------------------------------------------------------
    def annotate_batch(
        self,
        clips_dir: Path,
        pattern: str = "clip_*.npz",
        rate_limit_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Annotate all clips matching *pattern* in *clips_dir*.

        ``rate_limit_delay`` seconds are waited between API calls to stay
        within typical rate limits.
        """
        clip_paths = sorted(clips_dir.glob(pattern))
        if not clip_paths:
            logger.warning("No clips found in %s matching '%s'", clips_dir, pattern)
            return []

        results: List[Dict[str, Any]] = []
        for i, cp in enumerate(clip_paths):
            try:
                ann = self.annotate_clip(cp)
                if ann is not None:
                    results.append(ann)
            except Exception as exc:
                logger.error("Failed to annotate %s: %s", cp.name, exc)
            if i < len(clip_paths) - 1 and self.backend != "mock":
                time.sleep(rate_limit_delay)

        logger.info(
            "Batch annotation done: %d/%d clips annotated to %s",
            len(results), len(clip_paths), self.out_dir,
        )
        return results
