"""Shared, leakage-safe utilities for MetaUrban VLM post-training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from PIL import Image


LABELS = ("NEGATIVE_SOCIAL", "NEUTRAL", "POSITIVE_SOCIAL")
LABEL_SET = set(LABELS)
SYSTEM_PROMPT = (
    "You are a strict classifier of socially appropriate robot behavior in an urban "
    "shared space. Judge only the post-transition observation. Return valid JSON only."
)
USER_PROMPT_TEMPLATE = """Classify the social behavior visible in this single post-transition observation.

Use only:
- the RGB image as primary evidence
- ego_speed: {ego_speed:.4f} m/s
- ego_heading: {ego_heading:.4f} rad

Choose exactly one label: NEGATIVE_SOCIAL, NEUTRAL, or POSITIVE_SOCIAL.
If there is no clear pedestrian interaction or social signal, choose NEUTRAL.
Return exactly one JSON object with no explanation: {{"label":"<LABEL>"}}"""


def target_text(label: str) -> str:
    label = normalize_label(label)
    return json.dumps({"label": label}, separators=(",", ":"))


def normalize_label(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = {0: "NEGATIVE_SOCIAL", 1: "NEUTRAL", 2: "POSITIVE_SOCIAL"}.get(value, value)
    label = str(value).strip().upper()
    if label not in LABEL_SET:
        raise ValueError(f"Unsupported label {value!r}; expected one of {LABELS}")
    return label


def build_messages(example: Dict[str, Any], image: "Image.Image", include_target: bool) -> List[Dict[str, Any]]:
    prompt = USER_PROMPT_TEMPLATE.format(
        ego_speed=float(example["ego_speed"]),
        ego_heading=float(example["ego_heading"]),
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    if include_target:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text(example["label"])}],
            }
        )
    return messages


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def resolve_image(dataset_root: Path, example: Dict[str, Any]) -> "Image.Image":
    from PIL import Image
    relative = Path(example["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {relative}")
    image_path = Path(dataset_root) / relative
    with Image.open(image_path) as image:
        return image.convert("RGB")


def parse_prediction(text: str) -> Tuple[Optional[str], bool]:
    """Parse the exact target schema and report structured-output validity."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None, False
    if not isinstance(obj, dict) or set(obj) != {"label"}:
        return None, False
    try:
        return normalize_label(obj["label"]), True
    except ValueError:
        return None, False


def extract_label_lenient(text: str) -> Optional[str]:
    label, valid = parse_prediction(text)
    if valid:
        return label
    upper = str(text).upper()
    matches = [label for label in LABELS if re.search(rf"\b{label}\b", upper)]
    return matches[0] if len(matches) == 1 else None


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the training config") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    return config


def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_example_schema(rows: Iterable[Dict[str, Any]]) -> None:
    required = {"episode_id", "step_index", "image_path", "ego_speed", "ego_heading", "label"}
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(
                f"Example {index} fields must be exactly {sorted(required)}; got {sorted(row)}"
            )
        normalize_label(row["label"])
        float(row["ego_speed"])
        float(row["ego_heading"])
