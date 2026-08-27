#!/usr/bin/env python3
"""Join approved teacher labels to raw episodes and split by whole episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from common import LABELS, load_jsonl, normalize_label, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="Teacher JSONL with label and record identity")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def record_keys(row: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    if "episode_id" in row and "step_index" in row:
        yield ("episode_step", (str(row["episode_id"]), int(row["step_index"])))
    if row.get("record_id") is not None:
        yield ("record_id", str(row["record_id"]))
    if row.get("image_path") is not None:
        yield ("image_path", Path(str(row["image_path"])).as_posix())


def label_value(row: Dict[str, Any]) -> str:
    for field in ("label", "vlm_label_class", "target"):
        if field in row:
            return normalize_label(row[field])
    if "vlm_label_id" in row:
        return normalize_label(row["vlm_label_id"])
    raise ValueError("Annotation has no label/vlm_label_class/target/vlm_label_id")


def build_label_index(rows):
    index = {}
    for number, row in enumerate(rows, 1):
        label = label_value(row)
        keys = list(record_keys(row))
        if not keys:
            raise ValueError(f"Label row {number} has no episode+step, record_id, or image_path identity")
        for key in keys:
            previous = index.get(key)
            if previous is not None and previous != label:
                raise ValueError(f"Conflicting labels for {key}: {previous} vs {label}")
            index[key] = label
    return index


def choose_label(raw: Dict[str, Any], index):
    candidates = list(record_keys(raw))
    labels = {index[key] for key in candidates if key in index}
    if len(labels) > 1:
        raise ValueError(f"Conflicting annotation identities for {raw.get('episode_id')}:{raw.get('step_index')}")
    return next(iter(labels)) if labels else None


def allocate_splits(episode_ids, ratios, seed):
    ids = sorted(episode_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n < 3:
        raise ValueError("At least three labeled episodes are required for non-overlapping train/val/test splits")
    raw_counts = [n * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    for idx in sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[idx] += 1
    for idx in range(3):
        if counts[idx] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            if counts[donor] <= 1:
                raise ValueError("Cannot make every split non-empty")
            counts[donor] -= 1
            counts[idx] += 1
    train_end = counts[0]
    val_end = train_end + counts[1]
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def main():
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must be positive and sum to 1")

    label_index = build_label_index(load_jsonl(args.labels))
    examples = []
    unlabeled = 0
    records_paths = sorted((args.dataset_root / "episodes").glob("*/records.jsonl"))
    if not records_paths:
        raise FileNotFoundError(f"No episode records found under {args.dataset_root}")

    for records_path in records_paths:
        for raw in load_jsonl(records_path):
            label = choose_label(raw, label_index)
            if label is None:
                unlabeled += 1
                continue
            model_input = raw.get("model_input", {})
            if set(model_input) != {"image_path", "ego_speed", "ego_heading"}:
                raise ValueError(f"Unsafe model_input schema in {records_path}")
            image_path = Path(str(model_input["image_path"]))
            if image_path.is_absolute() or ".." in image_path.parts:
                raise ValueError(f"Unsafe image path in {records_path}: {image_path}")
            if not (args.dataset_root / image_path).is_file():
                raise FileNotFoundError(f"Missing image referenced by {records_path}: {image_path}")
            ego_speed = float(model_input["ego_speed"])
            ego_heading = float(model_input["ego_heading"])
            if not math.isfinite(ego_speed) or not math.isfinite(ego_heading):
                raise ValueError(f"Non-finite ego state in {records_path}")
            # No evaluation, reward, action, scenario, terminal, or teacher fields are copied.
            examples.append(
                {
                    "episode_id": str(raw["episode_id"]),
                    "step_index": int(raw["step_index"]),
                    "image_path": image_path.as_posix(),
                    "ego_speed": ego_speed,
                    "ego_heading": ego_heading,
                    "label": label,
                }
            )

    if not examples:
        raise ValueError("No raw records matched the supplied labels")
    by_episode = defaultdict(list)
    for example in examples:
        by_episode[example["episode_id"]].append(example)
    splits = allocate_splits(by_episode, ratios, args.seed)
    assignment = {episode: split for split, episodes in splits.items() for episode in episodes}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    class_counts = {}
    for split in ("train", "val", "test"):
        rows = sorted(
            (row for row in examples if assignment[row["episode_id"]] == split),
            key=lambda row: (row["episode_id"], row["step_index"]),
        )
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        split_counts[split] = len(rows)
        class_counts[split] = dict(sorted(Counter(row["label"] for row in rows).items()))

    source_digest = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    save_json(
        args.output_dir / "split_manifest.json",
        {
            "version": 1,
            "strategy": "deterministic whole-episode split",
            "seed": args.seed,
            "ratios": dict(zip(("train", "val", "test"), ratios)),
            "episode_assignments": dict(sorted(assignment.items())),
            "teacher_labels_sha256": source_digest,
        },
    )
    save_json(
        args.output_dir / "dataset_statistics.json",
        {
            "labels": list(LABELS),
            "labeled_examples": len(examples),
            "unlabeled_raw_records_excluded": unlabeled,
            "episodes": len(by_episode),
            "examples_per_split": split_counts,
            "class_counts_per_split": class_counts,
            "model_visible_fields": ["image_path", "ego_speed", "ego_heading"],
        },
    )
    print(json.dumps({"examples": len(examples), "episodes": len(by_episode), "splits": split_counts}, indent=2))


if __name__ == "__main__":
    main()
