#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

LABEL_MAP = {
    0: "NEGATIVE_SOCIAL",
    1: "NEUTRAL",
    2: "POSITIVE_SOCIAL",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample 30% of NEUTRAL items from recorded_dataset/new_labels, "
            "combine them with all non-NEUTRAL items, and copy merged_npy and rgb files "
            "into final output folders."
        )
    )
    parser.add_argument("--json-dir", type=str, default="./recorded_dataset/new_labels/json")
    parser.add_argument("--merged-npy-dir", type=str, default="./recorded_dataset/new_labels/merged_npy")
    parser.add_argument("--rgb-dir", type=str, default="./recorded_dataset/rgb_merged")
    parser.add_argument("--final-merged-npy-dir", type=str, default="./recorded_dataset/new_labels/final_merged_npy")
    parser.add_argument("--final-rgb-dir", type=str, default="./recorded_dataset/final_rgb_merged")
    parser.add_argument("--neutral-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def extract_idx_from_name(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        raise ValueError(f"Unexpected file name: {path.name}")
    return int(stem.split("_")[1])


def load_json_label(json_path: Path) -> Tuple[int, str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    idx = int(obj.get("idx", extract_idx_from_name(json_path)))
    label_id = obj.get("vlm_label_id", obj.get("label"))
    label_name = obj.get("vlm_label_class", LABEL_MAP.get(int(label_id), str(label_id)))

    if label_id is None:
        raise ValueError(f"Missing label id in {json_path}")

    return idx, str(label_name), int(label_id)


def ensure_empty_or_create(path: Path, clear_output: bool):
    if clear_output and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    json_dir = Path(args.json_dir)
    merged_npy_dir = Path(args.merged_npy_dir)
    rgb_dir = Path(args.rgb_dir)
    final_merged_npy_dir = Path(args.final_merged_npy_dir)
    final_rgb_dir = Path(args.final_rgb_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"json dir not found: {json_dir}")
    if not merged_npy_dir.exists():
        raise FileNotFoundError(f"merged_npy dir not found: {merged_npy_dir}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb dir not found: {rgb_dir}")

    ensure_empty_or_create(final_merged_npy_dir, args.clear_output)
    ensure_empty_or_create(final_rgb_dir, args.clear_output)

    json_files = sorted(json_dir.glob("step_*.json"))
    if not json_files:
        raise RuntimeError(f"No step_*.json files found in {json_dir}")

    neutral_items: List[Dict] = []
    non_neutral_items: List[Dict] = []

    for json_path in json_files:
        idx, label_name, label_id = load_json_label(json_path)

        merged_npy_path = merged_npy_dir / f"step_{idx:06d}.npy"
        rgb_path = rgb_dir / f"step_{idx:06d}.png"

        if not merged_npy_path.exists():
            print(f"[skip] missing merged npy: {merged_npy_path}")
            continue
        if not rgb_path.exists():
            print(f"[skip] missing rgb image: {rgb_path}")
            continue

        item = {
            "idx": idx,
            "label_name": label_name,
            "label_id": label_id,
            "json_path": json_path,
            "merged_npy_path": merged_npy_path,
            "rgb_path": rgb_path,
        }

        if label_id == 1 or label_name.upper() == "NEUTRAL":
            neutral_items.append(item)
        else:
            non_neutral_items.append(item)

    rng = random.Random(args.seed)

    neutral_count = len(neutral_items)
    sample_count = int(round(neutral_count * args.neutral_ratio))
    sample_count = max(0, min(sample_count, neutral_count))

    sampled_neutral = rng.sample(neutral_items, sample_count) if sample_count > 0 else []
    selected_items = sorted(sampled_neutral + non_neutral_items, key=lambda x: x["idx"])

    copied_npy = 0
    copied_rgb = 0

    for item in selected_items:
        dst_npy = final_merged_npy_dir / item["merged_npy_path"].name
        dst_rgb = final_rgb_dir / item["rgb_path"].name

        shutil.copy2(item["merged_npy_path"], dst_npy)
        copied_npy += 1

        shutil.copy2(item["rgb_path"], dst_rgb)
        copied_rgb += 1

    summary = {
        "json_dir": str(json_dir),
        "merged_npy_dir": str(merged_npy_dir),
        "rgb_dir": str(rgb_dir),
        "final_merged_npy_dir": str(final_merged_npy_dir),
        "final_rgb_dir": str(final_rgb_dir),
        "neutral_ratio": args.neutral_ratio,
        "seed": args.seed,
        "total_valid_items": len(neutral_items) + len(non_neutral_items),
        "neutral_total": len(neutral_items),
        "neutral_sampled": len(sampled_neutral),
        "non_neutral_kept": len(non_neutral_items),
        "final_total": len(selected_items),
    }

    summary_path = final_merged_npy_dir / "selection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    selected_index_path = final_merged_npy_dir / "selected_indices.txt"
    with open(selected_index_path, "w", encoding="utf-8") as f:
        for item in selected_items:
            f.write(f"{item['idx']:06d}\t{item['label_name']}\t{item['label_id']}\n")

    print("Done.")
    print(f"Neutral total:      {len(neutral_items)}")
    print(f"Neutral sampled:    {len(sampled_neutral)}")
    print(f"Non-neutral kept:   {len(non_neutral_items)}")
    print(f"Final total:        {len(selected_items)}")
    print(f"Copied npy files:   {copied_npy} -> {final_merged_npy_dir}")
    print(f"Copied rgb files:   {copied_rgb} -> {final_rgb_dir}")
    print(f"Saved summary to:   {summary_path}")
    print(f"Saved index list to:{selected_index_path}")


if __name__ == "__main__":
    main()
