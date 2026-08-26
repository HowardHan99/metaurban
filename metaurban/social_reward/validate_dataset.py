#!/usr/bin/env python3
"""Validate an episode-structured MetaUrban VLM raw dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image
import numpy as np


ALLOWED_MODEL_INPUT_FIELDS = {"image_path", "ego_speed", "ego_heading"}
PROHIBITED_MODEL_INPUT_FIELDS = {
    "action",
    "action_from_previous_state",
    "environment_reward",
    "vlm_reward",
    "social_reward",
    "min_agent_dist",
    "crash_human",
    "crash_vehicle",
    "crash_object",
    "out_of_road",
    "route_completion",
    "terminal",
    "truncation",
    "episode_id",
    "scenario_index",
    "seed",
    "step_index",
    "teacher",
    "label",
    "target",
}


def _all_numbers_finite(value) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(v) for v in value)
    return True


def _safe_relative_path(value) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value


def validate_dataset(dataset_root: Path):
    dataset_root = Path(dataset_root)
    errors = []
    invalid_records = 0
    missing_images = 0
    record_count = 0
    decoded_images = set()
    blank_images = 0
    speed_values = []
    heading_values = []
    scenario_distribution = Counter()

    manifest_path = dataset_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = None
        errors.append(f"manifest.json: {exc}")
    if manifest is not None:
        required_manifest_fields = {
            "dataset_version",
            "collector_version",
            "collector_git",
            "creation_timestamp_utc",
            "environment_configuration",
            "collection_configuration",
            "number_of_episodes",
            "number_of_records",
            "scenario_distribution",
            "episode_seeds",
        }
        missing_manifest_fields = required_manifest_fields - set(manifest)
        if missing_manifest_fields:
            errors.append(f"manifest is missing fields: {sorted(missing_manifest_fields)}")

    episodes_root = dataset_root / "episodes"
    episode_dirs = sorted(path for path in episodes_root.glob("*") if path.is_dir())

    for episode_dir in episode_dirs:
        episode_errors = []
        records_path = episode_dir / "records.jsonl"
        summary_path = episode_dir / "episode.json"
        try:
            episode_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            episode_summary = None
            episode_errors.append(f"{summary_path.relative_to(dataset_root)}: {exc}")

        previous_step = None
        first_step = None
        last_record = None
        episode_record_count = 0
        try:
            record_lines = records_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            record_lines = []
            episode_errors.append(f"{records_path.relative_to(dataset_root)}: {exc}")

        for line_number, line in enumerate(record_lines, start=1):
            record_errors = []
            record_count += 1
            episode_record_count += 1
            try:
                record = json.loads(line)
            except Exception as exc:
                invalid_records += 1
                errors.append(
                    f"{records_path.relative_to(dataset_root)}:{line_number}: invalid JSON: {exc}"
                )
                continue

            episode_id = record.get("episode_id")
            if not episode_id or episode_id != episode_dir.name:
                record_errors.append("missing or mismatched episode_id")
            for integer_field in ("scenario_index", "seed"):
                value = record.get(integer_field)
                if isinstance(value, bool) or not isinstance(value, int):
                    record_errors.append(f"{integer_field} is not an integer")

            step_index = record.get("step_index")
            if not isinstance(step_index, int) or isinstance(step_index, bool):
                record_errors.append("step_index is not an integer")
            elif previous_step is not None and step_index <= previous_step:
                record_errors.append("step_index is not strictly increasing")
            else:
                if first_step is None:
                    first_step = step_index
                previous_step = step_index
            last_record = record

            if record.get("scenario_index") != (
                episode_summary.get("scenario_index") if episode_summary else record.get("scenario_index")
            ):
                record_errors.append("scenario_index does not match episode.json")
            if record.get("seed") != (
                episode_summary.get("seed") if episode_summary else record.get("seed")
            ):
                record_errors.append("seed does not match episode.json")

            model_input = record.get("model_input")
            if not isinstance(model_input, dict):
                record_errors.append("model_input is missing or is not an object")
                model_input = {}
            model_fields = set(model_input)
            if model_fields != ALLOWED_MODEL_INPUT_FIELDS:
                record_errors.append(
                    f"model_input fields must be exactly {sorted(ALLOWED_MODEL_INPUT_FIELDS)}"
                )
            prohibited = model_fields & PROHIBITED_MODEL_INPUT_FIELDS
            if prohibited:
                record_errors.append(f"prohibited model_input fields: {sorted(prohibited)}")

            image_path_value = model_input.get("image_path")
            if not _safe_relative_path(image_path_value):
                record_errors.append("image_path is not a safe relative POSIX path")
            else:
                image_path = dataset_root / PurePosixPath(image_path_value)
                if not image_path.is_file():
                    missing_images += 1
                    record_errors.append("image does not exist")
                else:
                    try:
                        with Image.open(image_path) as image:
                            image.load()
                            if image.mode != "RGB" or image.width < 1 or image.height < 1:
                                raise ValueError(f"unexpected image mode/size: {image.mode} {image.size}")
                            pixels = np.asarray(image, dtype=np.uint8)
                            dynamic_range = int(pixels.max()) - int(pixels.min())
                            pixel_std = float(pixels.std())
                            zero_fraction = float(np.count_nonzero(pixels == 0) / pixels.size)
                            unique_colors = len(np.unique(pixels.reshape(-1, 3), axis=0))
                            if (
                                zero_fraction > 0.999
                                or (pixel_std < 1.0 and dynamic_range < 4)
                                or unique_colors < 4
                            ):
                                blank_images += 1
                                raise ValueError(
                                    "blank/uniform image "
                                    f"(std={pixel_std:.3f}, range={dynamic_range}, "
                                    f"zero_fraction={zero_fraction:.6f}, colors={unique_colors})"
                                )
                        decoded_images.add(image_path_value)
                    except Exception as exc:
                        record_errors.append(f"image cannot be decoded: {exc}")

            for field, values in (("ego_speed", speed_values), ("ego_heading", heading_values)):
                value = model_input.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    record_errors.append(f"{field} is missing or non-finite")
                else:
                    values.append(float(value))

            if not _all_numbers_finite(record):
                record_errors.append("record contains a non-finite numeric value")

            if record_errors:
                invalid_records += 1
                for problem in record_errors:
                    errors.append(f"{records_path.relative_to(dataset_root)}:{line_number}: {problem}")

        if episode_summary is not None:
            if episode_summary.get("episode_id") != episode_dir.name:
                episode_errors.append("episode.json has mismatched episode_id")
            if episode_summary.get("record_count") != episode_record_count:
                episode_errors.append("episode.json record_count does not match records.jsonl")
            if episode_summary.get("first_step_index") != first_step:
                episode_errors.append("episode.json first_step_index does not match records.jsonl")
            if episode_summary.get("last_step_index") != previous_step:
                episode_errors.append("episode.json last_step_index does not match records.jsonl")
            if last_record is not None:
                final_eval = last_record.get("evaluation_only", {})
                if bool(episode_summary.get("terminal")) != bool(final_eval.get("terminal")):
                    episode_errors.append("final record terminal flag does not match episode.json")
                if bool(episode_summary.get("truncation")) != bool(final_eval.get("truncation")):
                    episode_errors.append("final record truncation flag does not match episode.json")
            scenario_distribution[str(episode_summary.get("scenario_index"))] += 1

        errors.extend(episode_errors)

    image_count = sum(1 for path in episodes_root.glob("*/frames/*") if path.is_file())
    if manifest is not None:
        if manifest.get("number_of_episodes") != len(episode_dirs):
            errors.append("manifest number_of_episodes does not match episode directories")
        if manifest.get("number_of_records") != record_count:
            errors.append("manifest number_of_records does not match records.jsonl files")

    summary = {
        "dataset_root": str(dataset_root),
        "episodes": len(episode_dirs),
        "records": record_count,
        "valid_records": record_count - invalid_records,
        "invalid_records": invalid_records,
        "images": image_count,
        "decoded_images": len(decoded_images),
        "blank_images": blank_images,
        "missing_images": missing_images,
        "scenario_distribution": dict(sorted(scenario_distribution.items())),
        "speed_range": [min(speed_values), max(speed_values)] if speed_values else None,
        "heading_range": [min(heading_values), max(heading_values)] if heading_values else None,
        "dataset_errors": len(errors),
        "errors": errors,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args(argv)

    summary = validate_dataset(args.dataset_root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            if key != "errors":
                print(f"{key}: {value}")
        for error in summary["errors"]:
            print(f"ERROR: {error}")
    raise SystemExit(1 if summary["invalid_records"] or summary["missing_images"] or summary["dataset_errors"] else 0)


if __name__ == "__main__":
    main()
