#!/usr/bin/env python3
"""Compare an untouched base VLM with the same VLM plus a LoRA adapter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from common import LABELS, assert_example_schema, extract_label_lenient, load_config, load_jsonl, parse_prediction, resolve_image, save_json
from inference import load_model_and_processor, predict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen_vl_qlora.yaml")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    return parser.parse_args()


def metrics(rows):
    total = len(rows)
    correct = sum(row["prediction"] == row["target"] for row in rows)
    valid_count = sum(row["structured_output_valid"] for row in rows)
    per_class = {}
    f1_values = []
    for label in LABELS:
        tp = sum(row["target"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["target"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["target"] == label and row["prediction"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)
    columns = list(LABELS) + ["INVALID"]
    confusion = {
        target: {
            predicted: sum(
                row["target"] == target
                and ((row["prediction"] or "INVALID") == predicted)
                for row in rows
            )
            for predicted in columns
        }
        for target in LABELS
    }
    return {
        "examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": {"rows": "target", "columns": columns, "values": confusion},
        "structured_output_validity_rate": valid_count / total if total else 0.0,
    }


def run_variant(name, config, examples, dataset_root, adapter_path, output_dir, max_new_tokens):
    model, processor = load_model_and_processor(config, adapter_path)
    predictions = []
    raw_path = output_dir / f"predictions_{name}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            image = resolve_image(dataset_root, example)
            raw = predict(
                model,
                processor,
                image,
                float(example["ego_speed"]),
                float(example["ego_heading"]),
                max_new_tokens,
            )
            strict_label, valid = parse_prediction(raw)
            row = {
                "episode_id": example["episode_id"],
                "step_index": example["step_index"],
                "image_path": example["image_path"],
                "target": example["label"],
                "prediction": strict_label,
                "lenient_prediction": extract_label_lenient(raw),
                "structured_output_valid": valid,
                "raw_output": raw,
            }
            predictions.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
    result = {"variant": name, **metrics(predictions), "raw_predictions": raw_path.name}
    save_json(output_dir / f"evaluation_{name}.json", result)
    del model, processor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(args.dataset_dir / "test.jsonl")
    assert_example_schema(examples)
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise ValueError("--max_examples must be positive")
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("Held-out test split is empty")
    config = load_config(args.config)
    base = run_variant("base", config, examples, args.dataset_root, None, args.output_dir, args.max_new_tokens)
    lora = run_variant("lora", config, examples, args.dataset_root, args.adapter_path, args.output_dir, args.max_new_tokens)
    print(json.dumps({"base": base, "lora": lora}, indent=2))


if __name__ == "__main__":
    main()
