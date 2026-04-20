import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise ImportError(
        "Failed to import qwen_vl_utils.process_vision_info. Please install qwen-vl-utils."
    ) from e

from transformers import Qwen3VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Only use ego state indices 7 and 8
EGO_IDXS = [7, 8]
EGO_NAMES = [
    "yaw_rate_norm",
    "lateral_offset_norm",
]

LABEL_TO_ID = {
    "NEGATIVE_SOCIAL": 0,
    "NEUTRAL": 1,
    "POSITIVE_SOCIAL": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SYSTEM_PROMPT = (
    "You are a strict classifier for socially appropriate sidewalk driving. "
    "You evaluate exactly one timestep at a time. "
    "You ONLY judge social behavior, not physical safety. "
    "Use the image as primary evidence, and use ego state and action as supporting context. "
    "Return valid JSON only."
)

USER_PROMPT_TEMPLATE = """
You are evaluating ONE timestep from a sidewalk-driving agent.

Inputs:
1. Current RGB observation image.
2. Ego vehicle state only (2 normalized values):
   - yaw_rate_norm
   - lateral_offset_norm
3. Current action = [steer, throttle_or_brake], normalized to [-1, 1].

IMPORTANT:
- You ONLY evaluate SOCIAL behavior.
- DO NOT judge physical safety, collisions, or control stability.
- Focus only on pedestrians, human interactions, and social space.

Core principle:
- The image is the PRIMARY evidence for detecting pedestrians, groups, interpersonal spacing, and social interaction.
- Ego state and action are SECONDARY and only describe how the agent is moving relative to what is visible in the image.

How to use ego state and action (motion interpretation ONLY):
- Use current action to judge whether the agent is slowing down, yielding, or continuing forward.
- Use steer and yaw_rate_norm to judge whether the agent is turning away, detouring around people, or cutting into their space.
- Use lateral_offset_norm to judge whether the agent is shifting its path or leaving extra space while passing.

Strict constraints:
- Do NOT infer pedestrians, groups, or social interaction from ego state or action alone.
- If the image does NOT clearly show pedestrians or meaningful human social context, you MUST output NEUTRAL.
- Ego state and action may ONLY be used to interpret motion relative to people already visible in the image.
- If image evidence and motion evidence conflict, ALWAYS trust the image more.
- In uncertain cases, be conservative and prefer NEUTRAL or NEGATIVE over POSITIVE.

Rules for has_pedestrian:
- Set has_pedestrian to true only if one or more pedestrians are clearly visible in the image.
- Set has_pedestrian to false if no pedestrian is clearly visible.
- Visible pedestrian presence must be determined only from the image.
- If pedestrians are not clearly visible, set has_pedestrian=false and output NEUTRAL.

Your task:
Classify the CURRENT step using exactly ONE label from this closed set:

- POSITIVE_SOCIAL:
  The behavior shows clear social awareness and consideration.
  Requires BOTH:
  1. Visible pedestrians or social context in the image, AND
  2. Motion consistent with yielding, detouring, or leaving space.
  Examples:
  - going around pedestrians or groups
  - clearly leaving extra space
  - slowing down appropriately near people

- NEUTRAL:
  There is NO clear social signal.
  Use this when:
  - no pedestrians are visible
  - pedestrians are far away and unaffected
  - no clear interaction or social situation exists
  - motion changes (turning, slowing) are not clearly related to people

- NEGATIVE_SOCIAL:
  The behavior is socially inappropriate or intrusive.
  Requires BOTH:
  1. Visible pedestrians or interaction space in the image, AND
  2. Motion consistent with intrusion or insufficient yielding.
  Examples:
  - passing between people who are close together
  - cutting through a group
  - moving too close to pedestrians
  - not yielding in a clearly social situation

Critical social rules (VERY IMPORTANT):
- If there are NO pedestrians or meaningful social context:
  -> You MUST output NEUTRAL
  -> Do NOT hallucinate social behavior

- If multiple pedestrians appear to be interacting (facing each other, standing in a group):
  -> passing through them is NEGATIVE_SOCIAL
  -> going around them or leaving space can be POSITIVE_SOCIAL

- Passing between people who are close together is NEGATIVE_SOCIAL.

- The agent must respect invisible interpersonal spaces.

- If unsure whether people are interacting:
  -> assume they ARE interacting (conservative rule)

- Only output POSITIVE_SOCIAL when there is CLEAR evidence.
  Do NOT give positive labels easily.

- If both positive and negative cues exist:
  -> choose NEGATIVE_SOCIAL

Label id mapping (used later in storage, not for you to output directly):
- NEGATIVE_SOCIAL -> 0
- NEUTRAL -> 1
- POSITIVE_SOCIAL -> 2

Ego state:
{ego_state_text}

Current action:
{action_text}

Return JSON with exactly this schema:
{{
  "label": "POSITIVE_SOCIAL or NEUTRAL or NEGATIVE_SOCIAL",
  "has_pedestrian": true,
  "reason": "<one short sentence explaining why this label was chosen>"
}}
""".strip()


@dataclass
class StepRecord:
    idx: int
    image_path: Path
    npy_path: Path
    full_state: np.ndarray
    ego_state: np.ndarray
    action: np.ndarray
    original_reward: Optional[float]
    terminal: Optional[bool]
    trunc: Optional[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label simulator steps with Qwen-VL using image + selected ego state + action for 3-class social label dataset."
    )
    parser.add_argument("--img-dir", type=str, default="./recorded_dataset/rgb_merged")
    parser.add_argument("--data-dir", type=str, default="./recorded_dataset/data_merged")
    parser.add_argument("--out-dir", type=str, default="./recorded_dataset/new_labels")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-merged-npy", action="store_true", default=True)
    parser.add_argument("--no-save-merged-npy", action="store_false", dest="save_merged_npy")
    parser.add_argument("--prompt-path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_false", dest="verbose")

    parser.add_argument("--few-shot-neg-image", type=str, default="./recorded_dataset/few_shot/neg_social_example.png")
    parser.add_argument("--few-shot-pos-image", type=str, default="./recorded_dataset/few_shot/pos_social_example.png")
    parser.add_argument(
        "--few-shot-neg-text",
        type=str,
        default="Pedestrians are clearly visible, and the agent moves into their shared space or passes between them.",
    )
    parser.add_argument(
        "--few-shot-pos-text",
        type=str,
        default="Pedestrians are clearly visible, and the agent goes around them while leaving social space.",
    )

    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def find_step_pairs(img_dir: Path, data_dir: Path) -> List[Tuple[int, Path, Path]]:
    pairs: List[Tuple[int, Path, Path]] = []
    for img_path in sorted(img_dir.glob("step_*.png")):
        m = re.fullmatch(r"step_(\d+)\.png", img_path.name)
        if m is None:
            continue
        idx = int(m.group(1))
        npy_path = data_dir / f"step_{idx:06d}.npy"
        if npy_path.exists():
            pairs.append((idx, img_path, npy_path))
    return pairs


def format_named_ego_state(ego_state: np.ndarray, precision: int = 4) -> str:
    lines = []
    for i, name in enumerate(EGO_NAMES):
        val = float(ego_state[i]) if i < len(ego_state) else float("nan")
        lines.append(f"- {name}: {val:.{precision}f}")
    return "\n".join(lines)


def format_action(action: np.ndarray, precision: int = 4) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] >= 2:
        return (
            f"- steer: {float(action[0]):.{precision}f}\n"
            f"- throttle_or_brake: {float(action[1]):.{precision}f}"
        )
    return np.array2string(action, precision=precision, separator=", ")


def load_step_record(idx: int, image_path: Path, npy_path: Path) -> StepRecord:
    obj = np.load(npy_path, allow_pickle=True).item()
    full_state = np.asarray(obj["state"], dtype=np.float32).reshape(-1)

    needed_dim = max(EGO_IDXS) + 1
    if full_state.shape[0] < needed_dim:
        raise ValueError(
            f"State dim too small in {npy_path}: got {full_state.shape[0]}, need at least {needed_dim}"
        )

    ego_state = full_state[EGO_IDXS].copy()
    action = np.asarray(obj["action"], dtype=np.float32).reshape(-1)

    return StepRecord(
        idx=idx,
        image_path=image_path,
        npy_path=npy_path,
        full_state=full_state,
        ego_state=ego_state,
        action=action,
        original_reward=float(obj["reward"]) if "reward" in obj else None,
        terminal=bool(obj["terminal"]) if "terminal" in obj else None,
        trunc=bool(obj["trunc"]) if "trunc" in obj else None,
    )


def build_messages(
    image_path: Path,
    ego_state: np.ndarray,
    action: np.ndarray,
    prompt_template: str,
    few_shot_neg_image: Path,
    few_shot_pos_image: Path,
    few_shot_neg_text: str,
    few_shot_pos_text: str,
):
    ego_state_text = format_named_ego_state(ego_state)
    action_text = format_action(action)
    user_prompt = prompt_template.format(
        ego_state_text=ego_state_text,
        action_text=action_text,
    )

    few_shots = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(few_shot_neg_image)},
                {"type": "text", "text": """
        Ego state:
        - yaw_rate_norm: 0.054
        - lateral_offset_norm: 0.25

        Action:
        - steer: 0.04
        - throttle: 0.96

        Description:
        Pedestrians are visible and close together. The agent is moving forward with slight steering while at high speed.
        """}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "label": "NEGATIVE_SOCIAL",
                            "has_pedestrian": True,
                            "reason": "the agent intrudes into pedestrian interaction space",
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(few_shot_pos_image)},
                {"type": "text", "text": """
        Ego state:
        - yaw_rate_norm: 0.45
        - lateral_offset_norm: 0.35

        Action:
        - steer: 0.55
        - throttle: 0.40

        Description:
        Pedestrians or groups are visible. The agent is turning and moving away, leaving space.
        """}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "label": "POSITIVE_SOCIAL",
                            "has_pedestrian": True,
                            "reason": "the agent detours and leaves social space",
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        },
    ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *few_shots,
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages, user_prompt


def extract_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model output:\n{text}")
    return json.loads(match.group(0))


def classify_to_label_id(label: str) -> int:
    label = str(label).strip().upper()
    return LABEL_TO_ID.get(label, LABEL_TO_ID["NEUTRAL"])


def sanitize_result(result: Dict) -> Dict:
    label = str(result.get("label", "NEUTRAL")).strip().upper()
    valid_labels = {"POSITIVE_SOCIAL", "NEUTRAL", "NEGATIVE_SOCIAL"}
    if label not in valid_labels:
        label = "NEUTRAL"

    label_id = classify_to_label_id(label)

    has_pedestrian = result.get("has_pedestrian", False)
    if isinstance(has_pedestrian, str):
        has_pedestrian = has_pedestrian.strip().lower() in {"true", "1", "yes"}
    else:
        has_pedestrian = bool(has_pedestrian)

    reason = str(result.get("reason", "")).strip()

    if not has_pedestrian:
        label = "NEUTRAL"
        label_id = LABEL_TO_ID["NEUTRAL"]
        if reason == "":
            reason = "no pedestrians are clearly visible"

    return {
        "label": label,
        "label_id": label_id,
        "has_pedestrian": has_pedestrian,
        "reason": reason,
    }


def run_one(
    model,
    processor,
    image_path: Path,
    ego_state: np.ndarray,
    action: np.ndarray,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_template: str,
    few_shot_neg_image: Path,
    few_shot_pos_image: Path,
    few_shot_neg_text: str,
    few_shot_pos_text: str,
) -> Tuple[Dict, str, str]:
    messages, prompt_text = build_messages(
        image_path=image_path,
        ego_state=ego_state,
        action=action,
        prompt_template=prompt_template,
        few_shot_neg_image=few_shot_neg_image,
        few_shot_pos_image=few_shot_pos_image,
        few_shot_neg_text=few_shot_neg_text,
        few_shot_pos_text=few_shot_pos_text,
    )

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    raw = extract_json(output_text)
    result = sanitize_result(raw)
    return result, output_text, prompt_text


def maybe_save_merged_npy(step: StepRecord, out_dir: Path, label_payload: Dict):
    original = np.load(step.npy_path, allow_pickle=True).item()

    new_data = {
        "state": original["state"],
        "action": original["action"],
        "label": label_payload["label_id"],
    }

    merged_path = out_dir / "merged_npy" / step.npy_path.name
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(merged_path, new_data, allow_pickle=True)


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    few_shot_neg_image = Path(args.few_shot_neg_image)
    few_shot_pos_image = Path(args.few_shot_pos_image)

    if not few_shot_neg_image.exists():
        raise FileNotFoundError(f"Few-shot negative image not found: {few_shot_neg_image}")
    if not few_shot_pos_image.exists():
        raise FileNotFoundError(f"Few-shot positive image not found: {few_shot_pos_image}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    prompt_template = USER_PROMPT_TEMPLATE
    if args.prompt_path is not None:
        prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")

    pairs = find_step_pairs(img_dir, data_dir)
    pairs = [
        p for p in pairs
        if p[0] >= args.start_idx and (args.end_idx is None or p[0] <= args.end_idx)
    ]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    if not pairs:
        raise RuntimeError("No matching step_XXXXXX.png / step_XXXXXX.npy pairs found.")

    torch_dtype = resolve_dtype(args.dtype)
    _ = {
        "device_map": "auto",
        "torch_dtype": "auto" if torch_dtype == "auto" else torch_dtype,
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    jsonl_path = out_dir / "vlm_labels.jsonl"
    csv_path = out_dir / "vlm_labels.csv"
    label_npy_path = out_dir / "label_array.npy"
    prompt_dump_path = out_dir / "prompt_used.txt"

    results: List[Dict] = []
    done = set()

    if jsonl_path.exists() and not args.overwrite:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    results.append(row)
                    done.add(int(row["idx"]))
                except Exception:
                    continue

    pending_pairs = [p for p in pairs if p[0] not in done]

    if args.verbose:
        print(f"Found {len(pairs)} pairs, {len(pending_pairs)} pending, {len(done)} already done.")

    mode = "w" if args.overwrite else "a"
    with open(jsonl_path, mode, encoding="utf-8") as jf:
        if args.overwrite:
            results = []

        for idx, image_path, npy_path in pending_pairs:
            step = load_step_record(idx, image_path, npy_path)

            result, raw_text, prompt_text = run_one(
                model=model,
                processor=processor,
                image_path=step.image_path,
                ego_state=step.ego_state,
                action=step.action,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompt_template=prompt_template,
                few_shot_neg_image=few_shot_neg_image,
                few_shot_pos_image=few_shot_pos_image,
                few_shot_neg_text=args.few_shot_neg_text,
                few_shot_pos_text=args.few_shot_pos_text,
            )

            row = {
                "idx": step.idx,
                "image_path": str(step.image_path),
                "npy_path": str(step.npy_path),
                "original_env_reward": step.original_reward,
                "terminal": step.terminal,
                "trunc": step.trunc,
                "full_state_dim": int(step.full_state.shape[0]),
                "ego_state_dim": int(step.ego_state.shape[0]),
                "ego_state": step.ego_state.tolist(),
                "ego_state_named": {
                    name: float(step.ego_state[i]) for i, name in enumerate(EGO_NAMES)
                },
                "action": step.action.tolist(),
                "vlm_label_class": result["label"],
                "vlm_label_id": result["label_id"],
                "has_pedestrian": result["has_pedestrian"],
                "reason": result["reason"],
                "raw_model_output": raw_text,
            }

            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            results.append(row)

            single_json = out_dir / "json" / f"step_{idx:06d}.json"
            with open(single_json, "w", encoding="utf-8") as sf:
                json.dump(row, sf, ensure_ascii=False, indent=2)

            if args.save_merged_npy:
                maybe_save_merged_npy(step, out_dir, result)

            if args.verbose:
                print(
                    f"[{idx:06d}] "
                    f"label={result['label']} "
                    f"label_id={result['label_id']} "
                    f"has_pedestrian={result['has_pedestrian']} "
                    f"reason={result['reason']}"
                )

    results_sorted = sorted(results, key=lambda x: int(x["idx"]))
    label_array = np.asarray([int(r["vlm_label_id"]) for r in results_sorted], dtype=np.int64)
    np.save(label_npy_path, label_array)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx",
            "vlm_label_class",
            "vlm_label_id",
            "has_pedestrian",
            "reason",
            "original_env_reward",
            "terminal",
            "trunc",
            "image_path",
            "npy_path",
        ])
        for row in results_sorted:
            writer.writerow([
                row["idx"],
                row["vlm_label_class"],
                row["vlm_label_id"],
                row["has_pedestrian"],
                row["reason"],
                row["original_env_reward"],
                row["terminal"],
                row["trunc"],
                row["image_path"],
                row["npy_path"],
            ])

    prompt_dump_path.write_text(prompt_template, encoding="utf-8")

    print(f"Done. Saved JSONL to: {jsonl_path}")
    print(f"Saved CSV to:   {csv_path}")
    print(f"Saved labels:  {label_npy_path}")
    print(f"Saved prompt:  {prompt_dump_path}")


if __name__ == "__main__":
    main()