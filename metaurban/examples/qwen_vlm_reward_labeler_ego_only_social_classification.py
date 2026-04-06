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
    # from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except Exception as e:
    raise ImportError(
        "Failed to import Qwen2_5_VLForConditionalGeneration. Install a recent transformers build."
    ) from e

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise ImportError(
        "Failed to import qwen_vl_utils.process_vision_info. Please install qwen-vl-utils."
    ) from e


# DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # or "Qwen/Qwen2.5-VL-7B-Instruct" Qwen/Qwen2-VL-2B-Instruct

from transformers import Qwen3VLForConditionalGeneration, AutoModelForImageTextToText

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# transformers>=4.57.0



EGO_DIM = 9
EGO_NAMES = [
    "dist_to_left_side",
    "dist_to_right_side",
    "heading_diff",
    "speed_norm",
    "steering_norm",
    "last_action_0_norm",
    "last_action_1_norm",
    "yaw_rate_norm",
    "lateral_offset_norm",
]

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
2. Ego vehicle state only (9 normalized values):
   - dist_to_left_side
   - dist_to_right_side
   - heading_diff
   - speed_norm
   - steering_norm
   - last_action_0_norm
   - last_action_1_norm
   - yaw_rate_norm
   - lateral_offset_norm
3. Current action = [steer, throttle_or_brake], normalized to [-1, 1].

IMPORTANT:
- You ONLY evaluate SOCIAL behavior.
- DO NOT judge physical safety, collisions, or control stability.
- Focus on pedestrians, human interactions, and social space.
- Use the image as the main source of evidence.
- Use ego state and action only as supporting context for how the agent is moving relative to people.
- In uncertain scenes, stay conservative.

Your task:
Classify the CURRENT step using exactly ONE label from this closed set:

- POSITIVE_SOCIAL:
  The behavior shows clear social awareness, politeness, or consideration.
  Examples:
  - going around pedestrians or groups
  - leaving extra space for people
  - yielding or slowing down appropriately for people
  - clearly respecting human interaction space

- NEUTRAL:
  There is NO clear social signal.
  This includes:
  - no pedestrians in the image
  - pedestrians are far away and not affected by the agent
  - no visible human interaction or meaningful social situation
  - behavior is neither clearly considerate nor clearly intrusive

- NEGATIVE_SOCIAL:
  The behavior is socially inappropriate, intrusive, or uncomfortable.
  Examples:
  - passing between people who are close to each other
  - cutting through a group of pedestrians
  - getting too close to people
  - moving into personal or interaction space
  - not yielding when the social context clearly suggests it should

Critical social rules (VERY IMPORTANT):
- If there are NO pedestrians or no meaningful human social context:
  -> You MUST output NEUTRAL
  -> Do NOT hallucinate social issues or positive social behavior

- If multiple pedestrians are facing each other and appear to be interacting
  (e.g., standing in a group, talking, forming a circle or cluster):
  -> passing through the middle is NEGATIVE_SOCIAL
  -> going around them or clearly leaving space can be POSITIVE_SOCIAL

- Passing between people who are close to each other is NEGATIVE_SOCIAL.

- The agent should respect invisible social spaces and avoid breaking interpersonal interaction zones.

- If unsure whether people are interacting:
  -> assume they ARE interacting (conservative judgment)

- Only output POSITIVE_SOCIAL if there is CLEAR evidence of socially considerate behavior.
  Do NOT give positive labels easily.

- If both positive and negative cues exist:
  -> choose NEGATIVE_SOCIAL

Reward mapping rule (used later in storage, not for you to output directly):
- POSITIVE_SOCIAL -> +1.0
- NEUTRAL -> 0.0
- NEGATIVE_SOCIAL -> -1.0

Ego state:
{ego_state_text}

Current action:
{action_text}

Return JSON with exactly this schema:
{{
  "label": "POSITIVE_SOCIAL or NEUTRAL or NEGATIVE_SOCIAL",
  "confidence": <float in [0,1]>,
  "summary": "<one short sentence>"
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
        description="Label simulator steps with Qwen-VL using image + ego state + action for 3-level social reward."
    )
    parser.add_argument("--img-dir", type=str, default="./recorded_dataset/rgb")
    parser.add_argument("--data-dir", type=str, default="./recorded_dataset/data")
    parser.add_argument("--out-dir", type=str, default="./recorded_dataset/reward")
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
    if full_state.shape[0] < EGO_DIM:
        raise ValueError(
            f"State dim too small in {npy_path}: got {full_state.shape[0]}, need at least {EGO_DIM}"
        )
    ego_state = full_state[:EGO_DIM].copy()
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


def build_messages(image_path: Path, ego_state: np.ndarray, action: np.ndarray, prompt_template: str):
    ego_state_text = format_named_ego_state(ego_state)
    action_text = format_action(action)
    user_prompt = prompt_template.format(
        ego_state_text=ego_state_text,
        action_text=action_text,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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


def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def classify_to_reward(label: str) -> float:
    label = str(label).strip().upper()
    reward_table = {
        "POSITIVE_SOCIAL": 1.0,
        "NEUTRAL": 0.0,
        "NEGATIVE_SOCIAL": -1.0,
    }
    return reward_table.get(label, 0.0)


def sanitize_result(result: Dict) -> Dict:
    label = str(result.get("label", "NEUTRAL")).strip().upper()
    valid_labels = {"POSITIVE_SOCIAL", "NEUTRAL", "NEGATIVE_SOCIAL"}
    if label not in valid_labels:
        label = "NEUTRAL"
    reward = classify_to_reward(label)
    return {
        "label": label,
        "reward": reward,
        "confidence": clamp_float(result.get("confidence", 0.0), 0.0, 1.0),
        "summary": str(result.get("summary", ""))[:300],
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
) -> Tuple[Dict, str, str]:
    messages, prompt_text = build_messages(image_path, ego_state, action, prompt_template)
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
    original["vlm_reward"] = label_payload["reward"]
    original["vlm_label_class"] = label_payload["label"]
    original["vlm_label"] = label_payload
    merged_path = out_dir / "merged_npy" / step.npy_path.name
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(merged_path, original, allow_pickle=True)


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
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
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto" if torch_dtype == "auto" else torch_dtype,
    }

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     args.model_name,
    #     **model_kwargs,
    # )
    # model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    jsonl_path = out_dir / "vlm_rewards.jsonl"
    csv_path = out_dir / "vlm_rewards.csv"
    reward_npy_path = out_dir / "reward_array.npy"
    ego_npy_path = out_dir / "ego_state_array.npy"
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
                "vlm_reward": result["reward"],
                "confidence": result["confidence"],
                "summary": result["summary"],
                "raw_model_output": raw_text,
                "prompt_text": prompt_text,
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
                    f"reward={result['reward']:.4f} "
                    f"conf={result['confidence']:.3f} :: "
                    f"{result['summary']}"
                )

    results_sorted = sorted(results, key=lambda x: int(x["idx"]))
    reward_array = np.asarray([float(r["vlm_reward"]) for r in results_sorted], dtype=np.float32)
    ego_array = np.asarray([r["ego_state"] for r in results_sorted], dtype=np.float32)

    np.save(reward_npy_path, reward_array)
    np.save(ego_npy_path, ego_array)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx",
            "vlm_label_class",
            "vlm_reward",
            "confidence",
            "original_env_reward",
            "terminal",
            "trunc",
            "image_path",
            "npy_path",
            "summary",
        ])
        for row in results_sorted:
            writer.writerow([
                row["idx"],
                row["vlm_label_class"],
                row["vlm_reward"],
                row["confidence"],
                row["original_env_reward"],
                row["terminal"],
                row["trunc"],
                row["image_path"],
                row["npy_path"],
                row["summary"],
            ])

    prompt_dump_path.write_text(prompt_template, encoding="utf-8")

    print(f"Done. Saved JSONL to: {jsonl_path}")
    print(f"Saved CSV to:   {csv_path}")
    print(f"Saved rewards:  {reward_npy_path}")
    print(f"Saved ego arr:  {ego_npy_path}")
    print(f"Saved prompt:   {prompt_dump_path}")


if __name__ == "__main__":
    main()