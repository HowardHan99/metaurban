import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from google import genai

IMAGE_DIR = Path("./recorded_dataset/rgb_merged")
DATA_DIR = Path("./recorded_dataset/data_merged")

MODEL_NAME = "gemini-2.5-flash"

FIXED_STEPS = [7, 8, 9, 11, 12]  #[172, 173, 174, 175, 176]

EGO_IDXS = [7, 8]
EGO_NAMES = ["yaw_rate_norm", "lateral_offset_norm"]


def format_ego_state(ego_state):
    return {
        EGO_NAMES[i]: float(ego_state[i])
        for i in range(len(EGO_NAMES))
    }


def load_step(step):
    img_path = IMAGE_DIR / f"step_{step:06d}.png"
    npy_path = DATA_DIR / f"step_{step:06d}.npy"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing npy: {npy_path}")

    obj = np.load(npy_path, allow_pickle=True).item()

    full_state = np.asarray(obj["state"], dtype=np.float32).reshape(-1)
    ego_state = full_state[EGO_IDXS]
    action = np.asarray(obj["action"], dtype=np.float32).reshape(-1)

    frame_info = {
        "step": step,
        "image_path": str(img_path),
        "npy_path": str(npy_path),
        "ego_state": format_ego_state(ego_state),
        "action": {
            "steer": float(action[0]) if len(action) > 0 else None,
            "throttle_or_brake": float(action[1]) if len(action) > 1 else None,
        },
        "original_reward": float(obj["reward"]) if "reward" in obj else None,
        "terminal": bool(obj["terminal"]) if "terminal" in obj else None,
        "trunc": bool(obj["trunc"]) if "trunc" in obj else None,
    }

    image = Image.open(img_path)

    return image, frame_info


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY first.")

    client = genai.Client(api_key=api_key)

    contents = []
    frame_infos = []

    for step in FIXED_STEPS:
        image, frame_info = load_step(step)

        contents.append(f"Frame step_{step:06d}:")
        contents.append(image)

        frame_infos.append(frame_info)

    prompt = f"""
You are evaluating a short sequence from a sidewalk-driving robot.

The input contains 5 consecutive RGB frames and their corresponding metadata.
The metadata includes:
- ego_state: yaw_rate_norm and lateral_offset_norm
- action: steer and throttle_or_brake

Frame metadata:
{json.dumps(frame_infos, indent=2)}

Your task:
Analyze whether the robot understands the social scene across this sequence.

Important rules:
- Use the images as primary evidence.
- Use ego state and action only as supporting motion context.
- Do not hallucinate pedestrians.
- If no pedestrians are clearly visible, the social label should be NEUTRAL.
- Focus only on social behavior around pedestrians, not physical collision safety.

Return JSON only with this schema:

{{
  "scene_summary": "<brief description of the sequence>",
  "has_pedestrian": true,
  "pedestrian_motion_or_position": "<where pedestrians are and how they change across frames>",
  "robot_motion_inference": "<forward / turning / slowing / unclear>",
  "social_interaction": "<whether robot is interacting with pedestrians socially>",
  "social_risk": "LOW or MEDIUM or HIGH",
  "label": "POSITIVE_SOCIAL or NEUTRAL or NEGATIVE_SOCIAL",
  "reason": "<one long explanation>"
}}
""".strip()

    contents.append(prompt)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )

    print(response.text)


if __name__ == "__main__":
    main()
