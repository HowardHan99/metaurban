from metaurban.utils.math import wrap_to_pi

import copy
import json
import re
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# try:
#     from transformers import Qwen3VLForConditionalGeneration
# except Exception:
#     Qwen3VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

from metaurban.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metaurban.component.navigation_module.orca_navigation import ORCATrajectoryNavigation
from metaurban.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metaurban.component.map.base_map import BaseMap
from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.constants import DEFAULT_AGENT, TerminationState
from metaurban.envs.base_env import BaseEnv
from metaurban.manager.traffic_manager import TrafficMode
from metaurban.utils import clip, Config

METAURBAN_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map=3,  # int or string
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },
    store_map=True,
    crswalk_density=0.1,
    spawn_human_num=1,
    spawn_elderly_num=0,
    show_mid_block_map=False,

    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,
    random_traffic=False,
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.0,
    static_traffic_object=True,

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,

    # ===== Agent =====
    random_spawn_lane_index=True,
    vehicle_config=dict(
        navigation_module=NodeNetworkNavigation,
        ego_navigation_module=ORCATrajectoryNavigation,
    ),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    success_reward=5.0,
    out_of_road_penalty=5.0,
    on_lane_line_penalty=1.0,
    crash_vehicle_penalty=1.0,
    crash_object_penalty=1.0,
    crash_human_penalty=1.0,
    driving_reward=1.0,
    steering_range_penalty=0.5,
    heading_penalty=1.0,
    lateral_penalty=0.5,
    max_lateral_dist=2.0,
    no_negative_reward=True,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,
    crash_human_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    crash_human_done=False,
    relax_out_of_road_done=True,

    # ===== Online VLM Reward =====
    use_vlm_reward=True,
    vlm_model_name="Qwen/Qwen2-VL-2B-Instruct",
    vlm_adapter_path=None,
    vlm_device="cuda",
    vlm_dtype="bfloat16",
    vlm_query_interval=3,
    vlm_reward_weight=0.5,
    vlm_default_reward=0.0,
    vlm_max_new_tokens=128,
    vlm_temperature=0.0,
    vlm_log_response=False,
)


class SidewalkDynamicMetaUrbanEnv(BaseEnv):
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

    @classmethod
    def default_config(cls) -> Config:
        config = super(SidewalkDynamicMetaUrbanEnv, cls).default_config()
        config.update(METAURBAN_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(SidewalkDynamicMetaUrbanEnv, self).__init__(config)

        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios
        self.previous_agent_actions = {}

        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_ready = False
        self._vlm_last_reward = {}
        self._vlm_last_label = {}
        self._vlm_last_response = {}

    def _post_process_config(self, config):
        config = super(SidewalkDynamicMetaUrbanEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle) and not self._is_out_of_road(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
        }

        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE]
            or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
            or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.MAX_STEP]:
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True},
            )

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info["cost"], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        route_completion = vehicle.navigation.route_completion
        if route_completion > 0.95 or vehicle.navigation.reference_trajectory.length < 2:
            return True
        else:
            return False

    def _is_out_of_road(self, vehicle):
        if self.config["relax_out_of_road_done"]:
            lat = abs(vehicle.navigation.current_lateral)
            done = lat > self.config["max_lateral_dist"]
            return done
        return False

    def record_previous_agent_state(self, vehicle_id: str):
        self.previous_agent_actions[vehicle_id] = self.agents[vehicle_id].current_action

    def _lazy_init_vlm(self):
        if not self.config.get("use_vlm_reward", False):
            return False
        if self._vlm_ready:
            return True
        try:
            model_name = self.config["vlm_model_name"]
            adapter_path = self.config.get("vlm_adapter_path")
            peft_model_cls = None
            if adapter_path:
                try:
                    from peft import PeftConfig, PeftModel
                except ImportError as exc:
                    raise RuntimeError(
                        "vlm_adapter_path was supplied, but PEFT is not installed"
                    ) from exc
                adapter_config = PeftConfig.from_pretrained(adapter_path)
                if adapter_config.base_model_name_or_path:
                    model_name = adapter_config.base_model_name_or_path
                peft_model_cls = PeftModel

            dtype_name = str(self.config.get("vlm_dtype", "bfloat16")).lower()
            if dtype_name == "bfloat16":
                torch_dtype = torch.bfloat16
            elif dtype_name == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            if self.config["vlm_device"] == "cuda":
                self._vlm_model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                )
            else:
                self._vlm_model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                ).to(self.config["vlm_device"])

            if adapter_path:
                self._vlm_model = peft_model_cls.from_pretrained(
                    self._vlm_model,
                    adapter_path,
                )

            self._vlm_processor = AutoProcessor.from_pretrained(
                model_name,
                use_fast=True,
            )

            self._vlm_model.eval()
            self._vlm_ready = True
            if adapter_path:
                self.logger.info(f"Loaded VLM: {model_name} with PEFT adapter: {adapter_path}")
            else:
                self.logger.info(f"Loaded VLM: {model_name}")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to init VLM: {e}")
            return False

    def _capture_env_rgb(self):
        rgb = None

        for fn in [
            lambda: self.render(mode="rgb_array"),
            lambda: self.render("rgb_array"),
            lambda: self.render(),
        ]:
            try:
                out = fn()
                if isinstance(out, np.ndarray):
                    rgb = out
                    break
                if isinstance(out, (list, tuple)):
                    for item in out:
                        if isinstance(item, np.ndarray):
                            rgb = item
                            break
                    if rgb is not None:
                        break
            except Exception:
                pass

        if rgb is None:
            return None

        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_ego_state_9d(self, vehicle, vehicle_id):
        current_action = (
            vehicle.current_action
            if vehicle.current_action is not None
            else np.array([0.0, 0.0], dtype=np.float32)
        )
        current_action = np.asarray(current_action, dtype=np.float32).reshape(-1)

        prev_action = self.previous_agent_actions.get(vehicle_id, np.array([0.0, 0.0], dtype=np.float32))
        prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)

        heading_diff = float(
            wrap_to_pi(vehicle.heading_theta - vehicle.navigation.current_heading_theta_at_long) / np.pi
        )
        lateral_offset_norm = float(
            np.clip(vehicle.navigation.current_lateral / max(self.config["max_lateral_dist"], 1e-6), -1.0, 1.0)
        )

        dist_to_left_side = float(np.clip(1.0 - max(lateral_offset_norm, 0.0), -1.0, 1.0))
        dist_to_right_side = float(np.clip(1.0 + min(lateral_offset_norm, 0.0), -1.0, 1.0))
        speed_norm = float(np.clip(vehicle.speed / max(vehicle.max_speed_km_h, 1e-6), 0.0, 1.0))
        steering_norm = float(current_action[0]) if current_action.shape[0] > 0 else 0.0
        last_action_0_norm = float(prev_action[0]) if prev_action.shape[0] > 0 else 0.0
        last_action_1_norm = float(prev_action[1]) if prev_action.shape[0] > 1 else 0.0
        yaw_rate_norm = float(np.clip(getattr(vehicle, "yaw_rate", 0.0), -1.0, 1.0))

        return np.asarray(
            [
                dist_to_left_side,
                dist_to_right_side,
                heading_diff,
                speed_norm,
                steering_norm,
                last_action_0_norm,
                last_action_1_norm,
                yaw_rate_norm,
                lateral_offset_norm,
            ],
            dtype=np.float32,
        )

    def _format_named_ego_state(self, ego_state: np.ndarray, precision: int = 4) -> str:
        lines = []
        for i, name in enumerate(self.EGO_NAMES):
            val = float(ego_state[i]) if i < len(ego_state) else float("nan")
            lines.append(f"- {name}: {val:.{precision}f}")
        return "\n".join(lines)

    def _format_action(self, action: np.ndarray, precision: int = 4) -> str:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] >= 2:
            return (
                f"- steer: {float(action[0]):.{precision}f}\n"
                f"- throttle_or_brake: {float(action[1]):.{precision}f}"
            )
        return np.array2string(action, precision=precision, separator=", ")

    def _build_vlm_messages(self, vehicle, vehicle_id, pil_image):
        if self.config.get("vlm_adapter_path"):
            # The post-trained adapter was supervised on this deliberately narrow,
            # post-transition schema. Keep simulator/action/evaluation metadata out.
            system_prompt = (
                "You are a strict classifier of socially appropriate robot behavior in an urban "
                "shared space. Judge only the post-transition observation. Return valid JSON only."
            )
            user_prompt = (
                "Classify the social behavior visible in this single post-transition observation.\n\n"
                "Use only:\n"
                "- the RGB image as primary evidence\n"
                f"- ego_speed: {float(vehicle.speed):.4f} m/s\n"
                f"- ego_heading: {float(vehicle.heading_theta):.4f} rad\n\n"
                "Choose exactly one label: NEGATIVE_SOCIAL, NEUTRAL, or POSITIVE_SOCIAL.\n"
                "If there is no clear pedestrian interaction or social signal, choose NEUTRAL.\n"
                'Return exactly one JSON object with no explanation: {"label":"<LABEL>"}'
            )
            return [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ], user_prompt

        ego_state = self._get_ego_state_9d(vehicle, vehicle_id)
        ego_state_text = self._format_named_ego_state(ego_state)

        current_action = (
            vehicle.current_action
            if vehicle.current_action is not None
            else np.array([0.0, 0.0], dtype=np.float32)
        )
        action_text = self._format_action(current_action)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            ego_state_text=ego_state_text,
            action_text=action_text,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return messages, user_prompt

    def _classify_to_reward(self, label: str) -> float:
        label = str(label).strip().upper()
        return {
            "POSITIVE_SOCIAL": 1.0,
            "NEUTRAL": 0.0,
            "NEGATIVE_SOCIAL": -1.0,
        }.get(label, 0.0)

    def _parse_vlm_response(self, text):
        label = "NEUTRAL"
        confidence = 0.0
        summary = "parse_failed"

        if text is None:
            return {"label": label, "reward": 0.0, "confidence": confidence, "summary": summary}

        text = text.strip()
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                return {"label": label, "reward": 0.0, "confidence": confidence, "summary": summary}
            obj = json.loads(match.group(0))

        label = str(obj.get("label", "NEUTRAL")).strip().upper()
        if label not in {"POSITIVE_SOCIAL", "NEUTRAL", "NEGATIVE_SOCIAL"}:
            label = "NEUTRAL"

        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        summary = str(obj.get("summary", ""))[:300]

        return {
            "label": label,
            "reward": self._classify_to_reward(label),
            "confidence": confidence,
            "summary": summary,
        }

    @torch.no_grad()
    def _query_vlm_social_reward(self, vehicle_id: str):
        if not self._lazy_init_vlm():
            return 0.0, {
                "vlm_label": "NEUTRAL",
                "vlm_confidence": 0.0,
                "vlm_summary": "vlm_not_ready",
                "vlm_response": "",
            }

        rgb = self._capture_env_rgb()
        if rgb is None:
            return 0.0, {
                "vlm_label": "NEUTRAL",
                "vlm_confidence": 0.0,
                "vlm_summary": "capture_failed",
                "vlm_response": "",
            }

        pil_image = Image.fromarray(rgb)
        vehicle = self.agents[vehicle_id]
        messages, prompt_text = self._build_vlm_messages(vehicle, vehicle_id, pil_image)

        try:
            chat_text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._vlm_processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self._vlm_processor(
                    text=[chat_text],
                    images=[pil_image],
                    return_tensors="pt",
                )

            inputs = {
                k: v.to(self.config["vlm_device"]) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            generated_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=int(self.config.get("vlm_max_new_tokens", 128)),
                do_sample=False,
                temperature=float(self.config.get("vlm_temperature", 0.0)),
            )

            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self._vlm_processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            parsed = self._parse_vlm_response(response)
            return parsed["reward"], {
                "vlm_label": parsed["label"],
                "vlm_confidence": parsed["confidence"],
                "vlm_summary": parsed["summary"],
                "vlm_response": response,
                "vlm_prompt_text": prompt_text,
            }

        except Exception as e:
            return 0.0, {
                "vlm_label": "NEUTRAL",
                "vlm_confidence": 0.0,
                "vlm_summary": str(e),
                "vlm_response": "",
            }

    def _get_online_vlm_reward(self, vehicle_id: str):
        interval = max(1, int(self.config.get("vlm_query_interval", 5)))
        step_idx = self.episode_lengths[vehicle_id]

        if vehicle_id not in self._vlm_last_reward:
            self._vlm_last_reward[vehicle_id] = 0.0
            self._vlm_last_label[vehicle_id] = "NEUTRAL"
            self._vlm_last_response[vehicle_id] = ""

        if step_idx % interval != 0:
            return self._vlm_last_reward[vehicle_id], {
                "vlm_label": self._vlm_last_label[vehicle_id],
                "vlm_confidence": 0.0,
                "vlm_summary": "cached",
                "vlm_response": self._vlm_last_response[vehicle_id],
            }

        reward, info = self._query_vlm_social_reward(vehicle_id)
        self._vlm_last_reward[vehicle_id] = float(reward)
        self._vlm_last_label[vehicle_id] = info.get("vlm_label", "NEUTRAL")
        self._vlm_last_response[vehicle_id] = info.get("vlm_response", "")
        return float(reward), info

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        long_last = vehicle.navigation.last_longitude
        long_now = vehicle.navigation.current_longitude
        lateral_now = vehicle.navigation.current_lateral

        base_reward = 0.0

        progress_reward = self.config["driving_reward"] * (long_now - long_last)
        base_reward += progress_reward

        lateral_factor = abs(lateral_now) / self.config["max_lateral_dist"]
        lateral_penalty = -lateral_factor * self.config["lateral_penalty"]
        base_reward += lateral_penalty

        ref_line_heading = vehicle.navigation.current_heading_theta_at_long
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi
        heading_penalty = -heading_diff * self.config["heading_penalty"]
        base_reward += heading_penalty

        steering = abs(vehicle.current_action[0]) if vehicle.current_action is not None else 0.0
        allowed_steering = 1 / max(vehicle.speed, 1e-2)
        overflowed_steering = min((allowed_steering - steering), 0)
        steering_range_penalty = overflowed_steering * self.config["steering_range_penalty"]
        base_reward += steering_range_penalty

        if (
            vehicle_id not in self.previous_agent_actions
            or "steering_penalty" not in self.config
            or self.config["steering_penalty"] == 0
        ):
            steering_reward = 0.0
        else:
            steering_now = vehicle.current_action[0]
            prev_steering = self.previous_agent_actions[vehicle_id][0]
            steering_diff = abs(steering_now - prev_steering)
            steering_reward = -steering_diff * self.config["steering_penalty"]
        base_reward += steering_reward

        if self.config["no_negative_reward"]:
            base_reward = max(base_reward, 0.0)

        if vehicle.crash_vehicle:
            base_reward = -self.config["crash_vehicle_penalty"]
        if vehicle.crash_object:
            base_reward = -self.config["crash_object_penalty"]
        if vehicle.crash_human:
            base_reward = -self.config["crash_human_penalty"]

        vlm_reward = 0.0
        vlm_info = {
            "vlm_label": "NEUTRAL",
            "vlm_confidence": 0.0,
            "vlm_summary": "",
            "vlm_response": "",
        }

        if self.config.get("use_vlm_reward", False):
            vlm_reward, vlm_info = self._get_online_vlm_reward(vehicle_id)

        reward = base_reward + float(self.config.get("vlm_reward_weight", 0.5)) * float(vlm_reward)

        if self._is_arrive_destination(vehicle) and not self._is_out_of_road(vehicle):
            reward = self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]

        step_info["step_reward"] = float(reward)
        step_info["base_reward"] = float(base_reward)
        step_info["vlm_label_class"] = vlm_info.get("vlm_label", "NEUTRAL")
        step_info["vlm_confidence"] = vlm_info.get("vlm_confidence", 0.0)
        step_info["vlm_summary"] = vlm_info.get("vlm_summary", "")
        step_info["vlm_reward_raw"] = float(vlm_reward)
        step_info["vlm_reward_weighted"] = float(self.config.get("vlm_reward_weight", 0.5)) * float(vlm_reward)

        if self.config.get("vlm_log_response", False):
            step_info["vlm_response"] = vlm_info.get("vlm_response", "")

        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["carsize"] = [vehicle.WIDTH, vehicle.LENGTH]
        step_info["route_completion"] = vehicle.navigation.route_completion
        step_info["curriculum_level"] = self.engine.current_level
        step_info["scenario_index"] = self.engine.current_seed
        step_info["lateral_dist"] = lateral_now

        step_info["step_reward_progress"] = progress_reward
        step_info["step_reward_lateral"] = lateral_penalty
        step_info["step_reward_heading"] = heading_penalty
        step_info["step_reward_action_smooth"] = steering_range_penalty
        step_info["steering_reward"] = steering_reward

        self.record_previous_agent_state(vehicle_id)
        return float(reward), step_info

    def setup_engine(self):
        super(SidewalkDynamicMetaUrbanEnv, self).setup_engine()
        from metaurban.manager.traffic_manager import NewAssetPGTrafficManager
        from metaurban.manager.humanoid_manager import PGBackgroundSidewalkAssetsManager as PGHumanoidManager
        from metaurban.manager.pg_map_manager import PGMapManager
        from metaurban.manager.object_manager import TrafficObjectManager
        from metaurban.manager.sidewalk_manager import AssetManager

        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("asset_manager", AssetManager())
        self.engine.register_manager("traffic_manager", NewAssetPGTrafficManager())
        self.engine.register_manager("humanoid_manager", PGHumanoidManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

    def _get_agent_manager(self):
        if "agent_type" not in self.config:
            self.config["agent_type"] = "coco"
        if self.config["agent_type"] == "coco":
            from metaurban.manager.agent_manager import DeliveryRobotAgentManager
            return DeliveryRobotAgentManager(init_observations=self._get_observations())
        elif self.config["agent_type"] == "wheelchair":
            from metaurban.manager.agent_manager import WheelchairAgentManager
            return WheelchairAgentManager(init_observations=self._get_observations())


if __name__ == "__main__":

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = SidewalkDynamicMetaUrbanEnv()
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
