"""
Run a pretrained PPO policy in SidewalkDynamicMetaUrbanEnv.

Example:
python drive_with_pretrained_policy.py --policy ./pretrained_policy_576k --observation lidar

If you want to save images:
python drive_with_pretrained_policy.py --policy ./pretrained_policy_576k --observation all --save_img --out_dir saved_imgs
"""

from metaurban.constants import HELP_MESSAGE
import cv2
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file

from metaurban import SidewalkDynamicMetaUrbanEnv


def make_metadrive_env_fn(env_cfg):
    env = SidewalkDynamicMetaUrbanEnv(
        dict(
            log_level=50,
            **env_cfg,
        )
    )
    env = Monitor(env)
    return env


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 256, 128), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, action_log_std, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action, normal_log_density(action, action_mean, action_log_std, action_std)

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        idx = 0
        std_id = None
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = idx
                std_index = param_count
            param_count += param.view(-1).shape[0]
            idx += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.value_head(x)
        prob = torch.sigmoid(value)
        return prob


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=False,
        horizon=300,
        num_scenarios=100,
        random_spawn_lane_index=False,
        relax_out_of_road_done=True,
        max_lateral_dist=15.0,
        debug=False,
        window_size=(1200, 900),

        # agent
        agent_type="wheelchair",
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
            policy_reverse=False,
        ),

        # optional explicit flags
        out_of_route_done=False,
        crash_human_done=False,

        # dynamic sidewalk env related
        scene_type="commercial",
        scene_building_source="scene",
        spawn_robot_on_sidewalk=False,

        crossing_ped_num=8,
        signaling_ped_num=0,
        vulnerable_ped_num=4,
        spawn_elderly_num=2,
        group_ped_pair_num=3,
        spawn_wheelchairman_num=1,
        spawn_edog_num=0,
        spawn_erobot_num=0,
        spawn_drobot_num=0,
        max_actor_num=20,

        ped_ego_yield_radius=3.0,
        crossing_assertive_radius=4.0,
        vulnerable_yield_radius=5.0,
        vulnerable_speed_scale=0.70,

        group_spawn_near_ego=False,
        group_spawn_min_radius=5.0,
        group_spawn_max_radius=10.0,
        group_cluster_num=3,
        group_cluster_size_min=5,
        group_cluster_size_max=8,

        ignore_success_done=False,
        object_density=1.0,
        show_ego_navigation=False,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        default="./pretrained_policy_576k",
        help="Path to PPO policy .zip (with or without .zip).",
    )
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "all"])
    parser.add_argument("--out_dir", type=str, default="saved_imgs")
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    if args.observation == "all" or args.save_img:
        config.update(
            dict(
                sensors=dict(
                    rgb_camera=(RGBCamera, 1024, 576),
                    depth_camera=(DepthCamera, 1024, 576),
                    semantic_camera=(SemanticCamera, 1024, 576),
                ),
                norm_pixel=False,
            )
        )

    if args.save_img:
        os.makedirs(args.out_dir, exist_ok=True)

    env = SidewalkDynamicMetaUrbanEnv(config)
    obs, _ = env.reset(seed=args.seed)

    algo_config = dict(
        learning_rate=5e-5,
        n_steps=200,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=10.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log="./metaurban_ppo-single_scenario_per_process_1e8-tb_logs/",
    )

    expert = PPO(
        "MlpPolicy",
        env,
        **algo_config,
    )

    load_path = args.policy.rstrip(".zip")
    for p in (load_path, load_path + ".zip"):
        if os.path.exists(p):
            load_path_or_dict = p
            break
    else:
        print("ERROR: Policy file not found.")
        print(f"  Tried: {load_path}, {load_path}.zip")
        print("  Options:")
        print("    1. Use your own trained model")
        print("    2. Or provide a valid pretrained policy path")
        raise SystemExit(1)

    _, params, _ = load_from_zip_file(load_path_or_dict, device="cpu", load_data=False)
    expert.set_parameters(params, exact_match=True, device="cpu")

    action = [0.0, 0.0]
    scenario_t = 0
    start_t = 20

    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):
            obs, reward, terminated, truncated, info = env.step(action)

            obs_input = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action = expert.predict(obs_input, deterministic=True)[0]
            action = np.clip(action, a_min=-1.0, a_max=1.0)
            action = action[0].tolist()

            if args.save_img and scenario_t >= start_t:
                rgb_camera = env.engine.get_sensor("rgb_camera")
                rgb_front = rgb_camera.perceive(
                    to_float=config.get("norm_pixel", False),
                    new_parent_node=env.agent.origin,
                    position=[0, -7, 1.0],
                    hpr=[0, 0, 0],
                )
                max_rgb_value = rgb_front.max()
                rgb = rgb_front[..., ::-1]
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)

                depth_camera = env.engine.get_sensor("depth_camera")
                depth_front = depth_camera.perceive(
                    to_float=config.get("norm_pixel", False),
                    new_parent_node=env.agent.origin,
                    position=[0, -7, 1.0],
                    hpr=[0, 0, 0],
                ).reshape(576, 1024, -1)[..., -1]

                depth_normalized = cv2.normalize(depth_front, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_img = cv2.bitwise_not(depth_front)
                depth_img = depth_img[..., None]

                semantic_camera = env.engine.get_sensor("semantic_camera")
                semantic_front = semantic_camera.perceive(
                    to_float=config.get("norm_pixel", False),
                    new_parent_node=env.agent.origin,
                    position=[0, -7, 1.0],
                    hpr=[0, 0, 0],
                )
                max_semantic_value = semantic_front.max()
                semantic = semantic_front
                if max_semantic_value > 1:
                    semantic = semantic.astype(np.uint8)
                else:
                    semantic = (semantic * 255).astype(np.uint8)
                semantic = semantic[..., ::-1]

                base_name = f"seed_{env.current_seed:06d}_time_{scenario_t - start_t:06d}"
                cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_rgb.png"), rgb[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_semantic.png"), semantic[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_depth_colored.png"), depth_colored[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"{base_name}_depth_raw.png"), depth_img[..., ::-1])

            scenario_t += 1

            if terminated or truncated:
                obs, _ = env.reset(seed=env.current_seed + 1)
                action = [0.0, 0.0]
                scenario_t = 0

    finally:
        env.close()