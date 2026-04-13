import argparse
import os
import cv2
import numpy as np
import torch

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.save_util import load_from_zip_file

from metaurban.constants import HELP_MESSAGE
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation


def build_config(args):
    den_scale = 1
    return dict(
        crswalk_density=1,
        object_density=args.object_density,
        use_render=False,
        walk_on_all_regions=False,
        map=args.map,
        manual_control=False,
        drivable_area_extension=55,
        height_scale=1,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=args.horizon,
        on_continuous_line_done=False,
        out_of_route_done=True,
        relax_out_of_road_done=True,
        max_lateral_dist=15.0,
        show_sidewalk=True,
        show_crosswalk=True,
        random_spawn_lane_index=False,
        num_scenarios=args.num_scenarios,
        accident_prob=0,
        window_size=(1200, 900),

        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
            policy_reverse=False,
        ),

        image_observation=True,
        sensors=dict(
            rgb_camera=(RGBCamera, 256, 256),
            depth_camera=(DepthCamera, 256, 256),
            semantic_camera=(SemanticCamera, 256, 256),
        ),
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[],

        scene_type=args.scene_type,
        scene_building_source=args.scene_building_source,
        crossing_ped_num=args.crossing_ped_num,
        vulnerable_ped_num=args.vulnerable_ped_num,
        group_cluster_num=args.group_cluster_num,
        group_cluster_size_min=args.group_cluster_size_min,
        group_cluster_size_max=args.group_cluster_size_max,
        group_spawn_near_ego=args.group_spawn_near_ego,
        group_spawn_min_radius=args.group_spawn_min_radius,
        group_spawn_max_radius=args.group_spawn_max_radius,
        group_route_min_ego_distance=args.group_route_min_ego_distance,
        group_route_min_separation=args.group_route_min_separation,
        group_member_radius=args.group_member_radius,
        group_member_ring_step=args.group_member_ring_step,
        group_member_radius_jitter=args.group_member_radius_jitter,
        group_member_ring_step_jitter=args.group_member_ring_step_jitter,
        group_member_idle_shift_prob=args.group_member_idle_shift_prob,
        group_member_idle_shift_steps_mean=args.group_member_idle_shift_steps_mean,
        group_member_idle_shift_radius=args.group_member_idle_shift_radius,
        group_release_enable=args.group_release_enable,
        group_release_steps_mean=args.group_release_steps_mean,
        group_release_steps_std=args.group_release_steps_std,
        group_release_steps_min=args.group_release_steps_min,
        vulnerable_elderly_ratio=args.vulnerable_elderly_ratio,
        vulnerable_distracted_ratio=args.vulnerable_distracted_ratio,
        vulnerable_pause_prob=args.vulnerable_pause_prob,
        vulnerable_pause_steps_mean=args.vulnerable_pause_steps_mean,
        spawn_human_num=int(args.spawn_human_num * den_scale),
        spawn_wheelchairman_num=max(1, int(args.spawn_human_num // 20)),
        spawn_elderly_num=args.spawn_elderly_num,
        spawn_edog_num=0,
        spawn_erobot_num=0,
        spawn_drobot_num=0,
        max_actor_num=20,
        ignore_success_done=args.ignore_success_done,
        spawn_robot_on_sidewalk=args.spawn_robot_on_sidewalk,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", type=str, default="./pretrained_policy_576k")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)

    p.add_argument("--map", type=str, default="C")
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--num-scenarios", type=int, default=100)
    p.add_argument("--object-density", type=float, default=0.7)

    p.add_argument("--scene-type", type=str, default="default",
                   choices=["default", "commercial", "commute", "leisure", "constrained"])
    p.add_argument("--scene-building-source", type=str, default="default",
                   choices=["scene", "default"])

    p.add_argument("--crossing-ped-num", type=int, default=6)
    p.add_argument("--vulnerable-ped-num", type=int, default=8)
    p.add_argument("--group-cluster-num", type=int, default=4)
    p.add_argument("--group-cluster-size-min", type=int, default=3)
    p.add_argument("--group-cluster-size-max", type=int, default=5)
    p.add_argument("--group-spawn-near-ego", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-spawn-min-radius", type=float, default=6.0)
    p.add_argument("--group-spawn-max-radius", type=float, default=11.0)
    p.add_argument("--group-route-min-ego-distance", type=float, default=8.0)
    p.add_argument("--group-route-min-separation", type=float, default=5.5)
    p.add_argument("--group-member-radius", type=float, default=1.45)
    p.add_argument("--group-member-ring-step", type=float, default=0.62)
    p.add_argument("--group-member-radius-jitter", type=float, default=0.16)
    p.add_argument("--group-member-ring-step-jitter", type=float, default=0.12)
    p.add_argument("--group-member-idle-shift-prob", type=float, default=0.015)
    p.add_argument("--group-member-idle-shift-steps-mean", type=int, default=18)
    p.add_argument("--group-member-idle-shift-radius", type=float, default=0.22)
    p.add_argument("--group-release-enable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-release-steps-mean", type=int, default=180)
    p.add_argument("--group-release-steps-std", type=int, default=40)
    p.add_argument("--group-release-steps-min", type=int, default=60)

    p.add_argument("--vulnerable-elderly-ratio", type=float, default=0.6)
    p.add_argument("--vulnerable-distracted-ratio", type=float, default=0.4)
    p.add_argument("--vulnerable-pause-prob", type=float, default=0.02)
    p.add_argument("--vulnerable-pause-steps-mean", type=int, default=16)
    p.add_argument("--spawn-human-num", type=int, default=40)
    p.add_argument("--spawn-elderly-num", type=int, default=0)
    p.add_argument("--ignore-success-done", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--spawn-robot-on-sidewalk", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--save-dir", type=str, default="./recorded_dataset")
    p.add_argument("--save-merged-npy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def main():
    args = parse_args()
    config = build_config(args)

    env = SocialDynamicMetaUrbanEnv(config)
    o, _ = env.reset(seed=args.seed)

    save_dir = args.save_dir
    img_dir = os.path.join(save_dir, "rgb")
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

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

    expert = PPO("MlpPolicy", env, **algo_config)
    _, params, _ = load_from_zip_file(args.policy_path, device="cpu", load_data=False)
    expert.set_parameters(params, exact_match=True, device="cpu")

    global_steps = 0
    action = None

    try:
        print(HELP_MESSAGE)

        for _ in range(args.max_steps):
            obs_for_policy = o["state"] if isinstance(o, dict) and "state" in o else o
            obs_np = np.asarray(obs_for_policy, dtype=np.float32)

            action = expert.predict(torch.from_numpy(obs_np).reshape(1, -1), deterministic=True)[0]
            action = np.clip(action, a_min=-1.0, a_max=1.0)
            action = action[0].tolist()

            o_next, r, tm, tc, info = env.step(action)

            if isinstance(o, dict) and "image" in o:
                rgb = o["image"][:, :, :, 0]
                rgb = np.clip(rgb, 0, 1)
                rgb = (rgb * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(img_dir, f"step_{global_steps:06d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )

            action_to_save = info["action"] if ("action" in info) else action

            state_to_save = o["state"] if isinstance(o, dict) and "state" in o else o

            np.save(
                os.path.join(data_dir, f"step_{global_steps:06d}.npy"),
                {
                    "state": state_to_save,
                    "action": action_to_save,
                    "reward": r,
                    "terminal": tm,
                    "trunc": tc,
                    "info": info,
                },
                allow_pickle=True
            )

            if args.save_merged_npy:
                next_state_to_save = o_next["state"] if isinstance(o_next, dict) and "state" in o_next else o_next
                np.save(
                    os.path.join(data_dir, f"step_{global_steps:06d}_merged.npy"),
                    {
                        "state": state_to_save,
                        "next_state": next_state_to_save,
                        "action": action_to_save,
                        "reward": r,
                        "terminal": tm,
                        "trunc": tc,
                        "info": info,
                    },
                    allow_pickle=True
                )

            if args.verbose and global_steps % 20 == 0:
                print(f"[step] global_steps={global_steps} reward={r:.4f} tm={tm} tc={tc}")

            global_steps += 1
            o = o_next

            if tm or tc:
                o, _ = env.reset(((env.current_seed + 1) % config['num_scenarios'])
                                 + env.engine.global_config['start_seed'])

    finally:
        env.close()


if __name__ == "__main__":
    main()