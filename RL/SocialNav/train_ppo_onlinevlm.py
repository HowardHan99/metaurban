import argparse
import os

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file

from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation

from metaurban.envs.social_dynamic_env_onlinelearning import SocialDynamicMetaUrbanEnv


parser = argparse.ArgumentParser()
parser.add_argument("--unique_id", type=int, default=0)
parser.add_argument(
    "--pretrained_path",
    type=str,
    default="./pretrained_policy_576k.zip",
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--total_timesteps", type=int, default=int(3e5))

# ===== 对齐 drive_social_with_pretrained_policy.py =====
parser.add_argument("--map", type=str, default="C")
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--num_scenarios", type=int, default=100)
parser.add_argument("--object_density", type=float, default=0.7)

parser.add_argument(
    "--scene_type",
    type=str,
    default="default",
    choices=["default", "commercial", "commute", "leisure", "constrained"],
)
parser.add_argument(
    "--scene_building_source",
    type=str,
    default="default",
    choices=["scene", "default"],
)

parser.add_argument("--crossing_ped_num", type=int, default=6)
parser.add_argument("--vulnerable_ped_num", type=int, default=8)
parser.add_argument("--group_cluster_num", type=int, default=4)
parser.add_argument("--group_cluster_size_min", type=int, default=3)
parser.add_argument("--group_cluster_size_max", type=int, default=5)
parser.add_argument("--group_spawn_near_ego", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--group_spawn_min_radius", type=float, default=6.0)
parser.add_argument("--group_spawn_max_radius", type=float, default=11.0)
parser.add_argument("--group_route_min_ego_distance", type=float, default=8.0)
parser.add_argument("--group_route_min_separation", type=float, default=5.5)
parser.add_argument("--group_member_radius", type=float, default=1.45)
parser.add_argument("--group_member_ring_step", type=float, default=0.62)
parser.add_argument("--group_member_radius_jitter", type=float, default=0.16)
parser.add_argument("--group_member_ring_step_jitter", type=float, default=0.12)
parser.add_argument("--group_member_idle_shift_prob", type=float, default=0.015)
parser.add_argument("--group_member_idle_shift_steps_mean", type=int, default=18)
parser.add_argument("--group_member_idle_shift_radius", type=float, default=0.22)
parser.add_argument("--group_release_enable", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--group_release_steps_mean", type=int, default=180)
parser.add_argument("--group_release_steps_std", type=int, default=40)
parser.add_argument("--group_release_steps_min", type=int, default=60)

parser.add_argument("--vulnerable_elderly_ratio", type=float, default=0.6)
parser.add_argument("--vulnerable_distracted_ratio", type=float, default=0.4)
parser.add_argument("--vulnerable_pause_prob", type=float, default=0.02)
parser.add_argument("--vulnerable_pause_steps_mean", type=int, default=16)
parser.add_argument("--spawn_human_num", type=int, default=40)
parser.add_argument("--spawn_elderly_num", type=int, default=0)
parser.add_argument("--ignore_success_done", action=argparse.BooleanOptionalAction, default=False)

# online VLM extras
parser.add_argument("--spawn_robot_on_sidewalk", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

set_random_seed(args.unique_id)

os.makedirs("./RL_logs", exist_ok=True)
os.makedirs("./RL_logs/PPO", exist_ok=True)
exptid = f"{args.unique_id:04d}"

config = dict(
    env=dict(
        crswalk_density=1,
        object_density=args.object_density,
        use_render=True,
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
            use_saver=False,
            overtake_stat=False,
        ),

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
        spawn_human_num=args.spawn_human_num,
        spawn_wheelchairman_num=max(1, int(args.spawn_human_num // 20)),
        spawn_elderly_num=args.spawn_elderly_num,
        spawn_edog_num=0,
        spawn_erobot_num=0,
        spawn_drobot_num=0,
        max_actor_num=20,
        ignore_success_done=args.ignore_success_done,
        spawn_robot_on_sidewalk=args.spawn_robot_on_sidewalk,

        agent_observation=LidarStateObservation,

        # ===== Base reward =====
        success_reward=50.0,
        out_of_road_penalty=100.0,
        on_lane_line_penalty=1.0,
        crash_vehicle_penalty=2.0,
        crash_object_penalty=2.0,
        crash_human_penalty=4.0,
        crash_building_penalty=2.0,
        driving_reward=2.0,
        steering_range_penalty=0.5,
        heading_penalty=0.0,
        lateral_penalty=1.0,
        speed_reward=0.0,
        no_negative_reward=False,

        # ===== Online VLM reward =====
        use_vlm_reward=True,
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct",
        vlm_device=args.device,
        vlm_dtype="bfloat16",
        vlm_query_interval=5,
        vlm_reward_weight=0.5,
        vlm_default_reward=0.0,
        vlm_max_new_tokens=128,
        vlm_temperature=0.0,
        vlm_log_response=True,

        # ===== Cost =====
        crash_vehicle_cost=2.0,
        crash_object_cost=2.0,
        out_of_road_cost=2.0,
        crash_human_cost=2.0,
    ),
    algo=dict(
        learning_rate=5e-5,
        n_steps=200,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=10.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log=f"./RL_logs/PPO/metaurban_social_nav_{exptid}-tb_logs/",
        device=args.device,
    ),
)


def make_metaurban_env_fn(env_cfg, seed):
    def _thunk():
        env = SocialDynamicMetaUrbanEnv(
            dict(
                start_seed=seed,
                log_level=50,
                training=True,
                **env_cfg,
            )
        )
        env = Monitor(
            env,
            info_keywords=(
                "episode_env_reward",
                "episode_vlm_reward",
                "episode_total_reward",
            ),
        )
        return env
    return _thunk


def build_env():
    return DummyVecEnv([make_metaurban_env_fn(config["env"], 0)])


def build_model(env):
    model = PPO(
        "MlpPolicy",
        env,
        **config["algo"],
    )

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"[INFO] Loading pretrained params from: {args.pretrained_path}")
        _, params, _ = load_from_zip_file(
            args.pretrained_path,
            device=config["algo"]["device"],
            load_data=False,
        )
        model.set_parameters(
            params,
            exact_match=True,
            device=config["algo"]["device"],
        )
        print("[INFO] Pretrained PPO parameters loaded successfully.")
    else:
        print("[WARN] pretrained_path not found. Train from scratch.")

    return model


def train():
    print(HELP_MESSAGE)

    env = build_env()
    model = build_model(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"./RL_logs/PPO/metaurban_social_nav_{exptid}_ckpt_logs/",
        name_prefix=exptid,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([checkpoint_callback])

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=f"metaurban_social_nav_{exptid}",
    )

    save_path = f"./RL_logs/PPO/metaurban_social_nav_{exptid}_final"
    model.save(save_path)
    print(f"[DONE] Saved final model to {save_path}")
    env.close()


if __name__ == "__main__":
    train()
