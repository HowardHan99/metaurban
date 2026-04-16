"""
Shared environment configuration for all midterm project scripts.
All training/eval scripts import from here to ensure consistency.

Environment choice:
  - SidewalkStaticMetaUrbanEnv (used by train_ppo.py, train_sac.py): NO moving humans/robots.
    Static objects only (trees, lamps, etc. via object_density, crswalk_density).
    spawn_* params are ignored by the static env.

  - SidewalkDynamicMetaUrbanEnv: Spawns humans, robot dogs, delivery robots, etc.
    Uses spawn_human_num, spawn_edog_num, spawn_erobot_num, spawn_drobot_num,
    spawn_wheelchairman_num.

To run without any human/robot agents, use NO_HUMAN_ENV_CONFIG (all spawn counts = 0).
"""
from metaurban.obs.state_obs import LidarStateObservation


# Default config (static env - no moving humans/robots by design)
ENV_CONFIG = dict(
    use_render=True, # True False
    map="X",
    training=True,
    object_density=0.1, # 0.7
    crswalk_density=1,
    # Spawn counts (used only by SidewalkDynamicMetaUrbanEnv; ignored by Static)
    spawn_human_num=0,
    spawn_robotdog_num=0,
    spawn_deliveryrobot_num=0,
    spawn_wheelchairman_num=0,
    spawn_edog_num=0,
    spawn_erobot_num=0,
    spawn_drobot_num=0,
    walk_on_all_regions=False,
    show_mid_block_map=False,
    show_ego_navigation=False,
    debug=False,
    horizon=1000,
    on_continuous_line_done=False,
    out_of_route_done=True,
    vehicle_config=dict(
        show_lidar=True,
        show_navi_mark=True,
        show_line_to_navi_mark=False,
        show_dest_mark=False,
        use_saver=False,
        overtake_stat=False,
    ),
    show_sidewalk=True,
    show_crosswalk=True,
    random_spawn_lane_index=False,
    num_scenarios=1000,
    traffic_density=0,
    accident_prob=0,
    crash_vehicle_done=False,
    crash_object_done=False,
    relax_out_of_road_done=True,
    drivable_area_extension=75,

    # Reward scheme
    success_reward=8.0,
    out_of_road_penalty=3.0,
    on_lane_line_penalty=1.0,
    crash_vehicle_penalty=2.0,
    crash_object_penalty=2.0,
    crash_human_penalty=2.0,
    crash_building_penalty=2.0,
    driving_reward=2.0,
    steering_range_penalty=2.0,
    heading_penalty=0.0,
    lateral_penalty=2.0,
    max_lateral_dist=5.0,
    speed_reward=0.5,
    no_negative_reward=True,

    # Cost scheme
    crash_vehicle_cost=2.0,
    crash_object_cost=2.0,
    out_of_road_cost=2.0,
    crash_human_cost=2.0,

    agent_observation=LidarStateObservation,
)

# SubprocVecEnv start_seed for each worker in train_ppo / train_sac EvalCallback (10 envs).
# run_random.py measures the same seeds so the random success baseline matches eval logs.
EVAL_VEC_ENV_SEEDS = tuple(range(950, 960))
