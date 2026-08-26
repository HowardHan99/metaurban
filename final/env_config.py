"""
Shared environment configuration for all final-project scripts.

Mirrors midterm/env_config.py so that MBPO results on SidewalkStaticMetaUrbanEnv
are directly comparable to the midterm PPO and SAC runs.

The MBPO-specific reward overrides (MBPO_ENV_OVERRIDES) are the same ones that
unblocked the midterm SAC run (see midterm/train_sac.py): MBPO's inner policy is
a SAC agent that updates on (real + synthetic) transitions, so it inherits the
same "braking is the safest action" failure mode unless we rebalance rewards and
add the idle penalty.
"""
from metaurban.obs.state_obs import LidarStateObservation


ENV_CONFIG = dict(
    use_render=False,
    map="X",
    training=True,
    object_density=0.6,
    crswalk_density=1,
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

    # Reward scheme (base — overridden for MBPO below)
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

    crash_vehicle_cost=2.0,
    crash_object_cost=2.0,
    out_of_road_cost=2.0,
    crash_human_cost=2.0,

    agent_observation=LidarStateObservation,
)


# MBPO uses SAC as its inner policy, so apply the same reward rebalancing the
# midterm SAC run needed. Without these, the learned policy converges to
# "always brake" (reward clamped to 0 vs negative motion penalties).
MBPO_ENV_OVERRIDES = dict(
    no_negative_reward=False,
    driving_reward=3.0,
    success_reward=15.0,
    speed_reward=1.0,
    lateral_penalty=0.5,
    crash_vehicle_penalty=1.0,
    crash_object_penalty=1.0,
    crash_human_penalty=1.0,
    crash_building_penalty=1.0,
    out_of_road_penalty=2.0,
    steering_range_penalty=0.5,
)


# Same eval seeds as midterm so PPO/SAC/MBPO learning curves are plotted on
# identical evaluation scenarios. run_random.py in midterm used these too.
EVAL_VEC_ENV_SEEDS = tuple(range(950, 960))
