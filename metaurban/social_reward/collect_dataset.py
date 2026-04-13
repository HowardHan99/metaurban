#!/usr/bin/env python3
"""
collect_dataset.py — CLI entry point for building the offline social-reward dataset.

Usage examples
--------------
# Random policy, 100 episodes, no rendering (headless):
python collect_dataset.py --num-episodes 100 --out-dir /data/social_offline

# IDM policy, 200 episodes, save RGB frames, 6 s clips with 3 s stride:
python collect_dataset.py --policy idm --num-episodes 200 \\
    --capture-rgb --clip-len 6.0 --stride 3.0 \\
    --out-dir /data/social_offline

After collection the output directory will contain:
  <out-dir>/
      episodes/   ← one .npz per episode (full trajectory)
      clips/      ← one .npz per sliding-window clip
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("collect_dataset")


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _build_policy(name: str, env):
    """Return a callable ``obs -> action`` for the requested policy type."""
    if name == "random":
        return lambda obs: env.action_space.sample()

    if name == "idm":
        # IDM (Intelligent Driver Model) is MetaUrban's built-in autopilot.
        # The env ships with an IDMPolicy that can be used as follows:
        try:
            from metaurban.policy.idm_policy import IDMPolicy

            class _IDMWrapper:
                def __init__(self, env):
                    self._env = env

                def __call__(self, obs):
                    agent  = self._env.agent
                    # IDMPolicy expects an integer seed (or None), not a Generator.
                    policy_seed = getattr(self._env, "current_seed", None)
                    policy = IDMPolicy(agent, policy_seed)
                    return policy.act("default_agent")

            return _IDMWrapper(env)
        except Exception as exc:
            logger.warning("IDMPolicy unavailable (%s); falling back to random.", exc)
            return lambda obs: env.action_space.sample()

    raise ValueError(f"Unknown policy '{name}'. Choose from: random, idm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Collect a MetaUrban offline dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-episodes", type=int, default=50,
                   help="Number of episodes to collect.")
    p.add_argument("--out-dir", type=str, default="dataset",
                   help="Root output directory.")
    p.add_argument("--policy", type=str, default="random",
                   choices=["random", "idm"],
                   help="Policy used to drive the ego vehicle.")
    p.add_argument("--sim-hz", type=float, default=10.0,
                   help="Simulator decision frequency in Hz.")
    p.add_argument("--clip-len", type=float, default=4.0,
                   help="Sliding window clip length in seconds.")
    p.add_argument("--stride", type=float, default=2.0,
                   help="Stride between clip start positions in seconds.")
    p.add_argument("--capture-rgb", action="store_true",
                   help="Save RGB camera frames (requires render mode).")
    p.add_argument("--use-render", action="store_true",
                   help="Open a render window (implies headless=False).")
    p.add_argument("--map", type=str, default="XCS",
                   help="Map layout string passed to the environment.")
    p.add_argument("--object-density", type=float, default=1.0,
                   help="Static object spawn density used by sidewalk asset manager.")
    p.add_argument("--num-scenarios", type=int, default=1000,
                   help="Total number of scenarios in the environment pool.")
    p.add_argument("--env-mode", type=str, default="default",
                   choices=["base", "social", "default"],
                   help=(
                       "Environment type: base=original dynamic env, social=4-role social env, "
                       "default=social env with default building visuals."
                   ))
    p.add_argument("--crossing-ped-num", type=int, default=8,
                   help="Number of crossing-role pedestrians in social mode.")
    p.add_argument("--signaling-ped-num", type=int, default=0,
                   help="Reserved for future signaling role. Currently disabled.")
    p.add_argument("--vulnerable-ped-num", type=int, default=4,
                   help="Number of vulnerable-role pedestrians in social mode.")
    p.add_argument("--spawn-elderly-num", type=int, default=2,
                   help="Number of dedicated ElderlyPedestrian agents to spawn.")
    p.add_argument("--vulnerable-elderly-ratio", type=float, default=0.6,
                   help="Ratio of non-wheelchair vulnerable agents using elderly profile.")
    p.add_argument("--vulnerable-distracted-ratio", type=float, default=0.4,
                   help="Ratio of non-wheelchair vulnerable agents using distracted profile.")
    p.add_argument("--vulnerable-pause-prob", type=float, default=0.02,
                   help="Per-step pause probability for distracted vulnerable subtype.")
    p.add_argument("--vulnerable-pause-steps-mean", type=int, default=16,
                   help="Mean pause duration (steps) for distracted vulnerable subtype.")
    p.add_argument("--group-ped-pair-num", type=int, default=3,
                   help="Number of group pedestrian pairs in social mode.")
    p.add_argument("--group-cluster-num", type=int, default=4,
                   help="Number of multi-person group clusters (0 = derive from --group-ped-pair-num).")
    p.add_argument("--group-cluster-size-min", type=int, default=3,
                   help="Minimum number of members per group cluster.")
    p.add_argument("--group-cluster-size-max", type=int, default=5,
                   help="Maximum number of members per group cluster.")
    p.add_argument("--group-member-radius", type=float, default=1.45,
                   help="Base member radius (meters) for group conversation circles.")
    p.add_argument("--group-member-ring-step", type=float, default=0.62,
                   help="Ring spacing step (meters) for multi-ring group clusters.")
    p.add_argument("--group-member-radius-jitter", type=float, default=0.16,
                   help="Per-cluster random jitter applied to member radius.")
    p.add_argument("--group-member-ring-step-jitter", type=float, default=0.12,
                   help="Per-cluster random jitter applied to ring step spacing.")
    p.add_argument("--group-member-idle-shift-prob", type=float, default=0.015,
                   help="Per-step probability for a grouped member to make a subtle local move.")
    p.add_argument("--group-member-idle-shift-steps-mean", type=int, default=18,
                   help="Mean steps for each subtle grouped-member local move.")
    p.add_argument("--group-member-idle-shift-radius", type=float, default=0.22,
                   help="Maximum radius of subtle grouped-member local moves (meters).")
    p.add_argument(
        "--group-spawn-near-ego",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to initialize group clusters around ego for easier visualization.",
    )
    p.add_argument("--group-spawn-min-radius", type=float, default=5.0,
                   help="Minimum ego-centric radius for group cluster spawn (meters).")
    p.add_argument("--group-spawn-max-radius", type=float, default=10.0,
                   help="Maximum ego-centric radius for group cluster spawn (meters).")
    p.add_argument("--group-route-min-ego-distance", type=float, default=8.0,
                   help="Minimum ego distance when using route-aware group placement (meters).")
    p.add_argument("--group-route-min-separation", type=float, default=5.5,
                   help="Minimum separation between clusters along route-aware placement (meters).")
    p.add_argument(
        "--group-release-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether group clusters dissolve into normal pedestrians after a while.",
    )
    p.add_argument("--group-release-steps-mean", type=int, default=180,
                   help="Mean lifetime (steps) before a cluster dissolves.")
    p.add_argument("--group-release-steps-std", type=int, default=40,
                   help="Std of cluster lifetime in steps.")
    p.add_argument("--group-release-steps-min", type=int, default=60,
                   help="Minimum lifetime in steps before release.")
    p.add_argument("--scene-type", type=str, default="default",
                   choices=["default", "commercial", "commute", "leisure", "constrained"],
                   help="Scene type determines building asset pool distribution.")
    p.add_argument("--scene-building-source", type=str, default="scene",
                   choices=["scene", "default"],
                   help="Building visual source: scene-specific pool or default metadata models.")
    p.add_argument("--spawn-human-num", type=int, default=30,
                   help="Number of pedestrians spawned per scenario.")
    p.add_argument("--horizon", type=int, default=1000,
                   help="Maximum steps per episode.")
    p.add_argument("--skip-clips", action="store_true",
                   help="Skip the sliding-window clip extraction step.")
    p.add_argument("--seed", type=int, default=None,
                   help="Base random seed (None = fully random).")
    p.add_argument(
        "--spawn-robot-on-sidewalk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to initialize ego near sidewalk pedestrians in social mode.",
    )
    p.add_argument(
        "--ignore-success-done",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep episode running after arrive-destination to observe long-horizon social dynamics.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve output paths
    root      = Path(args.out_dir)
    ep_dir    = root / "episodes"
    clip_dir  = root / "clips"
    ep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory : %s", root.resolve())
    logger.info("Episodes dir     : %s", ep_dir)
    logger.info("Clips dir        : %s", clip_dir)

    # ------------------------------------------------------------------
    # Build environment config
    # ------------------------------------------------------------------
    env_config = dict(
        use_render=args.use_render,
        map=args.map,
        object_density=args.object_density,
        spawn_human_num=args.spawn_human_num,
        spawn_wheelchairman_num=max(1, args.spawn_human_num // 20),
        spawn_elderly_num=args.spawn_elderly_num,
        spawn_edog_num=0,
        spawn_erobot_num=0,
        spawn_drobot_num=0,
        max_actor_num=1,
        show_ego_navigation=False,
        horizon=args.horizon,
        num_scenarios=args.num_scenarios,
        decision_repeat=1,
        window_size=(960, 960) if args.use_render else (84, 84),
        vehicle_config=dict(show_lidar=False, show_navi_mark=True),
        crossing_ped_num=args.crossing_ped_num,
        signaling_ped_num=args.signaling_ped_num,
        vulnerable_ped_num=args.vulnerable_ped_num,
        vulnerable_elderly_ratio=args.vulnerable_elderly_ratio,
        vulnerable_distracted_ratio=args.vulnerable_distracted_ratio,
        vulnerable_pause_prob=args.vulnerable_pause_prob,
        vulnerable_pause_steps_mean=args.vulnerable_pause_steps_mean,
        group_ped_pair_num=args.group_ped_pair_num,
        group_cluster_num=args.group_cluster_num,
        group_cluster_size_min=args.group_cluster_size_min,
        group_cluster_size_max=args.group_cluster_size_max,
        group_member_radius=args.group_member_radius,
        group_member_ring_step=args.group_member_ring_step,
        group_member_radius_jitter=args.group_member_radius_jitter,
        group_member_ring_step_jitter=args.group_member_ring_step_jitter,
        group_member_idle_shift_prob=args.group_member_idle_shift_prob,
        group_member_idle_shift_steps_mean=args.group_member_idle_shift_steps_mean,
        group_member_idle_shift_radius=args.group_member_idle_shift_radius,
        group_spawn_near_ego=args.group_spawn_near_ego,
        group_spawn_min_radius=args.group_spawn_min_radius,
        group_spawn_max_radius=args.group_spawn_max_radius,
        group_route_min_ego_distance=args.group_route_min_ego_distance,
        group_route_min_separation=args.group_route_min_separation,
        group_release_enable=args.group_release_enable,
        group_release_steps_mean=args.group_release_steps_mean,
        group_release_steps_std=args.group_release_steps_std,
        group_release_steps_min=args.group_release_steps_min,
        scene_type=args.scene_type,
        scene_building_source=args.scene_building_source,
        spawn_robot_on_sidewalk=args.spawn_robot_on_sidewalk,
        ignore_success_done=args.ignore_success_done,
    )

    if args.env_mode == "default":
        env_config["scene_type"] = "default"
        env_config["scene_building_source"] = "default"

    # ------------------------------------------------------------------
    # Instantiate environment
    # ------------------------------------------------------------------
    logger.info("Importing MetaUrban …")
    try:
        if args.env_mode in ("social", "default"):
            from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv as EnvClass
        else:
            from metaurban.envs.sidewalk_dynamic_env import SidewalkDynamicMetaUrbanEnv as EnvClass
    except ImportError as exc:
        logger.error("Cannot import MetaUrban: %s", exc)
        sys.exit(1)

    logger.info("Creating environment …")
    env = EnvClass(env_config)
    logger.info("Environment mode: %s", args.env_mode)

    # ------------------------------------------------------------------
    # Build policy
    # ------------------------------------------------------------------
    obs, info = env.reset(seed=args.seed)
    policy_fn = _build_policy(args.policy, env)
    logger.info("Policy: %s", args.policy)

    # ------------------------------------------------------------------
    # Collect episodes
    # ------------------------------------------------------------------
    from metaurban.social_reward.dataset_collector import EpisodeBuffer, _get_rgb_frame

    written_episodes = []
    rng = np.random.default_rng(args.seed)

    for ep_idx in range(args.num_episodes):
        # Reset (seeded for reproducibility)
        ep_seed = int(rng.integers(0, 2**31 - 1)) if args.seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        scenario_idx = info.get("scenario_index", ep_idx)
        buf = EpisodeBuffer(scenario_index=scenario_idx, seed=ep_seed or ep_idx)

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            buf.record(obs, action, reward, info, env)

        ep_path = buf.flush(ep_dir)
        written_episodes.append(ep_path)
        logger.info(
            "[%d/%d]  scenario=%d  steps=%d  seed=%s  -> %s",
            ep_idx + 1, args.num_episodes,
            scenario_idx, len(buf), ep_seed, ep_path.name,
        )

    env.close()
    logger.info("Collection finished.  %d episodes saved to %s", len(written_episodes), ep_dir)

    # ------------------------------------------------------------------
    # Sliding-window clip extraction
    # ------------------------------------------------------------------
    if args.skip_clips:
        logger.info("--skip-clips set; skipping clip extraction.")
        return

    from metaurban.social_reward.dataset_collector import ClipExtractor

    window_steps = max(1, int(args.clip_len  * args.sim_hz))
    stride_steps = max(1, int(args.stride    * args.sim_hz))

    logger.info(
        "Extracting clips: window=%ds (%d steps), stride=%ds (%d steps) …",
        int(args.clip_len), window_steps, int(args.stride), stride_steps,
    )

    extractor = ClipExtractor(
        window_steps=window_steps,
        stride_steps=stride_steps,
        sim_hz=args.sim_hz,
    )
    all_clips = extractor.extract_batch(ep_dir, clip_dir)
    logger.info("Done.  %d clips saved to %s", len(all_clips), clip_dir)


if __name__ == "__main__":
    main()
