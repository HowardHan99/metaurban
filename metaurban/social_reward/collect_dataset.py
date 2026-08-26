#!/usr/bin/env python3
"""
collect_dataset.py — build an episode-structured VLM social-reward dataset.

Usage examples
--------------
# IDM policy, sample every five simulator decisions:
python collect_dataset.py --policy idm --num-episodes 200 \\
    --capture-rgb --sampling-interval 5 \\
    --out-dir /data/social_offline

After collection the output directory will contain:
  <out-dir>/
      manifest.json
      episodes/<episode-id>/
          episode.json
          records.jsonl
          frames/*.png
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
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

def _build_policy(name: str, env, policy_path: str = None):
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
                    self._agent = None
                    self._policy = None

                def __call__(self, obs):
                    agent = self._env.agent
                    if agent is not self._agent:
                        # IDMPolicy expects an integer seed (or None), not a Generator.
                        policy_seed = getattr(self._env, "current_seed", None)
                        self._agent = agent
                        self._policy = IDMPolicy(agent, policy_seed)
                    return self._policy.act("default_agent")

            return _IDMWrapper(env)
        except Exception as exc:
            logger.warning("IDMPolicy unavailable (%s); falling back to random.", exc)
            return lambda obs: env.action_space.sample()

    if name == "ppo":
        if not policy_path:
            raise ValueError("--policy-path is required for --policy ppo")
        from stable_baselines3 import PPO

        model = PPO.load(policy_path, device="cpu")

        def ppo_policy(obs):
            state = obs.get("state") if isinstance(obs, dict) else obs
            state = np.asarray(state, dtype=np.float32)
            if state.shape != model.observation_space.shape:
                raise ValueError(
                    f"PPO expects observation {model.observation_space.shape}, got {state.shape}"
                )
            action, _ = model.predict(state, deterministic=True)
            return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        return ppo_policy

    raise ValueError(f"Unknown policy '{name}'. Choose from: random, idm, ppo")


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
                   choices=["random", "idm", "ppo"],
                   help="Policy used to drive the ego vehicle.")
    p.add_argument("--policy-path", type=str, default=None,
                   help="Stable-Baselines3 PPO checkpoint used by --policy ppo.")
    p.add_argument("--sim-hz", type=float, default=10.0,
                   help="Expected simulator decision frequency in Hz (manifest metadata only).")
    p.add_argument("--sampling-interval", type=int, default=5,
                   help="Save every Nth simulator transition; terminal transitions are always saved.")
    p.add_argument("--capture-rgb", action="store_true",
                   help="Save RGB camera frames using the offscreen image observation.")
    p.add_argument("--image-width", type=int, default=512,
                   help="Saved RGB frame width.")
    p.add_argument("--image-height", type=int, default=288,
                   help="Saved RGB frame height.")
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


def _git_metadata(repo_root: Path):
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """Convert configuration objects to stable, human-readable JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv=None):
    args = parse_args(argv)
    if not args.capture_rgb:
        raise ValueError("--capture-rgb is required for the VLM raw dataset")
    if args.sampling_interval < 1:
        raise ValueError("--sampling-interval must be at least 1")
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("--image-width and --image-height must be positive")
    if args.num_scenarios < 1:
        raise ValueError("--num-scenarios must be at least 1")

    # On a Linux host without an X display, Panda3D's default GLX pipe cannot
    # create the offscreen window used by MainCamera.  Select Panda3D's bundled
    # EGL pipe before importing MetaUrban; surfaceless EGL works with both Mesa
    # software rendering and headless GPU drivers.
    if not args.use_render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "load-display p3headlessgl")

    # Resolve output paths
    root = Path(args.out_dir)
    ep_dir = root / "episodes"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (ep_dir.exists() and any(ep_dir.iterdir())):
        raise FileExistsError(f"Refusing to append to non-empty dataset directory: {root}")
    ep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory : %s", root.resolve())
    logger.info("Episodes dir     : %s", ep_dir)

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
        # The repository PPO checkpoint was trained with a 271-D state that
        # includes up to 50 surrounding actors.  Other policies do not consume
        # this state, so retain the smaller observation there.
        max_actor_num=50 if args.policy == "ppo" else 1,
        show_ego_navigation=False,
        horizon=args.horizon,
        num_scenarios=args.num_scenarios,
        decision_repeat=1,
        image_observation=True,
        # Panda3D's threaded Cull mode cannot create a surfaceless EGL
        # context on headless hosts.  Collection is I/O-bound at this scale,
        # so use the portable single-threaded renderer.
        multi_thread_render=False,
        norm_pixel=False,
        stack_size=1,
        window_size=(args.image_width, args.image_height),
        sensors=dict(main_camera=("MainCamera", args.image_width, args.image_height)),
        vehicle_config=dict(show_lidar=False, show_navi_mark=False, image_source="main_camera"),
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

    if args.policy == "ppo":
        from metaurban.social_reward.observations import MainCameraLidarStateObservation
        env_config["agent_observation"] = MainCameraLidarStateObservation

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
    policy_fn = _build_policy(args.policy, env, args.policy_path)
    logger.info("Policy: %s", args.policy)

    # ------------------------------------------------------------------
    # Collect episodes
    # ------------------------------------------------------------------
    from metaurban.social_reward.dataset_collector import (
        COLLECTOR_VERSION,
        RAW_DATASET_VERSION,
        StructuredEpisodeWriter,
    )

    episode_summaries = []
    rng = np.random.default_rng(args.seed)
    created_at = datetime.now(timezone.utc).isoformat()

    for ep_idx in range(args.num_episodes):
        # Reset (seeded for reproducibility)
        # MetaUrban uses ``seed`` as a scenario index and requires it to stay
        # within the configured scenario pool.  Walk that pool deterministically
        # from the requested base seed instead of generating an arbitrary int32.
        if args.seed is None:
            ep_seed = int(rng.integers(0, args.num_scenarios))
        else:
            ep_seed = int((args.seed + ep_idx) % args.num_scenarios)
        obs, info = env.reset(seed=ep_seed)
        scenario_idx = info.get("scenario_index", ep_idx)
        episode_id = f"episode_{ep_idx:06d}_s{int(scenario_idx):06d}_seed{ep_seed}"
        writer = StructuredEpisodeWriter(
            dataset_root=root,
            episode_id=episode_id,
            scenario_index=scenario_idx,
            seed=ep_seed,
            sampling_interval=args.sampling_interval,
        )

        terminated = truncated = False
        simulator_steps = 0
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            writer.record(
                step_index=simulator_steps,
                post_obs=obs,
                action_from_previous_state=action,
                environment_reward=reward,
                info=info,
                env=env,
                terminated=terminated,
                truncated=truncated,
            )
            simulator_steps += 1

        summary = writer.close(
            simulator_steps=simulator_steps,
            terminated=terminated,
            truncated=truncated,
        )
        episode_summaries.append(summary)
        logger.info(
            "[%d/%d] scenario=%d simulator_steps=%d records=%d seed=%s -> %s",
            ep_idx + 1, args.num_episodes,
            scenario_idx, simulator_steps, writer.record_count, ep_seed, episode_id,
        )

    env.close()
    scenario_distribution = Counter(str(s["scenario_index"]) for s in episode_summaries)
    repo_root = Path(__file__).resolve().parents[2]
    collector_source = Path(__file__).resolve()
    writer_source = collector_source.with_name("dataset_collector.py")
    manifest = {
        "dataset_version": RAW_DATASET_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "collector_git": _git_metadata(repo_root),
        "collector_source_sha256": {
            "collect_dataset.py": _sha256(collector_source),
            "dataset_collector.py": _sha256(writer_source),
        },
        "creation_timestamp_utc": created_at,
        "environment_configuration": _json_safe(env_config),
        "collection_configuration": {
            "policy": args.policy,
            "sampling_interval": args.sampling_interval,
            "expected_sim_hz": args.sim_hz,
            "base_seed": args.seed,
        },
        "number_of_episodes": len(episode_summaries),
        "number_of_records": sum(s["record_count"] for s in episode_summaries),
        "scenario_distribution": dict(sorted(scenario_distribution.items())),
        "episode_seeds": [s["seed"] for s in episode_summaries],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Collection finished. %d episodes and %d records saved to %s",
        manifest["number_of_episodes"], manifest["number_of_records"], root,
    )


if __name__ == "__main__":
    main()
