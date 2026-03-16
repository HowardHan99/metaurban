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
                    policy = IDMPolicy(agent, self._env.np_random)
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
    p.add_argument("--num-scenarios", type=int, default=1000,
                   help="Total number of scenarios in the environment pool.")
    p.add_argument("--spawn-human-num", type=int, default=30,
                   help="Number of pedestrians spawned per scenario.")
    p.add_argument("--horizon", type=int, default=1000,
                   help="Maximum steps per episode.")
    p.add_argument("--skip-clips", action="store_true",
                   help="Skip the sliding-window clip extraction step.")
    p.add_argument("--seed", type=int, default=None,
                   help="Base random seed (None = fully random).")
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
        spawn_human_num=args.spawn_human_num,
        spawn_wheelchairman_num=max(1, args.spawn_human_num // 20),
        horizon=args.horizon,
        num_scenarios=args.num_scenarios,
        decision_repeat=1,
        window_size=(960, 960) if args.use_render else (84, 84),
        vehicle_config=dict(show_lidar=False, show_navi_mark=True),
    )

    # ------------------------------------------------------------------
    # Instantiate environment
    # ------------------------------------------------------------------
    logger.info("Importing MetaUrban …")
    try:
        from metaurban.envs.sidewalk_dynamic_env import SidewalkDynamicMetaUrbanEnv
    except ImportError as exc:
        logger.error("Cannot import MetaUrban: %s", exc)
        sys.exit(1)

    logger.info("Creating environment …")
    env = SidewalkDynamicMetaUrbanEnv(env_config)

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
