"""
Run the trained PPO policy in the simulator with a live 3D window.
Watch the agent navigate in real time. Press H for help, R to reset, Esc to quit.

Usage:
    python run_ppo_live.py --model_path ./midterm_logs/PPO/ppo_seed0/best_model/best_model.zip
    python run_ppo_live.py --model_path ./midterm_logs/PPO/ppo_seed0/final_model.zip --episodes 5
"""
import argparse
import copy

from stable_baselines3 import PPO

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
from env_config import ENV_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO policy live in simulator")
    parser.add_argument(
        "--model_path", type=str, default="./midterm_logs/PPO/ppo_seed0/best_model/best_model.zip"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["training"] = False
    cfg["use_render"] = True
    cfg["window_size"] = (960, 960)
    cfg["vehicle_config"]["show_dest_mark"] = True
    cfg["vehicle_config"]["show_line_to_navi_mark"] = True

    env = SidewalkStaticMetaUrbanEnv(dict(start_seed=args.seed, log_level=50, **cfg))
    model = PPO.load(args.model_path)

    print(HELP_MESSAGE)
    print(f"\nRunning PPO from {args.model_path} for {args.episodes} episode(s).")
    print("Press H for help, R to reset scenario, Esc to quit.\n")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()

        print(f"Episode {ep + 1}/{args.episodes}: return={total_reward:.2f} "
              f"success={info.get('is_success', False)}")
        if ep < args.episodes - 1:
            env.reset(seed=args.seed + ep + 1)

    env.close()


if __name__ == "__main__":
    main()
