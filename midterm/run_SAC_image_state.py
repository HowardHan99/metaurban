import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from train_sac_image_state_test import (
    build_image_state_env_config,
    make_env,
)

CHECKPOINT_PATH = "./midterm_logs/SAC_image_state/sac_imgstate_seed0_0415_1149/checkpoints/sac_imgstate_260000_steps.zip"
SEED = 0
IMAGE_W = 80
IMAGE_H = 60
env_cfg = build_image_state_env_config(IMAGE_W, IMAGE_H, training=False)
env_cfg["use_render"] = True

env = DummyVecEnv([
    make_env(
        env_cfg,
        seed=SEED,
        image_width=IMAGE_W,
        image_height=IMAGE_H,
        use_idle_penalty=False
    )
])

model = SAC.load(CHECKPOINT_PATH, env=env, device="auto")

obs = env.reset()

for step in range(20000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done:
        print("Episode done")
        obs = env.reset()