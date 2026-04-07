#!/usr/bin/env python3
"""
Example: Social Navigation with Robot on Sidewalk + Group Clusters

This example shows how to:
1. Initialize the robot on the sidewalk (not on the road)
2. Create group pedestrian clusters that stand and chat on the sidewalk
3. Visualize the scene with onscreen rendering

Run this to see the robot and pedestrian groups interact!
"""

from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

# Configure the environment
config = dict(
    # ===== Render and Display =====
    use_render=True,           # Open rendering window
    window_size=(960, 960),
    manual_control=True,    # Set to True for manual keyboard control (WASD)
    
    # ===== Robot Spawn Location =====
    spawn_robot_on_sidewalk=True,  # NEW: Robot starts on sidewalk, not road
    
    # ===== Group Pedestrians (Chatting in Clusters) =====
    group_ped_pair_num=4,      # 4 pairs = 8 pedestrians standing together
    
    # ===== Regular Pedestrians =====
    spawn_human_num=20,        # Background crowd
    crossing_ped_num=0,        # No aggressive crossing pedestrians
    vulnerable_ped_num=0,      # No vulnerable pedestrians
    
    # ===== Episode Configuration =====
    horizon=500,               # Max steps per episode
    num_scenarios=100,
    
    # ===== Scene Type =====
    scene_type="commercial",   # Choose from: commercial, commute, leisure, constrained
)

# Create environment
env = SocialDynamicMetaUrbanEnv(config)

# Run the visualization
print("🚗 Robot spawning on sidewalk...")
print("👥 4 group pairs (8 pedestrians) clustering on sidewalk...")
print("Opening visualization window. Press ESC to exit or wait for episode to end.\n")

obs, info = env.reset()

for step in range(config["horizon"]):
    # Random policy (can replace with your own controller)
    # action = env.action_space.sample()
    action = [0,0]  # Move forward with moderate speed, no steering
    obs, reward, terminated, truncated, info = env.step(action)
    
    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}/{config['horizon']}")
    
    if terminated or truncated:
        print(f"Episode ended at step {step + 1}: {info.get('reason', 'unknown')}")
        break

env.close()
print("✓ Visualization complete!")
