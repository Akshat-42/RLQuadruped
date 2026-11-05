import argparse
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from quadruped_rl.envs.quadruped_env import QuadrupedEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize trained policy in MuJoCo")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip policy checkpoint")
    parser.add_argument("--target-velocity", type=float, default=0.4, help="Target velocity (m/s)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    args = parser.parse_args()

    # Create Environment
    env = QuadrupedEnv(target_velocity=args.target_velocity)

    # Load Model
    print(f"Loading trained policy from: {args.model_path}")
    model = PPO.load(args.model_path, env=env)

    # Launch MuJoCo Viewer
    print("Launching MuJoCo passive viewer. Press ESC or close window to exit.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            steps = 0
            t_start = time.time()

            while viewer.is_running():
                step_start = time.time()

                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                # Sync visualizer
                viewer.sync()

                # Maintain 50 Hz policy rate
                elapsed = time.time() - step_start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)

                if terminated or truncated:
                    break

            ep_duration = time.time() - t_start
            print(f"Episode {ep + 1}/{args.episodes}: Reward = {ep_reward:.2f}, Steps = {steps}, Duration = {ep_duration:.1f}s")
            if not viewer.is_running():
                break

    env.close()


if __name__ == "__main__":
    main()
