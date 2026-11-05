import argparse
import os
import sys

# Ensure package root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from quadruped_rl.envs.quadruped_env import QuadrupedEnv


def make_env(xml_path=None, target_velocity=0.4, rank=0, seed=0):
    def _init():
        env = QuadrupedEnv(xml_path=xml_path, target_velocity=target_velocity)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train 8-DOF Quadruped with PPO in MuJoCo using SB3")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--target-velocity", type=float, default=0.4, help="Target forward velocity (m/s)")
    parser.add_argument("--save-name", type=str, default="quadruped_ppo_mg996r", help="Policy checkpoint name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "policies")
    tb_log_dir = os.path.join(base_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"=== Starting Quadruped RL Training ===")
    print(f"Algorithm: PPO (Stable-Baselines3)")
    print(f"Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Parallel Envs: {args.num_envs}")
    print(f"Target Velocity: {args.target_velocity} m/s")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Checkpoint output: {models_dir}")

    # Vectorized Environments
    env_fns = [make_env(target_velocity=args.target_velocity, rank=i, seed=args.seed) for i in range(args.num_envs)]
    if args.num_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    # Evaluation Environment
    eval_env = DummyVecEnv([make_env(target_velocity=args.target_velocity, rank=999, seed=args.seed)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.num_envs, 1),
        save_path=models_dir,
        name_prefix=args.save_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=os.path.join(models_dir, "eval_logs"),
        eval_freq=max(20_000 // args.num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # PPO Policy Setup
    # Tuned for quadruped locomotion with servo dynamics
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ELU,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        device=args.device,
        verbose=1,
    )

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=args.save_name,
        reset_num_timesteps=True,
    )

    # Final Save
    final_save_path = os.path.join(models_dir, f"{args.save_name}_final")
    model.save(final_save_path)
    print(f"\nTraining completed! Saved final policy to: {final_save_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
