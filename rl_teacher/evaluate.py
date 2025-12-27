"""Evaluate a trained model and generate videos."""
import argparse
import os

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

from rl_teacher.envs import make_with_torque_removed


def record_video(model, env, video_path, num_episodes=3):
    """Record videos of the trained agent."""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    all_frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        frames = []
        total_reward = 0
        done = False

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}, Steps = {len(frames)}")
        all_frames.extend(frames)

        # Add a small gap between episodes
        if frames and episode < num_episodes - 1:
            for _ in range(10):
                all_frames.append(frames[-1])

    # Save video
    if all_frames:
        imageio.mimwrite(video_path, all_frames, fps=30, codec='libx264', quality=8)
        print(f"Video saved to: {video_path}")
    else:
        print("No frames captured!")

    return video_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model and record video")
    parser.add_argument("-m", "--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("-e", "--env_id", default="Hopper-v5", help="Environment ID")
    parser.add_argument("-o", "--output", default="/tmp/rl_teacher_eval.mp4", help="Output video path")
    parser.add_argument("-n", "--num_episodes", default=3, type=int, help="Number of episodes to record")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    print(f"Creating environment {args.env_id}...")
    env = make_with_torque_removed(args.env_id)

    print(f"Recording {args.num_episodes} episodes...")
    record_video(model, env, args.output, args.num_episodes)

    env.close()


if __name__ == "__main__":
    main()
