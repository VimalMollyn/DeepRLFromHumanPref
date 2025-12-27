"""Main training script for RL-Teacher.

Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)
Updated for 2025 with PyTorch and Stable-Baselines3.
"""
import os
import os.path as osp
import random
from collections import deque
from time import time, sleep

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode, make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.segment_sampling import sample_segment_from_path, segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef
from rl_teacher.video import SegmentVideoRecorder

CLIP_LENGTH = 1.5


class TraditionalRLRewardPredictor:
    """Predictor that always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)
        return path["original_rewards"]

    def path_callback(self, path):
        pass


class ComparisonRewardPredictor:
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment."""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule):
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule

        # Set up bookkeeping
        self.recent_segments = deque(maxlen=200)
        self._frames_per_segment = CLIP_LENGTH * env.fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = 1e2
        self._elapsed_predictor_training_iters = 0

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the model
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        self.model = FullyConnectedMLP(self.obs_shape, self.act_shape).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def _predict_rewards(self, obs_segments, act_segments):
        """
        Predict rewards for segments.

        Args:
            obs_segments: tensor with shape = (batch_size, segment_length, *obs_shape)
            act_segments: tensor with shape = (batch_size, segment_length, *act_shape)

        Returns:
            tensor with shape = (batch_size, segment_length)
        """
        batch_size, segment_length = obs_segments.shape[:2]

        # Reshape to process all timesteps at once
        obs_flat = obs_segments.reshape(-1, *self.obs_shape)
        acts_flat = act_segments.reshape(-1, *self.act_shape)

        # Run through the network
        rewards = self.model(obs_flat, acts_flat)

        # Reshape back
        return rewards.reshape(batch_size, segment_length)

    def predict_reward(self, path):
        """Predict the reward for each step in a given path."""
        self.model.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(np.asarray([path["obs"]])).to(self.device)
            acts = torch.FloatTensor(np.asarray([path["actions"]])).to(self.device)
            q_value = self._predict_rewards(obs, acts)
        return q_value[0].cpu().numpy()

    def path_callback(self, path):
        """Called after each episode path."""
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        self.agent_logger.log_episode(path)

        # Sample segments from the path for future comparisons
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment:
            self.recent_segments.append(segment)

        # If we need more comparisons, build them from recent segments
        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            if len(self.recent_segments) >= 2:
                self.comparison_collector.add_segment_pair(
                    random.choice(self.recent_segments),
                    random.choice(self.recent_segments),
                )

        # Train predictor periodically
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            self.train_predictor()
            self._steps_since_last_training = 0

    def train_predictor(self):
        """Train the reward predictor on labeled comparisons."""
        self.comparison_collector.label_unlabeled_comparisons()

        if len(self.comparison_collector.labeled_decisive_comparisons) == 0:
            return

        minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(
            self.comparison_collector.labeled_decisive_comparisons, minibatch_size
        )

        # Prepare batch data
        left_obs = torch.FloatTensor(
            np.asarray([comp["left"]["obs"] for comp in labeled_comparisons])
        ).to(self.device)
        left_acts = torch.FloatTensor(
            np.asarray([comp["left"]["actions"] for comp in labeled_comparisons])
        ).to(self.device)
        right_obs = torch.FloatTensor(
            np.asarray([comp["right"]["obs"] for comp in labeled_comparisons])
        ).to(self.device)
        right_acts = torch.FloatTensor(
            np.asarray([comp["right"]["actions"] for comp in labeled_comparisons])
        ).to(self.device)
        labels = torch.LongTensor(
            np.asarray([comp["label"] for comp in labeled_comparisons])
        ).to(self.device)

        # Training step
        self.model.train()
        self.optimizer.zero_grad()

        # Predict rewards for both segments
        left_rewards = self._predict_rewards(left_obs, left_acts)
        right_rewards = self._predict_rewards(right_obs, right_acts)

        # Sum rewards over segment length
        left_sum = left_rewards.sum(dim=1)
        right_sum = right_rewards.sum(dim=1)

        # Create logits for classification
        logits = torch.stack([left_sum, right_sum], dim=1)

        # Compute loss
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()

        self._elapsed_predictor_training_iters += 1
        self._write_training_summaries(loss.item())

    def _write_training_summaries(self, loss):
        """Write training summaries to TensorBoard."""
        self.agent_logger.log_simple("predictor/loss", loss)

        # Calculate correlation between true and predicted reward
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:
            self.model.eval()
            with torch.no_grad():
                validation_obs = torch.FloatTensor(
                    np.asarray([path["obs"] for path in recent_paths])
                ).to(self.device)
                validation_acts = torch.FloatTensor(
                    np.asarray([path["actions"] for path in recent_paths])
                ).to(self.device)
                q_value = self._predict_rewards(validation_obs, validation_acts)

            ep_reward_pred = q_value.sum(dim=1).cpu().numpy()
            reward_true = np.asarray([path["original_rewards"] for path in recent_paths])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons)
        )


class RewardPredictorWrapper(gym.Wrapper):
    """Gymnasium wrapper that replaces the environment reward with predicted reward."""

    def __init__(self, env, predictor):
        super().__init__(env)
        self.predictor = predictor
        self._current_path = None
        self._reset_path()

    def _reset_path(self):
        self._current_path = {
            "obs": [],
            "actions": [],
            "original_rewards": [],
            "human_obs": [],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_path()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store step data
        self._current_path["obs"].append(obs)
        self._current_path["actions"].append(action)
        self._current_path["original_rewards"].append(reward)
        self._current_path["human_obs"].append(info.get("human_obs"))

        # On episode end, compute predicted rewards and call callback
        if terminated or truncated:
            path = {k: np.array(v) for k, v in self._current_path.items()}
            predicted_rewards = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)

            # Return the last predicted reward
            if len(predicted_rewards) > 0:
                reward = predicted_rewards[-1]

            self._reset_path()

        return obs, reward, terminated, truncated, info


class RewardPredictorCallback(BaseCallback):
    """Stable-Baselines3 callback for integrating with the reward predictor."""

    def __init__(self, predictor, verbose=0):
        super().__init__(verbose)
        self.predictor = predictor

    def _on_step(self):
        return True


def make_env_fn(env_id, make_env_func, seed=0):
    """Create a function that makes an environment (without predictor wrapper)."""
    def _init():
        env = make_env_func(env_id)
        env.reset(seed=seed)
        return env
    return _init


def train_with_ppo(
    env_id,
    make_env,
    predictor,
    summary_writer,
    num_timesteps,
    seed=1,
    workers=4,
    use_predicted_reward=False,
):
    """Train an agent using PPO from Stable-Baselines3."""
    print(f"Training with PPO for {num_timesteps} timesteps...")

    if use_predicted_reward:
        # For learned reward, use DummyVecEnv with reward wrapper (can't pickle predictor)
        def make_wrapped_env():
            env = make_env(env_id)
            env.reset(seed=seed)
            return RewardPredictorWrapper(env, predictor)

        env = DummyVecEnv([make_wrapped_env])
        print("Using single environment with reward predictor (DummyVecEnv)")
    else:
        # For baseline RL, can use SubprocVecEnv for parallelism
        if workers > 1:
            env = SubprocVecEnv([
                make_env_fn(env_id, make_env, seed + i) for i in range(workers)
            ])
            print(f"Using {workers} parallel environments (SubprocVecEnv)")
        else:
            env = DummyVecEnv([make_env_fn(env_id, make_env, seed)])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=osp.expanduser("~/tb/rl-teacher/"),
        seed=seed,
    )

    # Create callback
    callback = RewardPredictorCallback(predictor) if predictor else None

    # Train
    model.learn(total_timesteps=num_timesteps, callback=callback)

    # Save the model
    save_path = osp.expanduser(f"~/tb/rl-teacher/models/{env_id}_{num_timesteps}.zip")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    env.close()
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RL-Teacher: Deep RL from Human Preferences")
    parser.add_argument("-e", "--env_id", required=True, help="Environment ID (e.g., Hopper-v5)")
    parser.add_argument("-p", "--predictor", required=True, choices=["rl", "synth", "human"],
                        help="Predictor type: rl (true reward), synth (synthetic labels), human (human labels)")
    parser.add_argument("-n", "--name", required=True, help="Experiment name")
    parser.add_argument("-s", "--seed", default=1, type=int, help="Random seed")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of parallel workers")
    parser.add_argument("-l", "--n_labels", default=None, type=int, help="Total number of labels to collect")
    parser.add_argument("-L", "--pretrain_labels", default=None, type=int, help="Number of pretraining labels")
    parser.add_argument("-t", "--num_timesteps", default=5e6, type=int, help="Total training timesteps")
    parser.add_argument("-i", "--pretrain_iters", default=10000, type=int, help="Predictor pretraining iterations")
    parser.add_argument("-V", "--no_videos", action="store_true", help="Disable video recording")
    args = parser.parse_args()

    print("Setting things up...")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.name, int(time()))
    summary_writer = make_summary_writer(run_name)

    env = make_with_torque_removed(env_id)

    num_timesteps = int(args.num_timesteps)
    experiment_name = slugify(args.name)

    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else (args.n_labels // 4 if args.n_labels else 50)

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels,
            )
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()
        elif args.predictor == "human":
            bucket = os.environ.get("RL_TEACHER_GCS_BUCKET")
            assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(env_id, experiment_name=experiment_name)
        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
        )

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id,
            make_with_torque_removed,
            n_desired_segments=pretrain_labels * 2,
            clip_length_in_seconds=CLIP_LENGTH,
            workers=args.workers,
        )

        for i in range(pretrain_labels):
            comparison_collector.add_segment_pair(
                pretrain_segments[i], pretrain_segments[i + pretrain_labels]
            )

        # Wait for labels
        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synth":
                print("%s synthetic labels generated... " % len(comparison_collector.labeled_comparisons))
            elif args.predictor == "human":
                print(
                    "%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... "
                    % (len(comparison_collector.labeled_comparisons), pretrain_labels)
                )
                sleep(5)

        # Pretrain the predictor
        for i in range(args.pretrain_iters):
            predictor.train_predictor()
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos
    if not args.no_videos and args.predictor != "rl":
        predictor = SegmentVideoRecorder(
            predictor, env, save_dir=osp.join("/tmp/rl_teacher_vids", run_name)
        )

    # Train with PPO
    print("Starting joint training of predictor and agent")
    use_predicted_reward = args.predictor != "rl"
    train_with_ppo(
        env_id=env_id,
        make_env=make_with_torque_removed,
        predictor=predictor if use_predicted_reward else None,
        summary_writer=summary_writer,
        num_timesteps=num_timesteps,
        seed=args.seed,
        workers=args.workers,
        use_predicted_reward=use_predicted_reward,
    )


if __name__ == "__main__":
    main()
