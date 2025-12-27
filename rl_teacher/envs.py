"""Environment wrappers for RL-Teacher.

Updated for Gymnasium (2024+) with modern MuJoCo bindings.
"""
from copy import copy

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit


class TransparentWrapper(gym.Wrapper):
    """Passes missing attributes through the wrapper stack."""

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self.env, attr)


class MjViewer(TransparentWrapper):
    """Adds a space-efficient human_obs to info that allows rendering videos subsequently."""

    def __init__(self, env, fps=40):
        super().__init__(env)
        self.fps = fps

    def _get_full_obs(self):
        """Get the full MuJoCo state (qpos and qvel)."""
        return (
            copy(self.unwrapped.data.qpos.flatten()),
            copy(self.unwrapped.data.qvel.flatten()),
        )

    def _set_full_obs(self, obs):
        """Set the full MuJoCo state."""
        qpos, qvel = obs[0], obs[1]
        self.unwrapped.set_state(qpos, qvel)

    def render_full_obs(self, full_obs):
        """Render a frame from a saved state."""
        old_obs = self._get_full_obs()
        self._set_full_obs(full_obs)

        # Render the frame
        frame = self.unwrapped.render()

        self._set_full_obs(old_obs)
        return frame

    def step(self, action):
        human_obs = self._get_full_obs()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["human_obs"] = human_obs
        return obs, reward, terminated, truncated, info


class UseReward(TransparentWrapper):
    """Use a reward other than the normal one for an environment.

    We do this because humans cannot see torque penalties.
    """

    def __init__(self, env, reward_info_key):
        super().__init__(env)
        self.reward_info_key = reward_info_key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, info[self.reward_info_key], terminated, truncated, info


class NeverDone(TransparentWrapper):
    """Environment that never returns a terminated signal (but can still truncate)."""

    def __init__(self, env, bonus=lambda a, data: 0.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self.bonus(action, self.unwrapped.data)
        reward = reward + bonus
        # Never terminate, but allow truncation from TimeLimit
        return obs, reward, False, truncated, info


class TimeLimitTransparent(TimeLimit, TransparentWrapper):
    """TimeLimit wrapper that passes through attributes."""

    pass


def limit(env, t):
    """Apply a time limit to an environment."""
    return TimeLimitTransparent(env, max_episode_steps=t)


def task_by_name(name, short=False):
    """Create an environment by name."""
    if name == "reacher":
        return reacher(short=short)
    elif name == "humanoid":
        return humanoid()
    elif name == "hopper":
        return hopper(short=short)
    elif name in ["walker"]:
        return walker(short=short)
    elif name == "swimmer":
        return swimmer()
    elif name == "ant":
        return ant()
    elif name in ["cheetah", "halfcheetah"]:
        return cheetah(short=short)
    elif name in ["pendulum"]:
        return pendulum()
    elif name in ["doublependulum"]:
        return double_pendulum()
    else:
        raise ValueError(f"Unknown environment: {name}")


def make_with_torque_removed(env_id):
    """Create an environment with torque penalty removed (for human feedback)."""
    if "-v" in env_id:
        env_id = env_id[: env_id.index("-v")].lower()
    if env_id.startswith("short"):
        env_id = env_id[len("short") :]
        short = True
    else:
        short = False
    return task_by_name(env_id, short)


def get_timesteps_per_episode(env):
    """Get the maximum timesteps per episode for an environment."""
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    if hasattr(env, "spec") and env.spec is not None:
        return env.spec.max_episode_steps
    if hasattr(env, "env"):
        return get_timesteps_per_episode(env.env)
    return None


def reacher(short=False):
    """Create Reacher environment with distance reward only."""
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = UseReward(env, reward_info_key="reward_dist")
    env = MjViewer(fps=10, env=env)
    return limit(t=20 if short else 50, env=env)


def hopper(short=False):
    """Create Hopper environment."""

    def bonus(a, data):
        height = data.qpos[1]  # height of hopper
        ctrl_cost = 1e-3 * np.square(a).sum()
        return (height - 1) + ctrl_cost

    env = gym.make("Hopper-v5", render_mode="rgb_array")
    env = MjViewer(fps=40, env=env)
    env = NeverDone(bonus=bonus, env=env)
    return limit(t=300 if short else 1000, env=env)


def humanoid(standup=True, short=False):
    """Create Humanoid environment."""
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env = MjViewer(env=env, fps=40)
    env = UseReward(env, reward_info_key="reward_linvel")
    if standup:

        def bonus(a, data):
            height = data.qpos[2]  # z position
            return 5 * (height - 1)

        env = NeverDone(env, bonus=bonus)
    return limit(env, 300 if short else 1000)


def double_pendulum():
    """Create Inverted Double Pendulum environment."""

    def bonus(a, data):
        # site_xpos[0][2] is the z-position of the tip
        tip_height = data.site_xpos[0][2]
        return 10 * (tip_height - 1)

    env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
    env = MjViewer(env=env, fps=10)
    env = NeverDone(env, bonus)
    return limit(env, 50)


def pendulum():
    """Create Inverted Pendulum environment."""

    def bonus(a, data):
        angle = data.qpos[1]
        return -np.square(angle)

    env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
    env = MjViewer(env=env, fps=10)
    env = NeverDone(env, bonus)
    return limit(env, 25)


def cheetah(short=False):
    """Create HalfCheetah environment with run reward only."""
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env = UseReward(env, reward_info_key="reward_run")
    env = MjViewer(env=env, fps=20)
    return limit(env, 300 if short else 1000)


def swimmer(short=False):
    """Create Swimmer environment with forward reward only."""
    env = gym.make("Swimmer-v5", render_mode="rgb_array")
    env = UseReward(env, reward_info_key="reward_fwd")
    env = MjViewer(env=env, fps=40)
    return limit(env, 300 if short else 1000)


def ant(standup=True, short=False):
    """Create Ant environment with forward reward only."""
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = UseReward(env, reward_info_key="reward_forward")
    env = MjViewer(env=env, fps=20)
    if standup:

        def bonus(a, data):
            height = data.qpos[2]
            return height - 1.2

        env = NeverDone(env, bonus)
    return limit(env, 300 if short else 1000)


def walker(short=False):
    """Create Walker2d environment."""

    def bonus(a, data):
        height = data.qpos[1]
        ctrl_cost = 1e-3 * np.square(a).sum()
        return height - 2.0 + ctrl_cost

    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = MjViewer(env=env, fps=30)
    env = NeverDone(env, bonus)
    return limit(env, 300 if short else 1000)
