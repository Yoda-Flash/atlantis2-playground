from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Atlantis2-v5", render_mode='human')

class Atlantis2Agent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        print(env.observation_space.shape[2])
        self.q_values = np.zeros((env.observation_space.shape[0], env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.total_rewards_episode = list()

    def get_action(self, obs: Box[0, 255, (210, 160, 3), np.uint8]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: Box[0, 255, (210, 160, 3), np.uint8],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Box[0, 255, (210, 160, 3), np.uint8],
    ):
        """Updates the Q-value of an action."""
        self.q_values[obs, action] = (1 - self.lr) * self.q_values[obs, action] + self.lr * (reward + self.discount_factor*max(self.q_values[next_obs, :]))
        self.total_rewards_episode.append(reward)
        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # temporal_difference = (
        #     reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        # )
        #
        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )
        # self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
