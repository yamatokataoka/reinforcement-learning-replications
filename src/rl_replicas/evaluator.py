from typing import List, Optional, Tuple

import gym
import numpy as np

from rl_replicas.policies import Policy


class Evaluator:
    """
    Evaluator

    :param seed: (int) Seed.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def evaluate(
        self, policy: Policy, env: gym.Env, num_episodes: int
    ) -> Tuple[List[float], List[int]]:
        """
        Evaluate the policy running evaluation episodes.

        :param policy: (Policy) Policy.
        :param env: (gym.Env) Environment.
        :param num_episodes: (int) The number of episodes.
        """
        episode_returns: List[float] = []
        episode_lengths: List[int] = []

        observation: np.ndarray = env.reset(seed=self.seed)

        for _ in range(num_episodes):
            done: bool = False
            episode_return: float = 0.0
            episode_length: int = 0

            while not done:
                action: np.ndarray = policy.get_action_numpy(observation)

                reward: float
                observation, reward, done, _ = env.step(action)

                episode_return += reward
                episode_length += 1

            observation = env.reset()

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        return episode_returns, episode_lengths
