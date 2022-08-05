from typing import List, Optional

import numpy as np


class Experience:
    """
    Experience

    N: The number of episodes.
    L: The length of each episode (it may vary).
    A^*: The shape of single action step.
    O^*: The shape of single observation step.

    :param observations: (Optional[List[List[np.ndarray]]]) A nested list of shape (N, L, O^*).
    :param actions: (Optional[List[List[np.ndarray]]]) A nested list of shape (N, L, A^*).
    :param rewards: (Optional[List[List[float]]]) A nested list of shape (N, L).
    :param last_observations: (Optional[List[np.ndarray]]) A list of np.ndarray (N, O^*).
    :param dones: (Optional[List[bool]]) A nested list shape (N, L).
    :param episode_returns: (Optional[List[float]]) A list with length (N).
    :param episode_lengths: (Optional[List[int]]) A list with length (N).
    """

    def __init__(
        self,
        observations: Optional[List[List[np.ndarray]]] = None,
        actions: Optional[List[List[np.ndarray]]] = None,
        rewards: Optional[List[List[float]]] = None,
        last_observations: Optional[List[np.ndarray]] = None,
        dones: Optional[List[List[bool]]] = None,
        episode_returns: Optional[List[float]] = None,
        episode_lengths: Optional[List[int]] = None,
    ):
        self.observations = observations if observations else []
        self.actions = actions if actions else []
        self.rewards = rewards if rewards else []
        self.last_observations = last_observations if last_observations else []
        self.dones = dones if dones else []
        self.episode_returns = episode_returns if episode_returns else []
        self.episode_lengths = episode_lengths if episode_lengths else []

    @property
    def observations_with_last_observation(self) -> List[List[np.ndarray]]:
        return [
            observations + [last_observation]
            for observations, last_observation in zip(
                self.observations, self.last_observations
            )
        ]

    @property
    def next_observations(self) -> List[List[np.ndarray]]:
        return [
            observations[1:] + [last_observation]
            for observations, last_observation in zip(
                self.observations, self.last_observations
            )
        ]

    @property
    def episode_dones(self) -> List[bool]:
        return [dones_per_episode[-1] for dones_per_episode in self.dones]

    @property
    def flattened_observations(self) -> List[np.ndarray]:
        return [
            observation
            for observations_per_episode in self.observations
            for observation in observations_per_episode
        ]

    @property
    def flattened_actions(self) -> List[np.ndarray]:
        return [
            action
            for actions_per_episode in self.actions
            for action in actions_per_episode
        ]

    @property
    def flattened_rewards(self) -> List[float]:
        return [
            reward
            for rewards_per_episode in self.rewards
            for reward in rewards_per_episode
        ]

    @property
    def flattened_next_observations(self) -> List[np.ndarray]:
        return [
            next_observation
            for next_observations_per_episode in self.next_observations
            for next_observation in next_observations_per_episode
        ]

    @property
    def flattened_dones(self) -> List[bool]:
        return [done for dones_per_episode in self.dones for done in dones_per_episode]
