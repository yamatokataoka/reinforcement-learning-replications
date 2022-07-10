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
    :param dones: (Optional[List[bool]]) A list with length (N).
    :param episode_returns: (Optional[List[float]]) A list with length (N).
    :param episode_lengths: (Optional[List[int]]) A list with length (N).
    """

    def __init__(
        self,
        observations: Optional[List[List[np.ndarray]]] = None,
        actions: Optional[List[List[np.ndarray]]] = None,
        rewards: Optional[List[List[float]]] = None,
        last_observations: Optional[List[np.ndarray]] = None,
        dones: Optional[List[bool]] = None,
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
    def next_observations(self) -> List[List[np.ndarray]]:
        return [
            observations[1:] + [last_observation]
            for observations, last_observation in zip(
                self.observations,
                self.last_observations,
            )
        ]
