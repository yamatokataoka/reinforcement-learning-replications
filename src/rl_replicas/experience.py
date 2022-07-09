from typing import List

import numpy as np
from typing_extensions import TypedDict


class Experience(TypedDict):
    """
    Experience

    N: The number of episodes.
    L: The length of each episode (it may vary).
    A^*: The shape of single action step.
    O^*: The shape of single observation step.

    :param observations: (List[List[np.ndarray]]) A nested list of shape (N, L, O^*).
    :param actions: (List[List[np.ndarray]]) A nested list of shape (N, L, A^*).
    :param rewards: (List[List[float]]) A nested list of shape (N, L).
    :param last_observations: (List[np.ndarray]) A list of np.ndarray (N, O^*).
    :param dones: (List[bool]) A list with length (N).
    :param episode_returns: (List[float]) A list with length (N).
    :param episode_lengths: (List[int]) A list with length (N).
    """

    observations: List[List[np.ndarray]]
    actions: List[List[np.ndarray]]
    rewards: List[List[float]]
    last_observations: List[np.ndarray]
    dones: List[bool]
    episode_returns: List[float]
    episode_lengths: List[int]
