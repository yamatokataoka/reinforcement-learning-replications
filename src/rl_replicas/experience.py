import numpy as np
from typing_extensions import TypedDict


class Experience(TypedDict):
    """
    Experience

    N: The number of episodes.
    L: The length of each episode (it may vary).
    A^*: The shape of single action step.
    O^*: The shape of single observation step.

    :param observations: (list[list[np.ndarray]]) A nested list of shape (N, L, O^*).
    :param actions: (list[list[np.ndarray]]) A nested list of shape (N, L, A^*).
    :param rewards: (list[list[float]]) A nested list of shape (N, L).
    :param last_observations: (list[np.ndarray]) A list of np.ndarray (N, O^*).
    :param dones: (list[bool]) A list with length (N).
    :param episode_returns: (list[float]) A list with length (N).
    :param episode_lengths: (list[int]) A list with length (N).
    """

    observations: list[list[np.ndarray]]
    actions: list[list[np.ndarray]]
    rewards: list[list[float]]
    last_observations: list[np.ndarray]
    dones: list[bool]
    episode_returns: list[float]
    episode_lengths: list[int]
