import numpy as np
from typing_extensions import TypedDict


class Experience(TypedDict):
    """
    Experience

    N: Number of episodes
    L: length of each episode (it may vary)
    A^*: Shape of single action step
    O^*: Shape of single observation step

    :param observations: (list[list[np.ndarray]]) A nested list of shape (N, L, O^*)
    :param actions: (list[list[np.ndarray]]) A nested list of shape (N, L, A^*)
    :param rewards: (list[list[float]]) A nested list of shape (N, L)
    :param last_observations: (list[np.ndarray]) A list of shape (N, O^*)
    :param dones: (list[bool]) A list of shape (N)
    :param episode_returns: (list[float]) A list of shape (N)
    :param episode_lengths: (list[int]) A list of shape (N)
    """
    observations: list[list[np.ndarray]]
    actions: list[list[np.ndarray]]
    rewards: list[list[float]]
    last_observations: list[np.ndarray]
    dones: list[bool]
    episode_returns: list[float]
    episode_lengths: list[int]
