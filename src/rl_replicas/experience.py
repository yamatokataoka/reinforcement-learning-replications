import numpy as np
from typing_extensions import TypedDict


class Experience(TypedDict):
    observations: list[list[np.ndarray]]
    actions: list[list[np.ndarray]]
    rewards: list[list[float]]
    last_observations: list[np.ndarray]
    dones: list[bool]
    episode_returns: list[float]
    episode_lengths: list[int]
