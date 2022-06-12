from typing import List

import numpy as np
from typing_extensions import TypedDict


class Experience(TypedDict):
    observations: List[List[np.ndarray]]
    actions: List[List[np.ndarray]]
    rewards: List[List[float]]
    last_observations: List[np.ndarray]
    dones: List[bool]
    episode_returns: List[float]
    episode_lengths: List[int]
