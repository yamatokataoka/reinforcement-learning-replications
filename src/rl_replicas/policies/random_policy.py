import numpy as np
import torch
from gym import Space
from torch import Tensor

from rl_replicas.policies import Policy


class RandomPolicy(Policy):
    """
    Random policy

    :param action_space: (Space) Action space.
    """

    def __init__(self, action_space: Space):
        self.action_space = action_space

    def get_action_tensor(self, observation: Tensor) -> Tensor:
        _ = observation  # Don't use observation
        action = self.action_space.sample()
        return torch.from_numpy(action)

    def get_action_numpy(self, observation: np.ndarray) -> np.ndarray:
        _ = observation  # Don't use observation
        action = self.action_space.sample()
        return action
