import torch
from gym import Space
from torch import Tensor

from rl_replicas.policies import Policy


class RandomPolicy(Policy):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def predict(self, observation: Tensor) -> Tensor:
        _ = observation  # Don't use observation
        action = self.action_space.sample()
        return torch.from_numpy(action)
