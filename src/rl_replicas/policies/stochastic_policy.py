from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Optimizer

from rl_replicas.policies.policy import Policy


class StochasticPolicy(Policy):
    """
    Abstract base class for stochastic policies

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__(network, optimizer)

    @abstractmethod
    def forward(self, observation: Tensor) -> Distribution:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation from the environment.
        :return: (Distribution) The distribution of action(s).
        """
        raise NotImplementedError

    def predict(self, observation: Tensor) -> Tensor:
        with torch.no_grad():
            distribution: Distribution = self.forward(observation)

        action: Tensor = distribution.sample()

        return action
