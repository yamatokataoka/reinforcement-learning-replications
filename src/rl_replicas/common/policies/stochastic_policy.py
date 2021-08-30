from abc import abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Distribution

from rl_replicas.common.policies.policy import Policy


class StochasticPolicy(Policy):
    """
    The abstract base class for stochastic policies.

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__(network, optimizer)

    @abstractmethod
    def forward(self, observation: torch.Tensor) -> Distribution:
        """
        Forward pass in policy

        :param observation: (torch.Tensor) The observation of the environment
        :return: (Distribution) The distribution of action(s).
        """
        raise NotImplementedError

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (torch.Tensor) The observation of the environment
        :return: (torch.Tensor) the action(s)
        """
        with torch.no_grad():
            distribution: Distribution = self.forward(observation)

        action: torch.Tensor = distribution.sample()

        return action
