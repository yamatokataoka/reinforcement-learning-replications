from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Optimizer

from rl_replicas.common.policies.policy import Policy


class StochasticPolicy(Policy):
    """
    The abstract base class for stochastic policies.

    :param network: (nn.Module) The network.
    :param optimizer: (Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__(network, optimizer)

    @abstractmethod
    def forward(self, observation: Tensor) -> Distribution:
        """
        Forward pass in policy

        :param observation: (Tensor) The observation of the environment
        :return: (Distribution) The distribution of action(s).
        """
        raise NotImplementedError

    def predict(self, observation: Tensor) -> Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (Tensor) The observation of the environment
        :return: (Tensor) the action(s)
        """
        with torch.no_grad():
            distribution: Distribution = self.forward(observation)

        action: Tensor = distribution.sample()

        return action
