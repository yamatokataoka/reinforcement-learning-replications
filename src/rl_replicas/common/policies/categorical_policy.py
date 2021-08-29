import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_replicas.common.policies import StochasticPolicy


class CategoricalPolicy(StochasticPolicy):
    """
    The categorical policy

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__(network, optimizer)

    def forward(self, observation: torch.Tensor) -> Categorical:
        """
        Forward pass in policy

        :param observation: (torch.Tensor) The observation of the environment
        :return: (Distribution) The distribution of action(s).
        """
        logits: torch.Tensor = self.network(observation)
        distribution: Categorical = Categorical(logits=logits)

        return distribution
