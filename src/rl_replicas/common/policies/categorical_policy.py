import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from rl_replicas.common.policies.stochastic_policy import StochasticPolicy


class CategoricalPolicy(StochasticPolicy):
    """
    The categorical policy

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__(network, optimizer)

    def forward(self, observation: Tensor) -> Categorical:
        """
        Forward pass in policy

        :param observation: (Tensor) The observation of the environment
        :return: (Distribution) The distribution of action(s).
        """
        logits: Tensor = self.network(observation)
        distribution: Categorical = Categorical(logits=logits)

        return distribution
