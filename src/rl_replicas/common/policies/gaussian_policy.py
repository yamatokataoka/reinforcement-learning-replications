import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from rl_replicas.common.policies.stochastic_policy import StochasticPolicy


class GaussianPolicy(StochasticPolicy):
    """
    The Gaussian policy

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    :param log_std: (torch.Tensor) The standard deviation of the distribution.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_std: nn.Parameter,
    ):
        super().__init__(network, optimizer)
        self.log_std = log_std

    def forward(self, observation: torch.Tensor) -> Independent:
        """
        Forward pass in policy

        :param observation: (torch.Tensor) The observation of the environment
        :return: (Independent) The normal (also called Gaussian) distribution of action(s).
        """
        mean: torch.Tensor = self.network(observation)
        std = torch.exp(self.log_std)
        # Use Independent for changing the shape of the result of log_prob()
        distribution: Independent = Independent(
            torch.distributions.Normal(loc=mean, scale=std), 1
        )

        return distribution
