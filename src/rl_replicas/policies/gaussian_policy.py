import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal
from torch.optim import Optimizer

from rl_replicas.policies.stochastic_policy import StochasticPolicy


class GaussianPolicy(StochasticPolicy):
    """
    Gaussian policy

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    :param log_std: (Tensor) The standard deviation of the distribution.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer, log_std: nn.Parameter):
        super().__init__()

        self.network = network
        self.optimizer = optimizer
        self.log_std = log_std

    def forward(self, observation: Tensor) -> Independent:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation(s) from the environment
        :return: (Independent) The normal (also called Gaussian) distribution of action(s).
        """
        mean: Tensor = self.network(observation)
        std = torch.exp(self.log_std)
        # Use Independent for changing the shape of the result of log_prob()
        distribution: Independent = Independent(Normal(loc=mean, scale=std), 1)

        return distribution
