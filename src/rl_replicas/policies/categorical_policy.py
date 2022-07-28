from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from rl_replicas.policies.stochastic_policy import StochasticPolicy


class CategoricalPolicy(StochasticPolicy):
    """
    Categorical policy

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor) -> Categorical:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation of the environment.
        :return: (Categorical) The distribution of action(s).
        """
        logits: Tensor = self.network(observation)
        distribution: Categorical = Categorical(logits=logits)

        return distribution
