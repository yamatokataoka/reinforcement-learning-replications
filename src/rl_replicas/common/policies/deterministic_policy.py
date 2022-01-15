import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rl_replicas.common.policies.policy import Policy


class DeterministicPolicy(Policy):
    """
    The deterministic policy

    :param network: (nn.Module) The network.
    :param optimizer: (Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__(network, optimizer)

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass in policy

        :param observation: (Tensor) The observation of the environment
        :return: (Tensor) The action(s).
        """
        action: Tensor = self.network(observation)

        return action

    def predict(self, observation: Tensor) -> Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (Tensor) The observation(s) of the environment
        :return: (Tensor) the action(s)
        """
        with torch.no_grad():
            action: Tensor = self.forward(observation)

        return action
