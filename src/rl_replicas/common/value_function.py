import torch
from torch import Tensor, nn


class ValueFunction(nn.Module):
    """
    The Value Function

    :param optimizer: (torch.optim.Optimizer) The optimizer.
    :param network: (nn.Module) The network.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass in the value function

        :param observation: (Tensor) The current observation of the environment
        :return value: (Tensor) The value
        """
        value: Tensor = self.network(observation)

        return value
