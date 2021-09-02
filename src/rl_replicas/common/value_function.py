import torch
import torch.nn as nn


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

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in the value function

        :param observation: (torch.Tensor) The current observation of the environment
        :return value: (torch.Tensor) The value
        """
        value: torch.Tensor = self.network(observation)

        return value
