import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class QFunction(nn.Module):
    """
    The Q Function

    :param network: (nn.Module) The network.
    :param optimizer: (Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """
        Forward pass in the Q-function

        :param observation: (Tensor) The observation of the environment
        :param action: (Tensor) The action of the environment
        :return squeezeed_q_value: (Tensor) The Q-value(s)
        """
        input: Tensor = torch.cat([observation, action], -1)
        q_value: Tensor = self.network(input)
        squeezeed_q_value: Tensor = torch.squeeze(q_value, -1)

        return squeezeed_q_value
