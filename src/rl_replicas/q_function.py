import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class QFunction(nn.Module):
    """
    Q Function

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        """
        Forward pass in the Q function

        :param observation: (Tensor) The observation(s) from the environment.
        :param action: (Tensor) Action(s) taken.
        :return squeezeed_q_value: (Tensor) The Q value(s).
        """
        input: Tensor = torch.cat([observation, action], -1)
        q_value: Tensor = self.network(input)
        squeezeed_q_value: Tensor = torch.squeeze(q_value, -1)

        return squeezeed_q_value
