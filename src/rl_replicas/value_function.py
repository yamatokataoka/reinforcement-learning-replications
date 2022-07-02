from torch import Tensor, nn
from torch.optim import Optimizer


class ValueFunction(nn.Module):
    """
    The Value Function

    :param optimizer: (Optimizer) The optimizer.
    :param network: (nn.Module) The network.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass in the value function

        :param observation: (Tensor) Observation(s) from the environment.
        :return value: (Tensor) The value(s).
        """
        value: Tensor = self.network(observation)

        return value
