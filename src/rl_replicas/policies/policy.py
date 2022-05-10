from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.optim import Optimizer


class Policy(nn.Module, ABC):
    """
    The base policy class

    :param network: (nn.Module) The network.
    :param optimizer: (Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    @abstractmethod
    def predict(self, observation: Tensor) -> Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (Tensor) The observation(s) of the environment
        :return: (Tensor) the action(s)
        """
        raise NotImplementedError
