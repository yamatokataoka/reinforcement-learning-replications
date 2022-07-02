from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.optim import Optimizer


class Policy(nn.Module, ABC):
    """
    Base policy class

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    @abstractmethod
    def predict(self, observation: Tensor) -> Tensor:
        """
        Selects action(s) given observation(s) from the environment

        :param observation: (Tensor) Observation(s) from the environment.
        :return: (Tensor) Action(s)
        """
        raise NotImplementedError
