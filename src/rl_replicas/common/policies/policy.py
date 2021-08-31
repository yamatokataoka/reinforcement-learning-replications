from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Policy(nn.Module, ABC):
    """
    The base policy class

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    @abstractmethod
    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (torch.Tensor) The observation(s) of the environment
        :return: (torch.Tensor) the action(s)
        """
        raise NotImplementedError
