from abc import ABC, abstractmethod

from torch import Tensor, nn


class Policy(nn.Module, ABC):
    """
    Base policy class
    """

    @abstractmethod
    def predict(self, observation: Tensor) -> Tensor:
        """
        Predict action(s) given observation(s) from the environment

        :param observation: (Tensor) Observation(s) from the environment.
        :return: (Tensor) Action(s)
        """
        raise NotImplementedError
