from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor, nn


class Policy(nn.Module, ABC):
    """
    Base policy class
    """

    @abstractmethod
    def get_action_tensor(self, observation: Tensor) -> Tensor:
        """
        Get action(s) given observation(s) from the environment

        :param observation: (Tensor) Observation(s) from the environment.
        :return: (Tensor) Action(s)
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_numpy(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action(s) given observation(s) from the environment

        :param observation: (np.ndarray) Observation(s) from the environment.
        :return: (np.ndarray) Action(s)
        """
        raise NotImplementedError
