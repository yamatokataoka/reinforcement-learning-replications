from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution

from rl_replicas.policies.policy import Policy


class StochasticPolicy(Policy):
    """
    Abstract base class for stochastic policies
    """

    @abstractmethod
    def forward(self, observation: Tensor) -> Distribution:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation from the environment.
        :return: (Distribution) The distribution of action(s).
        """
        raise NotImplementedError

    def get_action_tensor(self, observation: Tensor) -> Tensor:
        with torch.no_grad():
            distribution: Distribution = self.forward(observation)

        action: Tensor = distribution.sample()

        return action

    def get_action_numpy(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor: Tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            distribution: Distribution = self.forward(observation_tensor)

        action: Tensor = distribution.sample()

        return action.detach().numpy()
