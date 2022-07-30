import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rl_replicas.policies.policy import Policy


class DeterministicPolicy(Policy):
    """
    Deterministic policy

    :param network: (nn.Module) Network.
    :param optimizer: (Optimizer) Optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: Optimizer):
        super().__init__()

        self.network = network
        self.optimizer = optimizer

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation of the environment.
        :return: (Tensor) Action(s).
        """
        action: Tensor = self.network(observation)

        return action

    def get_action_tensor(self, observation: Tensor) -> Tensor:
        with torch.no_grad():
            action: Tensor = self.forward(observation)

        return action

    def get_action_numpy(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor: Tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            action: Tensor = self.forward(observation_tensor)

        return action.detach().numpy()
