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
        super().__init__(network, optimizer)

    def forward(self, observation: Tensor) -> Tensor:
        """
        Forward pass in policy

        :param observation: (Tensor) Observation of the environment.
        :return: (Tensor) Action(s).
        """
        action: Tensor = self.network(observation)

        return action

    def predict(self, observation: Tensor) -> Tensor:
        """
        Selects action(s) given observation(s) from the environment

        :param observation: (Tensor) Observation(s) from the environment.
        :return: (Tensor) Action(s).
        """
        with torch.no_grad():
            action: Tensor = self.forward(observation)

        return action
