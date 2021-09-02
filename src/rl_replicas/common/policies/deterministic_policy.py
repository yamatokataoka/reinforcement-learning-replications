import torch
import torch.nn as nn

from rl_replicas.common.policies.policy import Policy


class DeterministicPolicy(Policy):
    """
    The deterministic policy

    :param network: (nn.Module) The network.
    :param optimizer: (torch.optim.Optimizer) The optimizer.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__(network, optimizer)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in policy

        :param observation: (torch.Tensor) The observation of the environment
        :return: (torch.Tensor) The action(s).
        """
        action: torch.Tensor = self.network(observation)

        return action

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Selects the action(s) based on the observation of the environment.

        :param observation: (torch.Tensor) The observation(s) of the environment
        :return: (torch.Tensor) the action(s)
        """
        with torch.no_grad():
            action: torch.Tensor = self.forward(observation)

        return action
