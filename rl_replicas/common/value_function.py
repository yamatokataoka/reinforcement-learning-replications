from typing import List, Tuple, Type

import torch
import torch.nn as nn
import gym

from rl_replicas.common.torch_net import mlp

class ValueFunction(nn.Module):
  """
  A Value Function

  :param optimizer: (torch.optim.Optimizer) The optimizer.
  :param network: (nn.Module) The network.
  """

  def __init__(
    self,
    network: nn.Module,
    optimizer: torch.optim.Optimizer
  ):
    super().__init__()

    self.network = network
    self.optimizer = optimizer

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    """
    Forward pass in value function

    :param observation: (torch.Tensor) The current observation of the environment
    """
    value: torch.Tensor = self.network(observation)

    return value
