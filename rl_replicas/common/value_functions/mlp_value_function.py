from typing import List, Tuple, Type

import torch
import torch.nn as nn
import gym

from rl_replicas.common.torch_net import mlp

class MLPValueFunction(nn.Module):
  """
  A Value Function

  :param observation_space: (gym.spaces.Space) The observation space of the environment
  :param learning_rate: (float) The learning rate for the optimizer
  :param activation_function: (Type[nn.Module]) Activation function
  :param optimizer_class: (Type[torch.optim.Optimizer]) The optimizer class, `torch.optim.Adam` by default
  """

  def __init__(
    self,
    observation_space: gym.spaces.Space,
    learning_rate: float = 1e-3,
    activation_function: Type[nn.Module] = nn.Tanh,
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
  ):
    super().__init__()

    self.observation_space = observation_space
    self.learning_rate = learning_rate
    self.activation_function = activation_function
    self.optimizer_class = optimizer_class

    # Default
    self.network_architecture: List[int] = [64, 64]

    self._setup_network()
    self._setup_optimizer()

  def _setup_network(self):
    input_size: int = self.observation_space.shape[0]
    output_size: int = 1

    network_sizes: List[int] = [input_size] + self.network_architecture + [output_size]

    self.network: nn.Module = mlp(network_sizes, self.activation_function)

  def _setup_optimizer(self):
    self.optimizer: torch.optim.Optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    """
    Forward pass in value function

    :param observation: (torch.Tensor) The current observation of the environment
    """
    value: torch.Tensor = self.network(observation)

    return value
