from typing import List, Type

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym

from rl_replicas.common.torch_net import mlp

class MLPPolicy(nn.Module):
  """
  A MLP policy.

  The policy network selects action based on the observation of the environment.
  It uses a PyTorch neural network module to fit the function of pi(s).

  :param observation_space: (gym.spaces.Space) The observation space of the environment
  :param action_space: (gym.spaces.Space) The action space of the environment
  :param learning_rate: (float) The learning rate for the optimizer
  :param activation_function: (Type[nn.Module]) Activation function
  :param optimizer_class: (Type[torch.optim.Optimizer]) The optimizer class, `torch.optim.Adam` by default
  """

  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    learning_rate: float = 3e-4,
    activation_function: Type[nn.Module] = nn.Tanh,
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
  ):
    super().__init__()

    self.observation_space = observation_space
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.activation_function = activation_function
    self.optimizer_class = optimizer_class

    # Default
    self.network_architecture: List[int] = [64, 64]

    self._setup_network()
    self._setup_optimizer()

  def _setup_network(self):
    input_size: int = self.observation_space.shape[0]
    output_size: int = self.action_space.n

    network_sizes: List[int] = [input_size] + self.network_architecture + [output_size]

    self.network: nn.Module = mlp(network_sizes, self.activation_function)

  def _setup_optimizer(self):
    self.optimizer: torch.optim.Optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

  def forward(self, observation: torch.Tensor) -> Categorical:
    """
    Forward pass in policy

    :param observation: (torch.Tensor) The current observation of the environment
    """
    logits: torch.Tensor = self.network(observation)
    distribution: Categorical = Categorical(logits=logits)

    return distribution

  def predict(self, observation: torch.Tensor) -> torch.Tensor:
    """
    Selects action based on the observation of the environment.

    :param observation: (torch.Tensor) The current observation of the environment
    """
    with torch.no_grad():
      logits: torch.Tensor = self.network(observation)
    distribution: Categorical = Categorical(logits=logits)

    action: torch.Tensor = distribution.sample()

    return action
