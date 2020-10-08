from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym

from rl_replicas.common.torch_net import mlp

class ActorCriticPolicy(nn.Module):
  """
  Actor critic policy object

  :param observation_space: (gym.spaces.Space) The observation space of the environment
  :param action_space: (gym.spaces.Space) The action space of the environment
  :param learning_rate: (float) The learning rate for the optimizer
  :param activation_fn: (Type[nn.Module]) Activation function
  :param optimizer_class: (Type[torch.optim.Optimizer]) The optimizer class, `torch.optim.Adam` by default
  """

  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    learning_rate: float,
    activation_fn: Type[nn.Module] = nn.Tanh,
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
  ):
    super().__init__()

    self.observation_space = observation_space
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.activation_fn = activation_fn
    self.optimizer_class = optimizer_class

    # Default
    self.policy_net_arch: List[int] = [64, 64]
    self.value_fn_net_arch: List[int] = [64, 64]

    self._setup_policy()
    self._setup_value_fn()
    self._setup_optimizer()

  def _setup_policy(self):
    input_size: int = self.observation_space.shape[0]
    output_size: int = self.action_space.n

    net_sizes: List[int] = [input_size] + self.policy_net_arch + [output_size]

    self.policy_net: nn.Module = mlp(net_sizes, self.activation_fn)

  def _setup_value_fn(self):
    input_size: int = self.observation_space.shape[0]
    output_size: int = 1

    net_sizes: List[int] = [input_size] + self.value_fn_net_arch + [output_size]

    self.value_fn_net: nn.Module = mlp(net_sizes, self.activation_fn)

  def _setup_optimizer(self):
    self.optimizer: torch.optim.Optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

  def forward(self, observation: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
    """
    Forward pass in policy and value function

    :param observation: (torch.Tensor) The current observation of the environment
    """
    logits: torch.Tensor = self.policy_net(observation)
    policy_dist: Categorical = Categorical(logits=logits)

    value: torch.Tensor = self.value_fn_net(observation)

    return policy_dist, value

  def predict(self, observation: torch.Tensor) -> torch.Tensor:
    """
    Get the policy action from an observation.

    :param observation: (np.ndarray) The current observation of the environment
    """
    with torch.no_grad():
      logits: torch.Tensor = self.policy_net(observation)
    policy_dist: Categorical = Categorical(logits=logits)

    action: torch.Tensor = policy_dist.sample()

    return action
