import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Policy(nn.Module):
  """
  A policy.

  The policy network selects action based on the observation of the environment.
  It uses a PyTorch neural network module to fit the function of pi(s).

  :param network: (nn.Module) The network.
  :param optimizer: (torch.optim.Optimizer) The optimizer.
  """

  def __init__(
    self,
    network: nn.Module,
    optimizer: torch.optim.Optimizer
  ):
    super().__init__()

    self.network = network
    self.optimizer = optimizer

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
