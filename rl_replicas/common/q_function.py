import torch
import torch.nn as nn

class QFunction(nn.Module):
  """
  The Q Function

  :param network: (nn.Module) The network.
  :param optimizer: (torch.optim.Optimizer) The optimizer.
  """

  def __init__(
    self,
    network: nn.Module,
    optimizer: torch.optim.Optimizer
  ) -> None:
    super().__init__()

    self.network = network
    self.optimizer = optimizer

  def forward(
    self,
    observation: torch.Tensor,
    action: torch.Tensor
  ) -> torch.Tensor:
    """
    Forward pass in the Q-function

    :param observation: (torch.Tensor) The observation of the environment
    :param action: (torch.Tensor) The action of the environment
    :return squeezeed_q_value: (torch.Tensor) The Q-value(s)
    """
    input: torch.Tensor = torch.cat([observation, action], -1)
    q_value: torch.Tensor = self.network(input)
    squeezeed_q_value: torch.Tensor = torch.squeeze(q_value, -1)

    return squeezeed_q_value
