import random
from operator import itemgetter
from typing import Dict, List

import torch

class ReplayBuffer():
  """
  Replay buffer for off-policy algorithms.

  :param buffer_size: (int) Max number of element in the buffer
  """
  def __init__(
    self,
    buffer_size: int = int(1e6)
  ) -> None:
    self.buffer_size = buffer_size

    self.current_size: int = 0
    self.observations: List[torch.Tensor] = []
    self.actions: List[torch.Tensor] = []
    self.rewards: List[float] = []
    self.next_observations: List[torch.Tensor] = []
    self.dones: List[bool] = []

  def add_one_epoch_experience(
    self,
    observations: List[torch.Tensor],
    actions: List[torch.Tensor],
    rewards: List[float],
    next_observations: List[torch.Tensor],
    dones: List[bool]
  ):
    self.observations.extend(observations)
    self.actions.extend(actions)
    self.rewards.extend(rewards)
    self.next_observations.extend(next_observations)
    self.dones.extend(dones)

    self.current_size += len(observations)

    if self.current_size > self.buffer_size:
      num_exceeded_experinece: int = self.current_size - self.buffer_size

      for _ in range(num_exceeded_experinece):
        self.observations.pop()
        self.actions.pop()
        self.rewards.pop()
        self.next_observations.pop()
        self.dones.pop()

        self.current_size -= 1

  def sample_minibatch(
    self,
    minibatch_size: int = 32
  ):
    indices = random.sample(range(0, self.current_size), minibatch_size)

    sampled_observations: torch.Tensor = torch.stack(itemgetter(*indices)(self.observations))
    sampled_actions: torch.Tensor = torch.stack(itemgetter(*indices)(self.actions))
    sampled_rewards: torch.Tensor = torch.Tensor(itemgetter(*indices)(self.rewards))
    sampled_next_observations: torch.Tensor = torch.stack(itemgetter(*indices)(self.next_observations))
    sampled_dones: torch.Tensor = torch.Tensor(itemgetter(*indices)(self.dones))

    minibatch: Dict[str, torch.Tensor] = {'observations': sampled_observations,
                                          'actions': sampled_actions,
                                          'rewards': sampled_rewards,
                                          'next_observations': sampled_next_observations,
                                          'dones': sampled_dones}

    return minibatch
