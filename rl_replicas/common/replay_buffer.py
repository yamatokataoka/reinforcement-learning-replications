from typing import Dict

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
    self.observations: torch.Tensor = torch.Tensor()
    self.actions: torch.Tensor = torch.Tensor()
    self.rewards: torch.Tensor = torch.Tensor()
    self.next_observations: torch.Tensor = torch.Tensor()
    self.dones: torch.Tensor = torch.Tensor()

  def add_one_epoch_experience(
    self,
    one_epoch_observations: torch.Tensor,
    one_epoch_actions: torch.Tensor,
    one_epoch_rewards: torch.Tensor,
    one_epoch_next_observations: torch.Tensor,
    one_epoch_dones: torch.Tensor
  ):
    if self.current_size == 0:
      self.observations = one_epoch_observations
      self.actions = one_epoch_actions
      self.rewards = one_epoch_rewards
      self.next_observations = one_epoch_next_observations
      self.dones = one_epoch_dones
    else:
      self.observations = torch.cat([self.observations, one_epoch_observations])
      self.actions = torch.cat([self.actions, one_epoch_actions])
      self.rewards = torch.cat([self.rewards, one_epoch_rewards])
      self.next_observations = torch.cat([self.next_observations, one_epoch_next_observations])
      self.dones = torch.cat([self.dones, one_epoch_dones])

    self.current_size += len(one_epoch_observations)

    if self.current_size > self.buffer_size:
      num_exceeded_experinece: int = self.current_size - self.buffer_size

      one_epoch_observations = torch.narrow(self.observations,
                                            0,
                                            num_exceeded_experinece,
                                            self.buffer_size)
      one_epoch_actions = torch.narrow(self.actions,
                                       0,
                                       num_exceeded_experinece,
                                       self.buffer_size)
      one_epoch_rewards = torch.narrow(self.rewards,
                                       0,
                                       num_exceeded_experinece,
                                       self.buffer_size)
      one_epoch_next_observations = torch.narrow(self.next_observations,
                                                 0,
                                                 num_exceeded_experinece,
                                                 self.buffer_size)
      one_epoch_dones = torch.narrow(self.dones,
                                     0,
                                     num_exceeded_experinece,
                                     self.buffer_size)

      self.current_size = self.buffer_size

  def sample_minibatch(
    self,
    minibatch_size: int = 32
  ):
    indexes: torch.Tensor = torch.randint(low = 0,
                                          high = self.current_size,
                                          size = (minibatch_size,))

    minibatch: Dict[str, torch.Tensor] = dict(observations = self.observations[indexes],
                                              actions = self.actions[indexes],
                                              rewards = self.rewards[indexes],
                                              next_observations = self.next_observations[indexes],
                                              dones = self.dones[indexes])

    return minibatch
