from abc import ABC, abstractmethod
import copy
import os
import time
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_replicas import log
from rl_replicas.common.policies import Policy
from rl_replicas.common.q_function import QFunction
from rl_replicas.common.replay_buffer import ReplayBuffer
from rl_replicas.common.utils import seed_random_generators

logger = log.get_logger(__name__)

class OffPolicyAlgorithm(ABC):
  """
  The base of off-policy algorithms

  :param policy: (Policy) The policy
  :param q_function: (QFunction) The Q function
  :param env: (gym.Env) The environment to learn from
  :param gamma: (float) The discount factor for the cumulative return
  :param tau: (float) The interpolation factor in polyak averaging for target networks
  :param action_noise_scale: (float) The scale of the action noise (std)
  :param seed: (int) The seed for the pseudo-random generators
  """
  def __init__(
    self,
    policy: Policy,
    q_function: QFunction,
    env: gym.Env,
    gamma: float,
    tau: float,
    action_noise_scale: float,
    seed: Optional[int]
  ) -> None:
    self.policy = policy
    self.q_function = q_function
    self.env = env
    self.gamma = gamma
    self.tau = tau
    self.action_noise_scale = action_noise_scale
    if seed is not None:
      self.seed: int = seed

    self.action_limit: float = self.env.action_space.high[0]
    self.action_size: int = self.env.action_space.shape[0]
    self.target_policy = copy.deepcopy(self.policy)
    self.target_q_function = copy.deepcopy(self.q_function)
    if seed is not None:
      self._seed()

  def _seed(self) -> None:
    seed_random_generators(self.seed)
    self.env.action_space.seed(self.seed)
    self.env.seed(self.seed)

  def learn(
    self,
    epochs: int = 8000,
    steps_per_epoch: int = 200,
    replay_buffer_size: int = int(1e6),
    minibatch_size: int = 100,
    random_start_steps: int = 10000,
    steps_before_update: int = 1000,
    train_steps: int = 50,
    output_dir: str = '.',
    tensorboard: bool = False,
    model_saving: bool = False
  ) -> None:
    """
    Learn the model

    :param epochs: (int) The number of epochs to run and train.
    :param steps_per_epoch: (int) The number of steps to run per epoch; in other words, batch size is steps_per_epoch.
    :param replay_size: (int) The size of the replay buffer
    ;param minibatch_size: (int) The minibatch size for SGD.
    :param random_start_steps: (int) The number of steps for uniform-random action selection for exploration at the beginning.
    :param steps_before_update: (int) The number of steps to perform before policy is updated.
    :param train_steps: (int) The number of training steps on each epoch
    :param output_dir: (str) The directory of output
    :param tensorboard: (bool) Whether or not to log for tensorboard
    :param model_saving: (bool) Whether or not to save trained model (Save and overwrite at each end of epoch)
    """
    self.tensorboard = tensorboard

    start_time: float = time.time()
    self.current_total_steps: int = 0
    self.current_total_episodes: int = 0

    if self.tensorboard:
      logger.info('Set up tensorboard')
      os.makedirs(output_dir, exist_ok=True)
      tensorboard_path: str = os.path.join(output_dir, 'tensorboard')
      self.writer: SummaryWriter = SummaryWriter(tensorboard_path)

    self.replay_buffer: ReplayBuffer = ReplayBuffer(replay_buffer_size)

    for current_epoch in range(epochs):
      episode_returns: List[float]
      episode_lengths: List[int]
      episode_returns, episode_lengths = self.collect_one_epoch_experience(self.replay_buffer,
                                                                           steps_per_epoch,
                                                                           random_start_steps)

      if model_saving:
        logger.info('Set up model saving')
        os.makedirs(output_dir, exist_ok=True)
        model_path: str = os.path.join(output_dir, 'model.pt')

        logger.info('Save model')
        torch.save({
            'epoch': current_epoch,
            'total_steps': self.current_total_steps,
            'policy_state_dict': self.policy.network.state_dict(),
            'policy_optimizer_state_dict': self.policy.optimizer.state_dict(),
            'q_function_state_dict': self.q_function.network.state_dict(),
            'q_function_optimizer_state_dict': self.q_function.optimizer.state_dict(),
          },
          model_path)

      logger.info('Epoch: {}'.format(current_epoch))

      logger.info('Total env interactions: {:<8.3g}'.format(self.current_total_steps))
      logger.info('Total episodes:         {:<8.3g}'.format(self.current_total_episodes))

      if len(episode_returns) > 0:
        logger.info('Average Episode Return: {:<8.3g}'.format(np.mean(episode_returns)))
        logger.info('Episode Return STD:     {:<8.3g}'.format(np.std(episode_returns)))
        logger.info('Max Episode Return:     {:<8.3g}'.format(np.max(episode_returns)))
        logger.info('Min Episode Return:     {:<8.3g}'.format(np.min(episode_returns)))

      if len(episode_lengths) > 0:
        logger.info('Average Episode Length: {:<8.3g}'.format(np.mean(episode_lengths)))

      logger.info('Time:                   {:<8.3g}'.format(time.time()-start_time))

      if self.current_total_steps >= steps_before_update:
        self.train(self.replay_buffer, train_steps, minibatch_size)

    if self.tensorboard:
      self.writer.flush()
      self.writer.close()

  def collect_one_epoch_experience(
    self,
    replay_buffer: ReplayBuffer,
    steps_per_epoch: int,
    random_start_steps: int
  ) -> Tuple[List[float], List[int]]:
    observations_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    next_observations_list: List[torch.Tensor] = []

    rewards: List[float] = []
    dones: List[bool] = []

    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    # Variables on the current episode
    episode_length: int = 0
    episode_return: float = 0

    observation: np.ndarray = self.env.reset()
    observation_tensor: torch.Tensor = torch.from_numpy(observation).float()

    for current_step in range(steps_per_epoch):
      observations_list.append(observation_tensor)

      action: np.ndarray
      if self.current_total_steps > random_start_steps:
        action = self.action_limit * self.predict(observation)
        action += self.action_noise_scale * np.random.randn(self.action_size)
        action = np.clip(action, -self.action_limit, self.action_limit)
      else:
        action = self.env.action_space.sample()

      action_tensor: torch.Tensor = torch.from_numpy(action).float()
      actions_list.append(action_tensor)

      next_observation: np.ndarray
      reward: float
      episode_done: bool
      next_observation, reward, episode_done, _ = self.env.step(action)

      next_observation_tensor: torch.Tensor = torch.from_numpy(next_observation).float()
      next_observations_list.append(next_observation_tensor)

      observation = next_observation
      observation_tensor = next_observation_tensor

      rewards.append(reward)
      dones.append(episode_done)

      episode_length += 1
      episode_return += reward
      self.current_total_steps += 1

      epoch_ended: bool = current_step == steps_per_epoch-1

      if episode_done:
        self.current_total_episodes += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if self.tensorboard:
          self.writer.add_scalar(
            'env/episode_true_return',
            episode_return,
            self.current_total_steps
          )
          self.writer.add_scalar(
            'env/episode_length',
            episode_length,
            self.current_total_steps
          )

        observation, episode_length, episode_return = self.env.reset(), 0, 0

    this_epoch_observations: torch.Tensor = torch.stack(observations_list)
    this_epoch_actions: torch.Tensor = torch.stack(actions_list)
    this_epoch_rewards: torch.Tensor = torch.Tensor(rewards)
    this_epoch_next_observations: torch.Tensor = torch.stack(next_observations_list)
    this_epoch_dones: torch.Tensor = torch.Tensor(dones)

    replay_buffer.add_one_epoch_experience(this_epoch_observations,
                                           this_epoch_actions,
                                           this_epoch_rewards,
                                           this_epoch_next_observations,
                                           this_epoch_dones)

    return episode_returns, episode_lengths

  @abstractmethod
  def train(
    self,
    replay_buffer: ReplayBuffer,
    train_steps: int,
    minibatch_size: int
  ) -> None:
    """
    Train the algorithm with the experience.

    :param replay_buffer: (ReplayBuffer) The reply buffer
    :param train_steps: (int) The number of gradient descent updates
    :param minibatch_size: (int) The minibatch size
    """
    raise NotImplementedError

  def predict(
    self,
    observation: np.ndarray
  ) -> np.ndarray:
    """
    Get the action(s) from an observation which are sampled under the current policy.

    :param observation: (np.ndarray) The input observation
    :return: (np.ndarray) The action(s)
    """
    observation_tensor: torch.Tensor = torch.from_numpy(observation).float()
    action: torch.Tensor = self.policy.predict(observation_tensor)
    action_ndarray: np.ndarray = action.detach().numpy()

    return action_ndarray
