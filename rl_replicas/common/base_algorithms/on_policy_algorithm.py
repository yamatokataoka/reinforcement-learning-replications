from abc import ABC, abstractmethod
import os
import time
from typing import Any, Optional, List, Tuple
from typing_extensions import TypedDict

import gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from rl_replicas import log
from rl_replicas.common.policy import Policy
from rl_replicas.common.utils import seed_random_generators
from rl_replicas.common.value_function import ValueFunction
from rl_replicas.common.utils import discount_cumulative_sum, seed_random_generators, gae

logger = log.get_logger(__name__)

class OneEpochExperience(TypedDict):
  observations: torch.Tensor
  actions: torch.Tensor
  advantages: torch.Tensor
  discounted_returns: torch.Tensor
  episode_returns: List[float]
  episode_lengths: List[int]

class OnPolicyAlgorithm(ABC):
  """
  The base of RL algorithms

  :param policy: (Policy) The policy
  :param value_function: (ValueFunction) The value function
  :param env: (gym.Env or str) The environment to learn from
  :param gamma: (float) Discount factor
  :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
  :param seed: (int) The seed for the pseudo-random generators
  :param n_value_gradients (int): Number of gradient descent steps to take on value function per epoch.
  """
  def __init__(
    self,
    policy: Policy,
    value_function: ValueFunction,
    env: gym.Env,
    gamma: float,
    gae_lambda: float,
    seed: Optional[int],
    n_value_gradients: int
  ) -> None:
    self.policy = policy
    self.value_function = value_function
    self.env = env
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    if seed is not None:
      self.seed: int = seed
    self.n_value_gradients = n_value_gradients

    if seed is not None:
      self._seed()

  def _seed(self) -> None:
    seed_random_generators(self.seed)
    self.env.action_space.seed(self.seed)
    self.env.seed(self.seed)

  def learn(
    self,
    epochs: int = 50,
    steps_per_epoch: int = 4000,
    output_dir: str = '.',
    tensorboard: bool = False,
    model_saving: bool = False
  ) -> None:
    """
    Learn the model

    :param epochs: (int) The number of epochs (equivalent to number of policy updates) to perform
    :param steps_per_epoch: (int) The number of steps to run per epoch; in other words, batch size is steps.
    :param output_dir: (str) The directory of output
    :param tensorboard: (bool) Whether or not to log for tensorboard
    :param model_saving: (bool) Whether or not to save trained model (Save and overwrite at each end of epoch)
    """
    self.writer: SummaryWriter = None
    if tensorboard:
      logger.info('Set up tensorboard')
      os.makedirs(output_dir, exist_ok=True)
      tensorboard_path: str = os.path.join(output_dir, 'tensorboard')
      self.writer = SummaryWriter(tensorboard_path)

    start_time: float = time.time()
    self.current_total_steps: int = 0
    self.current_total_episodes: int = 0

    for current_epoch in range(epochs):

      one_epoch_experience: OneEpochExperience = self.collect_experience_one_epoch(steps_per_epoch)

      if model_saving:
        logger.info('Set up model saving')
        os.makedirs(output_dir, exist_ok=True)
        model_path: str = os.path.join(output_dir, 'model.pt')

        logger.info('Save model')
        torch.save({
            'epoch': current_epoch+1,
            'total_steps': self.current_total_steps,
            'policy_state_dict': self.policy.network.state_dict(),
            'policy_optimizer_state_dict': self.policy.optimizer.state_dict(),
            'value_fn_state_dict': self.value_function.network.state_dict(),
            'value_fn_optimizer_state_dict': self.value_function.optimizer.state_dict()
          },
          model_path)

      episode_returns: List[float] = one_epoch_experience['episode_returns']
      episode_lengths: List[int] = one_epoch_experience['episode_lengths']

      logger.info('Epoch: {}'.format(current_epoch+1))

      logger.info('Total env interactions: {:<8.3g}'.format(self.current_total_steps))
      logger.info('Total episodes:         {:<8.3g}'.format(self.current_total_episodes))

      logger.info('Average Episode Return: {:<8.3g}'.format(np.mean(episode_returns)))
      logger.info('Episode Return STD:     {:<8.3g}'.format(np.std(episode_returns)))
      logger.info('Max Episode Return:     {:<8.3g}'.format(np.max(episode_returns)))
      logger.info('Min Episode Return:     {:<8.3g}'.format(np.min(episode_returns)))

      logger.info('Average Episode Length: {:<8.3g}'.format(np.mean(episode_lengths)))

      self.train(one_epoch_experience)

      logger.info('Time:                   {:<8.3g}'.format(time.time()-start_time))

    if tensorboard:
      self.writer.flush()
      self.writer.close()

  def collect_experience_one_epoch(self, steps_per_epoch: int) -> OneEpochExperience:
    one_epoch_experience: OneEpochExperience = {
      'observations': None,
      'actions': None,
      'advantages': None,
      'discounted_returns': None,
      'episode_returns': [],
      'episode_lengths': []
    }

    observations_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []

    advantages_ndarray: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)
    discounted_returns_ndarray: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)

    episode_returns_list: List[float] = []
    episode_lengths_list: List[int] = []

    # Variables on the current episode
    rewards: List[float] = []
    values: List[float] = []
    episode_length: int = 0

    observation: np.ndarray = self.env.reset()

    for current_step in range(steps_per_epoch):
      observation_tensor: torch.Tensor = torch.from_numpy(observation).float()

      observations_list.append(observation_tensor)

      with torch.no_grad():
        policy_dist: Categorical = self.policy(observation_tensor)
        value: torch.Tensor = self.value_function(observation_tensor)

      values.append(value.detach().item())

      action: torch.Tensor = policy_dist.sample()

      actions_list.append(action)

      action_ndarray = action.detach().numpy()
      reward: float
      episode_done: bool
      observation, reward, episode_done, _ = self.env.step(action_ndarray)

      rewards.append(reward)

      episode_length += 1

      self.current_total_steps += 1

      epoch_ended: bool = current_step == steps_per_epoch-1

      if episode_done or epoch_ended:
        if epoch_ended and not(episode_done):
          logger.warn('The trajectory cut off at {} steps on the current episode'.format(episode_length))

        last_value_float: float
        if epoch_ended:
          observation_tensor = torch.from_numpy(observation).float()

          with torch.no_grad():
            last_value: torch.Tensor = self.value_function(observation_tensor)

          last_value_float = last_value.detach().item()
        else:
          last_value_float = 0.0
        values.append(last_value_float)
        rewards.append(last_value_float)

        episode_slice = slice(current_step-episode_length+1, current_step+1)

        # Calculate advantage over an episode
        values_ndarray: np.ndarray = np.asarray(values)
        rewards_ndarray: np.ndarray = np.asarray(rewards)
        episode_advantage: np.ndarray = gae(rewards_ndarray, self.gamma, values_ndarray, self.gae_lambda)

        advantages_ndarray[episode_slice] = episode_advantage

        # Calculate rewards-to-go over an episode, to be targets for the value function
        episode_discounted_return: np.ndarray = discount_cumulative_sum(rewards, self.gamma)[:-1]
        discounted_returns_ndarray[episode_slice] = episode_discounted_return

        episode_true_return: float = np.sum(rewards).item()

        episode_returns_list.append(episode_true_return)
        episode_lengths_list.append(episode_length)

        if episode_done and self.writer:
          self.writer.add_scalar(
            'env/episode_true_return',
            episode_true_return,
            self.current_total_steps
          )
          self.writer.add_scalar(
            'env/episode_length',
            episode_length,
            self.current_total_steps
          )

        if episode_done:
          self.current_total_episodes += 1

        observation, episode_length = self.env.reset(), 0
        rewards, values = [], []

    one_epoch_experience['observations'] = torch.stack(observations_list)
    one_epoch_experience['actions'] = torch.stack(actions_list)
    one_epoch_experience['advantages'] = torch.from_numpy(advantages_ndarray)
    one_epoch_experience['discounted_returns'] = torch.from_numpy(discounted_returns_ndarray)
    one_epoch_experience['episode_returns'] = episode_returns_list
    one_epoch_experience['episode_lengths'] = episode_lengths_list

    return one_epoch_experience

  @abstractmethod
  def train(
    self,
    one_epoch_experience: OneEpochExperience
  ) -> None:
    """
    Consume experience on the current epoch and update train algorithm.

    Implemented by individual algorithms.

    :param one_epoch_experience: (OneEpochExperience) Collected experience on one epoch.
    """
    raise NotImplementedError

  def predict(
    self,
    observation: np.ndarray
  ) -> np.ndarray:
    """
    Get the action(s) from an observation which are sampled under the current policy.

    :param observation: the input observation
    :return: the model's action
    """
    observation_tensor: torch.Tensor = torch.from_numpy(observation).float()
    action: torch.Tensor = self.policy.predict(observation_tensor)
    action_ndarray: np.ndarray = action.detach().numpy()

    return action_ndarray
