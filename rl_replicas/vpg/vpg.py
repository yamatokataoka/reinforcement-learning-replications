import os
import time
from typing import Optional, Type, Union, List

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np

from rl_replicas.common.policies import Policy
from rl_replicas.common.value_functions import MLPValueFunction
from rl_replicas.common.utils import discount_cumulative_sum, seed_random_generators, gae
from rl_replicas import log

logger = log.get_logger(__name__)

class VPG():
  """
  Vanilla Policy Gradient (REINFORCE) with GAE for advantage estimation

  VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

  :param policy: (Policy) The policy
  :param value_function: (MLPValueFunction) The value function
  :param env: (gym.Env or str) The environment to learn from
  :param gamma: (float) Discount factor
  :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
  :param seed: (int) The seed for the pseudo-random generators
  :param n_value_gradients (int): Number of gradient descent steps to take on value function per epoch.
  """
  def __init__(
    self,
    policy: Policy,
    value_function: MLPValueFunction,
    env: gym.Env,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    seed: Optional[int] = None,
    n_value_gradients: int = 80
  ) -> None:
    self.policy = policy
    self.value_function = value_function
    self.env = env
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    if seed is not None:
      self.seed: int = seed
    self.n_value_gradients = n_value_gradients

    self.action_space: gym.spaces.Space = self.env.action_space

    if seed is not None:
      self._seed()

  def _seed(self) -> None:
    seed_random_generators(self.seed)
    self.action_space.seed(self.seed)
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
    if tensorboard:
      logger.info('Set up tensorboard')
      os.makedirs(output_dir, exist_ok=True)
      tensorboard_path: str = os.path.join(output_dir, 'tensorboard')
      writer: SummaryWriter = SummaryWriter(tensorboard_path)

    policy_losses: List[float] = []
    value_losses: List[float] = []

    all_episode_returns: List[float] = []
    all_episode_lengths: List[int] = []

    all_entropies: List[float] = []
    all_values: List[float] = []

    self.start_time: float = time.time()
    current_total_steps: int = 0
    current_total_episodes: int = 0

    observation: np.ndarray = self.env.reset()

    for current_epoch in range(epochs):

      # Variables on the current epoch
      advantages_on_epoch: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)
      discounted_returns_on_epoch: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)

      observations_on_epoch: List[torch.Tensor] = []
      actions_on_epoch: List[torch.Tensor] = []

      # Variables on the current episode
      rewards: List[float] = []
      values: List[float] = []
      episode_length: int = 0

      for current_step in range(steps_per_epoch):
        observation_tensor: torch.Tensor = torch.from_numpy(observation).float()

        observations_on_epoch.append(observation_tensor)

        with torch.no_grad():
          policy_dist: Categorical = self.policy(observation_tensor)
          value: torch.Tensor = self.value_function(observation_tensor)

        values.append(value.detach().item())

        action: torch.Tensor = policy_dist.sample()

        actions_on_epoch.append(action)

        action_ndarray = action.detach().numpy()
        reward: float
        episode_done: bool
        observation, reward, episode_done, _ = self.env.step(action_ndarray)

        rewards.append(reward)

        episode_length += 1
        current_total_steps += 1

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

          advantages_on_epoch[episode_slice] = episode_advantage

          # Calculate rewards-to-go over an episode, to be targets for the value function
          episode_discounted_return: np.ndarray = discount_cumulative_sum(rewards, self.gamma)[:-1]
          discounted_returns_on_epoch[episode_slice] = episode_discounted_return

          episode_true_return: float = np.sum(rewards).item()

          all_episode_returns.append(episode_true_return)
          all_episode_lengths.append(episode_length)

          if episode_done and tensorboard:
            writer.add_scalar('env/episode_true_return',
                              episode_true_return,
                              current_total_steps)
            writer.add_scalar('env/episode_length',
                              episode_length,
                              current_total_steps)

          if episode_done:
            current_total_episodes += 1

          observation, episode_length = self.env.reset(), 0
          rewards, values = [], []

      # Update policy and value function on the current epoch
      observations_tensor: torch.Tensor = torch.stack(observations_on_epoch)
      actions_tensor: torch.Tensor = torch.stack(actions_on_epoch)

      policy_dist: Categorical = self.policy(observations_tensor)
      log_probs: torch.Tensor = policy_dist.log_prob(actions_tensor)

      advantages_tensor: torch.Tensor = torch.from_numpy(advantages_on_epoch)

      # Normalize advantage
      advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / advantages_tensor.std()

      policy_loss: torch.Tensor = -(log_probs * advantages_tensor).mean()

      # Train policy
      self.policy.optimizer.zero_grad()
      policy_loss.backward()
      self.policy.optimizer.step()

      # Train value function
      discounted_returns_tensor: torch.Tensor = torch.from_numpy(discounted_returns_on_epoch)
      values: torch.Tensor
      value_loss: torch.Tensor
      for _ in range(self.n_value_gradients):
        values = self.value_function(observations_tensor)
        squeezed_values = torch.squeeze(values, -1)
        self.value_function.optimizer.zero_grad()
        value_loss = nn.MSELoss()(squeezed_values, discounted_returns_tensor)
        value_loss.backward()
        self.value_function.optimizer.step()

      policy_losses.append(policy_loss.detach().item())
      value_losses.append(value_loss.detach().item())

      entropies: torch.Tensor = policy_dist.entropy()

      all_entropies.append(entropies.detach().tolist())
      all_values.append(squeezed_values.detach().tolist())

      # Stats over all epochs and episodes
      logger.info('Epoch: {}'.format(current_epoch+1))

      logger.info('Average Episode Return: {:<8.3g}'.format(np.mean(all_episode_returns)))
      logger.info('Std Episode Return:     {:<8.3g}'.format(np.std(all_episode_returns)))
      logger.info('Maximum Episode Return: {:<8.3g}'.format(np.max(all_episode_returns)))
      logger.info('Minimum Episode Return: {:<8.3g}'.format(np.min(all_episode_returns)))

      logger.info('Average Episode Length: {:<8.3g}'.format(np.mean(all_episode_lengths)))

      logger.info('Average Value:          {:<8.3g}'.format(np.mean(all_values)))
      logger.info('Std Value:              {:<8.3g}'.format(np.std(all_values)))
      logger.info('Maximum Value:          {:<8.3g}'.format(np.max(all_values)))
      logger.info('Minimum Value:          {:<8.3g}'.format(np.min(all_values)))

      logger.info('Total env interactions: {:<8.3g}'.format(current_total_steps))
      logger.info('Total episodes:         {:<8.3g}'.format(current_total_episodes))

      logger.info('Current Loss of policy:         {:<8.3g}'.format(policy_loss))
      logger.info('Current Loss of value function: {:<8.3g}'.format(value_loss))

      logger.info('Avarage Policy Loss:         {:<8.3g}'.format(np.mean(policy_losses)))
      logger.info('Avarage Value function Loss: {:<8.3g}'.format(np.mean(value_losses)))

      logger.info('Avarage Entropy:        {:<8.3g}'.format(np.mean(all_entropies)))

      logger.info('Time:                   {:<8.3g}'.format(time.time()-self.start_time))

      if tensorboard:
        writer.add_scalar('policy/loss',
                          policy_loss,
                          current_total_steps)
        writer.add_scalar('value/loss',
                          value_loss,
                          current_total_steps)
        writer.add_scalar('policy/avarage_entropy',
                          np.mean(all_entropies),
                          current_total_steps)
        writer.add_scalar('policy/log_prob_std',
                          log_probs.std(),
                          current_total_steps)

      if model_saving:
        logger.info('Set up model saving')
        os.makedirs(output_dir, exist_ok=True)
        model_path: str = os.path.join(output_dir, 'model.pt')

        logger.info('Save model')
        torch.save({
            'epoch': current_epoch+1,
            'total_steps': current_total_steps,
            'policy_state_dict': self.policy.network.state_dict(),
            'policy_optimizer_state_dict': self.policy.optimizer.state_dict(),
            'value_fn_state_dict': self.value_function.network.state_dict(),
            'value_fn_optimizer_state_dict': self.value_function.optimizer.state_dict()
          },
          model_path)

    if tensorboard:
      writer.flush()
      writer.close()

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
