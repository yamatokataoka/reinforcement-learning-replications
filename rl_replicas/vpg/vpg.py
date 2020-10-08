import time
import sys
from typing import Optional, Type, Union, List

import torch
from torch.distributions.categorical import Categorical
import gym
import numpy as np

from rl_replicas.common.policies import ActorCriticPolicy
from rl_replicas.common.utils import discount_cumulative_sum, seed_random_generators
from rl_replicas import log

logger = log.get_logger(__name__)

class VPG():
  """
  Vanilla Policy Gradient (REINFORCE) with GAE for advantage estimation

  :param policy_class: (Type[ActorCriticPolicy]) The policy model class
  :param env: (gym.Env or str) The environment to learn from
  :param learning_rate: (float) The learning rate for the optimizer
  :param gamma: (float) Discount factor
  :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
  :param seed: (int) The seed for the pseudo-random generators
  :param n_value_gradients (int): Number of gradient descent steps to take on value function per epoch.
  """
  def __init__(
    self,
    policy_class: Type[ActorCriticPolicy],
    env: Union[gym.Env, str],
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    seed: Optional[int] = None,
    n_value_gradients: int = 80
  ) -> None:
    if isinstance(policy_class, str):
      raise NotImplementedError
    else:
      self.policy_class = policy_class
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    if seed is not None:
      self.seed: int = seed
    self.n_value_gradients = n_value_gradients

    if env is not None:
      if isinstance(env, str):
        logger.info('Create environment from the given name: {}'.format(env))
        self.env = gym.make(env)

      self.env = env

    self.observation_space: gym.spaces.Space = self.env.observation_space
    self.action_space: gym.spaces.Space = self.env.action_space

    self.policy: ActorCriticPolicy = self.policy_class(self.observation_space,
                                                       self.action_space,
                                                       self.learning_rate)

    if self.seed is not None:
      self._seed()

  def _seed(self) -> None:
    seed_random_generators(self.seed)
    self.action_space.seed(self.seed)
    self.env.seed(self.seed)

  def learn(
    self,
    epochs: int = 50,
    steps_per_epoch: int = 4000,
  ) -> None:
    """
    Learn the model

    :param epochs: (int) The number of epochs (equivalent to number of policy updates) to perform
    :param steps_per_epoch: (int) The number of steps to run per epoch; in other words, batch size is steps.
    """
    epoch_policy_losses: List[np.ndarray] = []
    epoch_value_losses: List[np.ndarray] = []

    epoch_entropies: List[np.ndarray] = []

    previous_policy_loss: float = 0.0
    previous_value_loss: float = 0.0

    self.start_time = time.time()

    observation: np.ndarray = self.env.reset()

    # collect experiences
    for current_epoch in range(epochs):

      # variables on the current epoch
      # e.g. [ep_one_return, ep_one_return, (repeat the episode length times) ..., ep_two_return, ...]
      episode_advantages: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)
      discounted_returns: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)

      # e.g. [log_prob_one, log_prob_two ...]
      all_log_probs: List[torch.Tensor] = []
      all_values: List[torch.Tensor] = []

      all_entropies: np.ndarray = np.zeros(steps_per_epoch, dtype=np.float32)

      # list of the lengths of episode on the current epoch
      episode_lengths: List[int] = []
      episode_returns: List[float] = []

      # the variables on the current episode
      rewards: List[float] = []
      values: List[float] = []
      episode_length: int = 0

      for current_step in range(steps_per_epoch):
        observation_tensor: torch.Tensor = torch.from_numpy(observation).float()

        policy_dist: Categorical
        value: torch.Tensor
        policy_dist, value = self.policy(observation_tensor)

        all_values.append(value)

        values.append(value.detach().item())

        action: torch.Tensor = policy_dist.sample()
        log_prob: torch.Tensor = policy_dist.log_prob(action)

        all_log_probs.append(log_prob)

        entropy: np.ndarray = policy_dist.entropy().detach().numpy()

        all_entropies[current_step] = entropy

        action_ndarray = action.detach().numpy()

        reward: float
        episode_done: bool

        observation, reward, episode_done, _ = self.env.step(action_ndarray)

        rewards.append(reward)

        episode_length += 1

        epoch_ended = current_step == steps_per_epoch-1

        # At at the end of a trajectory or when one gets cut off by an epoch ending.
        if episode_done or epoch_ended:
          if epoch_ended and not(episode_done):
            logger.warn('The trajectory cut off at {} steps on the current episode'.format(episode_length))

          # if trajectory didn't reach terminal state, bootstrap value target
          if epoch_ended:
            with torch.no_grad():
              observation_tensor = torch.from_numpy(observation).float()
              last_value: torch.Tensor

              _, last_value = self.policy(observation_tensor)
              last_value_float = last_value.detach().item()
          else:
            last_value_float = 0.0
          values.append(last_value_float)
          rewards.append(last_value_float)

          episode_slice = slice(current_step-episode_length+1, current_step+1)

          # Calculate GAE advantage
          values_ndarray: np.ndarray = np.asarray(values)
          deltas: np.ndarray = rewards[:-1] + self.gamma * values_ndarray[1:] - values_ndarray[:-1]
          episode_advantage: np.ndarray = discount_cumulative_sum(deltas, self.gamma * self.gae_lambda)

          episode_advantages[episode_slice] = episode_advantage

          discounted_return: np.ndarray = discount_cumulative_sum(rewards, self.gamma)[:-1]
          discounted_returns[episode_slice] = discounted_return

          episode_returns.append(np.sum(rewards))
          episode_lengths.append(episode_length)

          observation, episode_length = self.env.reset(), 0
          rewards, values = [], []

      # Save model
      if current_epoch == epochs-1:
        logger.warn('Saving model is not implemented')

      # the advantage normalization
      # TODO: make it a function
      episode_advantages_tensor: torch.Tensor = torch.from_numpy(episode_advantages)

      episode_advantages_tensor = (episode_advantages_tensor - episode_advantages_tensor.mean()) / episode_advantages_tensor.std()

      all_log_probs_tensor: torch.Tensor = torch.stack(all_log_probs)

      policy_loss: torch.Tensor = -(all_log_probs_tensor * episode_advantages_tensor).mean()

      all_values_tensor: torch.Tensor = torch.stack(all_values)
      discounted_returns_tensor: torch.Tensor = torch.from_numpy(discounted_returns)

      value_loss: torch.Tensor = ((all_values_tensor - discounted_returns_tensor) ** 2).mean()

      self.policy.optimizer.zero_grad()
      # Train policy with a single step of gradient descent
      policy_loss.backward()

      # Value function learning
      value_loss.backward()

      self.policy.optimizer.step()

      mean_entropy: np.ndarray = all_entropies.mean()

      epoch_policy_losses.append(policy_loss.detach().numpy())
      epoch_value_losses.append(value_loss.detach().numpy())

      epoch_entropies.append(mean_entropy)

      logger.info('Loss of the current policy:         {:<8.3g}'.format(policy_loss))
      logger.info('Loss of the current value function: {:<8.3g}'.format(value_loss))

      if previous_policy_loss and previous_value_loss:
        logger.info('Difference of the previous policy loss:         {:<8.3g}'.format(policy_loss-previous_policy_loss))
        logger.info('Difference of the previous value function loss: {:<8.3g}'.format(value_loss-previous_value_loss))

      previous_policy_loss = policy_loss.detach().item()
      previous_value_loss = value_loss.detach().item()

      # info about the current learning
      logger.info('Epoch: {}'.format(current_epoch))

      logger.info('Average Episode Return: {:<8.3g}'.format(np.mean(episode_returns)))
      logger.info('Std Episode Return:     {:<8.3g}'.format(np.std(episode_returns)))
      logger.info('Maximum Episode Return: {}'.format(np.max(episode_returns)))
      logger.info('Minimum Episode Return: {}'.format(np.min(episode_returns)))

      logger.info('Average Episode Length: {:<8.3g}'.format(np.mean(episode_lengths)))

      logger.info('Average Episode Value:  {:<8.3g}'.format(all_values_tensor.detach().mean()))
      logger.info('Std Episode Value:      {:<8.3g}'.format(all_values_tensor.detach().std()))
      logger.info('Maximum Episode Value:  {:<8.3g}'.format(all_values_tensor.detach().max()))
      logger.info('Minimum Episode Value:  {:<8.3g}'.format(all_values_tensor.detach().min()))

      logger.info('Total env interactions: {}'.format((current_epoch+1) * steps_per_epoch))

      logger.info('Avarage Policy Loss:    {:<8.3g}'.format(np.mean(epoch_policy_losses)))
      logger.info('Avarage Value function Loss: {:8.3f}'.format(np.mean(epoch_value_losses)))
      logger.info('Avarage Entropy:        {:<8.3g}'.format(np.mean(epoch_entropies)))
      logger.info('Time:                   {:<8.3g}'.format(time.time()-self.start_time))

  def predict(
    self,
    observation: np.ndarray
  ) -> np.ndarray:
    """
    Get the action(s) from an observation which are sampled under the current policy.

    :param observation: the input observation
    :return: the model's action
    """
    return self.policy.predict(observation)
