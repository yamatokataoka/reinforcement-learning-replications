from typing import Optional

import torch
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import gym

from rl_replicas.common.base_algorithms.on_policy_algorithm import OnPolicyAlgorithm, OneEpochExperience
from rl_replicas.common.policy import Policy
from rl_replicas.common.value_function import ValueFunction
from rl_replicas import log

logger = log.get_logger(__name__)

class VPG(OnPolicyAlgorithm):
  """
  Vanilla Policy Gradient (REINFORCE) with GAE for advantage estimation

  VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

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
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    seed: Optional[int] = None,
    n_value_gradients: int = 80
  ) -> None:
    super().__init__(
      policy=policy,
      value_function=value_function,
      env=env,
      gamma=gamma,
      gae_lambda=gae_lambda,
      seed=seed,
      n_value_gradients=n_value_gradients
    )

  def train(
    self,
    one_epoch_experience: OneEpochExperience
  ) -> None:
    observations: torch.Tensor = one_epoch_experience['observations']
    actions: torch.Tensor = one_epoch_experience['actions']
    advantages: torch.Tensor = one_epoch_experience['advantages']

    # Normalize advantage
    advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

    policy_dist: Categorical = self.policy(observations)
    log_probs: torch.Tensor = policy_dist.log_prob(actions)

    policy_loss: torch.Tensor = -torch.mean(log_probs * advantages)

    # for logging
    policy_loss_before: torch.Tensor = policy_loss.detach()
    entropies: torch.Tensor = policy_dist.entropy().detach()

    # Train policy
    self.policy.optimizer.zero_grad()
    policy_loss.backward()
    self.policy.optimizer.step()

    discounted_returns: torch.Tensor = one_epoch_experience['discounted_returns']

    # for logging
    with torch.no_grad():
      value_loss_before: torch.Tensor = self.compute_value_loss(observations, discounted_returns)

    # Train value function
    for _ in range(self.n_value_gradients):
      value_loss: torch.Tensor = self.compute_value_loss(observations, discounted_returns)
      self.value_function.optimizer.zero_grad()
      value_loss.backward()
      self.value_function.optimizer.step()

    logger.info('Policy Loss:            {:<8.3g}'.format(policy_loss_before))
    logger.info('Avarage Entropy:        {:<8.3g}'.format(torch.mean(entropies)))
    logger.info('Log Prob STD:           {:<8.3g}'.format(torch.std(log_probs)))

    logger.info('Value Function Loss:    {:<8.3g}'.format(value_loss_before))

    if self.writer:
      self.writer.add_scalar(
        'policy/loss',
        policy_loss_before,
        self.current_total_steps
      )
      self.writer.add_scalar(
        'policy/avarage_entropy',
        torch.mean(entropies),
        self.current_total_steps
      )
      self.writer.add_scalar(
        'policy/log_prob_std',
        torch.std(log_probs),
        self.current_total_steps
      )

      self.writer.add_scalar(
        'value/loss',
        value_loss_before,
        self.current_total_steps
      )

  def compute_value_loss(
    self,
    observations: torch.Tensor,
    discounted_returns: torch.Tensor
  ) -> torch.Tensor:
    values: torch.Tensor = self.value_function(observations)
    squeezed_values: torch.Tensor = torch.squeeze(values, -1)
    value_loss: torch.Tensor = F.mse_loss(squeezed_values, discounted_returns)

    return value_loss
