import copy
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

class PPO(OnPolicyAlgorithm):
  """
  Proximal Policy Optimization (by clipping) with early stopping based on approximate KL

  :param policy: (Policy) The policy
  :param value_function: (ValueFunction) The value function
  :param env: (gym.Env or str) The environment to learn from
  :param clip_range: (float) The limit on the likelihood ratio between policies for clipping in the policy objective.
  :param max_kl_divergence: (float) The limit on the KL divergence between policies for early stopping.
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
    clip_range: float = 0.2,
    max_kl_divergence: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    seed: Optional[int] = None,
    n_policy_gradients: int = 80,
    n_value_gradients: int = 80,
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

    self.clip_range = clip_range
    self.n_policy_gradients = n_policy_gradients
    self.max_kl_divergence = max_kl_divergence

    self.old_policy: Policy = copy.deepcopy(self.policy)

  def train(
    self,
    one_epoch_experience: OneEpochExperience
  ) -> None:
    observations: torch.Tensor = one_epoch_experience['observations']
    actions: torch.Tensor = one_epoch_experience['actions']
    advantages: torch.Tensor = one_epoch_experience['advantages']

    # Normalize advantage
    advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

    # for logging
    with torch.no_grad():
      policy_dist: Categorical = self.policy(observations)
      policy_loss_before: torch.Tensor = self.compute_policy_loss(
                                           observations,
                                           actions,
                                           advantages
                                          ).detach()
    log_probs: torch.Tensor = policy_dist.log_prob(actions)
    entropies: torch.Tensor = policy_dist.entropy()

    # Train policy
    for i in range(self.n_policy_gradients):
      policy_loss: torch.Tensor = self.compute_policy_loss(observations, actions, advantages)
      approximate_kl_divergence: torch.Tensor = self.compute_approximate_kl_divergence(
                                                  observations,
                                                  actions
                                                ).detach()
      if approximate_kl_divergence > 1.5 * self.max_kl_divergence:
        logger.info('Early stopping at update {} due to reaching max KL divergence.'.format(i))
        break

      self.policy.optimizer.zero_grad()
      policy_loss.backward()
      self.policy.optimizer.step()

    self.old_policy.load_state_dict(self.policy.state_dict())

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
    logger.info('KL divergence:          {:<8.3g}'.format(approximate_kl_divergence))

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
        'policy/approximate_kl_divergence',
        approximate_kl_divergence,
        self.current_total_steps
      )

      self.writer.add_scalar(
        'value/loss',
        value_loss_before,
        self.current_total_steps
      )

  def compute_policy_loss(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor
  ) -> torch.Tensor:
    policy_dist: Categorical = self.policy(observations)
    log_probs: torch.Tensor = policy_dist.log_prob(actions)

    with torch.no_grad():
      old_policy_dist: Categorical = self.old_policy(observations)
      old_log_probs: torch.Tensor = old_policy_dist.log_prob(actions)

    # Calculate surrogate
    likelihood_ratio: torch.Tensor = torch.exp(log_probs - old_log_probs)
    surrogate: torch.Tensor = likelihood_ratio * advantages

    # Clipping the constraint
    likelihood_ratio_clip: torch.Tensor = torch.clamp(
                                            likelihood_ratio,
                                            min=1 - self.clip_range,
                                            max=1 + self.clip_range
                                          )

    # Calculate surrotate clip
    surrogate_clip: torch.Tensor = likelihood_ratio_clip * advantages

    policy_loss: torch.Tensor = -torch.min(surrogate, surrogate_clip).mean()

    return policy_loss

  def compute_approximate_kl_divergence(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor
  ) -> torch.Tensor:
    with torch.no_grad():
      policy_dist: Categorical = self.policy(observations)
      log_probs: torch.Tensor = policy_dist.log_prob(actions)

      old_policy_dist: Categorical = self.old_policy(observations)
      old_log_probs: torch.Tensor = old_policy_dist.log_prob(actions)

    approximate_kl_divergence: torch.Tensor = old_log_probs - log_probs

    return torch.mean(approximate_kl_divergence)

  def compute_value_loss(
    self,
    observations: torch.Tensor,
    discounted_returns: torch.Tensor
  ) -> torch.Tensor:
    values: torch.Tensor = self.value_function(observations)
    squeezed_values: torch.Tensor = torch.squeeze(values, -1)
    value_loss: torch.Tensor = F.mse_loss(squeezed_values, discounted_returns)

    return value_loss
