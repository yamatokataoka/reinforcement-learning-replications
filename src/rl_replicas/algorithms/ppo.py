import copy
import logging
from typing import List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import functional as F

from rl_replicas.base_algorithms.on_policy_algorithm import OnPolicyAlgorithm
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler
from rl_replicas.utils import (
    bootstrap_rewards_with_last_values,
    compute_values_numpy_list,
    discounted_cumulative_sums,
    gae,
)
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (by clipping) with early stopping based on approximate KL divergence

    :param policy: (Policy) Policy.
    :param value_function: (ValueFunction) Value function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param clip_range: (float) The limit on the likelihood ratio between policies for clipping in the policy objective.
    :param max_kl_divergence: (float) The limit on the KL divergence between policies for early stopping.
    :param gamma: (float) The discount factor for the cumulative return.
    :param gae_lambda: (float) The factor for trade-off of bias vs variance for GAE.
    :param num_policy_gradients (int): The number of gradient descent steps to take on policy per epoch.
    :param num_value_gradients (int): The number of gradient descent steps to take on value function per epoch.
    """

    def __init__(
        self,
        policy: Policy,
        value_function: ValueFunction,
        env: gym.Env,
        sampler: Sampler,
        clip_range: float = 0.2,
        max_kl_divergence: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        num_policy_gradients: int = 80,
        num_value_gradients: int = 80,
    ) -> None:
        super().__init__(
            policy=policy,
            value_function=value_function,
            env=env,
            sampler=sampler,
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_value_gradients=num_value_gradients,
        )

        self.clip_range = clip_range
        self.num_policy_gradients = num_policy_gradients
        self.max_kl_divergence = max_kl_divergence

        self.old_policy: Policy = copy.deepcopy(self.policy)

    def train(self, experience: Experience) -> None:
        values_numpy_list: np.ndarray = compute_values_numpy_list(
            experience.observations_with_last_observation, self.value_function
        )

        last_values: List[float] = [
            float(episode_values[-1]) for episode_values in values_numpy_list
        ]

        bootstrapped_rewards: List[List[float]] = bootstrap_rewards_with_last_values(
            experience.rewards, experience.episode_dones, last_values
        )

        # Calculate rewards-to-go over each episode, to be targets for the value function
        discounted_returns: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    discounted_cumulative_sums(episode_rewards, self.gamma)[:-1]
                    for episode_rewards in bootstrapped_rewards
                ]
            )
        ).float()

        observations: Tensor = torch.from_numpy(
            np.concatenate(experience.observations)
        ).float()
        actions: Tensor = torch.from_numpy(np.concatenate(experience.actions)).float()

        # Calculate advantages
        advantages: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    gae(
                        episode_rewards,
                        self.gamma,
                        episode_values,
                        self.gae_lambda,
                    )
                    for episode_rewards, episode_values in zip(
                        bootstrapped_rewards, values_numpy_list
                    )
                ]
            )
        ).float()

        # Normalize advantage
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        # For logging
        with torch.no_grad():
            policy_dist: Distribution = self.policy(observations)
            policy_loss_before: Tensor = self.compute_policy_loss(
                observations, actions, advantages
            ).detach()
            log_probs: Tensor = policy_dist.log_prob(actions)
            entropies: Tensor = policy_dist.entropy()

        # Train the policy
        for i in range(self.num_policy_gradients):
            policy_loss: Tensor = self.compute_policy_loss(
                observations, actions, advantages
            )
            approximate_kl_divergence: Tensor = self.compute_approximate_kl_divergence(
                observations, actions
            ).detach()
            if approximate_kl_divergence > 1.5 * self.max_kl_divergence:
                logger.info(
                    "Early stopping at update {} due to reaching max KL divergence.".format(
                        i
                    )
                )
                break

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

        # For logging
        with torch.no_grad():
            value_loss_before: Tensor = self.compute_value_loss(
                observations, discounted_returns
            )

        # Train value function
        for _ in range(self.num_value_gradients):
            value_loss: Tensor = self.compute_value_loss(
                observations, discounted_returns
            )
            self.value_function.optimizer.zero_grad()
            value_loss.backward()
            self.value_function.optimizer.step()

        logger.info("Policy Loss:            {:<8.3g}".format(policy_loss_before))
        logger.info("Avarage Entropy:        {:<8.3g}".format(torch.mean(entropies)))
        logger.info("Log Prob STD:           {:<8.3g}".format(torch.std(log_probs)))
        logger.info(
            "KL divergence:          {:<8.3g}".format(approximate_kl_divergence)
        )

        logger.info("Value Function Loss:    {:<8.3g}".format(value_loss_before))

        self.writer.add_scalar(
            "policy/loss",
            policy_loss_before,
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "policy/avarage_entropy",
            torch.mean(entropies),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "policy/log_prob_std",
            torch.std(log_probs),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "policy/approximate_kl_divergence",
            approximate_kl_divergence,
            self.current_total_steps,
        )

        self.writer.add_scalar(
            "value/loss",
            value_loss_before,
            self.current_total_steps,
        )

    def compute_policy_loss(
        self, observations: Tensor, actions: Tensor, advantages: Tensor
    ) -> Tensor:
        policy_dist: Distribution = self.policy(observations)
        log_probs: Tensor = policy_dist.log_prob(actions)

        with torch.no_grad():
            old_policy_dist: Distribution = self.old_policy(observations)
            old_log_probs: Tensor = old_policy_dist.log_prob(actions)

        # Calculate surrogate
        likelihood_ratio: Tensor = torch.exp(log_probs - old_log_probs)
        surrogate: Tensor = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip: Tensor = torch.clamp(
            likelihood_ratio,
            min=1 - self.clip_range,
            max=1 + self.clip_range,
        )

        # Calculate surrotate clip
        surrogate_clip: Tensor = likelihood_ratio_clip * advantages

        policy_loss: Tensor = -torch.min(surrogate, surrogate_clip).mean()

        return policy_loss

    def compute_approximate_kl_divergence(
        self, observations: Tensor, actions: Tensor
    ) -> Tensor:
        with torch.no_grad():
            policy_dist: Distribution = self.policy(observations)
            log_probs: Tensor = policy_dist.log_prob(actions)

            old_policy_dist: Distribution = self.old_policy(observations)
            old_log_probs: Tensor = old_policy_dist.log_prob(actions)

        approximate_kl_divergence: Tensor = old_log_probs - log_probs

        return torch.mean(approximate_kl_divergence)

    def compute_value_loss(
        self, observations: Tensor, discounted_returns: Tensor
    ) -> Tensor:
        values: Tensor = self.value_function(observations)
        squeezed_values: Tensor = torch.squeeze(values, -1)
        value_loss: Tensor = F.mse_loss(squeezed_values, discounted_returns)

        return value_loss
