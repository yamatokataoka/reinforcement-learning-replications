import copy
import logging
from typing import List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, kl
from torch.nn import functional as F

from rl_replicas.base_algorithms.on_policy_algorithm import OnPolicyAlgorithm
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler
from rl_replicas.utils import compute_values_numpy_list, discounted_cumulative_sums, gae
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class TRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization with GAE for advantage estimation

    :param policy: (Policy) Policy.
    :param value_function: (ValueFunction) Value function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param gamma: (float) The discount factor for the cumulative return.
    :param gae_lambda: (float) The factor for trade-off of bias vs variance for GAE.
    :param num_value_gradients (int): The number of gradient descent steps to take on value function per epoch.
    """

    def __init__(
        self,
        policy: Policy,
        value_function: ValueFunction,
        env: gym.Env,
        sampler: Sampler,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
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

        self.old_policy: Policy = copy.deepcopy(self.policy)

    def train(self, experience: Experience) -> None:
        values_numpy_list: np.ndarray = compute_values_numpy_list(
            experience.observations_with_last_observation, self.value_function
        )

        last_values: List[float] = [
            float(episode_values[-1]) for episode_values in values_numpy_list
        ]

        bootstrapped_rewards: List[List[float]] = self.bootstrap_rewards(
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

        # Normalize advantages
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        def compute_surrogate_loss() -> Tensor:
            policy_dist: Distribution = self.policy(observations)
            log_probs: Tensor = policy_dist.log_prob(actions)

            with torch.no_grad():
                old_policy_dist: Distribution = self.old_policy(observations)
                old_log_probs: Tensor = old_policy_dist.log_prob(actions)

            likelihood_ratio: Tensor = torch.exp(log_probs - old_log_probs)
            surrogate_loss: Tensor = -torch.mean(likelihood_ratio * advantages)

            return surrogate_loss

        def compute_kl_constraint() -> Tensor:
            policy_dist: Distribution = self.policy(observations)

            with torch.no_grad():
                old_policy_dist: Distribution = self.old_policy(observations)

            kl_constraint: Tensor = kl.kl_divergence(old_policy_dist, policy_dist)

            return torch.mean(kl_constraint)

        policy_loss: Tensor = compute_surrogate_loss()

        # For logging
        policy_loss_before: Tensor = policy_loss.detach()
        with torch.no_grad():
            policy_dist: Distribution = self.policy(observations)
        log_probs: Tensor = policy_dist.log_prob(actions)
        entropies: Tensor = policy_dist.entropy()

        # Train the policy
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step(compute_surrogate_loss, compute_kl_constraint)

        self.old_policy.load_state_dict(self.policy.state_dict())

        # For logging
        with torch.no_grad():
            value_loss_before: Tensor = self.compute_value_loss(
                observations, discounted_returns
            )

        # Train the value function
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

        logger.info("Value Function Loss:    {:<8.3g}".format(value_loss_before))

        self.writer.add_scalar(
            "policy/loss", policy_loss_before, self.current_total_steps
        )
        self.writer.add_scalar(
            "policy/avarage_entropy",
            torch.mean(entropies),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "policy/log_prob_std", torch.std(log_probs), self.current_total_steps
        )

        self.writer.add_scalar(
            "value/loss", value_loss_before, self.current_total_steps
        )

    def bootstrap_rewards(
        self,
        rewards_list: List[List[float]],
        episode_dones: List[bool],
        last_values: List[float],
    ) -> List[List[float]]:
        bootstrapped_rewards: List[List[float]] = []

        for episode_rewards, episode_done, last_value in zip(
            rewards_list, episode_dones, last_values
        ):
            episode_bootstrapped_rewards: List[float]
            if episode_done:
                episode_bootstrapped_rewards = episode_rewards + [0]
            else:
                episode_bootstrapped_rewards = episode_rewards + [last_value]
            bootstrapped_rewards.append(episode_bootstrapped_rewards)

        return bootstrapped_rewards

    def compute_value_loss(
        self, observations: Tensor, discounted_returns: Tensor
    ) -> Tensor:
        values: Tensor = self.value_function(observations)
        squeezed_values: Tensor = torch.squeeze(values, -1)
        value_loss: Tensor = F.mse_loss(squeezed_values, discounted_returns)

        return value_loss
