import copy
import logging
from typing import List, Optional

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import functional as F

from rl_replicas.base_algorithms.on_policy_algorithm import OnPolicyAlgorithm
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.utils import discounted_cumulative_sums, gae
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (by clipping) with early stopping based on approximate KL divergence

    :param policy: (Policy) Policy.
    :param value_function: (ValueFunction) Value function.
    :param env: (gym.Env) Environment.
    :param clip_range: (float) The limit on the likelihood ratio between policies for clipping in the policy objective.
    :param max_kl_divergence: (float) The limit on the KL divergence between policies for early stopping.
    :param gamma: (float) The discount factor for the cumulative return.
    :param gae_lambda: (float) The factor for trade-off of bias vs variance for GAE.
    :param seed: (int) The seed for the pseudo-random generators.
    :param num_policy_gradients (int): The number of gradient descent steps to take on policy per epoch.
    :param num_value_gradients (int): The number of gradient descent steps to take on value function per epoch.
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
        num_policy_gradients: int = 80,
        num_value_gradients: int = 80,
    ) -> None:
        super().__init__(
            policy=policy,
            value_function=value_function,
            env=env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            num_value_gradients=num_value_gradients,
        )

        self.clip_range = clip_range
        self.num_policy_gradients = num_policy_gradients
        self.max_kl_divergence = max_kl_divergence

        self.old_policy: Policy = copy.deepcopy(self.policy)

    def train(
        self,
        one_epoch_experience: Experience,
    ) -> None:
        observations_list: List[List[np.ndarray]] = one_epoch_experience.observations
        actions_list: List[List[np.ndarray]] = one_epoch_experience.actions
        rewards_list: List[List[float]] = one_epoch_experience.rewards
        observations_with_last_observation_list: List[
            List[np.ndarray]
        ] = one_epoch_experience.observations_with_last_observation
        episode_dones: List[bool] = one_epoch_experience.episode_dones

        values_tensor_list: List[Tensor] = self.compute_values_tensor_list(
            observations_with_last_observation_list
        )

        last_values: List[float] = [
            episode_values[-1].detach().item() for episode_values in values_tensor_list
        ]

        bootstrapped_rewards: List[List[float]] = self.bootstrap_rewards(
            rewards_list, episode_dones, last_values
        )

        # Calculate rewards-to-go over each episode, to be targets for the value function
        discounted_returns: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    discounted_cumulative_sums(one_episode_rewards, self.gamma)[:-1]
                    for one_episode_rewards in bootstrapped_rewards
                ]
            )
        ).float()

        observations: Tensor = torch.from_numpy(
            np.concatenate(observations_list)
        ).float()
        actions: Tensor = torch.from_numpy(np.concatenate(actions_list)).float()

        # Calculate advantages
        advantages: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    gae(
                        one_episode_rewards,
                        self.gamma,
                        one_episode_values.numpy(),
                        self.gae_lambda,
                    )
                    for one_episode_rewards, one_episode_values in zip(
                        bootstrapped_rewards, values_tensor_list
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

        if self.tensorboard:
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

    def compute_values_tensor_list(
        self, observations_with_last_observation_list: List[List[np.ndarray]]
    ) -> List[Tensor]:
        values_tensor_list: List[Tensor] = []
        with torch.no_grad():
            for (
                observations_with_last_observation
            ) in observations_with_last_observation_list:
                observations_with_last_observation_tensor = torch.from_numpy(
                    np.concatenate([observations_with_last_observation])
                ).float()
                values_tensor_list.append(
                    self.value_function(
                        observations_with_last_observation_tensor
                    ).flatten()
                )
        return values_tensor_list

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

    def compute_policy_loss(
        self,
        observations: Tensor,
        actions: Tensor,
        advantages: Tensor,
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
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            policy_dist: Distribution = self.policy(observations)
            log_probs: Tensor = policy_dist.log_prob(actions)

            old_policy_dist: Distribution = self.old_policy(observations)
            old_log_probs: Tensor = old_policy_dist.log_prob(actions)

        approximate_kl_divergence: Tensor = old_log_probs - log_probs

        return torch.mean(approximate_kl_divergence)

    def compute_value_loss(
        self,
        observations: Tensor,
        discounted_returns: Tensor,
    ) -> Tensor:
        values: Tensor = self.value_function(observations)
        squeezed_values: Tensor = torch.squeeze(values, -1)
        value_loss: Tensor = F.mse_loss(squeezed_values, discounted_returns)

        return value_loss
