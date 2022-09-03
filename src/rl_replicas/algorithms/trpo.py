import copy
import logging
import os
import time
from typing import Callable, List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, kl
from torch.nn import functional as F

from rl_replicas.experience import Experience
from rl_replicas.metrics_manager import MetricsManager
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler
from rl_replicas.utils import (
    bootstrap_rewards_with_last_values,
    compute_values,
    discounted_cumulative_sums,
    gae,
    normalize_tensor,
)
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class TRPO:
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
        self.policy = policy
        self.value_function = value_function
        self.env = env
        self.sampler = sampler
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_value_gradients = num_value_gradients

        self.old_policy: Policy = copy.deepcopy(self.policy)

    def learn(
        self,
        num_epochs: int = 50,
        batch_size: int = 4000,
        model_saving_interval: int = 4000,
        output_dir: str = ".",
    ) -> None:
        """
        Learn the model

        :param num_epochs: (int) The number of epochs to run and train.
        :param batch_size: (int) The number of steps to run per epoch.
        :param model_saving_interval: (int) The interval steps between model saving.
        :param output_dir: (str) The output directory.
        """
        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        os.makedirs(output_dir, exist_ok=True)

        self.metrics_manager: MetricsManager = MetricsManager(output_dir)

        for current_epoch in range(1, num_epochs + 1):
            experience: Experience = self.sampler.sample(batch_size, self.policy)

            episode_returns: List[float] = experience.episode_returns
            episode_lengths: List[int] = experience.episode_lengths

            self.current_total_steps += sum(experience.episode_lengths)
            self.current_total_episodes += sum(experience.episode_dones)

            self.metrics_manager.record_scalar("epoch", current_epoch)
            self.metrics_manager.record_scalar("total_steps", self.current_total_steps)
            self.metrics_manager.record_scalar(
                "total_episodes", self.current_total_episodes
            )

            self.metrics_manager.record_scalar(
                "sampling/average_episode_return",
                float(np.mean(episode_returns)),
                self.current_total_steps,
                tensorboard=True,
            )
            self.metrics_manager.record_scalar(
                "sampling/episode_return_std", float(np.std(episode_returns))
            )
            self.metrics_manager.record_scalar(
                "sampling/max_episode_return", float(np.max(episode_returns))
            )
            self.metrics_manager.record_scalar(
                "sampling/min_episode_return", float(np.min(episode_returns))
            )
            self.metrics_manager.record_scalar(
                "sampling/average_episode_length",
                float(np.mean(episode_lengths)),
                self.current_total_steps,
                tensorboard=True,
            )

            self.train(experience)

            if self.current_total_steps % model_saving_interval == 0:
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.debug("Save model")
                self.save_model(current_epoch, model_path)

            self.metrics_manager.record_scalar("time", time.time() - start_time)

            # Dump all metrics stored in this epoch
            self.metrics_manager.dump()

        self.metrics_manager.close()

    def train(self, experience: Experience) -> None:
        values: List[np.ndarray] = compute_values(
            experience.observations_with_last_observation, self.value_function
        )

        last_values: List[float] = [
            float(episode_values[-1]) for episode_values in values
        ]

        bootstrapped_rewards: List[np.ndarray] = bootstrap_rewards_with_last_values(
            experience.rewards, experience.episode_dones, last_values
        )

        discounted_returns: List[np.ndarray] = [
            discounted_cumulative_sums(episode_rewards, self.gamma)[:-1]
            for episode_rewards in bootstrapped_rewards
        ]
        flattened_discounted_returns: Tensor = torch.from_numpy(
            np.concatenate(discounted_returns)
        ).float()

        flattened_observations: Tensor = torch.from_numpy(
            np.concatenate(experience.observations)
        ).float()
        flattened_actions: Tensor = torch.from_numpy(
            np.concatenate(experience.actions)
        ).float()

        gaes: List[np.ndarray] = [
            gae(episode_rewards, self.gamma, episode_values, self.gae_lambda)
            for episode_rewards, episode_values in zip(bootstrapped_rewards, values)
        ]
        flattened_advantages: Tensor = torch.from_numpy(np.concatenate(gaes)).float()
        flattened_advantages = normalize_tensor(flattened_advantages)

        def compute_surrogate_loss() -> Tensor:
            policy_dist: Distribution = self.policy(flattened_observations)
            log_probs: Tensor = policy_dist.log_prob(flattened_actions)

            with torch.no_grad():
                old_policy_dist: Distribution = self.old_policy(flattened_observations)
                old_log_probs: Tensor = old_policy_dist.log_prob(flattened_actions)

            likelihood_ratio: Tensor = torch.exp(log_probs - old_log_probs)
            surrogate_loss: Tensor = -torch.mean(
                likelihood_ratio * flattened_advantages
            )

            return surrogate_loss

        def compute_kl_constraint() -> Tensor:
            policy_dist: Distribution = self.policy(flattened_observations)

            with torch.no_grad():
                old_policy_dist: Distribution = self.old_policy(flattened_observations)

            kl_constraint: Tensor = kl.kl_divergence(old_policy_dist, policy_dist)

            return torch.mean(kl_constraint)

        # For logging
        policy_loss_before: Tensor = compute_surrogate_loss()
        with torch.no_grad():
            policy_dist_before: Distribution = self.policy(flattened_observations)
        log_probs_before: Tensor = policy_dist_before.log_prob(flattened_actions)
        entropies_before: Tensor = policy_dist_before.entropy()

        self.train_policy(
            compute_surrogate_loss,
            compute_kl_constraint,
            flattened_observations,
            flattened_actions,
            flattened_advantages,
        )

        self.old_policy.load_state_dict(self.policy.state_dict())

        # Train value function
        value_function_losses: List[float] = []
        for _ in range(self.num_value_gradients):
            value_function_loss: Tensor = self.train_value_function(
                flattened_observations, flattened_discounted_returns
            )

            value_function_losses.append(value_function_loss.item())

        self.metrics_manager.record_scalar(
            "policy/loss",
            policy_loss_before.item(),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "policy/avarage_entropy",
            torch.mean(entropies_before).item(),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "policy/log_prob_std",
            torch.std(log_probs_before).item(),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "value_function/average_loss",
            float(np.mean(value_function_losses)),
            self.current_total_steps,
            tensorboard=True,
        )

    def train_policy(
        self,
        compute_surrogate_loss: Callable,
        compute_kl_constraint: Callable,
        flattened_observations: Tensor,
        flattened_actions: Tensor,
        flattened_advantages: Tensor,
    ) -> None:
        policy_loss: Tensor = compute_surrogate_loss()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step(compute_surrogate_loss, compute_kl_constraint)

    def train_value_function(
        self, flattened_observations: Tensor, flattened_discounted_returns: Tensor
    ) -> Tensor:
        value_function_loss: Tensor = self.compute_value_function_loss(
            flattened_observations, flattened_discounted_returns
        )

        self.value_function.optimizer.zero_grad()
        value_function_loss.backward()
        self.value_function.optimizer.step()

        return value_function_loss.detach()

    def compute_value_function_loss(
        self, observations: Tensor, discounted_returns: Tensor
    ) -> Tensor:
        values: Tensor = self.value_function(observations)
        squeezed_values: Tensor = torch.squeeze(values, -1)
        value_loss: Tensor = F.mse_loss(squeezed_values, discounted_returns)

        return value_loss

    def save_model(self, epoch: int, model_path: str) -> None:
        """
        Save model

        :param epoch: (int) The current epoch.
        :param model_path: (int) The path to save the model.
        """
        torch.save(
            {
                "epoch": epoch,
                "total_steps": self.current_total_steps,
                "policy_state_dict": self.policy.network.state_dict(),
                "policy_optimizer_state_dict": self.policy.optimizer.state_dict(),
                "value_function_state_dict": self.value_function.network.state_dict(),
                "value_function_optimizer_state_dict": self.value_function.optimizer.state_dict(),
            },
            model_path,
        )
