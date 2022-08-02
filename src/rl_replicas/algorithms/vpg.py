import logging
import os
import time
from typing import List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler
from rl_replicas.utils import (
    bootstrap_rewards_with_last_values,
    compute_values_numpy_list,
    discounted_cumulative_sums,
    gae,
    normalize_tensor,
)
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class VPG:
    """
    Vanilla Policy Gradient (REINFORCE) with Generalized Advantage Estimator (GAE)

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

    def learn(
        self,
        num_epochs: int = 50,
        batch_size: int = 4000,
        output_dir: str = ".",
        model_saving: bool = False,
    ) -> None:
        """
        Learn the model

        :param num_epochs: (int) The number of epochs to run and train.
        :param batch_size: (int) The number of steps to run per epoch.
        :param output_dir: (str) The output directory.
        :param model_saving: (bool) Whether or not to save trained model (Save and overwrite at each end of epoch).
        """
        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        os.makedirs(output_dir, exist_ok=True)

        tensorboard_path: str = os.path.join(output_dir, "tensorboard")
        self.writer: SummaryWriter = SummaryWriter(tensorboard_path)

        for current_epoch in range(num_epochs):
            experience: Experience = self.sampler.sample(batch_size, self.policy)

            if model_saving:
                os.makedirs(output_dir, exist_ok=True)
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.debug("Save model")
                self.save_model(current_epoch, model_path)

            episode_returns: List[float] = experience.episode_returns
            episode_lengths: List[int] = experience.episode_lengths

            self.current_total_steps += sum(experience.episode_lengths)
            self.current_total_episodes += sum(experience.episode_dones)

            logger.info("Epoch: {}".format(current_epoch))

            logger.info(
                "Total steps:            {:<8.3g}".format(self.current_total_steps)
            )
            logger.info(
                "Total episodes:         {:<8.3g}".format(self.current_total_episodes)
            )

            logger.info(
                "Average Episode Return: {:<8.3g}".format(np.mean(episode_returns))
            )
            logger.info(
                "Episode Return STD:     {:<8.3g}".format(np.std(episode_returns))
            )
            logger.info(
                "Max Episode Return:     {:<8.3g}".format(np.max(episode_returns))
            )
            logger.info(
                "Min Episode Return:     {:<8.3g}".format(np.min(episode_returns))
            )

            logger.info(
                "Average Episode Length: {:<8.3g}".format(np.mean(episode_lengths))
            )

            self.writer.add_scalar(
                "training/average_episode_return",
                np.mean(episode_returns),
                self.current_total_steps,
            )
            self.writer.add_scalar(
                "training/average_episode_length",
                np.mean(episode_lengths),
                self.current_total_steps,
            )

            self.train(experience)

            logger.info(
                "Time:                   {:<8.3g}".format(time.time() - start_time)
            )

        self.writer.close()

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
            for episode_rewards, episode_values in zip(
                bootstrapped_rewards, values_numpy_list
            )
        ]
        flattened_advantages: Tensor = torch.from_numpy(np.concatenate(gaes)).float()
        flattened_advantages = normalize_tensor(flattened_advantages)

        # For logging
        with torch.no_grad():
            policy_dist_before: Distribution = self.policy(flattened_observations)
        log_probs_before: Tensor = policy_dist_before.log_prob(flattened_actions)
        policy_loss_before: Tensor = -torch.mean(
            log_probs_before * flattened_advantages
        )
        entropies_before: Tensor = policy_dist_before.entropy()

        self.train_policy(
            flattened_observations, flattened_actions, flattened_advantages
        )

        # Train value function
        value_function_losses: List[float] = []
        for _ in range(self.num_value_gradients):
            value_function_loss: Tensor = self.compute_value_function_loss(
                flattened_observations, flattened_discounted_returns
            )
            self.value_function.optimizer.zero_grad()
            value_function_loss.backward()
            self.value_function.optimizer.step()

            value_function_losses.append(value_function_loss.detach().item())

        logger.info("Policy Loss:            {:<8.3g}".format(policy_loss_before))
        logger.info(
            "Avarage Entropy:        {:<8.3g}".format(torch.mean(entropies_before))
        )
        logger.info(
            "Log Prob STD:           {:<8.3g}".format(torch.std(log_probs_before))
        )

        logger.info(
            "Average Value Function Loss: {:<8.3g}".format(
                np.mean(value_function_losses)
            )
        )

        self.writer.add_scalar(
            "policy/loss", policy_loss_before, self.current_total_steps
        )
        self.writer.add_scalar(
            "policy/avarage_entropy",
            torch.mean(entropies_before),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "policy/log_prob_std", torch.std(log_probs_before), self.current_total_steps
        )

        self.writer.add_scalar(
            "value_function/average_loss",
            np.mean(value_function_losses),
            self.current_total_steps,
        )

    def train_policy(
        self,
        flattened_observations: Tensor,
        flattened_actions: Tensor,
        flattened_advantages: Tensor,
    ) -> None:
        policy_dist: Distribution = self.policy(flattened_observations)
        log_probs: Tensor = policy_dist.log_prob(flattened_actions)

        policy_loss: Tensor = -torch.mean(log_probs * flattened_advantages)

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

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
