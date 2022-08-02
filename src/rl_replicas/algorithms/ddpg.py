import copy
import logging
import os
import time
from typing import Dict, List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.evaluator import Evaluator
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import Sampler
from rl_replicas.utils import add_noise_to_get_action, polyak_average

logger = logging.getLogger(__name__)


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG)

    :param policy: (Policy) Policy.
    :param exploration_policy: (Policy) Exploration policy.
    :param q_function: (QFunction) Q function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param replay_buffer: (ReplayBuffer) Replay buffer.
    :param evaluator: (Evaluator) Evaluator.
    :param gamma: (float) The discount factor for the cumulative return.
    :param tau: (float) The interpolation factor in polyak averaging for target networks.
    :param action_noise_scale: (float) The scale of the noise (std).
    """

    def __init__(
        self,
        policy: Policy,
        exploration_policy: Policy,
        q_function: QFunction,
        env: gym.Env,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        evaluator: Evaluator,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_noise_scale: float = 0.1,
    ) -> None:
        self.policy = policy
        self.exploration_policy = exploration_policy
        self.q_function = q_function
        self.env = env
        self.sampler = sampler
        self.replay_buffer = replay_buffer
        self.evaluator = evaluator
        self.gamma = gamma
        self.tau = tau
        self.action_noise_scale = action_noise_scale

        self.noised_policy = add_noise_to_get_action(
            self.policy, self.env.action_space, self.action_noise_scale
        )
        self.evaluation_env = gym.make(env.spec.id)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_function = copy.deepcopy(self.q_function)

        for param in self.target_policy.network.parameters():
            param.requires_grad = False
        for param in self.target_q_function.network.parameters():
            param.requires_grad = False

    def learn(
        self,
        num_epochs: int = 2000,
        batch_size: int = 50,
        minibatch_size: int = 100,
        num_random_start_steps: int = 10000,
        num_steps_before_update: int = 1000,
        num_train_steps: int = 50,
        num_evaluation_episodes: int = 5,
        evaluation_interval: int = 4000,
        output_dir: str = ".",
        model_saving: bool = False,
    ) -> None:
        """
        Learn the model

        :param num_epochs: (int) The number of epochs to run and train.
        :param batch_size: (int) The number of steps to run per epoch.
        ;param minibatch_size: (int) The minibatch size for SGD.
        :param num_random_start_steps: (int) The number of steps for uniform-random action selection for exploration
            at the beginning.
        :param num_steps_before_update: (int) The number of steps to perform before policy is updated.
        :param num_train_steps: (int) The number of training steps on each epoch.
        :param num_evaluation_episodes: (int) The number of evaluation episodes.
        :param evaluation_interval: (int) The interval steps between evaluation.
        :param output_dir: (str) The output directory.
        :param model_saving: (bool) Whether or not to save the trained model (overwrite at each end of epoch).
        """
        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        os.makedirs(output_dir, exist_ok=True)

        tensorboard_path: str = os.path.join(output_dir, "tensorboard")
        self.writer: SummaryWriter = SummaryWriter(tensorboard_path)

        for current_epoch in range(num_epochs):
            experience: Experience
            if self.current_total_steps < num_random_start_steps:
                experience = self.sampler.sample(batch_size, self.exploration_policy)
            else:
                experience = self.sampler.sample(batch_size, self.noised_policy)

            self.replay_buffer.add_experience(
                experience.flattened_observations,
                experience.flattened_actions,
                experience.flattened_rewards,
                experience.flattened_next_observations,
                experience.flattened_dones,
            )

            if model_saving:
                os.makedirs(output_dir, exist_ok=True)
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.debug("Save model")
                self.save_model(current_epoch, model_path)

            episode_returns: List[float] = experience.episode_returns
            episode_lengths: List[int] = experience.episode_lengths

            self.current_total_steps += sum(experience.episode_lengths)
            self.current_total_episodes += sum(experience.flattened_dones)

            logger.info("Epoch: {}".format(current_epoch))

            logger.info(
                "Total steps:            {:<8.3g}".format(self.current_total_steps)
            )
            logger.info(
                "Total episodes:         {:<8.3g}".format(self.current_total_episodes)
            )

            if len(episode_lengths) > 0:
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

            if self.current_total_steps >= num_steps_before_update:
                self.train(self.replay_buffer, num_train_steps, minibatch_size)

            if (
                num_evaluation_episodes > 0
                and self.current_total_steps % evaluation_interval == 0
            ):
                evaluation_results: Dict[str, List] = self.evaluator.evaluate(
                    self.policy, self.evaluation_env, num_evaluation_episodes
                )
                logger.info(
                    "Average Evaluation Episode Return: {:<8.3g}".format(
                        np.mean(evaluation_results["episode_returns"])
                    )
                )
                logger.info(
                    "Evaluation Episode Return STD:     {:<8.3g}".format(
                        np.std(evaluation_results["episode_returns"])
                    )
                )
                logger.info(
                    "Max Evaluation Episode Return:     {:<8.3g}".format(
                        np.max(evaluation_results["episode_returns"])
                    )
                )
                logger.info(
                    "Min Evaluation Episode Return:     {:<8.3g}".format(
                        np.min(evaluation_results["episode_returns"])
                    )
                )
                logger.info(
                    "Average Evaluation Episode Length: {:<8.3g}".format(
                        np.mean(evaluation_results["episode_lengths"])
                    )
                )

                self.writer.add_scalar(
                    "evaluation/average_episode_return",
                    np.mean(evaluation_results["episode_returns"]),
                    self.current_total_steps,
                )
                self.writer.add_scalar(
                    "evaluation/average_episode_length",
                    np.mean(evaluation_results["episode_lengths"]),
                    self.current_total_steps,
                )

            logger.info(
                "Time:                   {:<8.3g}".format(time.time() - start_time)
            )

        self.writer.close()

    def train(
        self, replay_buffer: ReplayBuffer, num_train_steps: int, minibatch_size: int
    ) -> None:
        policy_losses: List[float] = []
        q_function_losses: List[float] = []
        all_q_values: List[float] = []

        for _ in range(num_train_steps):
            minibatch: Dict[str, np.ndarray] = replay_buffer.sample_minibatch(
                minibatch_size
            )

            observations: Tensor = torch.from_numpy(minibatch["observations"])
            actions: Tensor = torch.from_numpy(minibatch["actions"])
            rewards: Tensor = torch.from_numpy(minibatch["rewards"]).float()
            next_observations: Tensor = torch.from_numpy(minibatch["next_observations"])
            dones: Tensor = torch.from_numpy(minibatch["dones"]).int()

            q_values: Tensor = self.q_function(observations, actions)
            all_q_values.extend(q_values.tolist())

            with torch.no_grad():
                next_actions: Tensor = self.target_policy(next_observations)
                target_q_values: Tensor = self.target_q_function(
                    next_observations, next_actions
                )

                targets: Tensor = rewards + self.gamma * (1 - dones) * target_q_values

            q_function_loss: Tensor = F.mse_loss(q_values, targets)
            q_function_losses.append(q_function_loss.item())

            self.q_function.optimizer.zero_grad()
            q_function_loss.backward()
            self.q_function.optimizer.step()

            policy_actions: Tensor = self.policy(observations)

            # Freeze Q-network so you don't waste computational effort
            for param in self.q_function.network.parameters():
                param.requires_grad = False

            policy_q_values: Tensor = self.q_function(observations, policy_actions)

            policy_loss: Tensor = -torch.mean(policy_q_values)
            policy_losses.append(policy_loss.item())

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

            # Unfreeze Q-network
            for param in self.q_function.network.parameters():
                param.requires_grad = True

            polyak_average(
                self.policy.network.parameters(),
                self.target_policy.network.parameters(),
                self.tau,
            )
            polyak_average(
                self.q_function.network.parameters(),
                self.target_q_function.network.parameters(),
                self.tau,
            )

        logger.info("Average Policy Loss:    {:<8.3g}".format(np.mean(policy_losses)))
        logger.info(
            "Average Q Function Loss:{:<8.3g}".format(np.mean(q_function_losses))
        )

        logger.info("Average Q Value:        {:<8.3g}".format(np.mean(all_q_values)))
        logger.info("Max Q Value:            {:<8.3g}".format(np.max(all_q_values)))
        logger.info("Min Q Value:            {:<8.3g}".format(np.min(all_q_values)))

        self.writer.add_scalar(
            "policy/average_loss",
            np.mean(policy_losses),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "q-function/average_loss",
            np.mean(q_function_losses),
            self.current_total_steps,
        )
        self.writer.add_scalar(
            "q-function/avarage_q-value",
            torch.mean(q_values),
            self.current_total_steps,
        )

    def save_model(self, current_epoch: int, model_path: str) -> None:
        """
        Save model

        :param current_epoch: (int) The current epoch.
        :param model_path: (int) The path to save the model.
        """
        torch.save(
            {
                "epoch": current_epoch,
                "total_steps": self.current_total_steps,
                "policy_state_dict": self.policy.network.state_dict(),
                "policy_optimizer_state_dict": self.policy.optimizer.state_dict(),
                "target_policy_state_dict": self.target_policy.network.state_dict(),
                "q_function_state_dict": self.q_function.network.state_dict(),
                "q_function_optimizer_state_dict": self.q_function.optimizer.state_dict(),
                "target_q_function_state_dict": self.target_q_function.network.state_dict(),
            },
            model_path,
        )
