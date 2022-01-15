import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import gym
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.common.policies import Policy
from rl_replicas.common.q_function import QFunction
from rl_replicas.common.replay_buffer import ReplayBuffer
from rl_replicas.common.utils import seed_random_generators

logger = logging.getLogger(__name__)


class OffPolicyAlgorithm(ABC):
    """
    The base of off-policy algorithms

    :param policy: (Policy) The policy
    :param q_function: (QFunction) The Q function
    :param env: (gym.Env) The environment to learn from
    :param gamma: (float) The discount factor for the cumulative return
    :param tau: (float) The interpolation factor in polyak averaging for target networks
    :param action_noise_scale: (float) The scale of the action noise (std)
    :param seed: (int) The seed for the pseudo-random generators
    """

    def __init__(
        self,
        policy: Policy,
        q_function: QFunction,
        env: gym.Env,
        gamma: float,
        tau: float,
        action_noise_scale: float,
        seed: Optional[int],
    ) -> None:
        self.policy = policy
        self.q_function = q_function
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.action_noise_scale = action_noise_scale
        if seed is not None:
            self.seed: int = seed

        self.evaluation_env = gym.make(env.spec.id)
        self.action_limit: float = self.env.action_space.high[0]
        self.action_size: int = self.env.action_space.shape[0]
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_function = copy.deepcopy(self.q_function)
        if seed is not None:
            self._seed()

        for param in self.target_policy.network.parameters():
            param.requires_grad = False
        for param in self.target_q_function.network.parameters():
            param.requires_grad = False

    def _seed(self) -> None:
        seed_random_generators(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.seed(self.seed)

    def learn(
        self,
        epochs: int = 2000,
        steps_per_epoch: int = 50,
        replay_buffer_size: int = int(1e6),
        minibatch_size: int = 100,
        random_start_steps: int = 10000,
        steps_before_update: int = 1000,
        train_steps: int = 50,
        num_evaluation_episodes: int = 5,
        evaluation_interval: int = 4000,
        output_dir: str = ".",
        tensorboard: bool = False,
        model_saving: bool = False,
    ) -> None:
        """
        Learn the model

        :param epochs: (int) The number of epochs to run and train.
        :param steps_per_epoch: (int) The number of steps to run per epoch; in other words, batch size is
            steps_per_epoch.
        :param replay_size: (int) The size of the replay buffer
        ;param minibatch_size: (int) The minibatch size for SGD.
        :param random_start_steps: (int) The number of steps for uniform-random action selection for exploration
            at the beginning.
        :param steps_before_update: (int) The number of steps to perform before policy is updated.
        :param train_steps: (int) The number of training steps on each epoch
        :param num_evaluation_episodes: (int) The number of evaluation episodes
        :param evaluation_interval: (int) The interval steps of evaluation
        :param output_dir: (str) The directory of output
        :param tensorboard: (bool) Whether or not to log for tensorboard
        :param model_saving: (bool) Whether or not to save trained model (Save and overwrite at each end of epoch)
        """
        self.tensorboard = tensorboard

        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        if self.tensorboard:
            logger.info("Set up tensorboard")
            os.makedirs(output_dir, exist_ok=True)
            tensorboard_path: str = os.path.join(output_dir, "tensorboard")
            self.writer: SummaryWriter = SummaryWriter(tensorboard_path)

        self.replay_buffer: ReplayBuffer = ReplayBuffer(replay_buffer_size)

        for current_epoch in range(epochs):
            episode_returns: List[float]
            episode_lengths: List[int]
            episode_returns, episode_lengths = self.collect_one_epoch_experience(
                self.replay_buffer, steps_per_epoch, random_start_steps
            )

            if model_saving:
                logger.info("Set up model saving")
                os.makedirs(output_dir, exist_ok=True)
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.info("Save model")
                self.save_model(current_epoch, model_path)

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

                if self.tensorboard:
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

            if (
                num_evaluation_episodes > 0
                and self.current_total_steps % evaluation_interval == 0
            ):
                self.evaluate_policy(num_evaluation_episodes, self.evaluation_env)

            logger.info(
                "Time:                   {:<8.3g}".format(time.time() - start_time)
            )

            if self.current_total_steps >= steps_before_update:
                self.train(self.replay_buffer, train_steps, minibatch_size)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

    def collect_one_epoch_experience(
        self, replay_buffer: ReplayBuffer, steps_per_epoch: int, random_start_steps: int
    ) -> Tuple[List[float], List[int]]:
        observations_list: List[Tensor] = []
        actions_list: List[Tensor] = []
        next_observations_list: List[Tensor] = []

        rewards: List[float] = []
        dones: List[bool] = []

        episode_returns: List[float] = []
        episode_lengths: List[int] = []

        if not hasattr(self, "observation"):
            # Variables on the current episode
            self.episode_length: int = 0
            self.episode_return: float = 0

            self.observation: np.ndarray = self.env.reset()

        for current_step in range(steps_per_epoch):
            observation_tensor: Tensor = torch.from_numpy(self.observation).float()
            observations_list.append(observation_tensor)

            action: np.ndarray
            if self.current_total_steps < random_start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.action_limit * self.predict(self.observation)
                action += self.action_noise_scale * np.random.randn(self.action_size)
                action = np.clip(action, -self.action_limit, self.action_limit)

            action_tensor: Tensor = torch.from_numpy(action).float()
            actions_list.append(action_tensor)

            next_observation: np.ndarray
            reward: float
            episode_done: bool
            next_observation, reward, episode_done, _ = self.env.step(action)

            next_observation_tensor: Tensor = torch.from_numpy(next_observation).float()
            next_observations_list.append(next_observation_tensor)

            self.observation = next_observation

            rewards.append(reward)
            dones.append(episode_done)

            self.current_total_steps += 1
            self.episode_length += 1
            self.episode_return += reward

            if episode_done:
                self.current_total_episodes += 1

                episode_returns.append(self.episode_return)
                episode_lengths.append(self.episode_length)

                self.observation, self.episode_length, self.episode_return = (
                    self.env.reset(),
                    0,
                    0,
                )

        replay_buffer.add_one_epoch_experience(
            observations_list, actions_list, rewards, next_observations_list, dones
        )

        return episode_returns, episode_lengths

    @abstractmethod
    def train(
        self, replay_buffer: ReplayBuffer, train_steps: int, minibatch_size: int
    ) -> None:
        """
        Train the algorithm with the experience.

        :param replay_buffer: (ReplayBuffer) The reply buffer
        :param train_steps: (int) The number of gradient descent updates
        :param minibatch_size: (int) The minibatch size
        """
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Get the action(s) from an observation which are sampled under the current policy.

        :param observation: (np.ndarray) The input observation
        :return: (np.ndarray) The action(s)
        """
        observation_tensor: Tensor = torch.from_numpy(observation).float()
        action: Tensor = self.policy.predict(observation_tensor)
        action_ndarray: np.ndarray = action.detach().numpy()

        return action_ndarray

    def evaluate_policy(
        self,
        num_evaluation_episodes: int,
        evaluation_env: gym.Env,
    ) -> None:
        episode_returns: List[float] = []
        episode_lengths: List[int] = []

        for _ in range(num_evaluation_episodes):
            observation: np.ndarray = evaluation_env.reset()
            done: bool = False
            episode_return: float = 0.0
            episode_length: int = 0

            while not done:
                action: np.ndarray = self.predict(observation)

                reward: float
                observation, reward, done, _ = evaluation_env.step(action)

                episode_return += reward
                episode_length += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        logger.info(
            "Average Evaluation Episode Return: {:<8.3g}".format(
                np.mean(episode_returns)
            )
        )
        logger.info(
            "Evaluation Episode Return STD:     {:<8.3g}".format(
                np.std(episode_returns)
            )
        )
        logger.info(
            "Max Evaluation Episode Return:     {:<8.3g}".format(
                np.max(episode_returns)
            )
        )
        logger.info(
            "Min Evaluation Episode Return:     {:<8.3g}".format(
                np.min(episode_returns)
            )
        )

        logger.info(
            "Average Evaluation Episode Length: {:<8.3g}".format(
                np.mean(episode_lengths)
            )
        )

        if self.tensorboard:
            self.writer.add_scalar(
                "evaluation/average_episode_return",
                np.mean(episode_returns),
                self.current_total_steps,
            )
            self.writer.add_scalar(
                "evaluation/average_episode_length",
                np.mean(episode_lengths),
                self.current_total_steps,
            )

    def save_model(self, current_epoch: int, model_path: str) -> None:
        """
        Save model

        :param current_epoch: (int) The current epoch
        :param model_path: (int) The path to save the model
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
