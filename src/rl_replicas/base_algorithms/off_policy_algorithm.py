import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.evaluator import Evaluator
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import Sampler

logger = logging.getLogger(__name__)


class OffPolicyAlgorithm(ABC):
    """
    Base class for off-policy algorithms

    :param policy: (Policy) Policy.
    :param exploration_policy: (Policy) Exploration policy.
    :param q_function: (QFunction) Q function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param replay_buffer: (ReplayBuffer) Replay buffer.
    :param evaluator: (Evaluator) Evaluator.
    :param gamma: (float) The discount factor for the cumulative return.
    :param tau: (float) The interpolation factor in polyak averaging for target networks.
    :param action_noise_scale: (float) The scale of the action noise (std).
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
        gamma: float,
        tau: float,
        action_noise_scale: float,
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
        tensorboard: bool = False,
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
        :param tensorboard: (bool) Whether or not to log for tensorboard.
        :param model_saving: (bool) Whether or not to save the trained model (overwrite at each end of epoch).
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

        for current_epoch in range(num_epochs):
            experience: Experience
            if self.current_total_steps < num_random_start_steps:
                experience = self.sampler.sample(batch_size, self.exploration_policy)
            else:
                experience = self.sampler.sample(batch_size, self.policy)

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

                if self.tensorboard:
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

            if self.current_total_steps >= num_steps_before_update:
                self.train(self.replay_buffer, num_train_steps, minibatch_size)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

    @abstractmethod
    def train(
        self, replay_buffer: ReplayBuffer, num_train_steps: int, minibatch_size: int
    ) -> None:
        """
        Train the algorithm with the experience

        :param replay_buffer: (ReplayBuffer) Reply buffer.
        :param num_train_steps: (int) The number of gradient descent updates.
        :param minibatch_size: (int) The minibatch size.
        """
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action(s) given observation(s)

        :param observation: (np.ndarray) Observation(s).
        :return: (np.ndarray) Action(s).
        """
        action: np.ndarray = self.policy.get_action_numpy(observation)

        return action

    def select_action_with_noise(
        self, observation: np.ndarray, action_noise_scale: float
    ) -> np.ndarray:
        """
        Select action(s) with observation(s) and add noise.

        :param observation: (np.ndarray) Observation(s).
        :param action_noise_scale: (float) The scale of the action noise (std).
        :return: (np.ndarray) Action(s).
        """
        action_limit: float = self.env.action_space.high[0]
        action_size: int = self.env.action_space.shape[0]

        action = action_limit * self.predict(observation)
        action += action_noise_scale * np.random.randn(action_size)
        action = np.clip(action, -action_limit, action_limit)

        return action

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
