import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class OnPolicyAlgorithm(ABC):
    """
    Base class for on-policy algorithms

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
        gamma: float,
        gae_lambda: float,
        num_value_gradients: int,
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
        tensorboard: bool = False,
        model_saving: bool = False,
    ) -> None:
        """
        Learn the model

        :param num_epochs: (int) The number of epochs to run and train.
        :param batch_size: (int) The number of steps to run per epoch.
        :param output_dir: (str) The output directory.
        :param tensorboard: (bool) Whether or not to log for tensorboard.
        :param model_saving: (bool) Whether or not to save trained model (Save and overwrite at each end of epoch).
        """
        self.tensorboard = tensorboard

        if self.tensorboard:
            logger.info("Set up tensorboard")
            os.makedirs(output_dir, exist_ok=True)
            tensorboard_path: str = os.path.join(output_dir, "tensorboard")
            self.writer: SummaryWriter = SummaryWriter(tensorboard_path)

        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        for current_epoch in range(num_epochs):
            experience: Experience = self.sampler.sample(batch_size, self.policy)

            if model_saving:
                logger.info("Set up model saving")
                os.makedirs(output_dir, exist_ok=True)
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.info("Save model")
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

            logger.info(
                "Time:                   {:<8.3g}".format(time.time() - start_time)
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

            self.train(experience)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

    @abstractmethod
    def train(self, experience: Experience) -> None:
        """
        Train the algorithm with the experience

        :param experience: (Experience) Collected experience.
        """
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action(s) given observation(s)

        :param observation: (np.ndarray) Observation(s).
        :return: (np.ndarray) Action(s).
        """
        observation_tensor: Tensor = torch.from_numpy(observation).float()
        action: Tensor = self.policy.predict(observation_tensor)
        action_ndarray: np.ndarray = action.detach().numpy()

        return action_ndarray

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
