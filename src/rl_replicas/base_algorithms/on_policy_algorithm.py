import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import gym
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.utils import seed_random_generators
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class OnPolicyAlgorithm(ABC):
    """
    Base class for on-policy algorithms

    :param policy: (Policy) Policy.
    :param value_function: (ValueFunction) Value function.
    :param env: (gym.Env) Environment.
    :param gamma: (float) The discount factor for the cumulative return.
    :param gae_lambda: (float) The factor for trade-off of bias vs variance for GAE.
    :param seed: (int) The seed for the pseudo-random generators.
    :param n_value_gradients (int): The number of gradient descent steps to take on value function per epoch.
    """

    def __init__(
        self,
        policy: Policy,
        value_function: ValueFunction,
        env: gym.Env,
        gamma: float,
        gae_lambda: float,
        seed: Optional[int],
        n_value_gradients: int,
    ) -> None:
        self.policy = policy
        self.value_function = value_function
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        if seed is not None:
            self.seed: int = seed
        self.n_value_gradients = n_value_gradients

        if seed is not None:
            self._seed()

    def _seed(self) -> None:
        seed_random_generators(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.seed(self.seed)

    def learn(
        self,
        epochs: int = 50,
        steps_per_epoch: int = 4000,
        output_dir: str = ".",
        tensorboard: bool = False,
        model_saving: bool = False,
    ) -> None:
        """
        Learn the model

        :param epochs: (int) The number of epochs to run and train.
        :param steps_per_epoch: (int) The number of steps to run per epoch.
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

        for current_epoch in range(epochs):

            one_epoch_experience: Experience = self.collect_one_epoch_experience(
                steps_per_epoch
            )

            if model_saving:
                logger.info("Set up model saving")
                os.makedirs(output_dir, exist_ok=True)
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.info("Save model")
                self.save_model(current_epoch, model_path)

            episode_returns: List[float] = one_epoch_experience.episode_returns
            episode_lengths: List[int] = one_epoch_experience.episode_lengths

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

            self.train(one_epoch_experience)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

    def collect_one_epoch_experience(self, steps_per_epoch: int) -> Experience:
        """
        Collect experience for one epoch

        :param steps_per_epoch: (int) The number of steps to run per epoch; in other words, batch size is
            steps_per_epoch.
        :param random_start_steps: (int) The number of steps for uniform-random action selection for exploration
            at the beginning.
        :return: (Experience) Collected experience.
        """
        one_epoch_experience: Experience = Experience()

        # Variables on each episode
        episode_observations: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_return: float = 0.0
        episode_length: int = 0

        observation: np.ndarray = self.env.reset()

        for current_step in range(steps_per_epoch):
            episode_observations.append(observation)

            action: np.ndarray = self.predict(observation)
            episode_actions.append(action)

            reward: float
            episode_done: bool
            observation, reward, episode_done, _ = self.env.step(action)

            episode_return += reward
            episode_rewards.append(reward)

            episode_length += 1
            self.current_total_steps += 1
            epoch_ended: bool = current_step == steps_per_epoch - 1

            if episode_done or epoch_ended:
                if epoch_ended and not episode_done:
                    logger.debug(
                        "The trajectory cut off at {} steps on the current episode".format(
                            episode_length
                        )
                    )

                episode_last_observation: np.ndarray = observation

                one_epoch_experience.observations.append(episode_observations)
                one_epoch_experience.actions.append(episode_actions)
                one_epoch_experience.rewards.append(episode_rewards)
                one_epoch_experience.last_observations.append(episode_last_observation)
                one_epoch_experience.dones.append(episode_done)

                one_epoch_experience.episode_returns.append(episode_return)
                one_epoch_experience.episode_lengths.append(episode_length)

                if episode_done:
                    self.current_total_episodes += 1

                observation = self.env.reset()
                episode_return, episode_length = 0.0, 0
                (
                    episode_observations,
                    episode_actions,
                    episode_rewards,
                ) = ([], [], [])

        return one_epoch_experience

    @abstractmethod
    def train(self, one_epoch_experience: Experience) -> None:
        """
        Train the algorithm with the experience

        :param one_epoch_experience: (Experience) Collected experience for one epoch.
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
