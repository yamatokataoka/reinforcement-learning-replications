import logging
from typing import List, Optional

import gym
import numpy as np

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.samplers import Sampler

logger = logging.getLogger(__name__)


class BatchSampler(Sampler):
    """
    Batch sampler

    :param env: (gym.Env) Environment.
    :param seed: (int) Seed.
    :param is_continuous: (bool) If true, observation is retained and
        it samples experiences continuously across sample function calls.
    """

    def __init__(
        self, env: gym.Env, seed: Optional[int] = None, is_continuous: bool = False
    ):
        self.env = env
        self.seed = seed
        self.is_continuous = is_continuous

        self.observation: Optional[np.ndarray] = None

    def sample(self, num_samples: int, policy: Policy) -> Experience:
        """
        Sample experience

        :param num_samples: (int) The number of samples to collect.
        :param policy: (int) Policy.
        :return: (Experience) Sampled experience.
        """
        experience: Experience = Experience()

        # Variables on each episode
        episode_observations: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_dones: List[bool] = []
        episode_return: float = 0.0
        episode_length: int = 0

        if self.observation is None:
            # Reset env for the first function call
            self.observation = self.env.reset(seed=self.seed)
        elif not self.is_continuous:
            self.observation = self.env.reset()

        for current_step in range(num_samples):
            episode_observations.append(self.observation)

            action: np.ndarray = policy.get_action_numpy(self.observation)
            episode_actions.append(action)

            reward: float
            episode_done: bool
            self.observation, reward, episode_done, _ = self.env.step(action)

            episode_return += reward
            episode_rewards.append(reward)
            episode_dones.append(episode_done)

            episode_length += 1
            epoch_ended: bool = current_step == num_samples - 1

            if episode_done or epoch_ended:
                if epoch_ended and not episode_done:
                    logger.debug(
                        "The trajectory cut off at {} steps on the current episode".format(
                            episode_length
                        )
                    )

                episode_last_observation: np.ndarray = self.observation

                experience.observations.append(episode_observations)
                experience.actions.append(episode_actions)
                experience.rewards.append(episode_rewards)
                experience.last_observations.append(episode_last_observation)
                experience.dones.append(episode_dones)

                experience.episode_returns.append(episode_return)
                experience.episode_lengths.append(episode_length)

                if episode_done:
                    self.observation = self.env.reset()

                episode_return, episode_length = 0.0, 0
                (
                    episode_observations,
                    episode_actions,
                    episode_rewards,
                    episode_dones,
                ) = ([], [], [], [])

        return experience
