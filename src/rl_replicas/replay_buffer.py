from operator import itemgetter
from typing import Dict, List

import numpy as np

from rl_replicas.experience import Experience


class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms

    :param buffer_size: (int) The size of the replay buffer.
    """

    def __init__(self, buffer_size: int = int(1e6)) -> None:
        self.buffer_size = buffer_size

        self.current_size: int = 0
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add_experience(self, experience: Experience) -> None:
        """
        Add experience

        :param experience: (Experience) Experience.
        """
        self.observations.extend(experience.flattened_observations)
        self.actions.extend(experience.flattened_actions)
        self.rewards.extend(experience.flattened_rewards)
        self.next_observations.extend(experience.flattened_next_observations)
        self.dones.extend(experience.flattened_dones)

        self.current_size += len(experience.flattened_observations)

        if self.current_size > self.buffer_size:
            num_exceeded_experinece: int = self.current_size - self.buffer_size

            del self.observations[:num_exceeded_experinece]
            del self.actions[:num_exceeded_experinece]
            del self.rewards[:num_exceeded_experinece]
            del self.next_observations[:num_exceeded_experinece]
            del self.dones[:num_exceeded_experinece]

            self.current_size -= num_exceeded_experinece

    def sample_minibatch(self, minibatch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Sample minibatch

        :param minibatch_size: (int) The number of transitions to be sampled.
        :return: (Dict[str, np.ndarray]) Sampled transitions.
        """
        indices = np.random.randint(0, self.current_size, minibatch_size)

        sampled_observations: np.ndarray = np.vstack(
            itemgetter(*indices)(self.observations)
        )
        sampled_actions: np.ndarray = np.vstack(itemgetter(*indices)(self.actions))
        sampled_rewards: np.ndarray = np.asarray(itemgetter(*indices)(self.rewards))
        sampled_next_observations: np.ndarray = np.vstack(
            itemgetter(*indices)(self.next_observations)
        )
        sampled_dones: np.ndarray = np.asarray(itemgetter(*indices)(self.dones))

        minibatch: Dict[str, np.ndarray] = {
            "observations": sampled_observations,
            "actions": sampled_actions,
            "rewards": sampled_rewards,
            "next_observations": sampled_next_observations,
            "dones": sampled_dones,
        }

        return minibatch
