import random
from typing import Iterable, List

import numpy as np
import scipy.signal
import torch
from gym import Space
from torch import Tensor, nn

from rl_replicas.policies.policy import Policy
from rl_replicas.value_function import ValueFunction


def discounted_cumulative_sums(vector: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sums of vector

    :param vector: (np.ndarray) A target vector.
    e.g. [x0,
          x1,
          x2]
    :param discount: (float) The discount factor for the cumulative return.
    :return: (np.ndarray) The discounted cumulative sums of a vector.
    e.g. [x0 + discount * x1 + discount^2 * x2,
          x1 + discount * x2,
          x2]
    """
    return scipy.signal.lfilter([1], [1, -discount], vector[::-1], axis=0)[::-1]


def gae(
    rewards: np.ndarray, gamma: float, values: np.ndarray, gae_lambda: float
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE)

    :param rewards: (np.ndarray) Rewards for all states.
    :param gamma: (float) The discount factor for the cumulative return.
    :param values: (np.ndarray) Values for all states.
    :param gae_lambda: (float) A smoothing parameter for reducing the variance.
    :return gaes: (np.ndarray) The GAEs for all states.
    """
    deltas: np.ndarray = rewards[:-1] + gamma * values[1:] - values[:-1]
    gaes: np.ndarray = discounted_cumulative_sums(deltas, gamma * gae_lambda)

    return gaes


def polyak_average(
    params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter], rho: float
) -> None:
    """
    Perform Polyak averaging on target_params using params

    :param params: (Iterable[torch.nn.Parameter]) The parameters to use to update the target params.
    :param target_params: (Iterable[torch.nn.Parameter]) The parameters to update.
    :param rho: (float) The coefficient for polyak averaging (between 0 and 1).
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(
                torch.tensor(rho) * target_param.data
                + torch.tensor(1.0 - rho) * param.data
            )


def compute_values_numpy_list(
    observations_with_last_observation: List[List[np.ndarray]],
    value_function: ValueFunction,
) -> np.ndarray:
    values_numpy_list: List[np.ndarray] = []
    with torch.no_grad():
        for (
            episode_observations_with_last_observation
        ) in observations_with_last_observation:
            episode_observations_with_last_observation_tensor = torch.from_numpy(
                np.concatenate([episode_observations_with_last_observation])
            ).float()
            values_numpy_list.append(
                value_function(episode_observations_with_last_observation_tensor)
                .flatten()
                .numpy()
            )
    return values_numpy_list


def bootstrap_rewards_with_last_values(
    rewards: List[List[float]], episode_dones: List[bool], last_values: List[float]
) -> List[List[float]]:
    bootstrapped_rewards: List[List[float]] = []

    for episode_rewards, episode_done, last_value in zip(
        rewards, episode_dones, last_values
    ):
        episode_bootstrapped_rewards: List[float]
        if episode_done:
            episode_bootstrapped_rewards = episode_rewards + [0]
        else:
            episode_bootstrapped_rewards = episode_rewards + [last_value]
        bootstrapped_rewards.append(episode_bootstrapped_rewards)

    return bootstrapped_rewards


def normalize_tensor(vector: Tensor) -> Tensor:
    normalized_vector = (vector - torch.mean(vector)) / torch.std(vector)
    return normalized_vector


def add_noise_to_get_action(
    policy: Policy, action_space: Space, action_noise_scale: float
) -> Policy:
    noised_policy: Policy = _NoisedPolicy(policy, action_space, action_noise_scale)

    return noised_policy


class _NoisedPolicy(Policy):
    def __init__(
        self, base_policy: Policy, action_space: Space, action_noise_scale: float
    ):
        super().__init__()

        self.base_policy = base_policy
        self.action_space = action_space
        self.action_noise_scale = action_noise_scale

        self.action_limit = action_space.high[0]
        self.action_size = action_space.shape[0]

    def get_action_tensor(self, observation: Tensor) -> Tensor:
        action = self.base_policy.get_action_tensor(observation)
        action += self.action_noise_scale * torch.randn(self.action_size)
        action = torch.clip(action, -self.action_limit, self.action_limit)

        return action

    def get_action_numpy(self, observation: np.ndarray) -> np.ndarray:
        action = self.base_policy.get_action_numpy(observation)
        action += self.action_noise_scale * np.random.randn(self.action_size)
        action = np.clip(action, -self.action_limit, self.action_limit)

        return action


def set_seed_for_libraries(seed: int) -> None:
    """
    Set seed for random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
