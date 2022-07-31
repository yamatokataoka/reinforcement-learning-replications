from typing import Iterable, List

import numpy as np
import scipy.signal
import torch
from torch import Tensor, nn

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


def unflatten_tensors(
    flattened: np.ndarray, tensor_shapes: List[torch.Size]
) -> List[Tensor]:
    """
    Unflatten a flattened tensors into a list of tensors

    :param flattened: (np.ndarray) Flattened tensors.
    :param tensor_shapes: (List[torch.Size]) Tensor shapes.
    :return: (List[np.ndarray]) Unflattened list of tensors.
    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]

    return [
        np.reshape(pair[0], pair[1])
        for pair in zip(np.split(flattened, indices), tensor_shapes)
    ]


def polyak_average(
    params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter], tau: float
) -> None:
    """
    Perform Polyak averaging on target_params using params

    :param params: (Iterable[torch.nn.Parameter]) The parameters to use to update the target params.
    :param target_params: (Iterable[torch.nn.Parameter]) The parameters to update.
    :param tau: (float) The soft update coefficient ("Polyak update", between 0 and 1).
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


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
