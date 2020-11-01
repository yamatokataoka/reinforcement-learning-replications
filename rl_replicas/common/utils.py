import random
from typing import List, Tuple

import numpy as np
import scipy.signal
import torch

def discount_cumulative_sum(vector: np.ndarray, discount: float) -> np.ndarray:
  """
  Compute discounted cumulative sums of vector.

  :param vector: (np.ndarray) An target vector
  e.g. [x0,
        x1,
        x2]
  :param discount: (float) Discount factor
  :return: discounted cumulative sums of an vector
  e.g. [x0 + discount * x1 + discount^2 * x2,
        x1 + discount * x2,
        x2]
  """
  return scipy.signal.lfilter([1], [1, -discount], vector[::-1], axis=0)[::-1]

def seed_random_generators(seed: int) -> None:
    """
    Set the seed of the pseudo-random generators
    (python, numpy, pytorch)

    :param seed: (int)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def gae(
  rewards: np.ndarray,
  gamma: float,
  values: np.ndarray,
  gae_lambda: float
) -> np.ndarray:
  """
  Compute Generalized Advantage Estimation (GAE)

  :param rewards: (np.ndarray) Rewards for all states
  :param gamma: (float) Discount factor
  :param values: (np.ndarray) Values for all states
  :param gae_lambda: (float) A smoothing parameter for reducing the variance
  :return gaes: GAEs for all states
  """
  deltas: np.ndarray = rewards[:-1] + gamma * values[1:] - values[:-1]
  gaes: np.ndarray = discount_cumulative_sum(deltas, gamma * gae_lambda)

  return gaes

def unflatten_tensors(flattened: np.ndarray, tensor_shapes: List[torch.Size]) -> List[torch.Tensor]:
    """
    Unflatten a flattened tensors into a list of tensors.

    :param flattened: (np.ndarray) Flattened tensors.
    :param tensor_shapes: (List[torch.Size]) Tensor shapes.
    :return: (List[np.ndarray]) Unflattened list of tensors.
    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]

    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]
