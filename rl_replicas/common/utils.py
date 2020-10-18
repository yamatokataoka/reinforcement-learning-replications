import random

import scipy.signal
from typing import Optional

import numpy as np
import torch

def discount_cumulative_sum(vector, discount):
  """
  Compute discounted cumulative sums of vectors.

  input:
      vector
      e.g. [x0,
            x1,
            x2]
      discount

  output:
      [x0 + discount * x1 + discount^2 * x2,
       x1 + discount * x2,
       x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]

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
