import random

import numpy as np
import torch


class SeedManager:
    """
    Seed manager

    :param seed: (int) Seed.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def set_seed_for_libraries(self) -> None:
        """
        Set seed for random, numpy and torch.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
