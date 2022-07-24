from abc import ABC, abstractmethod

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy


class Sampler(ABC):
    """
    Base sampler class
    """

    @abstractmethod
    def sample(self, num_samples: int, policy: Policy) -> Experience:
        """
        Sample experience

        :param num_samples: (int) The number of samples to collect.
        :param policy: (Policy) Policy.
        :return: (Experience) Collected experience.
        """
        raise NotImplementedError
