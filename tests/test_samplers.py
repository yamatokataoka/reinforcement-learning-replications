from collections import namedtuple
from typing import List

import gymnasium as gym
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture

from rl_replicas.experience import Experience
from rl_replicas.policies import Policy, RandomPolicy
from rl_replicas.samplers import BatchSampler, Sampler

ExpectedExperience = namedtuple(
    "ExpectedExperience",
    [
        "observations",
        "actions",
        "rewards",
        "last_observations",
        "next_observations",
        "dones",
    ],
)


class TestSamplers:
    @fixture
    def env(self) -> gym.Env:
        env = gym.make("CartPole-v1")
        return env

    @fixture
    def expected_experience(self, seed: int, env: gym.Env) -> ExpectedExperience:
        env.action_space.seed(seed)
        num_samples: int = 1000

        expected_experience: ExpectedExperience = ExpectedExperience([], [], [], [], [], [])

        observation, _ = env.reset(seed=seed)

        for current_step in range(num_samples):
            expected_experience.observations.append(observation)

            action: np.ndarray = env.action_space.sample()
            expected_experience.actions.append(action)

            reward: float
            episode_done: bool
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated

            expected_experience.next_observations.append(observation)

            expected_experience.rewards.append(reward)
            expected_experience.dones.append(episode_done)

            epoch_ended: bool = current_step == num_samples - 1

            if episode_done or epoch_ended:
                last_observation: np.ndarray = observation
                expected_experience.last_observations.append(last_observation)

                if episode_done:
                    observation, _ = env.reset()

        return expected_experience

    def test_sample(self, seed: int, env: gym.Env, expected_experience: Experience) -> None:
        env.action_space.seed(seed)

        sampler: Sampler = BatchSampler(env, seed)
        policy: Policy = RandomPolicy(env.action_space)

        experience: Experience = sampler.sample(1000, policy)

        assert_array_equal(
            experience.flattened_observations,
            expected_experience.observations,
        )
        assert_array_equal(experience.flattened_actions, expected_experience.actions)
        assert_array_equal(experience.flattened_rewards, expected_experience.rewards)
        assert_array_equal(experience.last_observations, expected_experience.last_observations)
        assert_array_equal(experience.flattened_dones, expected_experience.dones)

    def test_sample_continuous(self, seed: int, env: gym.Env, expected_experience: Experience) -> None:
        env.action_space.seed(seed)

        sampler: Sampler = BatchSampler(env, seed, is_continuous=True)
        policy: Policy = RandomPolicy(env.action_space)

        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        next_observations: List[np.ndarray] = []
        dones: List[bool] = []
        for i in range(10):
            experience: Experience = sampler.sample(100, policy)

            observations.extend(experience.flattened_observations)
            actions.extend(experience.flattened_actions)
            rewards.extend(experience.flattened_rewards)
            next_observations.extend(experience.flattened_next_observations)
            dones.extend(experience.flattened_dones)

        assert_array_equal(observations, expected_experience.observations)
        assert_array_equal(actions, expected_experience.actions)
        assert_array_equal(rewards, expected_experience.rewards)
        assert_array_equal(next_observations, expected_experience.next_observations)
        assert_array_equal(dones, expected_experience.dones)
