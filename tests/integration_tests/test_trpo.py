import copy
import datetime
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from rl_replicas.algorithms import TRPO
from rl_replicas.evaluator import Evaluator
from rl_replicas.networks import MLP
from rl_replicas.optimizers import ConjugateGradientOptimizer
from rl_replicas.policies import CategoricalPolicy, GaussianPolicy, Policy
from rl_replicas.samplers import BatchSampler
from rl_replicas.value_function import ValueFunction


class TestTRPO:
    """
    Integration test for TRPO
    """

    def test_trpo_with_cartpole(self, seed: int) -> None:
        """
        Test TRPO with CartPole environment (discrete action spaces)
        """
        env = gym.make("CartPole-v1")
        env.action_space.seed(seed)

        evaluation_env: Env = copy.deepcopy(env)

        model: TRPO = self.create_trpo(env, seed)

        model.learn(
            num_epochs=5,
            batch_size=500,
            model_saving_interval=100,
            output_dir="/tmp/rl_replicas_tests/trpo-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 3)

        assert np.mean(episode_returns) > 50

    def test_trpo_with_pendulum(self, seed: int) -> None:
        """
        Test TRPO with Pendulum environment (continuous action spaces)
        """
        env = gym.make("Pendulum-v1")
        env.action_space.seed(seed)

        evaluation_env: Env = copy.deepcopy(env)

        model: TRPO = self.create_trpo(env, seed)

        model.learn(
            num_epochs=5,
            batch_size=500,
            model_saving_interval=100,
            output_dir="/tmp/rl_replicas_tests/trpo-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 3)

        assert np.mean(episode_returns) > -1300

    def create_trpo(self, env: Env, seed: int) -> TRPO:
        observation_size: int = env.observation_space.shape[0]
        action_size: int
        if isinstance(env.action_space, Discrete):
            action_size = env.action_space.n
        elif isinstance(env.action_space, Box):
            action_size = env.action_space.shape[0]

        policy_network: nn.Module = MLP(sizes=[observation_size] + [64, 64] + [action_size])

        value_function_network: nn.Module = MLP(sizes=[observation_size] + [64, 64] + [1])

        policy: Policy
        if isinstance(env.action_space, Discrete):
            policy = CategoricalPolicy(
                network=policy_network,
                optimizer=ConjugateGradientOptimizer(params=policy_network.parameters()),
            )
        elif isinstance(env.action_space, Box):
            policy = GaussianPolicy(
                network=policy_network,
                optimizer=ConjugateGradientOptimizer(params=policy_network.parameters()),
                log_std=nn.Parameter(-0.5 * torch.ones(action_size)),
            )

        model: TRPO = TRPO(
            policy,
            ValueFunction(
                network=value_function_network,
                optimizer=torch.optim.Adam(value_function_network.parameters(), lr=1e-3),
            ),
            env,
            BatchSampler(env, seed),
        )

        return model
