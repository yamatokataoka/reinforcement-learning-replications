import copy
import datetime
from typing import List

import gym
import torch
import torch.nn as nn
from gym import Env

from rl_replicas.algorithms import PPO
from rl_replicas.evaluator import Evaluator
from rl_replicas.networks import MLP
from rl_replicas.policies import CategoricalPolicy, GaussianPolicy
from rl_replicas.samplers import BatchSampler
from rl_replicas.seed_manager import SeedManager
from rl_replicas.value_function import ValueFunction


class TestPPO:
    """
    Integration test for PPO
    """

    def test_ppo_with_cartpole(self) -> None:
        """
        Test PPO with CartPole environment (discrete action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("CartPole-v0")
        env.action_space.seed(seed_manager.seed)

        evaluation_env: Env = copy.deepcopy(env)

        observation_size: int = env.observation_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [env.action_space.n]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: PPO = PPO(
            CategoricalPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(policy_network.parameters(), lr=3e-4),
            ),
            ValueFunction(
                network=value_function_network,
                optimizer=torch.optim.Adam(
                    value_function_network.parameters(), lr=1e-3
                ),
            ),
            env,
            BatchSampler(env, seed_manager),
        )

        model.learn(
            num_epochs=3,
            output_dir="/tmp/rl_replicas_tests/ppo-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed_manager)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 1)

        assert round(episode_returns[0], 2) == 60.0

    def test_ppo_with_pendulum(self) -> None:
        """
        Test PPO with Pendulum environment (continuous action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("Pendulum-v1")
        env.action_space.seed(seed_manager.seed)

        evaluation_env: Env = copy.deepcopy(env)

        observation_size: int = env.observation_space.shape[0]
        action_size: int = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [action_size]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: PPO = PPO(
            GaussianPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(policy_network.parameters(), lr=3e-4),
                log_std=nn.Parameter(-0.5 * torch.ones(action_size)),
            ),
            ValueFunction(
                network=value_function_network,
                optimizer=torch.optim.Adam(
                    value_function_network.parameters(), lr=1e-3
                ),
            ),
            env,
            BatchSampler(env, seed_manager),
        )

        model.learn(
            num_epochs=3,
            output_dir="/tmp/rl_replicas_tests/ppo-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed_manager)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 1)

        assert round(episode_returns[0], 2) == -986.35
