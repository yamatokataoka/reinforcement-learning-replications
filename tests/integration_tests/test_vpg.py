import copy
import datetime
from typing import List

import gym
import torch
import torch.nn as nn
from gym import Env
from gym.spaces import Box, Discrete
from pytest import approx

from rl_replicas.algorithms import VPG
from rl_replicas.evaluator import Evaluator
from rl_replicas.networks import MLP
from rl_replicas.policies import CategoricalPolicy, GaussianPolicy, Policy
from rl_replicas.samplers import BatchSampler
from rl_replicas.seed_manager import SeedManager
from rl_replicas.value_function import ValueFunction


class TestVPG:
    """
    Integration test for VPG
    """

    def test_vpg_with_cartpole(self) -> None:
        """
        Test VPG with CartPole environment (discrete action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("CartPole-v0")
        env.action_space.seed(seed_manager.seed)

        evaluation_env: Env = copy.deepcopy(env)

        model: VPG = self.create_vpg(env, seed_manager)

        model.learn(
            num_epochs=3,
            batch_size=100,
            model_saving_interval=100,
            output_dir="/tmp/rl_replicas_tests/vpg-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed_manager.seed)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 1)

        assert episode_returns[0] == approx(18.0, 0.01)

    def test_vpg_with_pendulum(self) -> None:
        """
        Test VPG with Pendulum environment (continuous action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("Pendulum-v1")
        env.action_space.seed(seed_manager.seed)

        evaluation_env: Env = copy.deepcopy(env)

        model: VPG = self.create_vpg(env, seed_manager)

        model.learn(
            num_epochs=3,
            batch_size=100,
            model_saving_interval=100,
            output_dir="/tmp/rl_replicas_tests/vpg-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed_manager.seed)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 1)

        assert episode_returns[0] == approx(-1021.275, 0.01)

    def create_vpg(self, env: Env, seed_manager: SeedManager) -> VPG:
        observation_size: int = env.observation_space.shape[0]
        action_size: int
        if isinstance(env.action_space, Discrete):
            action_size = env.action_space.n
        elif isinstance(env.action_space, Box):
            action_size = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [action_size]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        learning_rate: float = 3e-4
        policy: Policy
        if isinstance(env.action_space, Discrete):
            policy = CategoricalPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(
                    policy_network.parameters(), lr=learning_rate
                ),
            )
        elif isinstance(env.action_space, Box):
            policy = GaussianPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(
                    policy_network.parameters(), lr=learning_rate
                ),
                log_std=nn.Parameter(-0.5 * torch.ones(action_size)),
            )

        model: VPG = VPG(
            policy,
            ValueFunction(
                network=value_function_network,
                optimizer=torch.optim.Adam(
                    value_function_network.parameters(), lr=1e-3
                ),
            ),
            env,
            BatchSampler(env, seed_manager.seed),
        )

        return model
