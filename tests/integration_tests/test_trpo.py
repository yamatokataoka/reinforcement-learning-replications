import datetime

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import TRPO
from rl_replicas.networks import MLP
from rl_replicas.optimizers import ConjugateGradientOptimizer
from rl_replicas.policies import CategoricalPolicy, GaussianPolicy
from rl_replicas.samplers import BatchSampler
from rl_replicas.seed_manager import SeedManager
from rl_replicas.value_function import ValueFunction


class TestTRPO:
    """
    Integration test for TRPO
    """

    def test_trpo_with_cartpole(self) -> None:
        """
        Test TRPO with CartPole environment (discrete action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("CartPole-v0")
        env.action_space.seed(seed_manager.seed)

        observation_size: int = env.observation_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [env.action_space.n]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: TRPO = TRPO(
            CategoricalPolicy(
                network=policy_network,
                optimizer=ConjugateGradientOptimizer(
                    params=policy_network.parameters()
                ),
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
            output_dir="/tmp/rl_replicas_tests/trpo-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )

        # TODO: run evaluation against the trained model.

    def test_trpo_with_pendulum(self) -> None:
        """
        Test TRPO with Pendulum environment (continuous action spaces)
        """
        seed_manager: SeedManager = SeedManager(0)
        seed_manager.set_seed_for_libraries()

        env = gym.make("Pendulum-v1")
        env.action_space.seed(seed_manager.seed)

        observation_size: int = env.observation_space.shape[0]
        action_size: int = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [action_size]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: TRPO = TRPO(
            GaussianPolicy(
                network=policy_network,
                optimizer=ConjugateGradientOptimizer(
                    params=policy_network.parameters()
                ),
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
            output_dir="/tmp/rl_replicas_tests/trpo-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )
