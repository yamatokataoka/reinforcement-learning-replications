import datetime

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import VPG
from rl_replicas.common.networks import MLP
from rl_replicas.common.policies import CategoricalPolicy, GaussianPolicy
from rl_replicas.common.value_function import ValueFunction


class TestVPG:
    """
    Integration test for VPG
    """

    def test_vpg_with_cartpole(self) -> None:
        """
        Test VPG with CartPole environment (discrete action spaces)
        """
        env = gym.make("CartPole-v0")

        observation_size: int = env.observation_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [env.action_space.n]
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: VPG = VPG(
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
            seed=0,
        )

        model.learn(
            epochs=3,
            output_dir="/tmp/rl_replicas_tests/vpg-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )

    def test_vpg_with_pendulum(self) -> None:
        """
        Test VPG with Pendulum environment (continuous action spaces)
        """
        env = gym.make("Pendulum-v1")

        observation_size: int = env.observation_space.shape[0]
        action_size: int = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [action_size],
        )

        value_function_network: nn.Module = MLP(
            sizes=[observation_size] + [64, 64] + [1]
        )

        model: VPG = VPG(
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
            seed=0,
        )

        model.learn(
            epochs=3,
            output_dir="/tmp/rl_replicas_tests/vpg-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )
