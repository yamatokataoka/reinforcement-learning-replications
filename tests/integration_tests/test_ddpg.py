import datetime

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import DDPG
from rl_replicas.common.networks import MLP
from rl_replicas.common.policies import DeterministicPolicy
from rl_replicas.common.q_function import QFunction


class TestDDPG:
    """
    Integration test for DDPG
    """

    def test_ddpg_with_pendulum(self) -> None:
        """
        Test DDPG with Pendulum environment (continuous action spaces)
        """
        env = gym.make("Pendulum-v1")

        observation_size: int = env.observation_space.shape[0]
        action_size: int = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [256, 256] + [action_size],
            activation_function=nn.ReLU,
            output_activation_function=nn.Tanh,
        )

        q_function_network: nn.Module = MLP(
            sizes=[observation_size + action_size] + [256, 256] + [1],
            activation_function=nn.ReLU,
        )

        model: DDPG = DDPG(
            DeterministicPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(policy_network.parameters(), lr=1e-3),
            ),
            QFunction(
                network=q_function_network,
                optimizer=torch.optim.Adam(q_function_network.parameters(), lr=1e-3),
            ),
            env,
            seed=0,
        )

        model.learn(
            epochs=30,
            output_dir="/tmp/rl_replicas_tests/ddpg-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )
