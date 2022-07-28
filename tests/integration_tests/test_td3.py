import datetime

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import TD3
from rl_replicas.networks import MLP
from rl_replicas.policies import DeterministicPolicy, RandomPolicy
from rl_replicas.q_function import QFunction
from rl_replicas.samplers import BatchSampler, Sampler


class TestTD3:
    """
    Integration test for TD3
    """

    def test_td3_with_pendulum(self) -> None:
        """
        Test TD3 with Pendulum environment (continuous action spaces)
        """
        env = gym.make("Pendulum-v1")

        observation_size: int = env.observation_space.shape[0]
        action_size: int = env.action_space.shape[0]

        policy_network: nn.Module = MLP(
            sizes=[observation_size] + [256, 256] + [action_size],
            activation_function=nn.ReLU,
            output_activation_function=nn.Tanh,
        )

        q_function_learning_rate: float = 1e-3
        q_function_network_sizes = [observation_size + action_size] + [256, 256] + [1]
        q_function_1_network: nn.Module = MLP(
            sizes=q_function_network_sizes, activation_function=nn.ReLU
        )
        q_function_2_network: nn.Module = MLP(
            sizes=q_function_network_sizes, activation_function=nn.ReLU
        )

        sampler: Sampler = BatchSampler(env, is_continuous=True)

        model: TD3 = TD3(
            DeterministicPolicy(
                network=policy_network,
                optimizer=torch.optim.Adam(policy_network.parameters(), lr=1e-3),
            ),
            RandomPolicy(env.action_space),
            QFunction(
                network=q_function_1_network,
                optimizer=torch.optim.Adam(
                    q_function_1_network.parameters(), lr=q_function_learning_rate
                ),
            ),
            QFunction(
                network=q_function_2_network,
                optimizer=torch.optim.Adam(
                    q_function_2_network.parameters(), lr=q_function_learning_rate
                ),
            ),
            env,
            sampler,
            seed=0,
        )

        model.learn(
            num_epochs=30,
            output_dir="/tmp/rl_replicas_tests/td3-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            tensorboard=True,
            model_saving=True,
        )
