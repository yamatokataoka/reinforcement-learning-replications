import copy
import datetime
from typing import List

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import Env
from pytest import approx

from rl_replicas.algorithms import DDPG
from rl_replicas.evaluator import Evaluator
from rl_replicas.networks import MLP
from rl_replicas.policies import DeterministicPolicy, RandomPolicy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import BatchSampler


class TestDDPG:
    """
    Integration test for DDPG
    """

    def test_ddpg_with_pendulum(self, seed: int) -> None:
        """
        Test DDPG with Pendulum environment (continuous action spaces)
        """
        env = gym.make("Pendulum-v1")
        env.action_space.seed(seed)

        evaluation_env: Env = copy.deepcopy(env)

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
            RandomPolicy(env.action_space),
            QFunction(
                network=q_function_network,
                optimizer=torch.optim.Adam(q_function_network.parameters(), lr=1e-3),
            ),
            env,
            BatchSampler(env, seed, is_continuous=True),
            ReplayBuffer(int(1e6)),
            Evaluator(seed),
        )

        model.learn(
            num_epochs=6,
            num_start_steps=100,
            num_steps_before_update=200,
            evaluation_interval=100,
            model_saving_interval=100,
            output_dir="/tmp/rl_replicas_tests/ddpg-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        evaluator: Evaluator = Evaluator(seed)
        episode_returns: List[float]
        episode_returns, _ = evaluator.evaluate(model.policy, evaluation_env, 1)

        assert episode_returns[0] == approx(-1597.741, 0.01)
