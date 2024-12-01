import argparse
import os
from contextlib import redirect_stdout

import gymnasium as gym
import torch
from torch import nn

from rl_replicas.algorithms import TD3
from rl_replicas.evaluator import Evaluator
from rl_replicas.networks import MLP
from rl_replicas.policies import DeterministicPolicy, RandomPolicy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import BatchSampler
from rl_replicas.utils import set_seed_for_libraries

ALGORITHM_NAME: str = "td3"


def run_td3(environment_name: str, seed: int, output_dir: str) -> None:
    set_seed_for_libraries(seed)

    env = gym.make(environment_name)
    env.action_space.seed(seed)

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
        BatchSampler(env, seed, is_continuous=True),
        ReplayBuffer(int(1e6)),
        Evaluator(seed),
    )

    experiment_dir: str = os.path.join(
        output_dir, f"{environment_name}/{ALGORITHM_NAME}/seed-{seed}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "experiment.log"), "w") as f:
        with redirect_stdout(f):
            print(f"algorithm: {ALGORITHM_NAME}")
            print(f"environment_name: {environment_name}")
            print(f"seed: {seed}")
            model.learn(
                num_epochs=20000,
                evaluation_interval=10000,
                model_saving_interval=10000,
                output_dir=experiment_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment_name", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    run_td3(args.environment_name, args.seed, args.output_dir)
