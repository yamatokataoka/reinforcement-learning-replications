import argparse
import os
from contextlib import redirect_stdout

import gym
import torch
from torch import nn

from rl_replicas.algorithms import TRPO
from rl_replicas.networks import MLP
from rl_replicas.optimizers import ConjugateGradientOptimizer
from rl_replicas.policies import GaussianPolicy
from rl_replicas.samplers import BatchSampler
from rl_replicas.seed_manager import SeedManager
from rl_replicas.value_function import ValueFunction

ALGORITHM_NAME: str = "trpo"


def run_trpo(environment_name: str, seed: int, output_dir: str) -> None:
    seed_manager: SeedManager = SeedManager(seed)
    seed_manager.set_seed_for_libraries()

    env = gym.make(environment_name)
    env.action_space.seed(seed_manager.seed)

    observation_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.shape[0]

    policy_network: nn.Module = MLP(sizes=[observation_size] + [64, 32] + [action_size])
    value_function_network: nn.Module = MLP(sizes=[observation_size] + [64, 32] + [1])

    model: TRPO = TRPO(
        GaussianPolicy(
            network=policy_network,
            optimizer=ConjugateGradientOptimizer(params=policy_network.parameters()),
            log_std=nn.Parameter(-0.5 * torch.ones(action_size)),
        ),
        ValueFunction(
            network=value_function_network,
            optimizer=torch.optim.Adam(value_function_network.parameters(), lr=1e-3),
        ),
        env,
        BatchSampler(env, seed_manager),
    )

    experiment_dir: str = os.path.join(
        output_dir, f"{ALGORITHM_NAME}/{environment_name}/seed-{seed}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "experiment.log"), "w") as f:
        with redirect_stdout(f):
            print(f"algorithm: {ALGORITHM_NAME}")
            print(f"environment_name: {environment_name}")
            print(f"seed: {seed}")
            model.learn(
                num_epochs=750,
                output_dir=experiment_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment_name", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    run_trpo(args.environment_name, args.seed, args.output_dir)
