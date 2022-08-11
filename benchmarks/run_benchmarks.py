import argparse
from enum import Enum
from typing import Callable, List

from gym import envs
from run_ddpg import run_ddpg
from run_ppo import run_ppo
from run_td3 import run_td3
from run_trpo import run_trpo
from run_vpg import run_vpg


class AlgorithmNames(Enum):
    VPG = "vpg"
    TRPO = "trpo"
    PPO = "ppo"
    DDPG = "ddpg"
    TD3 = "td3"


def run_benchmarks(
    algorithm_names: List[str], environment_names: List[str], seeds: List[int]
) -> None:
    all_envs = envs.registry.values()
    env_ids = [env_spec.id for env_spec in all_envs]
    if not all(environment_name in env_ids for environment_name in environment_names):
        raise ValueError("Invalid environment name")

    if not all(
        algorithm_name in [member.value for member in AlgorithmNames]
        for algorithm_name in algorithm_names
    ):
        raise ValueError("Invalid algorithm name")

    for environment_name in environment_names:
        for algorithm_name in algorithm_names:
            run: Callable
            if algorithm_name == AlgorithmNames.VPG.value:
                run = run_vpg
            elif algorithm_name == AlgorithmNames.TRPO.value:
                run = run_trpo
            elif algorithm_name == AlgorithmNames.PPO.value:
                run = run_ppo
            elif algorithm_name == AlgorithmNames.DDPG.value:
                run = run_ddpg
            elif algorithm_name == AlgorithmNames.TD3.value:
                run = run_td3

            for seed in seeds:
                print(run, environment_name, seed)
                run(environment_name, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm_names", nargs="+", type=str)
    parser.add_argument("-e", "--environment_names", nargs="+", type=str)
    parser.add_argument("-s", "--seeds", nargs="+", type=int)

    args = parser.parse_args()

    run_benchmarks(args.algorithm_names, args.environment_names, args.seeds)
