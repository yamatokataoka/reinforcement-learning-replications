# Reinforcement Learning Replications
Reinforcement Learning Replications is a set of Pytorch implementations of reinforcement learning algorithms.


## Features

- Implement Algorithms
  - Vanilla Policy Gradient (VPG)
  - Trust Region Policy Optimization (TRPO)
  - Proximal Policy Optimization (PPO)
  - Deep Deterministic Policy Gradient (DDPG)
  - Twin Delayed DDPG (TD3)
- Use Python standard logging library
- Support TensorBoard


## Benchmarks

You can check the benchmark result [here](https://yamatokataoka.github.io/reinforcement-learning-replications/benchmarks/visualization.html).

This benchmark is conducted based on [the Benchmarks for Spinning Up Implementations](https://spinningup.openai.com/en/latest/spinningup/bench.html).

All experiments were run for 3 random seeds each. All the details such as tensorboard and experiment logs, training scripts and trained models are stored in the [benchmarks](https://github.com/yamatokataoka/reinforcement-learning-replications/tree/main/benchmarks) folder.

## Example Code

Here is the code to run the training of VPG on CartPole-v0 environment.

```python
import datetime
import logging
import sys

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import VPG
from rl_replicas.networks import MLP
from rl_replicas.policies import CategoricalPolicy
from rl_replicas.value_function import ValueFunction

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")

env_name = "CartPole-v0"
output_dir = "./runs/vpg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
num_epochs = 200
seed = 0

network_hidden_sizes = [64, 64]
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3

env = gym.make(env_name)

policy_network: nn.Module = MLP(
    sizes=[env.observation_space.shape[0]] + network_hidden_sizes + [env.action_space.n]
)

policy: CategoricalPolicy = CategoricalPolicy(
    network=policy_network,
    optimizer=torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)
)

value_function_network: nn.Module = MLP(
    sizes=[env.observation_space.shape[0]] + network_hidden_sizes + [1]
)
value_function: ValueFunction = ValueFunction(
    network=value_function_network,
    optimizer=torch.optim.Adam(
        value_function_network.parameters(), lr=value_function_learning_rate
    ),
)

model: VPG = VPG(policy, value_function, env, seed=seed)

model.learn(num_epochs=num_epochs, output_dir=output_dir, tensorboard=True, model_saving=True)

```

## Contributing

All contributions are welcome.

### Release Flow

1. A release branch with a version
1. A pull request from the release branch to the main branch (`master`)

Once the pull request is ready,

1. Merge the pull request
1. Create a release with the version. Once the release is published, packages will be uploaded.
