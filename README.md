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

Here is the code of training PPO on CartPole-v1 environment. You can run with [this Google Colab notebook](https://colab.research.google.com/drive/18MRw1FcDS4b_t3HAgfvyxBCi_1Z4lD__#scrollTo=A5GI_PJSchBn).

```python
import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import PPO
from rl_replicas.networks import MLP
from rl_replicas.policies import CategoricalPolicy
from rl_replicas.samplers import BatchSampler
from rl_replicas.value_function import ValueFunction

env_name = "CartPole-v1"
output_dir = "/content/ppo"
num_epochs = 80
seed = 0

network_hidden_sizes = [64, 64]
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3

env = gym.make(env_name)
env.action_space.seed(seed)

observation_size: int = env.observation_space.shape[0]
action_size: int = env.action_space.n

policy_network: nn.Module = MLP(
    sizes=[observation_size] + network_hidden_sizes + [action_size]
)

value_function_network: nn.Module = MLP(
    sizes=[observation_size] + network_hidden_sizes + [1]
)

model: PPO = PPO(
    CategoricalPolicy(
        network=policy_network,
        optimizer=torch.optim.Adam(policy_network.parameters(), lr=3e-4),
    ),
    ValueFunction(
        network=value_function_network,
        optimizer=torch.optim.Adam(value_function_network.parameters(), lr=1e-3),
    ),
    env,
    BatchSampler(env, seed),
)

model.learn(num_epochs=num_epochs, output_dir=output_dir)

```

## Contributing

All contributions are welcome.

### Release Flow

1. Create a release branch.
1. A pull request from the release branch to the `main` branch has the following:
   - Change logs in the body.
   - The `release` label.
   - Commit that bumps up the version in `VERSION`.
1. Once the pull request is ready, merge the pull request. The CI will upload the package and create the release.
