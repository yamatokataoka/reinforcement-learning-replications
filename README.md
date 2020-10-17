# Reinforcement Learning Replications
Reinforcement Learning Replications is a set of modular Pytorch implementations of reinforcement learning algorithms.

## Benchmarks

![CartPole-v0](./docs/CartPole-v0.gif)

### Vanilla Policy Gradient (REINFORCE)
#### CartPole-v0

![CartPole-v0 with VPG](./docs/vpg/CartPole-v0_3seeds.png)

You can change seed from 0 to 2.

```python
import datetime

import gym
from rl_replicas.vpg import VPG
from rl_replicas.common.policies import MLPPolicy
from rl_replicas.common.value_functions import MLPValueFunction

env = gym.make('CartPole-v0')
output_dir = './runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

policy: MLPPolicy = MLPPolicy(env.observation_space, env.action_space)

value_function: MLPValueFunction = MLPValueFunction(env.observation_space)

model = VPG(policy, value_function, env, seed=0)

model.learn(epochs=200, output_dir=output_dir, tensorboard=True, model_saving=True)
```
