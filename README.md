# Reinforcement Learning Replications
Reinforcement Learning Replications is a set of modular Pytorch implementations of reinforcement learning algorithms.

## Benchmarks

The Reinforcement Learning Replications is benchmarked in two environments from the OpenAI Gym: CartPole-v0 and LunarLander-v2.

All experiments were run for 3 random seeds each. Graphs show the each experiment (solid line) smoothing by 0.6 on TensorBoard.

|               CartPole-v0              |                LunarLander-v2                |
|:--------------------------------------:|:--------------------------------------------:|
| ![CartPole-v0](./docs/CartPole-v0.gif) | ![LunarLander-v2](./docs/LunarLander-v2.gif) |

### Vanilla Policy Gradient (REINFORCE)

##### example code

You can run each benchmark experiment changing `seed` and `env_name` to reproduce the results.

```python
import datetime

import gym
from rl_replicas.vpg import VPG
from rl_replicas.common.policies import MLPPolicy
from rl_replicas.common.value_functions import MLPValueFunction

env_name = 'CartPole-v0' # CartPole-v0 or LunarLander-v2
output_dir = './runs/vpg' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
epochs = 200
seed = 0 # from 0 to 2

env = gym.make(env_name)

policy: MLPPolicy = MLPPolicy(env.observation_space, env.action_space)

value_function: MLPValueFunction = MLPValueFunction(env.observation_space)

model = VPG(policy, value_function, env, seed=seed)

model.learn(epochs=epochs, output_dir=output_dir, tensorboard=True, model_saving=True)
```


#### CartPole-v0

Sample result and trained model stored at `./runs/vpg/CartPole-v0`.

![CartPole-v0 with VPG](./docs/vpg/CartPole-v0_3seeds.png)

#### LunarLander-v2

Sample result and trained model stored at `./runs/vpg/LunarLander-v2`.

![CartPole-v0 with VPG](./docs/vpg/LunarLander-v2_3seeds.png)
