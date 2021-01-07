import datetime
from typing import List

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import TD3
from rl_replicas.common.policies import DeterministicPolicy
from rl_replicas.common.q_function import QFunction
from rl_replicas.common.networks import MLP

env_name = 'LunarLanderContinuous-v2' # Pendulum-v0 or LunarLanderContinuous-v2
output_dir = './test/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

policy_network_architecture: List[int] = [256, 256]
q_function_network_architecture: List[int] = [256, 256]
policy_learning_rate: float = 1e-3
q_function_learning_rate: float = 1e-3

env: gym.Env = gym.make(env_name)

observation_size: int = env.observation_space.shape[0]
action_size: int = env.action_space.shape[0]
action_limit: float = env.action_space.high[0]

policy_network_architecture = [observation_size]+policy_network_architecture+[action_size]
policy_network: nn.Module = MLP(
  sizes = policy_network_architecture,
  activation_function = nn.ReLU,
  output_activation_function = nn.Tanh
)
policy: DeterministicPolicy = DeterministicPolicy(
  network = policy_network,
  optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)
)

q_function_network_architecture = [observation_size+action_size]+q_function_network_architecture+[1]
q_function_1_network: nn.Module = MLP(
  sizes = q_function_network_architecture,
  activation_function = nn.ReLU
)
q_function_2_network: nn.Module = MLP(
  sizes = q_function_network_architecture,
  activation_function = nn.ReLU
)
q_function_1: QFunction = QFunction(
  network = q_function_1_network,
  optimizer = torch.optim.Adam(q_function_1_network.parameters(), lr=q_function_learning_rate)
)
q_function_2: QFunction = QFunction(
  network = q_function_2_network,
  optimizer = torch.optim.Adam(q_function_2_network.parameters(), lr=q_function_learning_rate)
)

model: TD3 = TD3(policy, q_function_1, q_function_2, env, seed=0)

print(f'Experiment to {output_dir}')

model.learn(
  epochs = 2000,
  steps_per_epoch = 50,
  replay_buffer_size = int(1e6),
  minibatch_size = 100,
  random_start_steps = 10000,
  steps_before_update = 1000,
  train_steps = 50,
  output_dir = output_dir,
  num_evaluation_episodes = 3,
  evaluation_interval = 4000,
  tensorboard = True,
  model_saving = True
)

print(f'Experimented to {output_dir}')
