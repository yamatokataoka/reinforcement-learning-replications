import datetime
import logging
import os
import sys
from typing import List

import gym
import torch
import torch.nn as nn

from rl_replicas.algorithms import VPG, TRPO, PPO
from rl_replicas.common.base_algorithms import OnPolicyAlgorithm
from rl_replicas.common.policies import Policy, CategoricalPolicy
from rl_replicas.common.value_function import ValueFunction
from rl_replicas.common.optimizers import ConjugateGradientOptimizer
from rl_replicas.common.networks import MLP

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='vpg')
parser.add_argument('--environment', type=str, default='CartPole-v0')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--steps_per_epoch', type=int, default=4000)
parser.add_argument('--policy_network_arch', nargs='+', type=int, default=[64, 64])
parser.add_argument('--value_function_network_arch', nargs='+', type=int, default=[64, 64])
parser.add_argument('--policy_lr', type=float, default=3e-4)
parser.add_argument('--value_function_lr', type=float, default=1e-3)
parser.add_argument('--experiment_home', type=str, default='.')
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--model_saving', action='store_true')
args = parser.parse_args()

algorithm_name = args.algorithm
environment_name = args.environment
epochs = args.epochs
steps_per_epoch = args.steps_per_epoch
policy_network_architecture = args.policy_network_arch
value_function_network_architecture = args.value_function_network_arch
policy_learning_rate = args.policy_lr
value_function_learning_rate = args.value_function_lr
experiment_home = args.experiment_home
tensorboard = args.tensorboard
model_saving = args.model_saving

env = gym.make(environment_name)
output_dir: str = os.path.join(
  experiment_home,
  algorithm_name,
  environment_name,
  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

policy_network: nn.Module = MLP(
  sizes = [env.observation_space.shape[0]]+policy_network_architecture+[env.action_space.n]
)

policy: Policy
if algorithm_name == 'vpg':
  policy = CategoricalPolicy(
    network = policy_network,
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)
  )
elif algorithm_name == 'trpo':
  policy = CategoricalPolicy(
    network = policy_network,
    optimizer = ConjugateGradientOptimizer(params=policy_network.parameters())
  )
elif algorithm_name == 'ppo':
  policy = CategoricalPolicy(
    network = policy_network,
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)
  )
else:
  print('Invalid algorithm name: {}'.format(algorithm_name))

value_function_network: nn.Module = MLP(
  sizes = [env.observation_space.shape[0]]+value_function_network_architecture+[1]
)
value_function: ValueFunction = ValueFunction(
  network = value_function_network,
  optimizer = torch.optim.Adam(value_function_network.parameters(), lr=value_function_learning_rate)
)

model: OnPolicyAlgorithm
if algorithm_name == 'vpg':
  model = VPG(policy, value_function, env, seed=0)
elif algorithm_name == 'trpo':
  model = TRPO(policy, value_function, env, seed=0)
elif algorithm_name == 'ppo':
  model = PPO(policy, value_function, env, seed=0)
else:
  print('Invalid algorithm name: {}'.format(algorithm_name))

if tensorboard or model_saving:
  print('Start experiment to: {}'.format(output_dir))

print('epochs:              {}'.format(epochs))
print('steps_per_epoch:     {}'.format(steps_per_epoch))
print('algorithm:           {}'.format(algorithm_name))
print('environment:         {}'.format(environment_name))
print('seed:                {}'.format(model.seed))

print('value_function_learning_rate: {}'.format(value_function_learning_rate))
if algorithm_name != 'trpo':
  print('policy_learning_rate: {}'.format(policy_learning_rate))

model.learn(
  epochs=epochs,
  steps_per_epoch=steps_per_epoch,
  output_dir=output_dir,
  tensorboard=tensorboard,
  model_saving=model_saving
)
