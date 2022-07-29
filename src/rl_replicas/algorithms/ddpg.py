import logging
from typing import Dict, List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from rl_replicas.base_algorithms import OffPolicyAlgorithm
from rl_replicas.policies import Policy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import Sampler
from rl_replicas.utils import polyak_average

logger = logging.getLogger(__name__)


class DDPG(OffPolicyAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG)

    :param policy: (Policy) Policy.
    :param exploration_policy: (Policy) Exploration policy.
    :param q_function: (QFunction) Q function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param replay_buffer: (ReplayBuffer) Replay buffer.
    :param gamma: (float) The discount factor for the cumulative return.
    :param tau: (float) The interpolation factor in polyak averaging for target networks.
    :param action_noise_scale: (float) The scale of the noise (std).
    """

    def __init__(
        self,
        policy: Policy,
        exploration_policy: Policy,
        q_function: QFunction,
        env: gym.Env,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_noise_scale: float = 0.1,
    ) -> None:
        super().__init__(
            policy=policy,
            exploration_policy=exploration_policy,
            q_function=q_function,
            env=env,
            sampler=sampler,
            replay_buffer=replay_buffer,
            gamma=gamma,
            tau=tau,
            action_noise_scale=action_noise_scale,
        )

    def train(
        self, replay_buffer: ReplayBuffer, num_train_steps: int, minibatch_size: int
    ) -> None:
        policy_losses: List[float] = []
        q_function_losses: List[float] = []
        all_q_values: List[float] = []

        for _ in range(num_train_steps):
            minibatch: Dict[str, np.ndarray] = replay_buffer.sample_minibatch(
                minibatch_size
            )

            observations: Tensor = torch.from_numpy(minibatch["observations"])
            actions: Tensor = torch.from_numpy(minibatch["actions"])
            rewards: Tensor = torch.from_numpy(minibatch["rewards"]).float()
            next_observations: Tensor = torch.from_numpy(minibatch["next_observations"])
            dones: Tensor = torch.from_numpy(minibatch["dones"]).int()

            q_values: Tensor = self.q_function(observations, actions)
            all_q_values.extend(q_values.tolist())

            with torch.no_grad():
                next_actions: Tensor = self.target_policy(next_observations)
                target_q_values: Tensor = self.target_q_function(
                    next_observations, next_actions
                )

                targets: Tensor = rewards + self.gamma * (1 - dones) * target_q_values

            q_function_loss: Tensor = F.mse_loss(q_values, targets)
            q_function_losses.append(q_function_loss.item())

            self.q_function.optimizer.zero_grad()
            q_function_loss.backward()
            self.q_function.optimizer.step()

            policy_actions: Tensor = self.policy(observations)

            # Freeze Q-network so you don't waste computational effort
            for param in self.q_function.network.parameters():
                param.requires_grad = False

            policy_q_values: Tensor = self.q_function(observations, policy_actions)

            policy_loss: Tensor = -torch.mean(policy_q_values)
            policy_losses.append(policy_loss.item())

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

            # Unfreeze Q-network
            for param in self.q_function.network.parameters():
                param.requires_grad = True

            polyak_average(
                self.policy.network.parameters(),
                self.target_policy.network.parameters(),
                self.tau,
            )
            polyak_average(
                self.q_function.network.parameters(),
                self.target_q_function.network.parameters(),
                self.tau,
            )

        if self.tensorboard:
            self.writer.add_scalar(
                "policy/loss",
                policy_loss,
                self.current_total_steps,
            )
            self.writer.add_scalar(
                "q-function/loss",
                q_function_loss,
                self.current_total_steps,
            )
            self.writer.add_scalar(
                "q-function/avarage_q-value",
                torch.mean(q_values),
                self.current_total_steps,
            )

        logger.info("Policy Loss:            {:<8.3g}".format(np.mean(policy_losses)))
        logger.info(
            "Q Function Loss:        {:<8.3g}".format(np.mean(q_function_losses))
        )

        logger.info("Average Q Value:        {:<8.3g}".format(np.mean(all_q_values)))
        logger.info("Max Q Value:            {:<8.3g}".format(np.max(all_q_values)))
        logger.info("Min Q Value:            {:<8.3g}".format(np.min(all_q_values)))
