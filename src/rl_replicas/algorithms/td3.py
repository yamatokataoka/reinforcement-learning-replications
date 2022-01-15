import copy
import logging
from typing import Dict, List, Optional

import gym
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from rl_replicas.common.base_algorithms import OffPolicyAlgorithm
from rl_replicas.common.policies import Policy
from rl_replicas.common.q_function import QFunction
from rl_replicas.common.replay_buffer import ReplayBuffer
from rl_replicas.common.utils import polyak_average

logger = logging.getLogger(__name__)


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    :param policy: (Policy) The policy
    :param q_function_1: (QFunction) The Q function
    :param q_function_2: (QFunction) The Q function
    :param env: (gym.Env) The environment to learn from
    :param gamma: (float) The discount factor for the cumulative return
    :param tau: (float) The interpolation factor in polyak averaging for target networks
    :param action_noise_scale: (float) The scale of the noise (std) for the policy to explore better
    :param target_noise_scale: (float) The scale of the smoothing noise (std) for the target policy to exploit harder
    :param target_noise_clip: (float) The limit for absolute value of the target policy smoothing noise
    :param policy_delay: (int) The policy will only be updated once every policy_delay times for each update of
        the Q-networks.
    :param seed: (int) The seed for the pseudo-random generators
    """

    def __init__(
        self,
        policy: Policy,
        q_function_1: QFunction,
        q_function_2: QFunction,
        env: gym.Env,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_noise_scale: float = 0.1,
        target_noise_scale: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            policy=policy,
            q_function=q_function_1,
            env=env,
            gamma=gamma,
            tau=tau,
            action_noise_scale=action_noise_scale,
            seed=seed,
        )

        self.target_noise_scale = target_noise_scale
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self.q_function_1 = self.q_function
        self.q_function_2 = q_function_2
        self.target_q_function_1 = self.target_q_function
        self.target_q_function_2 = copy.deepcopy(q_function_2)

        for param in self.target_q_function_2.network.parameters():
            param.requires_grad = False

    def train(
        self, replay_buffer: ReplayBuffer, train_steps: int, minibatch_size: int
    ) -> None:
        policy_losses: List[float] = []
        q_function_1_losses: List[float] = []
        q_function_2_losses: List[float] = []
        all_q_values_1: List[float] = []
        all_q_values_2: List[float] = []

        for train_step in range(train_steps):
            minibatch: Dict[str, Tensor] = replay_buffer.sample_minibatch(
                minibatch_size
            )

            observations: Tensor = minibatch["observations"]
            actions: Tensor = minibatch["actions"]
            rewards: Tensor = minibatch["rewards"]
            next_observations: Tensor = minibatch["next_observations"]
            dones: Tensor = minibatch["dones"]

            q_values_1: Tensor = self.q_function_1(observations, actions)
            q_values_2: Tensor = self.q_function_2(observations, actions)
            all_q_values_1.extend(q_values_1.tolist())
            all_q_values_2.extend(q_values_2.tolist())

            with torch.no_grad():
                next_actions: Tensor = self.target_policy(next_observations)
                epsilon: Tensor = self.target_noise_scale * torch.randn_like(
                    next_actions
                )
                epsilon = torch.clamp(
                    epsilon, -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = next_actions + epsilon
                next_actions = torch.clamp(
                    next_actions, -self.action_limit, self.action_limit
                )

                target_q_values_1: Tensor = self.target_q_function_1(
                    next_observations, next_actions
                )
                target_q_values_2: Tensor = self.target_q_function_2(
                    next_observations, next_actions
                )
                target_q_values: Tensor = torch.min(
                    target_q_values_1, target_q_values_2
                )

                targets: Tensor = rewards + self.gamma * (1 - dones) * target_q_values

            q_function_1_loss: Tensor = F.mse_loss(q_values_1, targets)
            q_function_2_loss: Tensor = F.mse_loss(q_values_2, targets)
            q_function_1_losses.append(q_function_1_loss.item())
            q_function_2_losses.append(q_function_2_loss.item())

            self.q_function_1.optimizer.zero_grad()
            q_function_1_loss.backward()
            self.q_function_1.optimizer.step()

            self.q_function_2.optimizer.zero_grad()
            q_function_2_loss.backward()
            self.q_function_2.optimizer.step()

            if train_step % self.policy_delay == 0:
                # Freeze Q-networks
                for param in self.q_function_1.network.parameters():
                    param.requires_grad = False
                for param in self.q_function_2.network.parameters():
                    param.requires_grad = False

                policy_actions: Tensor = self.policy(observations)
                policy_q_values: Tensor = self.q_function_1(
                    observations, policy_actions
                )

                policy_loss: Tensor = -torch.mean(policy_q_values)
                policy_losses.append(policy_loss.item())

                self.policy.optimizer.zero_grad()
                policy_loss.backward()
                self.policy.optimizer.step()

                # Unfreeze Q-networks
                for param in self.q_function_1.network.parameters():
                    param.requires_grad = True
                for param in self.q_function_2.network.parameters():
                    param.requires_grad = True

                polyak_average(
                    self.policy.network.parameters(),
                    self.target_policy.network.parameters(),
                    self.tau,
                )
                polyak_average(
                    self.q_function_1.network.parameters(),
                    self.target_q_function_1.network.parameters(),
                    self.tau,
                )
                polyak_average(
                    self.q_function_2.network.parameters(),
                    self.target_q_function_2.network.parameters(),
                    self.tau,
                )

                if self.tensorboard:
                    self.writer.add_scalar(
                        "policy/loss", policy_loss, self.current_total_steps
                    )

            if self.tensorboard:
                self.writer.add_scalar(
                    "q-function_1/loss", q_function_1_loss, self.current_total_steps
                )
                self.writer.add_scalar(
                    "q-function_2/loss", q_function_2_loss, self.current_total_steps
                )
                self.writer.add_scalar(
                    "q-function_1/avarage_q-value",
                    torch.mean(q_values_1),
                    self.current_total_steps,
                )
                self.writer.add_scalar(
                    "q-function_2/avarage_q-value",
                    torch.mean(q_values_2),
                    self.current_total_steps,
                )

        logger.info("Policy Loss:            {:<8.3g}".format(np.mean(policy_losses)))
        logger.info(
            "Q Function Loss (1):    {:<8.3g}".format(np.mean(q_function_1_losses))
        )
        logger.info(
            "Q Function Loss (2):    {:<8.3g}".format(np.mean(q_function_2_losses))
        )

        logger.info("Average Q Value (1):    {:<8.3g}".format(np.mean(all_q_values_1)))
        logger.info("Max Q Value (1):        {:<8.3g}".format(np.max(all_q_values_1)))
        logger.info("Min Q Value (1):        {:<8.3g}".format(np.min(all_q_values_1)))

        logger.info("Average Q Value (2):    {:<8.3g}".format(np.mean(all_q_values_2)))
        logger.info("Max Q Value (2):        {:<8.3g}".format(np.max(all_q_values_2)))
        logger.info("Min Q Value (2):        {:<8.3g}".format(np.min(all_q_values_2)))

    def save_model(self, current_epoch: int, model_path: str) -> None:
        """
        Save model

        :param current_epoch: (int) The current epoch
        :param model_path: (int) The path to save the model
        """
        torch.save(
            {
                "epoch": current_epoch,
                "total_steps": self.current_total_steps,
                "policy_state_dict": self.policy.network.state_dict(),
                "policy_optimizer_state_dict": self.policy.optimizer.state_dict(),
                "target_policy_state_dict": self.target_policy.network.state_dict(),
                "q_function_1_state_dict": self.q_function_1.network.state_dict(),
                "q_function_1_optimizer_state_dict": self.q_function_1.optimizer.state_dict(),
                "target_q_function_1_state_dict": self.target_q_function_1.network.state_dict(),
                "q_function_2_state_dict": self.q_function_2.network.state_dict(),
                "q_function_2_optimizer_state_dict": self.q_function_2.optimizer.state_dict(),
                "target_q_function_2_state_dict": self.target_q_function_2.network.state_dict(),
            },
            model_path,
        )
