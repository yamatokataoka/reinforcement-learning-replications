import copy
import logging
import os
import time
from typing import Dict, List

import gym
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from rl_replicas.evaluator import Evaluator
from rl_replicas.experience import Experience
from rl_replicas.metrics_manager import MetricsManager
from rl_replicas.policies import Policy
from rl_replicas.q_function import QFunction
from rl_replicas.replay_buffer import ReplayBuffer
from rl_replicas.samplers import Sampler
from rl_replicas.utils import add_noise_to_get_action, polyak_average

logger = logging.getLogger(__name__)


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    :param policy: (Policy) Policy.
    :param exploration_policy: (Policy) Exploration policy.
    :param q_function_1: (QFunction) Q function.
    :param q_function_2: (QFunction) Q function.
    :param env: (gym.Env) Environment.
    :param sampler: (Sampler) Sampler.
    :param replay_buffer: (ReplayBuffer) Replay buffer.
    :param evaluator: (Evaluator) Evaluator.
    :param gamma: (float) The discount factor for the cumulative return.
    :param polyak_rho: (float) The interpolation factor in polyak averaging for target networks.
    :param action_noise_scale: (float) The scale of the noise (std) for the policy to explore better.
    :param target_noise_scale: (float) The scale of the smoothing noise (std) for the target policy to exploit harder.
    :param target_noise_clip: (float) The limit for absolute value of the target policy smoothing noise.
    :param policy_delay: (int) The policy will only be updated once every policy_delay times for each update of
        the Q-networks.
    """

    def __init__(
        self,
        policy: Policy,
        exploration_policy: Policy,
        q_function_1: QFunction,
        q_function_2: QFunction,
        env: gym.Env,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        evaluator: Evaluator,
        gamma: float = 0.99,
        polyak_rho: float = 0.995,
        action_noise_scale: float = 0.1,
        target_noise_scale: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
    ) -> None:
        self.policy = policy
        self.exploration_policy = exploration_policy
        self.q_function_1 = q_function_1
        self.q_function_2 = q_function_2
        self.env = env
        self.sampler = sampler
        self.replay_buffer = replay_buffer
        self.evaluator = evaluator
        self.gamma = gamma
        self.polyak_rho = polyak_rho
        self.action_noise_scale = action_noise_scale

        self.noised_policy = add_noise_to_get_action(
            self.policy, self.env.action_space, self.action_noise_scale
        )
        self.evaluation_env = gym.make(env.spec.id)
        self.target_policy = copy.deepcopy(self.policy)

        for param in self.target_policy.network.parameters():
            param.requires_grad = False

        self.target_noise_scale = target_noise_scale
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self.target_q_function_1 = copy.deepcopy(self.q_function_1)
        self.target_q_function_2 = copy.deepcopy(self.q_function_2)

        for param in self.target_q_function_1.network.parameters():
            param.requires_grad = False
        for param in self.target_q_function_2.network.parameters():
            param.requires_grad = False

    def learn(
        self,
        num_epochs: int = 2000,
        batch_size: int = 50,
        minibatch_size: int = 100,
        num_start_steps: int = 10000,
        num_steps_before_update: int = 1000,
        num_train_steps: int = 50,
        num_evaluation_episodes: int = 5,
        evaluation_interval: int = 4000,
        model_saving_interval: int = 4000,
        output_dir: str = ".",
    ) -> None:
        """
        Learn the model

        :param num_epochs: (int) The number of epochs to run and train.
        :param batch_size: (int) The number of steps to run per epoch.
        ;param minibatch_size: (int) The minibatch size for SGD.
        :param num_start_steps: (int) The number of steps for exploration action selection at the beginning.
        :param num_steps_before_update: (int) The number of steps to perform before policy is updated.
        :param num_train_steps: (int) The number of training steps on each epoch.
        :param num_evaluation_episodes: (int) The number of evaluation episodes.
        :param evaluation_interval: (int) The interval steps between evaluation.
        :param model_saving_interval: (int) The interval steps between model saving.
        :param output_dir: (str) The output directory.
        """
        start_time: float = time.time()
        self.current_total_steps: int = 0
        self.current_total_episodes: int = 0

        os.makedirs(output_dir, exist_ok=True)

        self.metrics_manager: MetricsManager = MetricsManager(output_dir)

        for current_epoch in range(1, num_epochs + 1):
            experience: Experience
            if self.current_total_steps < num_start_steps:
                experience = self.sampler.sample(batch_size, self.exploration_policy)
            else:
                experience = self.sampler.sample(batch_size, self.noised_policy)

            self.replay_buffer.add_experience(
                experience.flattened_observations,
                experience.flattened_actions,
                experience.flattened_rewards,
                experience.flattened_next_observations,
                experience.flattened_dones,
            )

            episode_returns: List[float] = experience.episode_returns
            episode_lengths: List[int] = experience.episode_lengths

            self.current_total_steps += sum(experience.episode_lengths)
            self.current_total_episodes += sum(experience.flattened_dones)

            self.metrics_manager.record_scalar("epoch", current_epoch)
            self.metrics_manager.record_scalar("total_steps", self.current_total_steps)
            self.metrics_manager.record_scalar(
                "total_episodes", self.current_total_episodes
            )

            if len(episode_lengths) > 0:
                self.metrics_manager.record_scalar(
                    "sampling/average_episode_return",
                    float(np.mean(episode_returns)),
                    self.current_total_steps,
                    tensorboard=True,
                )
                self.metrics_manager.record_scalar(
                    "sampling/episode_return_std", float(np.std(episode_returns))
                )
                self.metrics_manager.record_scalar(
                    "sampling/max_episode_return", float(np.max(episode_returns))
                )
                self.metrics_manager.record_scalar(
                    "sampling/min_episode_return", float(np.min(episode_returns))
                )
                self.metrics_manager.record_scalar(
                    "sampling/average_episode_length",
                    float(np.mean(episode_lengths)),
                    self.current_total_steps,
                    tensorboard=True,
                )

            if self.current_total_steps >= num_steps_before_update:
                self.train(self.replay_buffer, num_train_steps, minibatch_size)

            if (
                num_evaluation_episodes > 0
                and self.current_total_steps % evaluation_interval == 0
            ):
                evaluation_episode_returns: List[float]
                evaluation_episode_lengths: List[int]
                (
                    evaluation_episode_returns,
                    evaluation_episode_lengths,
                ) = self.evaluator.evaluate(
                    self.policy, self.evaluation_env, num_evaluation_episodes
                )

                self.metrics_manager.record_scalar(
                    "evaluation/average_episode_return",
                    float(np.mean(evaluation_episode_returns)),
                    self.current_total_steps,
                    tensorboard=True,
                )
                self.metrics_manager.record_scalar(
                    "evaluation/episode_return_std",
                    float(np.std(evaluation_episode_returns)),
                )
                self.metrics_manager.record_scalar(
                    "evaluation/max_episode_return",
                    float(np.max(evaluation_episode_returns)),
                )
                self.metrics_manager.record_scalar(
                    "evaluation/min_episode_return",
                    float(np.min(evaluation_episode_returns)),
                )
                self.metrics_manager.record_scalar(
                    "evaluation/average_episode_length",
                    float(np.mean(evaluation_episode_lengths)),
                    self.current_total_steps,
                    tensorboard=True,
                )

            if self.current_total_steps % model_saving_interval == 0:
                model_path: str = os.path.join(output_dir, "model.pt")

                logger.debug("Save model")
                self.save_model(current_epoch, model_path)

            self.metrics_manager.record_scalar("time", time.time() - start_time)

            # Dump all metrics stored in this epoch
            self.metrics_manager.dump()

        self.metrics_manager.close()

    def train(
        self, replay_buffer: ReplayBuffer, num_train_steps: int, minibatch_size: int
    ) -> None:
        policy_losses: List[float] = []
        q_function_1_losses: List[float] = []
        q_function_2_losses: List[float] = []
        all_q_values_1: List[float] = []
        all_q_values_2: List[float] = []

        for train_step in range(num_train_steps):
            minibatch: Dict[str, np.ndarray] = replay_buffer.sample_minibatch(
                minibatch_size
            )

            observations: Tensor = torch.from_numpy(minibatch["observations"]).float()
            actions: Tensor = torch.from_numpy(minibatch["actions"]).float()
            rewards: Tensor = torch.from_numpy(minibatch["rewards"]).float()
            next_observations: Tensor = torch.from_numpy(
                minibatch["next_observations"]
            ).float()
            dones: Tensor = torch.from_numpy(minibatch["dones"]).int()

            # For logging
            with torch.no_grad():
                q_values_1: Tensor = self.q_function_1(observations, actions)
                q_values_2: Tensor = self.q_function_2(observations, actions)
            all_q_values_1.extend(q_values_1.tolist())
            all_q_values_2.extend(q_values_2.tolist())

            targets: Tensor = self.compute_targets(next_observations, rewards, dones)

            q_function_1_loss: Tensor = self.train_q_function(
                self.q_function_1, observations, actions, targets
            )
            q_function_2_loss: Tensor = self.train_q_function(
                self.q_function_2, observations, actions, targets
            )
            q_function_1_losses.append(q_function_1_loss.item())
            q_function_2_losses.append(q_function_2_loss.item())

            if train_step % self.policy_delay == 0:
                policy_loss: Tensor = self.train_policy(observations)
                policy_losses.append(policy_loss.item())

                # Update targets
                polyak_average(
                    self.policy.network.parameters(),
                    self.target_policy.network.parameters(),
                    self.polyak_rho,
                )
                polyak_average(
                    self.q_function_1.network.parameters(),
                    self.target_q_function_1.network.parameters(),
                    self.polyak_rho,
                )
                polyak_average(
                    self.q_function_2.network.parameters(),
                    self.target_q_function_2.network.parameters(),
                    self.polyak_rho,
                )

        self.metrics_manager.record_scalar(
            "policy/average_loss",
            float(np.mean(policy_losses)),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "q-function_1/average_loss",
            float(np.mean(q_function_1_losses)),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "q-function_2/average_loss",
            float(np.mean(q_function_2_losses)),
            self.current_total_steps,
            tensorboard=True,
        )

        self.metrics_manager.record_scalar(
            "q-function_1/avarage_q-value",
            float(np.mean(all_q_values_1)),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "q-function_1/max_q-value", float(np.max(all_q_values_1))
        )
        self.metrics_manager.record_scalar(
            "q-function_1/min_q-value", float(np.min(all_q_values_1))
        )
        self.metrics_manager.record_scalar(
            "q-function_2/avarage_q-value",
            float(np.mean(all_q_values_2)),
            self.current_total_steps,
            tensorboard=True,
        )
        self.metrics_manager.record_scalar(
            "q-function_2/max_q-value", float(np.max(all_q_values_2))
        )
        self.metrics_manager.record_scalar(
            "q-function_2/min_q-value", float(np.min(all_q_values_2))
        )

    def train_policy(self, observations: Tensor) -> Tensor:
        # Freeze Q-networks
        for param in self.q_function_1.network.parameters():
            param.requires_grad = False
        for param in self.q_function_2.network.parameters():
            param.requires_grad = False

        policy_actions: Tensor = self.policy(observations)
        policy_q_values: Tensor = self.q_function_1(observations, policy_actions)

        policy_loss: Tensor = -torch.mean(policy_q_values)

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        # Unfreeze Q-networks
        for param in self.q_function_1.network.parameters():
            param.requires_grad = True
        for param in self.q_function_2.network.parameters():
            param.requires_grad = True

        return policy_loss.detach()

    def compute_targets(
        self, next_observations: Tensor, rewards: Tensor, dones: Tensor
    ) -> Tensor:
        with torch.no_grad():
            next_actions: Tensor = self.target_policy(next_observations)
        epsilon: Tensor = self.target_noise_scale * torch.randn_like(next_actions)
        epsilon = torch.clamp(epsilon, -self.target_noise_clip, self.target_noise_clip)
        next_actions = next_actions + epsilon
        action_limit: float = self.env.action_space.high[0]
        next_actions = torch.clamp(next_actions, -action_limit, action_limit)

        with torch.no_grad():
            target_q_values_1: Tensor = self.target_q_function_1(
                next_observations, next_actions
            )
            target_q_values_2: Tensor = self.target_q_function_2(
                next_observations, next_actions
            )
        target_q_values: Tensor = torch.min(target_q_values_1, target_q_values_2)

        targets: Tensor = rewards + self.gamma * (1 - dones) * target_q_values

        return targets

    def train_q_function(
        self,
        q_function: QFunction,
        observations: Tensor,
        actions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        q_values: Tensor = q_function(observations, actions)

        q_function_loss: Tensor = F.mse_loss(q_values, targets)

        q_function.optimizer.zero_grad()
        q_function_loss.backward()
        q_function.optimizer.step()

        return q_function_loss.detach()

    def save_model(self, current_epoch: int, model_path: str) -> None:
        """
        Save model

        :param current_epoch: (int) The current epoch.
        :param model_path: (int) The path to save the model.
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
