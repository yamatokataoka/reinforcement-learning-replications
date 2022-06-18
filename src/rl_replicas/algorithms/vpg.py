import logging
from typing import List, Optional

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import functional as F

from rl_replicas.base_algorithms.on_policy_algorithm import OnPolicyAlgorithm
from rl_replicas.experience import Experience
from rl_replicas.policies import Policy
from rl_replicas.utils import discounted_cumulative_sums, gae
from rl_replicas.value_function import ValueFunction

logger = logging.getLogger(__name__)


class VPG(OnPolicyAlgorithm):
    """
    Vanilla Policy Gradient (REINFORCE) with GAE for advantage estimation

    :param policy: (Policy) The policy
    :param value_function: (ValueFunction) The value function
    :param env: (gym.Env) The environment to learn from
    :param gamma: (float) The discount factor for the cumulative return
    :param gae_lambda: (float) The factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param seed: (int) The seed for the pseudo-random generators
    :param n_value_gradients (int): Number of gradient descent steps to take on value function per epoch.
    """

    def __init__(
        self,
        policy: Policy,
        value_function: ValueFunction,
        env: gym.Env,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        seed: Optional[int] = None,
        n_value_gradients: int = 80,
    ) -> None:
        super().__init__(
            policy=policy,
            value_function=value_function,
            env=env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            n_value_gradients=n_value_gradients,
        )

    def train(self, one_epoch_experience: Experience) -> None:
        observations_list: List[List[np.ndarray]] = one_epoch_experience["observations"]
        actions_list: List[List[np.ndarray]] = one_epoch_experience["actions"]
        rewards_list: List[List[float]] = one_epoch_experience["rewards"]
        last_observations_list: List[np.ndarray] = one_epoch_experience[
            "last_observations"
        ]
        dones: List[bool] = one_epoch_experience["dones"]

        values_tensor_list: List[Tensor] = []
        with torch.no_grad():
            for (observations, last_observation) in zip(
                observations_list, last_observations_list
            ):
                observations_with_last_observation = torch.from_numpy(
                    np.concatenate([observations, [last_observation]])
                ).float()
                values_tensor_list.append(
                    self.value_function(observations_with_last_observation).flatten()
                )

        bootstrapped_rewards_list: List[List[float]] = []
        for episode_rewards, episode_done, values_tensor in zip(
            rewards_list, dones, values_tensor_list
        ):
            last_value_float: float = 0
            if not episode_done:
                last_value_float = values_tensor[-1].detach().item()
            bootstrapped_rewards_list.append(episode_rewards + [last_value_float])

        # Calculate rewards-to-go over each episode, to be targets for the value function
        discounted_returns: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    discounted_cumulative_sums(one_episode_rewards, self.gamma)[:-1]
                    for one_episode_rewards in bootstrapped_rewards_list
                ]
            )
        ).float()

        # Calculate advantages
        observations: Tensor = torch.from_numpy(
            np.concatenate(observations_list)
        ).float()
        actions: Tensor = torch.from_numpy(np.concatenate(actions_list)).float()

        advantages: Tensor = torch.from_numpy(
            np.concatenate(
                [
                    gae(
                        one_episode_rewards,
                        self.gamma,
                        one_episode_values.numpy(),
                        self.gae_lambda,
                    )
                    for one_episode_rewards, one_episode_values in zip(
                        bootstrapped_rewards_list, values_tensor_list
                    )
                ]
            )
        ).float()

        # Normalize advantage
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        policy_dist: Distribution = self.policy(observations)
        log_probs: Tensor = policy_dist.log_prob(actions)

        policy_loss: Tensor = -torch.mean(log_probs * advantages)

        # for logging
        policy_loss_before: Tensor = policy_loss.detach()
        entropies: Tensor = policy_dist.entropy().detach()

        # Train policy
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        # for logging
        with torch.no_grad():
            value_loss_before: Tensor = self.compute_value_loss(
                observations, discounted_returns
            )

        # Train value function
        for _ in range(self.n_value_gradients):
            value_loss: Tensor = self.compute_value_loss(
                observations, discounted_returns
            )
            self.value_function.optimizer.zero_grad()
            value_loss.backward()
            self.value_function.optimizer.step()

        logger.info("Policy Loss:            {:<8.3g}".format(policy_loss_before))
        logger.info("Avarage Entropy:        {:<8.3g}".format(torch.mean(entropies)))
        logger.info("Log Prob STD:           {:<8.3g}".format(torch.std(log_probs)))

        logger.info("Value Function Loss:    {:<8.3g}".format(value_loss_before))

        if self.tensorboard:
            self.writer.add_scalar(
                "policy/loss", policy_loss_before, self.current_total_steps
            )
            self.writer.add_scalar(
                "policy/avarage_entropy",
                torch.mean(entropies),
                self.current_total_steps,
            )
            self.writer.add_scalar(
                "policy/log_prob_std", torch.std(log_probs), self.current_total_steps
            )

            self.writer.add_scalar(
                "value/loss", value_loss_before, self.current_total_steps
            )

    def compute_value_loss(
        self, observations: Tensor, discounted_returns: Tensor
    ) -> Tensor:
        values: Tensor = self.value_function(observations)
        squeezed_values: Tensor = torch.squeeze(values, -1)
        value_loss: Tensor = F.mse_loss(squeezed_values, discounted_returns)

        return value_loss
