from collections import deque
from statistics import fmean
from typing import Self, cast

import gymnasium as gym
import torch
import torch.nn as nn
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dqn.buffers.replay_buffer import ReplayBuffer, Transition, Transitions
from dqn.envs.atari import make_atari_env
from dqn.loggers.logger import Logger, TrainingMetrics
from dqn.network.dqn import DQN


class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        env_prefix="DQN_",
    )

    # Training
    total_steps: int = Field(default=1_000_000, ge=0)
    seed: int = Field(default=2137, ge=0)

    # Environment
    env_id: str = Field(default="ALE/Pong-v5", pattern=r"^ALE/")

    # Replay buffer
    replay_warmup_steps: int = Field(default=5_000, ge=0)
    replay_capacity: int = Field(default=10_000, gt=0)

    # Optimizer
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-4, ge=0.0)

    # Exploration
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay_steps: int = Field(default=250_000, gt=0)

    # Target network
    tau_soft_update: float = Field(default=0.01, ge=0.0)

    # Rewards
    reward_clip: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    # Infrastructure
    device: str = Field(default="cpu", pattern=r"^cpu|cuda|mps$")

    # Logging
    log_interval_steps: int = Field(default=1_000, gt=0)
    episode_window_size: int = Field(default=100, gt=0)
    loss_window_size: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot be greater than epsilon_start")

        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot be greater than replay_capacity")

        if self.batch_size > self.replay_warmup_steps:
            raise ValueError("batch_size cannot be greater than replay_warmup_steps")

        if self.replay_warmup_steps > self.replay_capacity:
            raise ValueError("replay_warmup_steps cannot be greater than replay_capacity")

        if self.replay_warmup_steps >= self.total_steps:
            raise ValueError("replay_warmup_steps must be smaller than total_steps")

        if self.replay_warmup_steps + self.epsilon_decay_steps > self.total_steps:
            raise ValueError("warmup and epsilon decay must finish within total_steps")

        return self


class Trainer:
    """
    Trains a DQN agent using the given configuration.

    Typically, components such as the policy network and replay buffer are created
    separately and passed to the trainer. This project keeps them inside the trainer
    for simplicity, since it is a standalone implementation rather than a training framework.
    """

    def __init__(self, config: TrainerConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.env = make_atari_env(config.env_id)
        action_space = cast(gym.spaces.Discrete, self.env.action_space)

        self.policy_network = DQN(int(action_space.n), config.device)
        self.target_network = self.policy_network.build_target_network()

        self.replay_buffer = ReplayBuffer(config.replay_capacity)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _epsilon_at_step(self, step: int) -> float:
        """Linear decay of epsilon over the course of training after replay warmup steps."""

        if step < self.config.replay_warmup_steps:
            return self.config.epsilon_start

        step -= self.config.replay_warmup_steps
        step = min(step, self.config.epsilon_decay_steps)
        diff = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
        return self.config.epsilon_start - step * diff

    def _train_step(self, batch_transitions: Transitions) -> float:
        states, actions, rewards, next_states, dones = batch_transitions

        # Transferring uint8 to GPU is 4x faster than floats
        states = states.to(self.device).float().div_(255.0)
        next_states = next_states.to(self.device).float().div_(255.0)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        predicted_q_values = self.policy_network(states).gather(index=actions[:, None], dim=1).squeeze(1)
        with torch.no_grad():
            max_q_values, _ = self.target_network(next_states).max(dim=1)
            target_q_values = rewards + self.config.gamma * (~dones).float() * max_q_values

        loss = self.loss_fn(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update (Polyak)
        self.target_network.update_from(self.policy_network, tau=self.config.tau_soft_update)

        return loss.item()

    def _train(self) -> None:
        returns = deque[float](maxlen=self.config.episode_window_size)
        episode_lengths = deque[int](maxlen=self.config.episode_window_size)
        losses = deque[float](maxlen=self.config.loss_window_size)
        episode_return = 0
        episode_length = 0

        state, _ = self.env.reset(seed=self.config.seed)
        for step in range(1, self.config.total_steps + 1):
            epsilon = self._epsilon_at_step(step)
            episode_length += 1

            # Add batch dimension since `sample_actions` expects a batch of states, but we use a single env
            batch_state = state.unsqueeze(0).to(self.device).float().div_(255.0)
            batch_actions = self.policy_network.sample_actions(batch_state, epsilon=epsilon)
            action = batch_actions.squeeze(0).cpu()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

            transition = Transition(state=state, action=int(action.item()), reward=float(reward), next_state=next_state, done=done)
            self.replay_buffer.add(transition)

            if done:
                returns.append(episode_return)
                episode_lengths.append(episode_length)

                state, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
            else:
                state = next_state

            if len(self.replay_buffer) >= self.config.replay_warmup_steps:
                batch_transitions = self.replay_buffer.sample(self.config.batch_size)
                loss = self._train_step(batch_transitions)
                losses.append(loss)

            if step % self.config.log_interval_steps == 0:
                self.logger.log(
                    step=step,
                    metrics=TrainingMetrics(
                        mean_rolling_loss=fmean(losses) if losses else None,
                        mean_episode_length=fmean(episode_lengths) if episode_lengths else None,
                        mean_recent_return=fmean(returns) if returns else None,
                        epsilon=epsilon,
                        replay_buffer_size=len(self.replay_buffer),
                    ),
                )

    def train(self) -> None:
        try:
            self._train()
        finally:
            self.env.close()
            self.logger.close()
