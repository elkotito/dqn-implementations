import os
import random
from collections import deque
from pathlib import Path
from statistics import fmean
from typing import Literal, Self, cast

import gymnasium as gym
import torch
import torch.nn as nn
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

from dqn.buffers.replay_buffer import ReplayBuffer, Transition, Transitions
from dqn.envs.atari import make_atari_env
from dqn.loggers.logger import EvaluationMetrics, TrainingMetrics
from dqn.loggers.wandb import WandbLogger
from dqn.network.dqn import DQN


class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        cli_hide_none_type=True,
        cli_kebab_case=True,
        env_prefix="DQN_",
    )

    config: Path | None = Field(default=None, exclude=True, description="Path to a YAML configuration file")

    # Training
    total_steps: int = Field(default=10_000_000, ge=0)
    train_frequency_steps: int = Field(
        default=4,
        gt=0,
        description="Run one gradient update every N environment steps. It improves training speed. It is not related to 4 frames forming a single state.",
    )

    # Environment
    env_id: str = Field(default="ALE/Pong-v5", pattern=r"^ALE/")

    # Replay buffer
    replay_warmup_steps: int = Field(default=50_000, ge=0)
    replay_capacity: int = Field(default=100_000, gt=0)

    # Optimizer
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-4, ge=0.0)

    # Exploration
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay_steps: int = Field(default=1_000_000, gt=0)

    # Target network
    target_sync_interval_steps: int = Field(default=10_000, gt=0)

    # Rewards
    reward_clip: float = Field(default=1.0, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    # Infrastructure & meta
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    seed: int = Field(default=2137, ge=0)

    # Output
    policy_dir: Path | None = Field(default=None, description="Directory to store policy.pt and best_policy.pt")

    # Evaluation
    eval_interval_steps: int = Field(default=250_000, gt=0)
    eval_episodes: int = Field(default=10, gt=0)

    # Logging
    wandb_project: str = Field(default="pong", min_length=1)
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    log_interval_steps: int = Field(default=10_000, gt=0)
    episode_window_size: int = Field(default=100, gt=0)
    loss_window_size: int = Field(default=100, gt=0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        cli_settings = CliSettingsSource(settings_cls, cli_parse_args=True)
        config_path = cli_settings().get("config")
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_path)

        # Order determines priority
        return cli_settings, init_settings, env_settings, dotenv_settings, file_secret_settings, yaml_settings

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.config is not None and not self.config.is_file():
            raise ValueError(f"config file does not exist: {self.config}")

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

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.env = make_atari_env(config.env_id)
        self.eval_env = make_atari_env(config.env_id)
        action_space = cast(gym.spaces.Discrete, self.env.action_space)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.policy_network = DQN(int(action_space.n), config.device)
        self.target_network = self.policy_network.build_target_network()

        self.replay_buffer = ReplayBuffer(config.replay_capacity)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.best_eval_mean_episode_return = float("-inf")
        self.latest_train_mean_episode_return: float | None = None
        self.logger = WandbLogger(
            project=config.wandb_project,
            config=config.model_dump(mode="json"),
            mode=config.wandb_mode,
        )

    def _save_policy(
        self,
        name: str,
        *,
        step: int,
        train_mean_episode_return: float | None,
        eval_mean_episode_return: float | None,
    ) -> Path | None:
        if self.config.policy_dir is None:
            return None

        self.config.policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = self.config.policy_dir / name
        temporary_path = policy_path.with_suffix(".pt.tmp")
        checkpoint = {
            "policy_state_dict": {
                parameter_name: parameter.detach().cpu() for parameter_name, parameter in self.policy_network.state_dict().items()
            },
            "env_id": self.config.env_id,
            "step": step,
            "train_mean_episode_return": train_mean_episode_return,
            "eval_mean_episode_return": eval_mean_episode_return,
        }

        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, policy_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return policy_path

    def _evaluate(self) -> EvaluationMetrics:
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(self.config.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                batch_state = state.unsqueeze(0).to(self.device).float().div_(255.0)
                action = self.policy_network.greedy_actions(batch_state).squeeze(0).cpu()
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_return += float(reward)
                episode_length += 1
                done = terminated or truncated

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        return EvaluationMetrics(
            mean_episode_return=fmean(episode_returns),
            min_episode_return=min(episode_returns),
            max_episode_return=max(episode_returns),
            mean_episode_length=fmean(episode_lengths),
        )

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

        return loss.item()

    def _train(self) -> None:
        episode_returns = deque[float](maxlen=self.config.episode_window_size)
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
            batch_actions = self.policy_network.eps_greedy_actions(batch_state, epsilon=epsilon)
            action = batch_actions.squeeze(0).cpu()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

            # We use raw reward to report `episode_return`, but rewards in `ReplayBuffer` are used for training hence clipping there
            clipped_reward = max(-self.config.reward_clip, min(self.config.reward_clip, float(reward)))
            # A time limit does not make the state worthless; the agent could still take valuable actions from it => `done=truncated`
            transition = Transition(state=state, action=int(action.item()), reward=clipped_reward, next_state=next_state, done=terminated)
            self.replay_buffer.add(transition)

            if done:
                state, _ = self.env.reset()
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                episode_return = 0
                episode_length = 0

                mean_episode_return = fmean(episode_returns)
                self.latest_train_mean_episode_return = mean_episode_return
            else:
                state = next_state

            if len(self.replay_buffer) >= self.config.replay_warmup_steps and step % self.config.train_frequency_steps == 0:
                batch_transitions = self.replay_buffer.sample(self.config.batch_size)
                loss = self._train_step(batch_transitions)
                losses.append(loss)

            if step % self.config.target_sync_interval_steps == 0:
                self.target_network.sync(self.policy_network)

            if step % self.config.log_interval_steps == 0:
                training_metrics = TrainingMetrics(
                    mean_rolling_loss=fmean(losses) if losses else None,
                    mean_episode_length=fmean(episode_lengths) if episode_lengths else None,
                    mean_episode_return=fmean(episode_returns) if episode_returns else None,
                    epsilon=epsilon,
                    replay_buffer_size=len(self.replay_buffer),
                )
                self.logger.log_train(metrics=training_metrics, step=step)

            if step % self.config.eval_interval_steps == 0 or step == self.config.total_steps:
                evaluation_metrics = self._evaluate()
                self.logger.log_eval(metrics=evaluation_metrics, step=step)
                self._save_policy(
                    "policy.pt",
                    step=step,
                    train_mean_episode_return=self.latest_train_mean_episode_return,
                    eval_mean_episode_return=evaluation_metrics.mean_episode_return,
                )

                if evaluation_metrics.mean_episode_return > self.best_eval_mean_episode_return:
                    self.best_eval_mean_episode_return = evaluation_metrics.mean_episode_return
                    self._save_policy(
                        "best_policy.pt",
                        step=step,
                        train_mean_episode_return=self.latest_train_mean_episode_return,
                        eval_mean_episode_return=evaluation_metrics.mean_episode_return,
                    )

    def train(self) -> None:
        try:
            self._train()
        finally:
            self.env.close()
            self.eval_env.close()
            self.logger.close()
