from pathlib import Path
from typing import Literal, Self

import gymnasium as gym
import torch
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource


class PrioritizedDQNConfig(BaseSettings):
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
    total_steps: int = Field(default=10_000_000, gt=0)
    gradient_update_interval_steps: int = Field(
        default=4,
        gt=0,
        description=(
            "Run one gradient update every N environment steps as it improves training speed. "
            "It is not related to 4 frames forming a single state."
        ),
    )

    # Environment
    env_id: str = Field(default="ALE/Pong-v5", pattern=r"^ALE/")

    # Replay buffer
    replay_warmup_steps: int = Field(default=50_000, ge=0)
    replay_capacity: int = Field(default=100_000, gt=0)
    replay_alpha: float = Field(default=0.6, ge=0.0, le=1.0, allow_inf_nan=False)
    replay_beta_start: float = Field(default=0.4, ge=0.0, le=1.0, allow_inf_nan=False)
    replay_beta_end: float = Field(default=0.8, ge=0.0, le=1.0, allow_inf_nan=False)
    replay_beta_anneal_steps: int = Field(default=1_000_000, gt=0)

    # Optimizer
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=6.25e-5, gt=0.0, allow_inf_nan=False)
    adam_epsilon: float = Field(default=1.5e-4, gt=0.0, allow_inf_nan=False)
    max_grad_norm: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)

    # Exploration
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0, allow_inf_nan=False)
    epsilon_decay_steps: int = Field(default=1_000_000, gt=0)

    # Target network
    target_network_sync_interval_steps: int = Field(default=10_000, gt=0)

    # Rewards
    reward_clip: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, allow_inf_nan=False)

    # Infrastructure & meta
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    seed: int = Field(default=2137, ge=0, le=2**32 - 1)

    # Output
    policy_dir: Path = Field(default=Path("./policies"), strict=False)

    # Evaluation
    eval_interval_steps: int = Field(default=250_000, gt=0)
    eval_episodes: int = Field(default=10, gt=0)

    # WandB
    wandb_project: str = Field(default="pong", min_length=1)
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    # Logging
    train_log_interval_steps: int = Field(default=10_000, gt=0)
    episode_window: int = Field(default=100, gt=0)
    gradient_window: int = Field(default=100, gt=0)

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

        try:
            gym.spec(self.env_id)
        except gym.error.Error as error:
            raise ValueError(f"environment does not exist: {self.env_id}") from error

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device is not available")

        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device is not available")

        if self.policy_dir.exists() and not self.policy_dir.is_dir():
            raise ValueError(f"policy_dir is not a directory: {self.policy_dir}")

        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot be greater than epsilon_start")

        if self.replay_beta_end < self.replay_beta_start:
            raise ValueError("beta_end cannot be smaller than beta_start")

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

        if self.replay_warmup_steps + self.replay_beta_anneal_steps > self.total_steps:
            raise ValueError("warmup and beta steps must finish within total_steps")

        return self
