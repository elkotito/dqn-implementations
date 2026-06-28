from pathlib import Path
from typing import Any, Literal, Self, cast

import gymnasium as gym
import torch
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dqn.envs.atari import make_atari_env
from dqn.evaluation.greedy_policy_evaluator import GreedyPolicyEvaluator
from dqn.network.dqn import DQN


class RecordVideoConfig(BaseSettings):
    """Record greedy episodes from a saved DQN policy."""

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        cli_hide_none_type=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        cli_parse_args=True,
    )

    policy: Path = Field(description="Path to policy.pt or best_policy.pt", strict=False)
    output_dir: Path = Field(default=Path("videos"), strict=False)
    episodes: int = Field(default=1, gt=0)
    device: Literal["cpu", "cuda", "mps"] = "mps"
    seed: int = Field(default=2137, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.policy.is_file():
            raise ValueError(f"policy file does not exist: {self.policy}")

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"output_dir is not a directory: {self.output_dir}")

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device is not available")

        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device is not available")

        return self


def main() -> None:
    config = RecordVideoConfig()  # pyright: ignore[reportCallIssue]
    evaluator = GreedyPolicyEvaluator(episodes=config.episodes, seed=config.seed, device=torch.device(config.device))

    checkpoint = torch.load(config.policy, map_location=config.device, weights_only=True)
    env_id = str(checkpoint["env_id"])
    env = make_atari_env(env_id, video_dir=config.output_dir, video_name_prefix=config.policy.stem)

    try:
        action_space = cast(gym.spaces.Discrete, env.action_space)
        policy = DQN(int(action_space.n), config.device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()

        evaluator.evaluate(env=env, policy_network=policy)
    finally:
        env.close()


if __name__ == "__main__":
    main()
