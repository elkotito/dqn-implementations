import os
from pathlib import Path

import torch
import torch.nn as nn


class PolicyCheckpointStore:
    def __init__(self, *, policy_dir: Path, env_id: str) -> None:
        self.policy_dir = policy_dir
        self.env_id = env_id

    def save(
        self,
        name: str,
        *,
        policy_network: nn.Module,
        step: int,
        train_mean_episode_return: float | None,
        eval_mean_episode_return: float | None,
    ) -> Path:
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = self.policy_dir / name
        temporary_path = policy_path.with_suffix(".pt.tmp")
        checkpoint = {
            "policy_state_dict": {
                parameter_name: parameter.detach().cpu() for parameter_name, parameter in policy_network.state_dict().items()
            },
            "env_id": self.env_id,
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
