from copy import deepcopy
from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class DQN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    @torch.no_grad()
    def update_from(self, source: Self, *, tau: float) -> None:
        """Used to update Target Network"""
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.lerp_(source_param, weight=tau)

    def create_target_network(self) -> Self:
        target = deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target

    def forward(
        self,
        states: Float[Tensor, "batch n_frames x y"],  # noqa: F722
    ) -> Float[Tensor, "batch q_values"]:  # noqa: F722
        return self.model(states)
