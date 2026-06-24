from copy import deepcopy
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor


class DQN(nn.Module):
    def __init__(self, n_actions: int, device: str) -> None:
        super().__init__()
        self.n_actions = n_actions
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
        ).to(device)

    @torch.no_grad()
    def update_from(self, source: Self, *, tau: float) -> None:
        """Used to update Target Network"""
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.lerp_(source_param, weight=tau)

    def build_target_network(self) -> Self:
        target = deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target

    def forward(self, states: Tensor) -> Tensor:
        return self.model(states)

    @torch.no_grad()
    def sample_actions(self, states: Tensor, *, epsilon: float) -> Tensor:
        batch_size = states.shape[0]

        greedy = self.forward(states).argmax(dim=1)
        random = torch.randint(self.n_actions, size=(batch_size,), device=states.device)
        explore = torch.rand(batch_size, device=states.device) < epsilon

        return torch.where(explore, random, greedy)
