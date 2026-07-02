from copy import deepcopy
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor


class DuelingDQN(nn.Module):
    def __init__(self, n_actions: int, device: str) -> None:
        super().__init__()
        self.n_actions = n_actions
        self._model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.LeakyReLU(negative_slope=0.01),
        ).to(device)

        self._value_head = nn.Linear(512, 1).to(device)
        self._advantage_head = nn.Linear(512, n_actions).to(device)

    @torch.no_grad()
    def update_from(self, source: Self, *, tau: float) -> None:
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.lerp_(source_param, weight=tau)

    @torch.no_grad()
    def sync(self, source: Self) -> None:
        self.load_state_dict(source.state_dict())

    def build_target_network(self) -> Self:
        target = deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target

    def forward(self, states: Tensor) -> Tensor:
        features = self._model(states)
        value = self._value_head(features)
        advantage = self._advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def greedy_actions(self, states: Tensor) -> Tensor:
        return self.forward(states).argmax(dim=1)

    @torch.no_grad()
    def eps_greedy_actions(self, states: Tensor, *, epsilon: float) -> Tensor:
        batch_size = states.shape[0]

        greedy = self.greedy_actions(states)
        random = torch.randint(self.n_actions, size=(batch_size,), device=states.device)
        explore = torch.rand(batch_size, device=states.device) < epsilon

        return torch.where(explore, random, greedy)
