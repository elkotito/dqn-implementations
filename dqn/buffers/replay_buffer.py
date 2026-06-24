from dataclasses import dataclass
from random import sample

import torch
from torch import Tensor


@dataclass
class Transition:
    state: Tensor
    reward: float
    next_state: Tensor
    done: bool


class ReplayBuffer:
    """Store raw pixels (uint8) in ReplayBuffer to save memory and convert to float only after sampling."""

    def __init__(self, max_size: int) -> None:
        # Array is better than deque for random access
        self.buffer: list[Transition | None] = [None] * max_size
        self.max_size = max_size
        self.curr_size = 0
        self.idx = 0
        self.device = device

    def add(self, transition: Transition):
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sampled_idxs = sample(range(self.curr_size), k=batch_size)
        states, next_states, rewards, dones = [], [], [], []
        for idx in sampled_idxs:
            transition = self.buffer[idx]
            assert transition is not None
            states.append(transition.state)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            dones.append(transition.done)

        states = torch.stack([state for state in states])
        next_states = torch.stack([next_state for next_state in next_states])
        rewards = torch.tensor([reward for reward in rewards])
        dones = torch.tensor([done for done in dones])
        return states.div_(255.0), rewards, next_states.div_(255.0), dones

    def __len__(self) -> int:
        return self.curr_size
