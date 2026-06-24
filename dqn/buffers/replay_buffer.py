from dataclasses import dataclass
from random import sample

import torch
from torch import Tensor


@dataclass
class Transition:
    state: Tensor
    action: int
    reward: float
    next_state: Tensor
    done: bool


type Transitions = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class ReplayBuffer:
    """Store raw CPU uint8 observations and return unnormalized CPU batches."""

    def __init__(self, max_size: int) -> None:
        # Array is better than deque for random access
        self.buffer: list[Transition | None] = [None] * max_size
        self.max_size = max_size
        self.curr_size = 0
        self.idx = 0

    def add(self, transition: Transition) -> None:
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size: int) -> Transitions:
        sampled_idxs = sample(range(self.curr_size), k=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in sampled_idxs:
            transition = self.buffer[idx]
            assert transition is not None
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return self.curr_size
