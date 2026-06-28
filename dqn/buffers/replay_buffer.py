from random import sample
from typing import NamedTuple

import torch
from torch import Tensor


class Transition(NamedTuple):
    state: Tensor
    action: int
    reward: float
    next_state: Tensor
    done: bool


type Transitions = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class NaiveReplayBuffer:
    """Store raw CPU uint8 observations and return unnormalized CPU batches."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.curr_size = 0
        self.curr_buffer_idx = 0

        self.states: list[Tensor | None] = [None] * max_size
        self.actions: list[int | None] = [None] * max_size
        self.rewards: list[float | None] = [None] * max_size
        self.next_states: list[Tensor | None] = [None] * max_size
        self.dones: list[bool | None] = [None] * max_size

    def add(self, transition: Transition) -> None:
        state, action, reward, next_state, done = transition
        self.states[self.curr_buffer_idx] = state
        self.actions[self.curr_buffer_idx] = action
        self.rewards[self.curr_buffer_idx] = reward
        self.next_states[self.curr_buffer_idx] = next_state
        self.dones[self.curr_buffer_idx] = done

        self.curr_buffer_idx = (self.curr_buffer_idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size: int) -> Transitions:
        sampled_idxs = sample(range(self.curr_size), k=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in sampled_idxs:
            state = self.states[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            next_state = self.next_states[idx]
            done = self.dones[idx]

            assert state is not None
            assert action is not None
            assert reward is not None
            assert next_state is not None
            assert done is not None

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return self.curr_size
