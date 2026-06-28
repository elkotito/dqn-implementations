import random
from typing import TypedDict

import torch
from torch import Tensor

from src.buffers.replay_buffer import Transition


class Transitions(TypedDict):
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor
    tree_idxs: Tensor
    weights: Tensor


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.max_size = capacity
        self.curr_size = 0
        self.curr_buffer_idx = 0

        self.tree = [0.0] * (2 * capacity - 1)
        # Use this offset to convert between circular replay buffer indices to tree leaves and vice versa
        self.n_tree_internal_nodes = self.max_size - 1

    def total(self) -> float:
        return self.tree[0]

    def update(self, idx: int, value: float) -> None:
        # Propagate the change up to the root
        change = value - self.tree[idx]
        while idx >= 0:
            self.tree[idx] += change
            idx = (idx - 1) // 2

    def add(self, value: float) -> None:
        # Convert the circular replay buffer index to its leaf position in the tree
        tree_idx = self.curr_buffer_idx + self.n_tree_internal_nodes
        self.update(tree_idx, value)

        self.curr_buffer_idx = (self.curr_buffer_idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def get(self, value: float) -> tuple[int, int, float]:
        # Prefix sum search; first prefix >= value
        tree_idx = 0

        # We stop iterating at a leaf node
        while tree_idx < self.n_tree_internal_nodes:
            left = 2 * tree_idx + 1
            right = left + 1

            # e.g. Root = [0, 10), Left = [0, 3), Right = [3, 10) => left value is a splitter
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right

        # Convert the leaf position in a tree back to the raw replay buffer index
        data_idx = tree_idx - self.n_tree_internal_nodes
        return tree_idx, data_idx, self.tree[tree_idx]


class PrioritizedReplayBuffer:
    """Store raw CPU uint8 observations and return unnormalized CPU batches."""

    def __init__(self, max_size: int, alpha: float = 0.6) -> None:
        self.sum_tree = SumTree(max_size)

        self.epsilon = 1e-6
        self.alpha = alpha
        self.max_priority = 1.0
        self.max_size = max_size
        self.curr_size = 0
        self.curr_buffer_idx = 0

        # Replay buffer arrays
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

        self.sum_tree.add(self.max_priority)
        self.curr_buffer_idx = (self.curr_buffer_idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size: int, *, beta: float) -> Transitions:
        total_priority = self.sum_tree.total()
        segment = total_priority / batch_size
        data_idxs: list[int] = []
        tree_idxs: list[int] = []
        sampling_probabilities: list[float] = []

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            tree_idx, data_idx, priority = self.sum_tree.get(value)
            data_idxs.append(data_idx)
            tree_idxs.append(tree_idx)
            sampling_probabilities.append(priority / total_priority)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in data_idxs:
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

        weights = (self.curr_size * torch.tensor(sampling_probabilities, dtype=torch.float32)).pow(-beta)
        weights.div_(weights.max())

        return {
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.int64),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.stack(next_states),
            "dones": torch.tensor(dones, dtype=torch.bool),
            "tree_idxs": torch.tensor(tree_idxs, dtype=torch.int64),
            "weights": weights,
        }

    def update_priorities(self, tree_idxs: Tensor, td_errors: Tensor) -> None:
        tree_idxs = tree_idxs.detach().cpu().to(torch.int64).flatten()
        td_errors = td_errors.detach().cpu().to(torch.float32).flatten()
        priorities = (td_errors.abs() + self.epsilon).pow(self.alpha)

        for tree_idx, priority in zip(tree_idxs.tolist(), priorities.tolist(), strict=True):
            self.sum_tree.update(int(tree_idx), float(priority))
            self.max_priority = max(self.max_priority, float(priority))

    def __len__(self) -> int:
        return self.curr_size
