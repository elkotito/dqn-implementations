from dataclasses import dataclass
from random import sample


@dataclass
class Transition[State]:
    state: State
    reward: float
    next_state: State
    done: bool


class ReplayBuffer[State]:
    def __init__(self, max_size: int) -> None:
        self.buffer: list[Transition | None] = [None] * max_size
        self.max_size = max_size
        self.curr_size = 0
        self.idx = 0

    def add(self, transition: Transition[State]):
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, k: int) -> list[Transition[State]]:
        sampled_idxs = sample(range(self.curr_size), k=k)
        results: list[Transition[State]] = []
        for idx in sampled_idxs:
            transition = self.buffer[idx]
            assert transition is not None
            results.append(transition)

        return results

    def __len__(self):
        return self.curr_size
