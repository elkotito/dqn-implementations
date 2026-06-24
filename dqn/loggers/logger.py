from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    mean_rolling_loss: float | None
    mean_episode_length: float | None
    mean_recent_return: float | None
    epsilon: float
    replay_buffer_size: int


class Logger(Protocol):
    def log(self, *, metrics: TrainingMetrics, step: int) -> None: ...
    def close(self) -> None: ...
