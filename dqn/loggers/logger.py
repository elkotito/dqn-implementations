from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    mean_rolling_loss: float | None
    mean_episode_length: float | None
    mean_episode_return: float | None
    epsilon: float
    replay_buffer_size: int


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mean_episode_return: float
    min_episode_return: float
    max_episode_return: float
    mean_episode_length: float


class Logger(Protocol):
    def log_train(self, *, metrics: TrainingMetrics, step: int) -> None: ...
    def log_eval(self, *, metrics: EvaluationMetrics, step: int) -> None: ...

    def close(self) -> None: ...
