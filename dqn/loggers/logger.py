from typing import Protocol

from dqn.training.metrics import EvaluationMetrics, TrainingMetrics


class Logger(Protocol):
    def log_train(self, *, metrics: TrainingMetrics, step: int) -> None: ...
    def log_eval(self, *, metrics: EvaluationMetrics, step: int) -> None: ...

    def close(self) -> None: ...
