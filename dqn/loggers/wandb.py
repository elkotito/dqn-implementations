from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal

import wandb
from dqn.metrics.metrics_tracker import EvaluationMetrics, TrainingMetrics


class WandbLogger:
    def __init__(
        self,
        *,
        project: str,
        config: Mapping[str, object],
        entity: str | None = None,
        run_name: str | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
    ) -> None:
        self._run = wandb.init(project=project, entity=entity, name=run_name, config=dict(config), mode=mode)
        self._closed = False

    def log_train(self, *, metrics: TrainingMetrics, step: int) -> None:
        self._run.log({f"train/{name}": value for name, value in asdict(metrics).items()}, step=step)

    def log_eval(self, *, metrics: EvaluationMetrics, step: int) -> None:
        self._run.log({f"eval/{name}": value for name, value in asdict(metrics).items()}, step=step)

    def close(self) -> None:
        if not self._closed:
            self._run.finish()
            self._closed = True
