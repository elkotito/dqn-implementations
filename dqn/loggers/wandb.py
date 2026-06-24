from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal

import wandb

from dqn.loggers.logger import TrainingMetrics


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

    def log(self, *, metrics: TrainingMetrics, step: int) -> None:
        self._run.log(asdict(metrics), step=step)

    def close(self) -> None:
        if not self._closed:
            self._run.finish()
            self._closed = True
