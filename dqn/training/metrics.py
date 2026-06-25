from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode_return: float
    episode_length: int


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    loss: float
    gradient_norm: float
    predicted_q: float
    predicted_q_max: float
    target_q: float
    target_q_max: float
    abs_td_error: float


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    mean_rolling_loss: float | None
    mean_rolling_gradient_norm: float | None
    mean_rolling_predicted_q: float | None
    mean_rolling_predicted_q_max: float | None
    mean_rolling_target_q: float | None
    mean_rolling_target_q_max: float | None
    mean_rolling_abs_td_error: float | None
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


class MetricsTracker:
    def __init__(self, *, episode_window_size: int, optimization_window_size: int) -> None:
        self._episodes = deque[EpisodeMetrics](maxlen=episode_window_size)
        self._optimizations = deque[OptimizationMetrics](maxlen=optimization_window_size)
        self._latest_evaluation: EvaluationMetrics | None = None
        self._best_evaluation: EvaluationMetrics | None = None

    def record_episode(self, metrics: EpisodeMetrics) -> None:
        self._episodes.append(metrics)

    def record_optimization(self, metrics: OptimizationMetrics) -> None:
        self._optimizations.append(metrics)

    def record_evaluation(self, metrics: EvaluationMetrics) -> bool:
        self._latest_evaluation = metrics
        if self._best_evaluation is None or metrics.mean_episode_return > self._best_evaluation.mean_episode_return:
            self._best_evaluation = metrics
            return True
        return False

    @property
    def mean_episode_return(self) -> float | None:
        return self._mean(self._episodes, lambda metrics: metrics.episode_return)

    @property
    def latest_evaluation(self) -> EvaluationMetrics | None:
        return self._latest_evaluation

    @property
    def best_evaluation(self) -> EvaluationMetrics | None:
        return self._best_evaluation

    def training_snapshot(self, *, epsilon: float, replay_buffer_size: int) -> TrainingMetrics:
        return TrainingMetrics(
            mean_rolling_loss=self._mean(self._optimizations, lambda metrics: metrics.loss),
            mean_rolling_gradient_norm=self._mean(self._optimizations, lambda metrics: metrics.gradient_norm),
            mean_rolling_predicted_q=self._mean(self._optimizations, lambda metrics: metrics.predicted_q),
            mean_rolling_predicted_q_max=self._mean(self._optimizations, lambda metrics: metrics.predicted_q_max),
            mean_rolling_target_q=self._mean(self._optimizations, lambda metrics: metrics.target_q),
            mean_rolling_target_q_max=self._mean(self._optimizations, lambda metrics: metrics.target_q_max),
            mean_rolling_abs_td_error=self._mean(self._optimizations, lambda metrics: metrics.abs_td_error),
            mean_episode_length=self._mean(self._episodes, lambda metrics: metrics.episode_length),
            mean_episode_return=self.mean_episode_return,
            epsilon=epsilon,
            replay_buffer_size=replay_buffer_size,
        )

    def _mean[T](self, metrics: deque[T], selector: Callable[[T], float | int]) -> float | None:
        if not metrics:
            return None
        return fmean(selector(item) for item in metrics)
