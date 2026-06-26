class LinearEpsilonScheduler:
    def __init__(
        self,
        *,
        replay_warmup_steps: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
    ) -> None:
        self.replay_warmup_steps = replay_warmup_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def __call__(self, step: int) -> float:
        """Linear decay of epsilon over the course of training after replay warmup steps."""

        if step < self.replay_warmup_steps:
            return self.epsilon_start

        step -= self.replay_warmup_steps
        step = min(step, self.epsilon_decay_steps)
        diff = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        return self.epsilon_start - step * diff
