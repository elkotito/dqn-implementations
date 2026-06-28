class LinearScheduler:
    def __init__(self, *, warmup_steps: int, start: float, end: float, num_steps: int) -> None:
        self.replay_warmup_steps = warmup_steps
        self.start = start
        self.end = end
        self.num_steps = num_steps

    def __call__(self, step: int) -> float:
        if step < self.replay_warmup_steps:
            return self.start

        step -= self.replay_warmup_steps
        step = min(step, self.num_steps)
        diff = (self.start - self.end) / self.num_steps
        return self.start - step * diff
