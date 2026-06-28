from statistics import fmean
from typing import Protocol

import torch
from torch import Tensor

from src.envs.atari import AtariEnv
from src.metrics.metrics_tracker import EvaluationMetrics


class GreedyPolicy(Protocol):
    def greedy_actions(self, states: Tensor) -> Tensor: ...


class GreedyPolicyEvaluator:
    def __init__(self, *, episodes: int, seed: int, device: torch.device) -> None:
        self.episodes = episodes
        self.seed = seed
        self.device = device

    def evaluate(self, *, env: AtariEnv, policy_network: GreedyPolicy) -> EvaluationMetrics:
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        with torch.inference_mode():
            for episode in range(self.episodes):
                state, _ = env.reset(seed=self.seed + episode)
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done:
                    batch_state = state.unsqueeze(0).to(self.device).float().div_(255.0)
                    action = policy_network.greedy_actions(batch_state).squeeze(0).cpu()
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = terminated or truncated

                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)

        return EvaluationMetrics(
            mean_episode_return=fmean(episode_returns),
            min_episode_return=min(episode_returns),
            max_episode_return=max(episode_returns),
            mean_episode_length=fmean(episode_lengths),
        )
