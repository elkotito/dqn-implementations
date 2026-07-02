import random
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.algorithms.dueling_dqn.config import DuelingDQNConfig
from src.buffers.replay_buffer import NaiveReplayBuffer, Transition, Transitions
from src.checkpoints.policy_checkpoint_store import PolicyCheckpointStore
from src.envs.atari import make_atari_env
from src.evaluation.greedy_policy_evaluator import GreedyPolicyEvaluator
from src.loggers.wandb import WandbLogger
from src.metrics.metrics_tracker import EpisodeMetrics, GradientUpdateMetrics, MetricsTracker
from src.network.dueling_dqn import DuelingDQN
from src.schedules.linear_scheduler import LinearScheduler


class DuelingDQNTrainer:
    def __init__(self, config: DuelingDQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.env = make_atari_env(config.env_id)
        self.eval_env = make_atari_env(config.env_id)
        action_space = cast(gym.spaces.Discrete, self.env.action_space)

        self.replay_buffer = NaiveReplayBuffer(config.replay_capacity)
        self.policy_network = DuelingDQN(int(action_space.n), config.device)
        self.target_network = self.policy_network.build_target_network()
        self.epsilon_scheduler = LinearScheduler(
            warmup_steps=config.replay_warmup_steps,
            start=config.epsilon_start,
            end=config.epsilon_end,
            num_steps=config.epsilon_decay_steps,
        )

        # A larger epsilon limits Adam updates when the second-moment estimate is very small, reducing abrupt Q-value changes that can destabilize DQN training.
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
        self.loss_fn = nn.SmoothL1Loss()

        self.evaluator = GreedyPolicyEvaluator(episodes=config.eval_episodes, seed=config.seed, device=self.device)
        self.checkpoints = PolicyCheckpointStore(policy_dir=config.policy_dir, env_id=config.env_id)
        self.metrics = MetricsTracker(episode_window=config.episode_window, gradient_window=config.gradient_window)
        self.logger = WandbLogger(project=config.wandb_project, config=config.model_dump(mode="json"), mode=config.wandb_mode)

    def _gradient_update(self, transitions: Transitions) -> GradientUpdateMetrics:
        states, actions, rewards, next_states, dones = transitions

        # Transferring uint8 to GPU is 4x faster than floats
        states = states.to(self.device).float().div_(255.0)
        next_states = next_states.to(self.device).float().div_(255.0)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        predicted_q_values = self.policy_network(states).gather(index=actions[:, None], dim=1).squeeze(1)
        with torch.no_grad():
            max_q_values, _ = self.target_network(next_states).max(dim=1)
            target_q_values = rewards + self.config.gamma * (~dones).float() * max_q_values
            abs_td_error = (target_q_values - predicted_q_values).abs()  # Debugging only

        loss = self.loss_fn(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        return GradientUpdateMetrics(
            loss=loss.detach().cpu().item(),
            gradient_norm=gradient_norm.detach().cpu().item(),
            predicted_q=predicted_q_values.detach().mean().cpu().item(),
            predicted_q_max=predicted_q_values.detach().max().cpu().item(),
            target_q=target_q_values.detach().mean().cpu().item(),
            target_q_max=target_q_values.detach().max().cpu().item(),
            abs_td_error=abs_td_error.detach().mean().cpu().item(),
        )

    def _train(self) -> None:
        episode_return = 0
        episode_length = 0

        state, _ = self.env.reset(seed=self.config.seed)
        for step in range(1, self.config.total_steps + 1):
            epsilon = self.epsilon_scheduler(step)
            episode_length += 1

            # Add batch dimension since `sample_actions` expects a batch of states, but we use a single env
            batch_state = state.unsqueeze(0).to(self.device).float().div_(255.0)
            batch_actions = self.policy_network.eps_greedy_actions(batch_state, epsilon=epsilon)
            action = batch_actions.squeeze(0).cpu()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

            # We use raw reward to report `episode_return`, but rewards in `ReplayBuffer` are used for training hence clipping there
            clipped_reward = np.clip(float(reward), -self.config.reward_clip, self.config.reward_clip)
            # Only true terminations stop bootstrapping; time-limit truncations do not.
            transition = Transition(state=state, action=int(action.item()), reward=clipped_reward, next_state=next_state, done=terminated)
            self.replay_buffer.add(transition)

            if done:
                state, _ = self.env.reset()
                episode_metrics = EpisodeMetrics(episode_return=episode_return, episode_length=episode_length)
                self.metrics.record_episode(episode_metrics)
                episode_return = 0
                episode_length = 0
            else:
                state = next_state

            if len(self.replay_buffer) >= self.config.replay_warmup_steps and step % self.config.gradient_update_interval_steps == 0:
                batch_transitions = self.replay_buffer.sample(self.config.batch_size)
                gradient_update_metrics = self._gradient_update(batch_transitions)
                self.metrics.record_gradient_update(gradient_update_metrics)

            if step % self.config.target_network_sync_interval_steps == 0:
                self.target_network.sync(self.policy_network)

            if step % self.config.train_log_interval_steps == 0 or step == self.config.total_steps:
                metrics = self.metrics.training_snapshot(epsilon=epsilon, replay_buffer_size=len(self.replay_buffer))
                self.logger.log_train(step=step, metrics=metrics)

            if step % self.config.eval_interval_steps == 0 or step == self.config.total_steps:
                evaluation_metrics = self.evaluator.evaluate(env=self.eval_env, policy_network=self.policy_network)
                is_best_evaluation = self.metrics.record_evaluation(evaluation_metrics)
                self.logger.log_eval(step=step, metrics=evaluation_metrics)
                self.checkpoints.save(
                    "policy.pt",
                    policy_network=self.policy_network,
                    step=step,
                    train_mean_episode_return=self.metrics.mean_episode_return,
                    eval_mean_episode_return=evaluation_metrics.mean_episode_return,
                )

                if is_best_evaluation:
                    self.checkpoints.save(
                        "best_policy.pt",
                        policy_network=self.policy_network,
                        step=step,
                        train_mean_episode_return=self.metrics.mean_episode_return,
                        eval_mean_episode_return=evaluation_metrics.mean_episode_return,
                    )

    def train(self) -> None:
        try:
            self._train()
        finally:
            self.env.close()
            self.eval_env.close()
            self.logger.close()
