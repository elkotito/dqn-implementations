import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from jaxtyping import Float
from torch import Tensor

from dqn.buffers.replay_buffer import ReplayBuffer, Transition
from dqn.network.dqn import DQN

total_steps = 10000
warmup_steps = 1000

batch_size = 32
device = "cpu"
replay_buffer = ReplayBuffer[Float[Tensor, ""]](10000)


def train_step(batch: list[Transition[Tensor]]):
    pass


def main():
    env = gym.make("ALE/Pong-v5", frameskip=1)  # TODO: I want to add typing!
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )

    env = FrameStackObservation(env, stack_size=4)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    policy_network = DQN(n_actions).to(device)
    target_network = policy_network.create_target_network()
    optimizer = torch.optim.AdamW(policy_network.parameters())

    state, _ = env.reset()
    for _ in range(total_steps):
        # TODO: Sample epsilon greedy action
        next_state, reward, terminated, truncated, _ = env.step(2)
        done = terminated or truncated
        replay_buffer.add(Transition(state, float(reward), next_state, done))
        state = next_state
        if done:
            state, _ = env.reset()

        if len(replay_buffer) >= warmup_steps:
            batch = replay_buffer.sample(batch_size)
            train_step(batch)


if __name__ == "__main__":
    main()
