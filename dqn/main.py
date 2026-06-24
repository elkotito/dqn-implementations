from typing import Annotated

import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field,
from torch import Tensor

from dqn.buffers.replay_buffer import ReplayBuffer, Transition
from dqn.network.dqn import DQN

total_steps = 10000
warmup_steps = 1000
eps_greedy = 0.1

batch_size = 32
device = "cpu"
replay_buffer = ReplayBuffer[Float[Tensor, ""]](10000)

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    eps_greedy: Probability


def train_step(batch: list[Transition[Tensor]]):
    pass

type Observation = NDArray[np.uint8]
type Action = np.int64
type AtariEnv = gym.Env[Observation, Action]

def make_env() -> AtariEnv:
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)
    env = NumpyToTorch(env, device=device)

    return cast(AtariEnv, env)

# Buffer Replay, Target Network, Reward Clipping
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
    target_network = policy_network.build_target_network()
    optimizer = torch.optim.AdamW(policy_network.parameters())

    state, _ = env.reset()
    for _ in range(total_steps):
        actions = policy_network.sample_actions(state, eps_greedy=eps_greedy)

        next_state, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        replay_buffer.add(Transition(state, float(reward), next_state, done))
        if done:
            state, _ = env.reset()

        state = next_state

        if len(replay_buffer) >= warmup_steps:
            batch_transitions = replay_buffer.sample(batch_size)
            train_step(batch_transitions)


if __name__ == "__main__":
    main()
