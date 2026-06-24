from typing import cast

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from torch import Tensor

type Observation = Tensor
type Action = Tensor
type AtariEnv = gym.Env[Observation, Action]

gym.register_envs(ale_py)


def make_atari_env(env_id: str) -> AtariEnv:
    env = gym.make(env_id, frameskip=1)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise TypeError(f"{env_id!r} must have a discrete action space")

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,  # We store raw uint8 in replay buffer
    )
    env = FrameStackObservation(env, stack_size=4)
    env = NumpyToTorch(env, device="cpu")

    # Wrapper `AtariPreprocessing` and `FrameStackObservation` erase the concrete generic parameters
    return cast(AtariEnv, env)
