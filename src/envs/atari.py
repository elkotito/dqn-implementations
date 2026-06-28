from pathlib import Path
from typing import cast

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from torch import Tensor

type Observation = Tensor
type Action = Tensor
type AtariEnv = gym.Env[Observation, Action]

gym.register_envs(ale_py)


def make_atari_env(env_id: str, *, video_dir: Path | None = None, video_name_prefix: str = "policy") -> AtariEnv:
    render_mode = "rgb_array" if video_dir is not None else None
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise TypeError(f"{env_id!r} must have a discrete action space")

    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_folder=str(video_dir), episode_trigger=lambda _: True, name_prefix=video_name_prefix)

    # We store raw uint8 in replay buffer hence we don't scale obs
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)
    env = NumpyToTorch(env, device="cpu")

    # Wrapper `AtariPreprocessing` and `FrameStackObservation` erase the concrete generic parameters
    return cast(AtariEnv, env)
