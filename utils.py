import torch
import numpy as np
import os
import gym
from typing import Optional, Tuple, Union
import json
from imageio import mimsave
import random

from modules import Actor
from dataset import ReplayBuffer


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def seed_everything(seed: int,
                    env: Optional[gym.Env] = None,
                    use_deterministic_algos: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)

def rollout(batch_size,
            horizon,
            transition,
            policy,
            env_buffer,
            model_buffer,
            exploration_noise=0.1,
            max_action=1):
    raise NotImplementedError()


def parse_json_dataset(filename: str) -> Tuple[int, int, float]:
    max_action = 1.0

    if not filename.endswith('.json'):
        filename = filename + '.json'

    filename_ = os.path.join("json_datasets", filename)
    with open(filename_) as f:
        obj = json.load(f)
    
    states = np.array(obj["observations"])
    actions = np.array(obj["actions"])

    return states.shape[1], actions.shape[1], max_action


def wrap_env(env: gym.Env,
             state_mean: Union[np.ndarray, float] = 0.0,
             state_std: Union[np.ndarray, float] = 1.0,
             reward_scale: float = 1.0) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


@torch.no_grad()
def eval_actor(env: gym.Env,
               actor: Actor,
               device: str,
               num_episodes: int,
               seed: int) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    
    for _ in range(num_episodes):

        state, done = env.reset(), False
        episode_reward = 0.0

        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


class VideoRecorder:
    def __init__(self, dir_name, height=512, width=512, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env:gym.Env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            mimsave(path, self.frames, fps=self.fps)
