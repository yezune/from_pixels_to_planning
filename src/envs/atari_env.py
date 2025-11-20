import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from gymnasium.wrappers import ResizeObservation
import ale_py

gym.register_envs(ale_py)

class AtariPixelEnv(gym.Wrapper):
    def __init__(self, env_id, image_size=64, device='cpu'):
        # Initialize environment
        # We use 'rgb_array' render mode to get pixel observations if needed, 
        # but standard Atari envs return pixels in step() anyway.
        try:
            env = gym.make(env_id, render_mode='rgb_array')
        except gym.error.Error:
            # Fallback for older gym versions or if ID is different
            env = gym.make(env_id)
            
        # Resize to target size
        env = ResizeObservation(env, (image_size, image_size))
        
        super().__init__(env)
        self.image_size = image_size
        self.device = device
        
        # Define observation space as (3, H, W) float [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, image_size, image_size), dtype=np.float32
        )
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(), # Converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
        ])

    def _process_obs(self, obs):
        # obs is numpy array (H, W, C) from ResizeObservation
        if isinstance(obs, np.ndarray):
            return self.transform(obs).to(self.device)
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), float(reward), terminated, truncated, info
