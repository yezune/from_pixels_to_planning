import gymnasium as gym
import numpy as np
import torch
from src.utils.preprocessing import preprocess_observation

class ActiveInferenceEnv:
    def __init__(self, env_id, render_mode='rgb_array', target_size=(64, 64), grayscale=True, device='cpu'):
        """
        Wrapper for Gymnasium environments to be used with Active Inference agents.
        
        Args:
            env_id (str): The gymnasium environment ID.
            render_mode (str): Render mode ('rgb_array' or 'human').
            target_size (tuple): Target size for observation resizing.
            grayscale (bool): Whether to convert observations to grayscale.
            device (str): Device to put tensors on.
        """
        self.env = gym.make(env_id, render_mode=render_mode)
        self.target_size = target_size
        self.grayscale = grayscale
        self.device = device
        
        # Action space information
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None):
        """
        Resets the environment and returns the preprocessed observation.
        """
        obs, info = self.env.reset(seed=seed)
        
        # Determine if we need to render to get pixels
        # 1. MiniGrid special case
        if hasattr(self.env.spec, 'id') and 'MiniGrid' in self.env.spec.id:
            img = self.env.get_frame()
        # 2. If observation is not image-like (e.g. vector), try to render
        elif len(obs.shape) < 2: 
            img = self.env.render()
        # 3. Default: assume observation is the image
        else:
            img = obs
            
        processed_obs = preprocess_observation(img, self.target_size, self.grayscale)
            
        return processed_obs.to(self.device), info

    def step(self, action):
        """
        Takes a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            obs (torch.Tensor): Preprocessed observation.
            reward (float): Reward.
            terminated (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode is truncated.
            info (dict): Additional info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Determine if we need to render to get pixels
        if hasattr(self.env.spec, 'id') and 'MiniGrid' in self.env.spec.id:
            img = self.env.get_frame()
        elif len(obs.shape) < 2:
            img = self.env.render()
        else:
            img = obs
            
        processed_obs = preprocess_observation(img, self.target_size, self.grayscale)
            
        return processed_obs.to(self.device), reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def sample_action(self):
        return self.env.action_space.sample()
