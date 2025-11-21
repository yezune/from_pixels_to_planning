import torch
import numpy as np
from src.utils.buffer import ReplayBuffer

class BaseTrainer:
    def __init__(self, env, agent, buffer_size=10000, batch_size=64, lr=1e-3, device='cpu'):
        self.env = env
        self.agent = agent
        self.device = device
        self.batch_size = batch_size
        
        # Determine observation shape
        if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
            obs_shape = env.observation_space.shape
        else:
            # Fallback default
            obs_shape = (1, 64, 64)

        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_shape=obs_shape,
            action_dim=agent.action_dim,
            device=device
        )
        
        # Fix buffer obs shape if env wrapper is used (e.g. for resizing/grayscale)
        if hasattr(env, 'target_size'):
             # (C, H, W)
             c = 1 if getattr(env, 'grayscale', False) else 3
             self.buffer.obs_shape = (c, *env.target_size)
             # Re-initialize buffers with correct shape
             self.buffer.obs_buf = np.zeros((buffer_size, c, *env.target_size), dtype=np.float32)
             self.buffer.next_obs_buf = np.zeros((buffer_size, c, *env.target_size), dtype=np.float32)

    def collect_data(self, num_steps):
        """
        Interacts with the environment and stores data in the buffer.
        """
        obs, _ = self.env.reset()
        self.agent.reset()
        
        for _ in range(num_steps):
            # Select action (using current policy/planning)
            action = self.agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store in buffer
            self.buffer.add(obs, action, reward, next_obs, float(terminated))
            
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.agent.reset()
