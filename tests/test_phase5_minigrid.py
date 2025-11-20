import unittest
import numpy as np
import torch
import gymnasium as gym
from src.envs.minigrid_env import PixelMiniGridEnv
from src.models.vae import VAE

class TestPixelMiniGridEnv(unittest.TestCase):
    def setUp(self):
        # We'll use a simple empty room for testing
        self.env_id = "MiniGrid-Empty-5x5-v0"
        
    def test_initialization(self):
        env = PixelMiniGridEnv(self.env_id, render_mode='rgb_array')
        self.assertIsNotNone(env)
        env.close()

    def test_observation_shape(self):
        env = PixelMiniGridEnv(self.env_id, render_mode='rgb_array', image_size=64)
        obs, info = env.reset()
        
        # Check type
        self.assertTrue(isinstance(obs, torch.Tensor), "Observation should be a torch Tensor")
        
        # Check shape (C, H, W)
        self.assertEqual(obs.shape, (3, 64, 64))
        
        # Check value range [0, 1]
        self.assertTrue(obs.min() >= 0.0)
        self.assertTrue(obs.max() <= 1.0)
        env.close()

    def test_step(self):
        env = PixelMiniGridEnv(self.env_id, render_mode='rgb_array', image_size=64)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (3, 64, 64))
        env.close()

    def test_vae_integration(self):
        """
        Test if the observation from the environment can be processed by the VAE.
        """
        # Initialize VAE
        vae = VAE(input_shape=(3, 64, 64), latent_dim=32)
        
        # Get observation
        env = PixelMiniGridEnv(self.env_id, render_mode='rgb_array', image_size=64)
        obs, _ = env.reset()
        
        # Add batch dimension
        obs_batch = obs.unsqueeze(0)
        
        # Pass through VAE
        recon, mu, logvar = vae(obs_batch)
        
        self.assertEqual(recon.shape, (1, 3, 64, 64))
        self.assertEqual(mu.shape, (1, 32))
        env.close()

if __name__ == '__main__':
    unittest.main()
