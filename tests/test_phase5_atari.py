import unittest
import torch
import numpy as np
from src.envs.atari_env import AtariPixelEnv

class TestAtariEnv(unittest.TestCase):
    def setUp(self):
        # Use a standard Atari game
        self.env_id = "ALE/Breakout-v5" 
        
    def test_initialization(self):
        try:
            env = AtariPixelEnv(self.env_id, image_size=64)
            self.assertIsNotNone(env)
            env.close()
        except Exception as e:
            self.fail(f"Initialization failed: {e}")

    def test_observation_shape(self):
        env = AtariPixelEnv(self.env_id, image_size=64)
        obs, _ = env.reset()
        
        # Check type
        self.assertTrue(isinstance(obs, torch.Tensor), "Observation should be a torch Tensor")
        
        # Check shape (C, H, W) - Expecting RGB (3 channels)
        self.assertEqual(obs.shape, (3, 64, 64))
        
        # Check range [0, 1]
        self.assertTrue(obs.min() >= 0.0)
        self.assertTrue(obs.max() <= 1.0)
        env.close()

    def test_step(self):
        env = AtariPixelEnv(self.env_id, image_size=64)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (3, 64, 64))
        self.assertIsInstance(reward, float)
        env.close()

if __name__ == '__main__':
    unittest.main()
