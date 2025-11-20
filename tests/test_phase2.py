import unittest
import numpy as np
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.preprocessing import preprocess_observation
from src.envs.env_wrapper import ActiveInferenceEnv

class TestPhase2(unittest.TestCase):
    def test_preprocessing_shape_and_type(self):
        print("\nTesting Preprocessing...")
        # Mock image (H, W, C)
        raw_obs = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test Grayscale
        processed = preprocess_observation(raw_obs, target_size=(64, 64), grayscale=True)
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (1, 64, 64))
        self.assertTrue(processed.max() <= 1.0)
        self.assertTrue(processed.min() >= 0.0)
        print("  - Grayscale preprocessing: OK")

        # Test RGB
        processed_rgb = preprocess_observation(raw_obs, target_size=(32, 32), grayscale=False)
        self.assertEqual(processed_rgb.shape, (3, 32, 32))
        print("  - RGB preprocessing: OK")

    @patch('src.envs.env_wrapper.gym.make')
    def test_env_wrapper_initialization(self, mock_make):
        print("\nTesting Env Wrapper (Mock Env)...")
        
        # Setup Mock Env
        mock_env = MagicMock()
        mock_env.spec.id = 'MockEnv-v0'
        mock_env.reset.return_value = (np.zeros(4), {}) # Vector obs
        mock_env.step.return_value = (np.zeros(4), 1.0, False, False, {})
        mock_env.render.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_env.action_space.sample.return_value = 0
        
        mock_make.return_value = mock_env

        try:
            env = ActiveInferenceEnv(env_id='MockEnv-v0', target_size=(64, 64), grayscale=True)
            
            # Test Reset
            obs, info = env.reset()
            self.assertEqual(obs.shape, (1, 64, 64))
            self.assertIsInstance(obs, torch.Tensor)
            print("  - Reset & Observation shape: OK")
            
            # Test Step
            action = env.sample_action()
            obs, reward, terminated, truncated, info = env.step(action)
            
            self.assertEqual(obs.shape, (1, 64, 64))
            print("  - Step & Observation shape: OK")
            
            env.close()
        except Exception as e:
            self.fail(f"Env wrapper failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
