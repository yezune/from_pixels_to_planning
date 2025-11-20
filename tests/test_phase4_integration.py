import unittest
import torch
import sys
import os
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.models.agent import ActiveInferenceAgent
    from src.models.vae import VAE
    from src.models.transition import TransitionModel
    from src.envs.env_wrapper import ActiveInferenceEnv
except ImportError:
    ActiveInferenceAgent = None

class TestPhase4Integration(unittest.TestCase):
    def setUp(self):
        if ActiveInferenceAgent is None:
            self.skipTest("Modules not implemented")
            
        # Setup Mock Env to avoid pygame dependency issues in CI/Test environment
        self.mock_env_patcher = patch('src.envs.env_wrapper.gym.make')
        self.mock_make = self.mock_env_patcher.start()
        
        self.mock_env = MagicMock()
        self.mock_env.spec.id = 'MockEnv-v0'
        self.mock_env.reset.return_value = (np.zeros(4), {})
        self.mock_env.step.return_value = (np.zeros(4), 1.0, False, False, {})
        self.mock_env.render.return_value = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.mock_env.action_space.n = 2
        self.mock_env.action_space.sample.return_value = 0
        
        self.mock_make.return_value = self.mock_env
        
        # Use 64x64 to match VAE default architecture stride requirements better
        self.target_size = (64, 64)
        self.env = ActiveInferenceEnv(env_id='MockEnv-v0', target_size=self.target_size, grayscale=True)
        
        self.latent_dim = 16
        self.action_dim = self.env.action_space.n
        self.hidden_dim = 32
        
        self.vae = VAE(input_shape=(1, 64, 64), latent_dim=self.latent_dim)
        self.transition = TransitionModel(latent_dim=self.latent_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        
        self.agent = ActiveInferenceAgent(self.vae, self.transition, action_dim=self.action_dim)
        
        self.output_dir = os.path.join(os.path.dirname(__file__), 'integration_test_output')
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.env.close()
        self.mock_env_patcher.stop()
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_full_loop_execution(self):
        print("\n[Integration] Testing Full Loop (Env + Agent)...")
        
        # 1. Reset Env
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (1, 64, 64))
        
        # 2. Run a few steps
        max_steps = 5
        total_reward = 0
        
        for t in range(max_steps):
            # Agent selects action
            action = self.agent.select_action(obs)
            
            # Env steps
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check shapes
            self.assertEqual(next_obs.shape, (1, 64, 64))
            
            obs = next_obs
            total_reward += reward
            
            if terminated or truncated:
                break
                
        print(f"  - Ran {t+1} steps successfully. Total Reward: {total_reward}")
        self.assertTrue(total_reward >= 0) 

if __name__ == '__main__':
    unittest.main()
