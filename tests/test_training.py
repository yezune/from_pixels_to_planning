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
    from src.trainer import ActiveInferenceTrainer
    from src.models.agent import ActiveInferenceAgent
    from src.models.vae import VAE
    from src.models.transition import TransitionModel
    from src.envs.env_wrapper import ActiveInferenceEnv
except ImportError:
    ActiveInferenceTrainer = None

class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        if ActiveInferenceTrainer is None:
            self.skipTest("Trainer module not yet implemented")
            
        # Setup Mock Env
        self.mock_env_patcher = patch('src.envs.env_wrapper.gym.make')
        self.mock_make = self.mock_env_patcher.start()
        
        self.mock_env = MagicMock()
        self.mock_env.spec.id = 'MockEnv-v0'
        self.mock_env.reset.return_value = (np.zeros(4), {})
        self.mock_env.step.return_value = (np.zeros(4), 1.0, False, False, {})
        # Return random image 64x64x3
        self.mock_env.render.return_value = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.mock_env.action_space.n = 2
        self.mock_env.action_space.sample.return_value = 0
        
        self.mock_make.return_value = self.mock_env
        
        self.target_size = (64, 64)
        self.env = ActiveInferenceEnv(env_id='MockEnv-v0', target_size=self.target_size, grayscale=True)
        
        self.latent_dim = 16
        self.action_dim = self.env.action_space.n
        self.hidden_dim = 32
        
        self.vae = VAE(input_shape=(1, 64, 64), latent_dim=self.latent_dim)
        self.transition = TransitionModel(latent_dim=self.latent_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.agent = ActiveInferenceAgent(self.vae, self.transition, action_dim=self.action_dim)
        
        self.trainer = ActiveInferenceTrainer(
            env=self.env,
            agent=self.agent,
            buffer_size=100,
            batch_size=4,
            lr=1e-3
        )
        
        self.output_dir = os.path.join(os.path.dirname(__file__), 'training_test_output')
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.env.close()
        self.mock_env_patcher.stop()
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_collect_data(self):
        print("\n[Training] Testing Data Collection...")
        steps = 10
        self.trainer.collect_data(num_steps=steps)
        
        self.assertEqual(len(self.trainer.buffer), steps)
        print(f"  - Collected {len(self.trainer.buffer)} steps.")

    def test_train_step(self):
        print("\n[Training] Testing Training Step...")
        # Pre-fill buffer
        self.trainer.collect_data(num_steps=10)
        
        initial_vae_loss = self.trainer.train_vae(epochs=1)
        initial_trans_loss = self.trainer.train_transition(epochs=1)
        
        self.assertIsInstance(initial_vae_loss, float)
        self.assertIsInstance(initial_trans_loss, float)
        print(f"  - VAE Loss: {initial_vae_loss}, Transition Loss: {initial_trans_loss}")

if __name__ == '__main__':
    unittest.main()
