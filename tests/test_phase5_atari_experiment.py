import unittest
import torch
import gymnasium as gym
from src.envs.atari_env import AtariPixelEnv
from src.models.hierarchical_agent import HierarchicalAgent
from src.models.vae import VAE
from src.models.mlp_vae import MlpVAE
from src.models.transition import TransitionModel
from src.hierarchical_trainer import HierarchicalTrainer
import os

class TestAtariExperiment(unittest.TestCase):
    def setUp(self):
        self.env_id = "BreakoutNoFrameskip-v4"
        self.image_size = 64
        self.env = AtariPixelEnv(self.env_id, image_size=self.image_size)
        
        # Model parameters
        self.input_dim = (3, 64, 64)
        self.action_dim = self.env.action_space.n
        self.latent_dim1 = 64
        self.latent_dim2 = 64 # Must match latent_dim1 for direct goal comparison
        self.hidden_dim = 128
        
        # Level 1 Models
        self.vae1 = VAE(input_shape=self.input_dim, latent_dim=self.latent_dim1)
        self.trans1 = TransitionModel(latent_dim=self.latent_dim1, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        
        # Level 2 Models
        self.vae2 = MlpVAE(input_dim=self.latent_dim1, latent_dim=self.latent_dim2, hidden_dim=self.hidden_dim)
        self.trans2 = TransitionModel(latent_dim=self.latent_dim2, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        
        self.agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=self.action_dim
        )
        
        self.trainer = HierarchicalTrainer(
            env=self.env,
            agent=self.agent,
            buffer_size=1000,
            batch_size=4,
            lr=1e-4
        )

    def test_agent_components(self):
        """Test if the agent components are correctly initialized."""
        self.assertIsInstance(self.agent.vae1, VAE)
        self.assertIsInstance(self.agent.trans1, TransitionModel)
        self.assertIsInstance(self.agent.vae2, MlpVAE)
        self.assertIsInstance(self.agent.trans2, TransitionModel)

    def test_training_loop(self):
        """Test the training loop (collect data and train step)."""
        # Collect some data
        self.trainer.collect_data(num_steps=10)
        self.assertGreaterEqual(self.trainer.buffer.size, 10)
        
        # Perform one update
        losses = self.trainer.train_step()
        
        self.assertIn('vae1_loss', losses)
        self.assertIn('trans1_loss', losses)
        self.assertIn('vae2_loss', losses)
        self.assertIn('trans2_loss', losses)
        
        self.assertIsInstance(losses['vae1_loss'], float)

if __name__ == '__main__':
    unittest.main()
