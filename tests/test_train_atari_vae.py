"""
Tests for Atari VAE training pipeline.
"""

import unittest
import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vae import VAE
from src.envs.atari_env import AtariPixelEnv


class TestAtariVAETraining(unittest.TestCase):
    """Test Atari VAE training pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = 'cpu'
        self.env = AtariPixelEnv("BreakoutNoFrameskip-v4", image_size=64)
        self.latent_dim = 32
        self.vae = VAE(input_shape=(3, 64, 64), latent_dim=self.latent_dim).to(self.device)
    
    def test_vae_initialization(self):
        """Test VAE can be initialized for Atari."""
        self.assertIsNotNone(self.vae)
        self.assertEqual(self.vae.latent_dim, self.latent_dim)
        self.assertEqual(self.vae.c, 3)  # RGB channels
    
    def test_collect_atari_data(self):
        """Test data collection from Atari environment."""
        num_steps = 10
        observations = []
        
        obs, _ = self.env.reset()
        for _ in range(num_steps):
            observations.append(obs)
            action = self.env.action_space.sample()
            obs, _, done, truncated, _ = self.env.step(action)
            
            if done or truncated:
                obs, _ = self.env.reset()
        
        self.assertEqual(len(observations), num_steps)
        self.assertEqual(observations[0].shape, (3, 64, 64))
    
    def test_vae_forward_pass(self):
        """Test VAE forward pass with Atari observation."""
        obs, _ = self.env.reset()
        obs_batch = obs.unsqueeze(0).to(self.device)  # (1, 3, 64, 64)
        
        with torch.no_grad():
            recon, mu, logvar = self.vae(obs_batch)
        
        self.assertEqual(recon.shape, obs_batch.shape)
        self.assertEqual(mu.shape, (1, self.latent_dim))
        self.assertEqual(logvar.shape, (1, self.latent_dim))
    
    def test_vae_loss_computation(self):
        """Test VAE loss can be computed."""
        obs, _ = self.env.reset()
        obs_batch = obs.unsqueeze(0).to(self.device)
        
        recon, mu, logvar = self.vae(obs_batch)
        loss_dict = self.vae.loss_function(recon, obs_batch, mu, logvar)
        
        self.assertIn('loss', loss_dict)
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('kld_loss', loss_dict)
        self.assertIsInstance(loss_dict['loss'].item(), float)
    
    def test_vae_training_step(self):
        """Test single training step updates parameters."""
        obs, _ = self.env.reset()
        obs_batch = obs.unsqueeze(0).to(self.device)
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        # Get initial parameter
        initial_param = next(self.vae.parameters()).clone()
        
        # Training step
        self.vae.train()
        optimizer.zero_grad()
        recon, mu, logvar = self.vae(obs_batch)
        loss_dict = self.vae.loss_function(recon, obs_batch, mu, logvar)
        loss_dict['loss'].backward()
        optimizer.step()
        
        # Check parameter changed
        updated_param = next(self.vae.parameters())
        self.assertFalse(torch.equal(initial_param, updated_param))
    
    def test_vae_save_load(self):
        """Test VAE can be saved and loaded."""
        import tempfile
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        try:
            # Train a bit first to have non-random weights
            obs, _ = self.env.reset()
            obs_batch = obs.unsqueeze(0).to(self.device)
            optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
            
            for _ in range(3):
                optimizer.zero_grad()
                recon, mu, logvar = self.vae(obs_batch)
                loss_dict = self.vae.loss_function(recon, obs_batch, mu, logvar)
                loss_dict['loss'].backward()
                optimizer.step()
            
            # Save
            torch.save(self.vae.state_dict(), save_path)
            
            # Load into new VAE
            new_vae = VAE(input_shape=(3, 64, 64), latent_dim=self.latent_dim).to(self.device)
            new_vae.load_state_dict(torch.load(save_path, weights_only=True))
            
            # Compare parameters
            for p1, p2 in zip(self.vae.parameters(), new_vae.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == '__main__':
    unittest.main()
