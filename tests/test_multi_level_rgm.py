import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.base_vae import BaseVAE
from src.models.transition import TransitionModel


class TestMultiLevelRGM(unittest.TestCase):
    """
    Test suite for Multi-level Hierarchical RGM.
    
    Tests the core functionality of N-level (3+) hierarchical generative models:
    - Bottom-up inference through all levels
    - Top-down prediction from upper to lower levels
    - Hierarchical loss computation
    - Temporal resolution per level
    """
    
    def setUp(self):
        self.device = 'cpu'
        self.num_levels = 3
        
        # Level 0: Pixels (64 latent dim, τ=1)
        # Level 1: Features (32 latent dim, τ=4)
        # Level 2: Paths (16 latent dim, τ=16)
        
        self.level_configs = [
            {'latent_dim': 64, 'temporal_resolution': 1},   # Level 0
            {'latent_dim': 32, 'temporal_resolution': 4},   # Level 1
            {'latent_dim': 16, 'temporal_resolution': 16},  # Level 2
        ]
        
        self.observation_shape = (3, 64, 64)  # (C, H, W)
        self.action_dim = 4
    
    def test_3_level_initialization(self):
        """Test that a 3-level RGM initializes correctly."""
        from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
        
        # Create level configs
        configs = []
        for i, cfg in enumerate(self.level_configs):
            # Mock VAE and Transition for each level
            vae = MagicMock(spec=BaseVAE)
            vae.latent_dim = cfg['latent_dim']
            
            transition = MagicMock(spec=TransitionModel)
            
            level_cfg = LevelConfig(
                latent_dim=cfg['latent_dim'],
                temporal_resolution=cfg['temporal_resolution'],
                vae=vae,
                transition=transition
            )
            configs.append(level_cfg)
        
        # Initialize MultiLevelRGM
        model = MultiLevelRGM(configs, device=self.device)
        
        # Assertions
        self.assertEqual(model.num_levels, 3)
        self.assertEqual(len(model.levels), 3)
        
        # Check each level's config
        for i, level in enumerate(model.levels):
            self.assertEqual(level.latent_dim, self.level_configs[i]['latent_dim'])
            self.assertEqual(level.temporal_resolution, self.level_configs[i]['temporal_resolution'])
    
    def test_bottom_up_inference(self):
        """Test that bottom-up inference passes through all levels."""
        from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
        
        # Setup mocked models
        configs = []
        for i, cfg in enumerate(self.level_configs):
            vae = MagicMock(spec=BaseVAE)
            vae.latent_dim = cfg['latent_dim']
            
            # Mock encode to return (mu, logvar)
            mu = torch.randn(1, cfg['latent_dim'])
            logvar = torch.randn(1, cfg['latent_dim'])
            vae.encode.return_value = (mu, logvar)
            vae.reparameterize.return_value = torch.randn(1, cfg['latent_dim'])
            
            transition = MagicMock(spec=TransitionModel)
            
            level_cfg = LevelConfig(
                latent_dim=cfg['latent_dim'],
                temporal_resolution=cfg['temporal_resolution'],
                vae=vae,
                transition=transition
            )
            configs.append(level_cfg)
        
        model = MultiLevelRGM(configs, device=self.device)
        
        # Create dummy observation
        observation = torch.randn(1, *self.observation_shape)
        
        # Run bottom-up inference
        result = model.bottom_up_inference(observation)
        
        # Assertions
        self.assertIn('z', result)
        self.assertEqual(len(result['z']), 3)  # Should have latents for all 3 levels
        
        # Check that each level's VAE was called
        for i, level in enumerate(model.levels):
            if i == 0:
                # Level 0 encodes the observation
                level.vae.encode.assert_called()
            else:
                # Higher levels encode the lower level's latent
                level.vae.encode.assert_called()
    
    def test_top_down_prediction(self):
        """Test that top-down prediction provides priors for lower levels."""
        from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
        
        # Setup
        configs = []
        for i, cfg in enumerate(self.level_configs):
            vae = MagicMock(spec=BaseVAE)
            vae.latent_dim = cfg['latent_dim']
            vae.decode = MagicMock(return_value=torch.randn(1, cfg['latent_dim']))
            
            transition = MagicMock(spec=TransitionModel)
            
            level_cfg = LevelConfig(
                latent_dim=cfg['latent_dim'],
                temporal_resolution=cfg['temporal_resolution'],
                vae=vae,
                transition=transition
            )
            configs.append(level_cfg)
        
        model = MultiLevelRGM(configs, device=self.device)
        
        # Test top-down prediction from Level 2 -> Level 1
        z_upper = torch.randn(1, self.level_configs[2]['latent_dim'])  # Level 2 latent
        prior = model.top_down_prediction(z_upper, target_level=1)
        
        # Assertions
        self.assertIsNotNone(prior)
        self.assertEqual(prior.shape[1], self.level_configs[1]['latent_dim'])  # Should match Level 1 dim
    
    def test_hierarchical_loss_computation(self):
        """Test that hierarchical loss is computed correctly across all levels."""
        from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
        
        # Setup with real loss computation
        configs = []
        for i, cfg in enumerate(self.level_configs):
            vae = MagicMock(spec=BaseVAE)
            vae.latent_dim = cfg['latent_dim']
            
            # Mock for forward pass
            mu = torch.randn(1, cfg['latent_dim'])
            logvar = torch.randn(1, cfg['latent_dim'])
            z = torch.randn(1, cfg['latent_dim'])
            
            vae.encode.return_value = (mu, logvar)
            vae.reparameterize.return_value = z
            
            if i == 0:
                # Level 0 reconstructs observation
                vae.decode.return_value = torch.randn(1, *self.observation_shape)
            else:
                # Higher levels reconstruct lower level latents
                vae.decode.return_value = torch.randn(1, self.level_configs[i-1]['latent_dim'])
            
            transition = MagicMock(spec=TransitionModel)
            
            level_cfg = LevelConfig(
                latent_dim=cfg['latent_dim'],
                temporal_resolution=cfg['temporal_resolution'],
                vae=vae,
                transition=transition
            )
            configs.append(level_cfg)
        
        model = MultiLevelRGM(configs, device=self.device)
        
        # Create dummy data
        observation = torch.randn(1, *self.observation_shape)
        
        # Compute loss
        loss_dict = model.compute_hierarchical_loss(observation)
        
        # Assertions
        self.assertIn('total_loss', loss_dict)
        self.assertIn('recon_loss_0', loss_dict)  # Level 0 reconstruction
        self.assertIn('kl_loss_0', loss_dict)
        self.assertIn('kl_loss_1', loss_dict)  # Level 1 KL with prior from Level 2
        self.assertIn('kl_loss_2', loss_dict)  # Level 2 KL with uniform prior
        
        # Total loss should be sum of all components
        self.assertIsInstance(loss_dict['total_loss'], torch.Tensor)
        self.assertTrue(loss_dict['total_loss'].item() >= 0)
    
    def test_temporal_resolution_scheduling(self):
        """
        Test that levels with different temporal resolutions are updated correctly.
        
        Level 0 (τ=1): updates every step
        Level 1 (τ=4): updates every 4 steps
        Level 2 (τ=16): updates every 16 steps
        """
        from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
        
        # Setup
        configs = []
        for i, cfg in enumerate(self.level_configs):
            vae = MagicMock(spec=BaseVAE)
            vae.latent_dim = cfg['latent_dim']
            transition = MagicMock(spec=TransitionModel)
            
            level_cfg = LevelConfig(
                latent_dim=cfg['latent_dim'],
                temporal_resolution=cfg['temporal_resolution'],
                vae=vae,
                transition=transition
            )
            configs.append(level_cfg)
        
        model = MultiLevelRGM(configs, device=self.device)
        
        # Simulate 16 timesteps
        for t in range(16):
            should_update = model.should_update_level(level=0, timestep=t)
            self.assertTrue(should_update)  # Level 0 updates every step
        
        # Level 1 should update at t=0, 4, 8, 12
        for t in [0, 4, 8, 12]:
            self.assertTrue(model.should_update_level(level=1, timestep=t))
        
        for t in [1, 2, 3, 5, 6, 7]:
            self.assertFalse(model.should_update_level(level=1, timestep=t))
        
        # Level 2 should only update at t=0
        self.assertTrue(model.should_update_level(level=2, timestep=0))
        for t in range(1, 16):
            self.assertFalse(model.should_update_level(level=2, timestep=t))


if __name__ == '__main__':
    unittest.main()
