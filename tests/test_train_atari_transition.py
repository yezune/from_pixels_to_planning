"""
Tests for Atari Transition Model Training.

This test suite validates the training pipeline for learning temporal dynamics
in the latent space: z_{t+1} = f(z_t, a_t)
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.envs.atari_env import AtariPixelEnv


class TestAtariTransitionTraining(unittest.TestCase):
    """Test suite for Atari Transition Model training."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 32
        self.action_dim = 4
        self.device = 'cpu'
        
        # Initialize VAE
        self.vae = VAE(input_shape=(3, 64, 64), latent_dim=self.latent_dim).to(self.device)
        self.vae.eval()
        
        # Initialize Transition Model
        self.transition = TransitionModel(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Initialize environment
        self.env = AtariPixelEnv(env_id='Breakout', device=self.device)
    
    def test_transition_initialization(self):
        """Test Transition Model can be initialized."""
        self.assertIsNotNone(self.transition)
        self.assertEqual(self.transition.latent_dim, self.latent_dim)
        self.assertEqual(self.transition.action_dim, self.action_dim)
    
    def test_collect_transition_data(self):
        """Test data collection for transition learning."""
        # Collect a few transitions
        transitions = []
        obs, _ = self.env.reset()
        
        for _ in range(5):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Encode observations
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(self.device)
                next_obs_batch = next_obs.unsqueeze(0).to(self.device)
                
                _, z_t, _ = self.vae(obs_batch)
                _, z_next, _ = self.vae(next_obs_batch)
            
            transitions.append({
                'z_t': z_t.squeeze(0),
                'action': action,
                'z_next': z_next.squeeze(0)
            })
            
            obs = next_obs
            if terminated or truncated:
                break
        
        # Verify data collection
        self.assertGreater(len(transitions), 0)
        for trans in transitions:
            self.assertEqual(trans['z_t'].shape, (self.latent_dim,))
            self.assertEqual(trans['z_next'].shape, (self.latent_dim,))
            # Action can be int or numpy int64
            self.assertIn(type(trans['action']).__name__, ['int', 'int64', 'int32'])
    
    def test_transition_forward_pass(self):
        """Test Transition Model forward pass."""
        batch_size = 4
        z_t = torch.randn(batch_size, self.latent_dim).to(self.device)
        actions = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        
        # Forward pass
        (z_mu, z_logvar), hidden = self.transition(z_t, actions)
        
        # Check output shapes
        self.assertEqual(z_mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(z_logvar.shape, (batch_size, self.latent_dim))
        self.assertEqual(hidden.shape, (1, batch_size, self.transition.hidden_dim))
    
    def test_transition_loss_computation(self):
        """Test transition loss can be computed."""
        batch_size = 4
        z_t = torch.randn(batch_size, self.latent_dim).to(self.device)
        actions = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        z_next = torch.randn(batch_size, self.latent_dim).to(self.device)
        
        # Predict
        (z_mu, z_logvar), _ = self.transition(z_t, actions)
        
        # Compute MSE loss (use mean prediction)
        loss = nn.functional.mse_loss(z_mu, z_next)
        
        # Verify loss
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_transition_training_step(self):
        """Test single training step updates parameters."""
        # Get initial parameters
        initial_params = [p.clone() for p in self.transition.parameters()]
        
        # Create batch
        batch_size = 4
        z_t = torch.randn(batch_size, self.latent_dim).to(self.device)
        actions = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        z_next = torch.randn(batch_size, self.latent_dim).to(self.device)
        
        # Training step
        optimizer = optim.Adam(self.transition.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        (z_mu, z_logvar), _ = self.transition(z_t, actions)
        loss = nn.functional.mse_loss(z_mu, z_next)
        loss.backward()
        optimizer.step()
        
        # Check parameters updated
        params_changed = False
        for p_init, p_new in zip(initial_params, self.transition.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        
        self.assertTrue(params_changed, "Parameters should be updated after training step")
    
    def test_transition_save_load(self):
        """Test Transition Model can be saved and loaded."""
        import tempfile
        
        # Train a bit first
        batch_size = 4
        optimizer = optim.Adam(self.transition.parameters(), lr=1e-3)
        
        for _ in range(3):
            z_t = torch.randn(batch_size, self.latent_dim).to(self.device)
            actions = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
            z_next = torch.randn(batch_size, self.latent_dim).to(self.device)
            
            optimizer.zero_grad()
            (z_mu, z_logvar), _ = self.transition(z_t, actions)
            loss = nn.functional.mse_loss(z_mu, z_next)
            loss.backward()
            optimizer.step()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        try:
            torch.save(self.transition.state_dict(), save_path)
            
            # Load into new model
            new_transition = TransitionModel(
                latent_dim=self.latent_dim,
                action_dim=self.action_dim
            ).to(self.device)
            new_transition.load_state_dict(torch.load(save_path, weights_only=True))
            
            # Compare parameters
            for p1, p2 in zip(self.transition.parameters(), new_transition.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_prediction_accuracy_improves(self):
        """Test that prediction accuracy improves with training."""
        # Create fixed test data
        torch.manual_seed(42)
        test_z_t = torch.randn(10, self.latent_dim).to(self.device)
        test_actions = torch.randint(0, self.action_dim, (10,)).to(self.device)
        test_z_next = torch.randn(10, self.latent_dim).to(self.device)
        
        # Initial loss
        with torch.no_grad():
            (z_mu, _), _ = self.transition(test_z_t, test_actions)
            initial_loss = nn.functional.mse_loss(z_mu, test_z_next).item()
        
        # Train for a few steps
        optimizer = optim.Adam(self.transition.parameters(), lr=1e-2)
        for _ in range(50):
            optimizer.zero_grad()
            (z_mu, _), _ = self.transition(test_z_t, test_actions)
            loss = nn.functional.mse_loss(z_mu, test_z_next)
            loss.backward()
            optimizer.step()
        
        # Final loss
        with torch.no_grad():
            (z_mu, _), _ = self.transition(test_z_t, test_actions)
            final_loss = nn.functional.mse_loss(z_mu, test_z_next).item()
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss * 0.5, 
                       f"Loss should decrease significantly (initial: {initial_loss:.4f}, final: {final_loss:.4f})")


if __name__ == '__main__':
    unittest.main()
