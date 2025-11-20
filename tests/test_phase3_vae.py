import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import will fail initially, which is expected in TDD
try:
    from src.models.vae import VAE
except ImportError:
    VAE = None

class TestPhase3VAE(unittest.TestCase):
    def setUp(self):
        if VAE is None:
            self.skipTest("VAE module not yet implemented")
        
        self.input_shape = (1, 64, 64) # C, H, W
        self.latent_dim = 32
        self.model = VAE(input_shape=self.input_shape, latent_dim=self.latent_dim)

    def test_vae_initialization(self):
        self.assertIsInstance(self.model, torch.nn.Module)
        print("\n[VAE] Initialization: OK")

    def test_forward_pass_shape(self):
        batch_size = 8
        dummy_input = torch.randn(batch_size, *self.input_shape)
        
        recon, mu, logvar = self.model(dummy_input)
        
        # Check reconstruction shape
        self.assertEqual(recon.shape, dummy_input.shape)
        
        # Check latent shapes
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
        print("[VAE] Forward pass shapes: OK")

    def test_loss_function(self):
        batch_size = 4
        dummy_input = torch.randn(batch_size, *self.input_shape)
        recon, mu, logvar = self.model(dummy_input)
        
        # Simple loss check
        loss_dict = self.model.loss_function(recon, dummy_input, mu, logvar)
        
        self.assertIn('loss', loss_dict)
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('kld_loss', loss_dict)
        self.assertTrue(loss_dict['loss'] > 0)
        print("[VAE] Loss function: OK")

if __name__ == '__main__':
    unittest.main()
