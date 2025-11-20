import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.models.transition import TransitionModel
except ImportError:
    TransitionModel = None

class TestPhase3Transition(unittest.TestCase):
    def setUp(self):
        if TransitionModel is None:
            self.skipTest("Transition module not yet implemented")
            
        self.latent_dim = 32
        self.action_dim = 4
        self.hidden_dim = 64
        self.model = TransitionModel(
            latent_dim=self.latent_dim, 
            action_dim=self.action_dim, 
            hidden_dim=self.hidden_dim
        )

    def test_initialization(self):
        self.assertIsInstance(self.model, torch.nn.Module)
        print("\n[Transition] Initialization: OK")

    def test_forward_pass(self):
        batch_size = 8
        
        # Inputs
        z_t = torch.randn(batch_size, self.latent_dim)
        action = torch.randint(0, self.action_dim, (batch_size,))
        # Convert action to one-hot or embedding inside the model usually, 
        # but let's assume the model takes raw indices or one-hot. 
        # We'll pass indices for now.
        
        # Initial hidden state (optional, usually None for first step)
        hidden = None
        
        # Forward
        next_z_dist, next_hidden = self.model(z_t, action, hidden)
        
        # Check outputs
        # next_z_dist should be a tuple (mu, logvar) or a distribution object
        # Here we assume it returns (mu, logvar) for the next state distribution
        mu, logvar = next_z_dist
        
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
        
        # Check hidden state shape (num_layers, batch, hidden_dim) for LSTM/GRU
        # Assuming 1 layer GRU/LSTM
        self.assertEqual(next_hidden.shape[1], batch_size)
        self.assertEqual(next_hidden.shape[2], self.hidden_dim)
        
        print("[Transition] Forward pass: OK")

if __name__ == '__main__':
    unittest.main()
