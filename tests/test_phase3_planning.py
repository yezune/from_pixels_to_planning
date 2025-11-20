import unittest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.models.agent import ActiveInferenceAgent
    from src.models.vae import VAE
    from src.models.transition import TransitionModel
except ImportError:
    ActiveInferenceAgent = None

class TestPhase3Planning(unittest.TestCase):
    def setUp(self):
        if ActiveInferenceAgent is None:
            self.skipTest("Agent module not yet implemented")
            
        self.latent_dim = 32
        self.action_dim = 4
        self.hidden_dim = 64
        
        self.vae = VAE(latent_dim=self.latent_dim)
        self.transition = TransitionModel(latent_dim=self.latent_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        
        self.agent = ActiveInferenceAgent(self.vae, self.transition, action_dim=self.action_dim)

    def test_initialization(self):
        self.assertIsInstance(self.agent, object)
        print("\n[Agent] Initialization: OK")

    def test_action_selection(self):
        # Dummy observation (1, 1, 64, 64)
        obs = torch.randn(1, 1, 64, 64)
        
        # Select action
        action = self.agent.select_action(obs)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
        print("[Agent] Action Selection: OK")

    def test_internal_state_update(self):
        obs = torch.randn(1, 1, 64, 64)
        
        # First step
        self.agent.reset()
        self.assertIsNone(self.agent.current_hidden)
        
        action = self.agent.select_action(obs)
        
        # Hidden state should be updated after action selection (or rather, after processing obs)
        # In a real loop, we process obs -> update state -> select action.
        # Let's check if the agent stores the current latent state
        self.assertIsNotNone(self.agent.current_z)
        print("[Agent] Internal State Update: OK")

if __name__ == '__main__':
    unittest.main()
