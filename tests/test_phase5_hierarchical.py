import unittest
import torch
from unittest.mock import MagicMock
from src.models.hierarchical_agent import HierarchicalAgent

class TestHierarchicalAgent(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.action_dim = 4
        
        # Mock Level 1 Models (Pixel Level)
        self.vae1 = MagicMock()
        self.vae1.encode.return_value = (torch.zeros(1, 32), torch.zeros(1, 32))
        self.vae1.reparameterize.return_value = torch.zeros(1, 32)
        self.vae1.to.return_value = None
        
        self.trans1 = MagicMock()
        # Returns (next_mu, next_logvar), next_hidden
        self.trans1.return_value = ((torch.zeros(1, 32), torch.zeros(1, 32)), torch.zeros(1, 64))
        self.trans1.to.return_value = None

        # Mock Level 2 Models (Path/Abstract Level)
        self.vae2 = MagicMock()
        self.vae2.encode.return_value = (torch.zeros(1, 32), torch.zeros(1, 32))
        self.vae2.reparameterize.return_value = torch.zeros(1, 32)
        self.vae2.to.return_value = None
        
        self.trans2 = MagicMock()
        self.trans2.return_value = ((torch.zeros(1, 32), torch.zeros(1, 32)), torch.zeros(1, 64))
        self.trans2.to.return_value = None

    def test_initialization(self):
        agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=self.action_dim,
            device=self.device
        )
        self.assertIsNotNone(agent)
        self.assertEqual(agent.action_dim, self.action_dim)

    def test_select_action_returns_int(self):
        agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Dummy observation (C, H, W)
        obs = torch.zeros(3, 64, 64)
        action = agent.select_action(obs)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)

    def test_hierarchy_update(self):
        """
        Test if both levels are involved in the process.
        This is a behavioral test.
        """
        agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=self.action_dim,
            device=self.device
        )
        
        obs = torch.zeros(3, 64, 64)
        agent.select_action(obs)
        
        # Level 1 VAE should be called to encode observation
        self.vae1.encode.assert_called()
        
        # Level 1 Transition should be called to predict next state/evaluate actions
        self.trans1.assert_called()
        
        # Level 2 Transition should be called to update high-level plan/prior
        # We expect the agent to consult the higher level
        self.trans2.assert_called()

    def test_goal_directed_behavior(self):
        """
        Test if the agent selects action that minimizes distance to the goal provided by Level 2.
        """
        agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=2,
            device=self.device
        )
        
        # Setup:
        # Current state z1 is 0.
        # Action 0 leads to z1 = [10, 10, ...]
        # Action 1 leads to z1 = [-10, -10, ...]
        # Level 2 sets goal = [-10, -10, ...]
        # Expected: Agent chooses Action 1.
        
        # Mock Level 1 Transition
        def trans1_side_effect(z, action, hidden):
            # action is tensor([a])
            a = action.item()
            if a == 0:
                mu = torch.ones(1, 32) * 10
            else:
                mu = torch.ones(1, 32) * -10
            logvar = torch.zeros(1, 32)
            return (mu, logvar), hidden
            
        self.trans1.side_effect = trans1_side_effect
        
        # Mock Level 2 Transition (Goal)
        # Returns goal = -10
        self.trans2.return_value = ((torch.ones(1, 32) * -10, torch.zeros(1, 32)), None)
        
        obs = torch.zeros(3, 64, 64)
        action = agent.select_action(obs)
        
        self.assertEqual(action, 1)

if __name__ == '__main__':
    unittest.main()
