import unittest
import torch
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig


class TestMultiLevelAgent(unittest.TestCase):
    """
    Test suite for Multi-level Agent.
    
    Tests the agent's ability to:
    - Perform multi-level state inference
    - Calculate Expected Free Energy (EFE) across all levels
    - Select actions based on hierarchical goals
    - Handle temporal resolution (different update rates per level)
    """
    
    def setUp(self):
        self.device = 'cpu'
        self.action_dim = 4
        self.num_levels = 3
        self.observation_shape = (3, 64, 64)
        
        # Mock RGM
        self.mock_rgm = MagicMock(spec=MultiLevelRGM)
        self.mock_rgm.num_levels = self.num_levels
        self.mock_rgm.device = self.device
        
        # Mock level configs
        self.mock_rgm.levels = [
            MagicMock(latent_dim=64, temporal_resolution=1),   # Level 0
            MagicMock(latent_dim=32, temporal_resolution=4),   # Level 1
            MagicMock(latent_dim=16, temporal_resolution=16),  # Level 2
        ]
    
    def test_agent_initialization(self):
        """Test that MultiLevelAgent initializes correctly."""
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(
            rgm=self.mock_rgm,
            action_dim=self.action_dim,
            device=self.device
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.device, self.device)
        self.assertEqual(len(agent.hidden_states), self.num_levels)
        self.assertEqual(len(agent.current_z), self.num_levels)
    
    def test_reset_clears_internal_state(self):
        """Test that reset() clears all internal states."""
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, self.action_dim, self.device)
        
        # Set some dummy states
        agent.current_z = [torch.randn(1, 64), torch.randn(1, 32), torch.randn(1, 16)]
        agent.hidden_states = [torch.randn(1, 128), None, None]
        
        # Reset
        agent.reset()
        
        # All states should be None
        for z in agent.current_z:
            self.assertIsNone(z)
        for h in agent.hidden_states:
            self.assertIsNone(h)
    
    def test_infer_state_calls_bottom_up_inference(self):
        """Test that infer_state properly calls RGM's bottom_up_inference."""
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, self.action_dim, self.device)
        
        # Mock bottom_up_inference
        mock_z_list = [
            torch.randn(1, 64),
            torch.randn(1, 32),
            torch.randn(1, 16)
        ]
        self.mock_rgm.bottom_up_inference.return_value = {
            'z': mock_z_list,
            'mu': mock_z_list,
            'logvar': [torch.zeros_like(z) for z in mock_z_list]
        }
        
        observation = torch.randn(1, *self.observation_shape)
        z_list = agent.infer_state(observation)
        
        # Should call RGM's bottom_up_inference
        self.mock_rgm.bottom_up_inference.assert_called_once()
        
        # Should return and store latents
        self.assertEqual(len(z_list), self.num_levels)
        self.assertEqual(agent.current_z, mock_z_list)
    
    def test_multi_level_efe_calculation(self):
        """Test that EFE is calculated across all levels."""
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, self.action_dim, self.device)
        
        # Set current states
        current_z = [
            torch.zeros(1, 64),
            torch.zeros(1, 32),
            torch.zeros(1, 16)
        ]
        agent.current_z = current_z
        
        # Mock predict_next_state
        self.mock_rgm.predict_next_state.return_value = (torch.randn(1, 64), None)
        
        # Calculate EFE for action 0
        action = 0
        efe = agent.calculate_multi_level_efe(action, current_z)
        
        # Should return a scalar
        self.assertIsInstance(efe, float)
        
        # RGM's predict_next_state should be called for Level 0 at least
        self.mock_rgm.predict_next_state.assert_called()
    
    def test_goal_directed_behavior_3_levels(self):
        """
        Test goal-directed action selection with 3 levels.
        
        Setup:
        - Level 2 (highest) sets a goal
        - Level 1 provides sub-goal
        - Level 0 selects primitive action to achieve sub-goal
        
        Expected:
        - Agent should select action that minimizes distance to goal hierarchy
        """
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, action_dim=2, device=self.device)
        
        # Mock bottom_up_inference to set current state
        current_z = [
            torch.zeros(1, 64),  # Level 0: at origin
            torch.zeros(1, 32),  # Level 1: at origin
            torch.zeros(1, 16)   # Level 2: at origin
        ]
        self.mock_rgm.bottom_up_inference.return_value = {
            'z': current_z,
            'mu': current_z,
            'logvar': [torch.zeros_like(z) for z in current_z]
        }
        
        # Mock predict_next_state
        # Action 0 -> moves to positive state
        # Action 1 -> moves to negative state
        def mock_predict(level, z_current, action, hidden=None):
            a = action.item() if isinstance(action, torch.Tensor) else action
            if a == 0:
                next_mu = torch.ones_like(z_current) * 10.0
            else:
                next_mu = torch.ones_like(z_current) * -10.0
            return next_mu, None
        
        self.mock_rgm.predict_next_state.side_effect = mock_predict
        
        # Set Level 2 goal to negative state (so Level 1 should also prefer negative)
        current_z[2] = torch.ones(1, 16) * -10.0
        current_z[1] = torch.ones(1, 32) * -5.0
        
        observation = torch.randn(1, *self.observation_shape)
        
        # Agent should choose action 1 (moves to negative state, closer to goal)
        action = agent.select_action(observation)
        
        self.assertEqual(action, 1)
    
    def test_temporal_resolution_awareness(self):
        """
        Test that agent respects temporal resolution when updating levels.
        
        Level 0 (τ=1): updates every step
        Level 1 (τ=4): updates every 4 steps
        Level 2 (τ=16): updates every 16 steps
        """
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, self.action_dim, self.device)
        
        # Mock should_update_level from RGM
        def mock_should_update(level, timestep=None):
            if timestep is None:
                timestep = 0
            tau = [1, 4, 16][level]
            return (timestep % tau) == 0
        
        self.mock_rgm.should_update_level.side_effect = mock_should_update
        
        # At timestep 0, all levels should update
        for level in range(3):
            should_update = agent.should_update_level(level, timestep=0)
            self.assertTrue(should_update)
        
        # At timestep 1, only Level 0 should update
        self.assertTrue(agent.should_update_level(0, timestep=1))
        self.assertFalse(agent.should_update_level(1, timestep=1))
        self.assertFalse(agent.should_update_level(2, timestep=1))
        
        # At timestep 4, Level 0 and 1 should update
        self.assertTrue(agent.should_update_level(0, timestep=4))
        self.assertTrue(agent.should_update_level(1, timestep=4))
        self.assertFalse(agent.should_update_level(2, timestep=4))
    
    def test_action_selection_updates_internal_state(self):
        """Test that select_action properly updates agent's internal state."""
        from src.models.multi_level_agent import MultiLevelAgent
        
        agent = MultiLevelAgent(self.mock_rgm, self.action_dim, self.device)
        
        # Mock bottom_up_inference
        new_z = [torch.randn(1, 64), torch.randn(1, 32), torch.randn(1, 16)]
        self.mock_rgm.bottom_up_inference.return_value = {
            'z': new_z,
            'mu': new_z,
            'logvar': [torch.zeros_like(z) for z in new_z]
        }
        
        # Mock predict_next_state
        self.mock_rgm.predict_next_state.return_value = (torch.randn(1, 64), torch.randn(1, 128))
        
        observation = torch.randn(1, *self.observation_shape)
        
        # Initially, current_z should be None
        self.assertTrue(all(z is None for z in agent.current_z))
        
        # Select action
        action = agent.select_action(observation)
        
        # After selection, current_z should be updated
        self.assertIsNotNone(agent.current_z[0])
        self.assertEqual(agent.current_z, new_z)


if __name__ == '__main__':
    unittest.main()
