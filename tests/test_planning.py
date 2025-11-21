import unittest
import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vae import VAE
from src.models.transition import TransitionModel


class TestMCTSPlanner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 16
        self.action_dim = 4
        self.device = 'cpu'
        
        # Create simple models
        self.vae = VAE(input_shape=(1, 32, 32), latent_dim=self.latent_dim)
        self.transition = TransitionModel(
            latent_dim=self.latent_dim, 
            action_dim=self.action_dim, 
            hidden_dim=32
        )
        
    def test_mcts_initialization(self):
        """Test MCTS planner can be initialized."""
        try:
            from src.planning.mcts import MCTSPlanner
            
            planner = MCTSPlanner(
                transition_model=self.transition,
                action_dim=self.action_dim,
                num_simulations=10,
                device=self.device
            )
            
            self.assertIsNotNone(planner)
            self.assertEqual(planner.action_dim, self.action_dim)
            print("\n[MCTS] Initialization: PASS")
        except ImportError:
            self.skipTest("MCTSPlanner not yet implemented")
    
    def test_mcts_rollout(self):
        """Test MCTS can perform rollout from a given state."""
        try:
            from src.planning.mcts import MCTSPlanner
            
            planner = MCTSPlanner(
                transition_model=self.transition,
                action_dim=self.action_dim,
                num_simulations=5,
                device=self.device
            )
            
            # Create initial state
            initial_state = torch.randn(1, self.latent_dim)
            
            # Perform rollout
            best_action = planner.plan(initial_state, depth=3)
            
            self.assertIsInstance(best_action, int)
            self.assertTrue(0 <= best_action < self.action_dim)
            print(f"\n[MCTS] Rollout: PASS (best_action={best_action})")
        except ImportError:
            self.skipTest("MCTSPlanner not yet implemented")
    
    def test_mcts_with_goal(self):
        """Test MCTS can plan towards a goal state."""
        try:
            from src.planning.mcts import MCTSPlanner
            
            planner = MCTSPlanner(
                transition_model=self.transition,
                action_dim=self.action_dim,
                num_simulations=10,
                device=self.device
            )
            
            initial_state = torch.randn(1, self.latent_dim)
            goal_state = torch.randn(1, self.latent_dim)
            
            best_action = planner.plan(initial_state, goal_state=goal_state, depth=3)
            
            self.assertIsInstance(best_action, int)
            print(f"\n[MCTS] Goal-directed planning: PASS (best_action={best_action})")
        except ImportError:
            self.skipTest("MCTSPlanner not yet implemented")


class TestTrajectoryOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 16
        self.action_dim = 4
        self.device = 'cpu'
        
        self.vae = VAE(input_shape=(1, 32, 32), latent_dim=self.latent_dim)
        self.transition = TransitionModel(
            latent_dim=self.latent_dim, 
            action_dim=self.action_dim, 
            hidden_dim=32
        )
    
    def test_trajectory_optimizer_initialization(self):
        """Test trajectory optimizer can be initialized."""
        try:
            from src.planning.trajectory_optimizer import TrajectoryOptimizer
            
            optimizer = TrajectoryOptimizer(
                transition_model=self.transition,
                action_dim=self.action_dim,
                horizon=5,
                device=self.device
            )
            
            self.assertIsNotNone(optimizer)
            self.assertEqual(optimizer.horizon, 5)
            print("\n[TrajectoryOptimizer] Initialization: PASS")
        except ImportError:
            self.skipTest("TrajectoryOptimizer not yet implemented")
    
    def test_trajectory_optimization(self):
        """Test trajectory optimizer can optimize action sequence."""
        try:
            from src.planning.trajectory_optimizer import TrajectoryOptimizer
            
            optimizer = TrajectoryOptimizer(
                transition_model=self.transition,
                action_dim=self.action_dim,
                horizon=5,
                device=self.device
            )
            
            initial_state = torch.randn(1, self.latent_dim)
            goal_state = torch.randn(1, self.latent_dim)
            
            # Optimize trajectory
            action_sequence = optimizer.optimize(
                initial_state=initial_state,
                goal_state=goal_state,
                num_iterations=10
            )
            
            self.assertIsInstance(action_sequence, (list, np.ndarray, torch.Tensor))
            self.assertEqual(len(action_sequence), 5)  # horizon
            print(f"\n[TrajectoryOptimizer] Optimization: PASS")
        except ImportError:
            self.skipTest("TrajectoryOptimizer not yet implemented")
    
    def test_trajectory_optimizer_returns_valid_actions(self):
        """Test that optimized actions are within valid range."""
        try:
            from src.planning.trajectory_optimizer import TrajectoryOptimizer
            
            optimizer = TrajectoryOptimizer(
                transition_model=self.transition,
                action_dim=self.action_dim,
                horizon=3,
                device=self.device
            )
            
            initial_state = torch.randn(1, self.latent_dim)
            goal_state = torch.randn(1, self.latent_dim)
            
            action_sequence = optimizer.optimize(
                initial_state=initial_state,
                goal_state=goal_state,
                num_iterations=5
            )
            
            # Check all actions are valid
            for action in action_sequence:
                self.assertTrue(0 <= action < self.action_dim)
            
            print(f"\n[TrajectoryOptimizer] Valid actions: PASS")
        except ImportError:
            self.skipTest("TrajectoryOptimizer not yet implemented")


if __name__ == '__main__':
    unittest.main()
