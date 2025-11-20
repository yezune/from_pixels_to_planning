import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.models.agent import ActiveInferenceAgent
from src.models.hierarchical_agent import HierarchicalAgent
from src.experiments.comparison_runner import ComparisonRunner

class TestComparisonExperiment(unittest.TestCase):
    def setUp(self):
        # Mock Environment
        self.env = MagicMock()
        self.env.reset.return_value = (torch.randn(3, 64, 64), {})
        self.env.step.return_value = (torch.randn(3, 64, 64), 1.0, False, False, {})
        self.env.action_space.n = 4
        
        # Mock Models for Flat Agent
        self.vae = MagicMock()
        self.vae.encode.return_value = (torch.randn(1, 32), torch.randn(1, 32))
        self.vae.reparameterize.return_value = torch.randn(1, 32)
        self.trans = MagicMock()
        self.trans.return_value = ((torch.randn(1, 32), torch.randn(1, 32)), torch.randn(1, 64))
        
        self.flat_agent = ActiveInferenceAgent(
            vae=self.vae,
            transition_model=self.trans,
            action_dim=4
        )
        
        # Mock Models for Hierarchical Agent
        self.h_vae1 = MagicMock()
        self.h_vae1.encode.return_value = (torch.randn(1, 32), torch.randn(1, 32))
        self.h_vae1.reparameterize.return_value = torch.randn(1, 32)
        self.h_trans1 = MagicMock()
        self.h_trans1.return_value = ((torch.randn(1, 32), torch.randn(1, 32)), torch.randn(1, 64))
        
        self.h_vae2 = MagicMock()
        self.h_vae2.encode.return_value = (torch.randn(1, 32), torch.randn(1, 32))
        self.h_vae2.reparameterize.return_value = torch.randn(1, 32)
        self.h_trans2 = MagicMock()
        self.h_trans2.return_value = ((torch.randn(1, 32), torch.randn(1, 32)), torch.randn(1, 64))
        
        self.hierarchical_agent = HierarchicalAgent(
            level1_models=(self.h_vae1, self.h_trans1),
            level2_models=(self.h_vae2, self.h_trans2),
            action_dim=4
        )
        
        self.runner = ComparisonRunner(
            env=self.env,
            agents={
                'Flat': self.flat_agent,
                'Hierarchical': self.hierarchical_agent
            }
        )

    def test_initialization(self):
        """Test if the runner is initialized correctly."""
        self.assertEqual(len(self.runner.agents), 2)
        self.assertIn('Flat', self.runner.agents)
        self.assertIn('Hierarchical', self.runner.agents)

    def test_run_episode(self):
        """Test running a single episode."""
        # Configure mock to terminate after 5 steps
        self.env.step.side_effect = [
            (torch.randn(3, 64, 64), 1.0, False, False, {}),
            (torch.randn(3, 64, 64), 1.0, False, False, {}),
            (torch.randn(3, 64, 64), 1.0, False, False, {}),
            (torch.randn(3, 64, 64), 1.0, False, False, {}),
            (torch.randn(3, 64, 64), 1.0, True, False, {})
        ]
        
        reward, steps = self.runner.run_episode(self.flat_agent)
        self.assertEqual(steps, 5)
        self.assertEqual(reward, 5.0)

    def test_evaluate(self):
        """Test evaluating all agents."""
        # Mock run_episode to avoid complex env interactions in this test
        self.runner.run_episode = MagicMock(return_value=(10.0, 100))
        
        results = self.runner.evaluate(num_episodes=2)
        
        self.assertIn('Flat', results)
        self.assertIn('Hierarchical', results)
        self.assertEqual(len(results['Flat']['rewards']), 2)
        self.assertEqual(len(results['Hierarchical']['rewards']), 2)
        self.assertEqual(results['Flat']['mean_reward'], 10.0)

if __name__ == '__main__':
    unittest.main()
