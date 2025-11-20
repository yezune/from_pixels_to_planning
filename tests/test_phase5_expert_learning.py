import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.utils.expert_collector import ExpertCollector
from src.hierarchical_trainer import HierarchicalTrainer
from src.models.hierarchical_agent import HierarchicalAgent

class TestExpertLearning(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock()
        self.env.reset.return_value = (torch.zeros(3, 64, 64), {})
        self.env.step.return_value = (torch.zeros(3, 64, 64), 1.0, False, False, {})
        
        # Mock Expert Agent (always returns action 0)
        self.expert_agent = MagicMock()
        self.expert_agent.select_action.return_value = 0
        
        self.device = 'cpu'

    def test_collect_expert_trajectory(self):
        collector = ExpertCollector(self.env, self.expert_agent, device=self.device)
        trajectories = collector.collect_episodes(n_episodes=2, max_steps=10)
        
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(len(trajectories[0]['obs']), 10)
        self.assertEqual(len(trajectories[0]['actions']), 10)
        # Check if observations are stored
        self.assertTrue(isinstance(trajectories[0]['obs'][0], np.ndarray) or isinstance(trajectories[0]['obs'][0], torch.Tensor))

    def test_train_level2_on_expert(self):
        # Setup Hierarchical Agent
        vae1 = MagicMock()
        vae1.parameters.return_value = [torch.randn(1)] # Fix empty param list
        vae1.encode.return_value = (torch.zeros(1, 32), torch.zeros(1, 32))
        vae1.reparameterize.return_value = torch.zeros(1, 32)
        
        trans1 = MagicMock()
        trans1.parameters.return_value = [torch.randn(1)] # Fix empty param list
        
        vae2 = MagicMock()
        vae2.parameters.return_value = [torch.randn(1, requires_grad=True)]
        vae2.encode.return_value = (torch.zeros(1, 16), torch.zeros(1, 16))
        vae2.reparameterize.return_value = torch.zeros(1, 16)
        # Mock VAE2 call (forward)
        # When called as self.agent.vae2(z1_seq), it calls __call__, which calls forward.
        # MagicMock's __call__ returns return_value by default.
        # We need to set side_effect on the mock object itself, not just forward, 
        # because we are calling the object directly.
        vae2.side_effect = lambda x: (torch.zeros(x.shape[0], 32, requires_grad=True), torch.zeros(x.shape[0], 16, requires_grad=True), torch.zeros(x.shape[0], 16, requires_grad=True))
        
        trans2 = MagicMock()
        trans2.parameters.return_value = [torch.randn(1, requires_grad=True)]
        # Transition is also called directly: self.agent.trans2(...)
        trans2.side_effect = lambda z, a, h: ((torch.zeros(z.shape[0], 16, requires_grad=True), torch.zeros(z.shape[0], 16, requires_grad=True)), None)
        
        agent = HierarchicalAgent(
            level1_models=(vae1, trans1),
            level2_models=(vae2, trans2),
            action_dim=4,
            device=self.device
        )
        
        # Mock Trajectories (List of dicts)
        # Each trajectory has 'obs' which is a list of tensors/arrays
        trajectories = [
            {
                'obs': [torch.zeros(3, 64, 64) for _ in range(5)],
                'actions': [0 for _ in range(5)]
            }
        ]
        
        # Mock VAE1 encode/reparameterize to return correct shape (T=5)
        vae1.encode.return_value = (torch.zeros(5, 32), torch.zeros(5, 32))
        vae1.reparameterize.return_value = torch.zeros(5, 32)
        
        # Mock VAE2 encode/reparameterize to return correct shape (T=5)
        vae2.encode.return_value = (torch.zeros(5, 16), torch.zeros(5, 16))
        vae2.reparameterize.return_value = torch.zeros(5, 16)
        
        # We need a trainer or a specialized function to train on this
        # Let's assume we add a method to HierarchicalTrainer or use a new class
        from src.hierarchical_trainer import HierarchicalTrainer
        trainer = HierarchicalTrainer(self.env, agent, device=self.device)
        
        # This method needs to be implemented
        losses = trainer.train_on_expert_trajectories(trajectories, epochs=1)
        
        self.assertIn('level2_loss', losses)

if __name__ == '__main__':
    unittest.main()
