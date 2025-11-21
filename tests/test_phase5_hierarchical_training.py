import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from src.models.hierarchical_agent import HierarchicalAgent
from src.hierarchical_trainer import HierarchicalTrainer

class TestHierarchicalTrainer(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.action_dim = 4
        
        # Mock Models
        self.vae1 = MagicMock()
        self.vae1.parameters.return_value = [torch.randn(1)]
        self.vae1.encode.return_value = (torch.zeros(1, 32), torch.zeros(1, 32))
        self.vae1.reparameterize.return_value = torch.zeros(1, 32)
        
        self.trans1 = MagicMock()
        self.trans1.parameters.return_value = [torch.randn(1)]
        self.trans1.return_value = ((torch.zeros(1, 32), torch.zeros(1, 32)), None)
        
        self.vae2 = MagicMock()
        self.vae2.parameters.return_value = [torch.randn(1)]
        self.vae2.encode.return_value = (torch.zeros(1, 32), torch.zeros(1, 32))
        self.vae2.reparameterize.return_value = torch.zeros(1, 32)
        
        self.trans2 = MagicMock()
        self.trans2.parameters.return_value = [torch.randn(1)]
        self.trans2.return_value = ((torch.zeros(1, 32), torch.zeros(1, 32)), None)
        
        self.agent = HierarchicalAgent(
            level1_models=(self.vae1, self.trans1),
            level2_models=(self.vae2, self.trans2),
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Mock Environment
        self.env = MagicMock()
        # Ensure target_size is not present to avoid BaseTrainer overriding buffer shape
        del self.env.target_size
        self.env.observation_space.shape = (3, 64, 64)
        self.env.reset.return_value = (torch.zeros(3, 64, 64), {})
        self.env.step.return_value = (torch.zeros(3, 64, 64), 0.0, False, False, {})
        
    def test_initialization(self):
        trainer = HierarchicalTrainer(self.env, self.agent, device=self.device)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.vae1_optimizer)
        self.assertIsNotNone(trainer.trans1_optimizer)
        self.assertIsNotNone(trainer.vae2_optimizer)
        self.assertIsNotNone(trainer.trans2_optimizer)

    def test_collect_data(self):
        trainer = HierarchicalTrainer(self.env, self.agent, device=self.device)
        trainer.collect_data(num_steps=10)
        
        self.assertEqual(self.env.step.call_count, 10)
        self.assertEqual(trainer.buffer.size, 10)

    def test_train_step(self):
        trainer = HierarchicalTrainer(self.env, self.agent, batch_size=2, device=self.device)
        
        # Add dummy data to buffer
        for _ in range(10):
            trainer.buffer.add(
                torch.zeros(3, 64, 64), 
                0, 
                0.0, 
                torch.zeros(3, 64, 64), 
                0.0
            )
            
        # Mock loss calculation to return a tensor with grad
        # We need to mock the internal update methods or the models' forward passes
        # For this test, let's mock the models' forward to return tensors
        
        # VAE1 forward: recon, mu, logvar
        self.vae1.return_value = (torch.zeros(2, 3, 64, 64, requires_grad=True), torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True))
        self.vae1.encode.return_value = (torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True))
        self.vae1.reparameterize.return_value = torch.zeros(2, 32, requires_grad=True)
        
        # Trans1 forward: (mu, logvar), hidden
        self.trans1.return_value = ((torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True)), None)
        
        # VAE2 forward
        self.vae2.return_value = (torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True))
        self.vae2.encode.return_value = (torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True))
        self.vae2.reparameterize.return_value = torch.zeros(2, 32, requires_grad=True)
        
        # Trans2 forward
        self.trans2.return_value = ((torch.zeros(2, 32, requires_grad=True), torch.zeros(2, 32, requires_grad=True)), None)

        # Run train step
        losses = trainer.train_step()
        
        self.assertIn('vae1_loss', losses)
        self.assertIn('trans1_loss', losses)
        # Level 2 might not be trained every step or needs specific logic, but let's assume it is for now
        self.assertIn('vae2_loss', losses)
        self.assertIn('trans2_loss', losses)

    def test_integration_with_real_models(self):
        """
        Test training step with real model instances to ensure shape compatibility.
        """
        from src.models.vae import VAE
        from src.models.mlp_vae import MlpVAE
        from src.models.transition import TransitionModel
        
        # Real Models
        vae1 = VAE(input_shape=(3, 64, 64), latent_dim=32)
        trans1 = TransitionModel(latent_dim=32, action_dim=4, hidden_dim=64)
        
        vae2 = MlpVAE(input_dim=32, latent_dim=16, hidden_dim=64)
        trans2 = TransitionModel(latent_dim=16, action_dim=4, hidden_dim=64)
        
        agent = HierarchicalAgent(
            level1_models=(vae1, trans1),
            level2_models=(vae2, trans2),
            action_dim=4,
            device=self.device
        )
        
        trainer = HierarchicalTrainer(self.env, agent, batch_size=2, device=self.device)
        
        # Add dummy data
        for _ in range(4):
            trainer.buffer.add(
                torch.zeros(3, 64, 64), 
                0, 
                0.0, 
                torch.zeros(3, 64, 64), 
                0.0
            )
            
        # Run train step
        losses = trainer.train_step()
        
        self.assertIn('vae1_loss', losses)
        self.assertIn('trans1_loss', losses)
        self.assertIn('vae2_loss', losses)
        self.assertIn('trans2_loss', losses)

if __name__ == '__main__':
    unittest.main()
