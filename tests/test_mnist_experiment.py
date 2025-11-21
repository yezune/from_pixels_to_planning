import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spatial_rgm import SpatialRGM
from src.experiments.mnist_experiment import MNISTExperiment

class TestMNISTExperiment(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.img_size = 28
        self.num_classes = 10
        self.device = torch.device("cpu")

    def test_spatial_rgm_initialization(self):
        """Test if SpatialRGM initializes correctly."""
        model = SpatialRGM(input_channels=1, hidden_dim=32, latent_dim=16, num_classes=10)
        self.assertIsInstance(model, nn.Module)
        # Check for hierarchical layers
        self.assertTrue(hasattr(model, 'enc1'))
        self.assertTrue(hasattr(model, 'z1_proj'))
        self.assertTrue(hasattr(model, 'enc2'))
        self.assertTrue(hasattr(model, 'dec2'))
        self.assertTrue(hasattr(model, 'dec1'))

    def test_spatial_rgm_forward(self):
        """Test forward pass of SpatialRGM."""
        model = SpatialRGM(input_channels=1, hidden_dim=32, latent_dim=16, num_classes=10)
        dummy_input = torch.randn(self.batch_size, 1, self.img_size, self.img_size)
        
        # Forward pass should return reconstruction, z2_logits, and loss_dict
        recon, logits, loss_dict = model(dummy_input)
        
        self.assertEqual(recon.shape, dummy_input.shape)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('total_loss', loss_dict)
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('kl_z1', loss_dict)
        self.assertIn('kl_z2', loss_dict)

    def test_spatial_rgm_generate(self):
        """Test generation capability."""
        model = SpatialRGM(input_channels=1, hidden_dim=32, latent_dim=16, num_classes=10)
        digit_class = 5
        
        img = model.generate(digit_class, self.device)
        
        # Output should be (1, 1, 28, 28)
        self.assertEqual(img.shape, (1, 1, 28, 28))

    @patch('src.experiments.mnist_experiment.DataLoader')
    @patch('src.experiments.mnist_experiment.MNIST')
    def test_experiment_initialization(self, mock_mnist, mock_loader):
        """Test initialization of the experiment class."""
        experiment = MNISTExperiment(batch_size=self.batch_size, epochs=1, device=self.device)
        self.assertIsInstance(experiment.model, SpatialRGM)
        self.assertIsNotNone(experiment.optimizer)

    @patch('src.experiments.mnist_experiment.DataLoader')
    @patch('src.experiments.mnist_experiment.MNIST')
    def test_train_step(self, mock_mnist, mock_loader):
        """Test a single training step."""
        experiment = MNISTExperiment(batch_size=self.batch_size, epochs=1, device=self.device)
        
        # Mock data
        dummy_imgs = torch.randn(self.batch_size, 1, self.img_size, self.img_size)
        dummy_labels = torch.randint(0, 10, (self.batch_size,))
        
        loss = experiment.train_step(dummy_imgs, dummy_labels)
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()
