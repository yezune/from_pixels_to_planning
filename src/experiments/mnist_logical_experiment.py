import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os
from src.models.logical_spatial_rgm import LogicalSpatialRGM
from src.l_fep.loss import LogicalDivergenceLoss

class MNISTLogicalExperiment:
    def __init__(self, batch_size=64, epochs=5, lr=0.01, device=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Data Setup
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # L-FEP often works better with normalized inputs, but [0,1] is usually fine.
            # Standard MNIST normalization:
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Model Setup
        # Use LogicalSpatialRGM for both classification and generation
        self.model = LogicalSpatialRGM(input_channels=1, latent_dim=32, num_classes=10).to(self.device)
        
        # L-FEP Components
        # Optimizer: Adadelta is used in the reference implementation
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.criterion = LogicalDivergenceLoss()
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)

    def get_loaders(self):
        train_dataset = MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=self.transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return self.train_loader, self.test_loader

    def train_step(self, images, labels):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass through RGM
        recon, z1, z2, z1_prior = self.model(images)
        
        # Calculate L-AGI Loss
        total_loss, _, _, _ = self.model.get_loss(recon, images, z1, z1_prior, z2, labels)
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                _, _, z2, _ = self.model(images)
                
                # z2 is amplitude, z2^2 is probability
                probs = z2 ** 2
                _, predicted = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def run(self):
        train_loader, test_loader = self.get_loaders()
        
        print(f"Starting Phase 1: Deep Logical Learning on {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                loss = self.train_step(data, target)
                total_loss += loss
                
            avg_loss = total_loss / len(train_loader)
            accuracy = self.evaluate(test_loader)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
            
        return accuracy
