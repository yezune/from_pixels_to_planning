import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
import os
from src.models.spatial_rgm import SpatialRGM

class MNISTExperiment:
    def __init__(self, batch_size=64, epochs=5, lr=1e-3, device=None):
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
        ])
        
        # Model Setup
        self.model = SpatialRGM(latent_dim=32, num_classes=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.cls_criterion = nn.CrossEntropyLoss()
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)

    def get_loaders(self):
        # Download=True might be an issue in some environments, but standard for local
        train_dataset = MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def train_step(self, images, labels, epoch=0, total_epochs=10):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # --- Annealing Schedule ---
        # Temperature: 1.0 -> 0.5
        temp = max(0.5, 1.0 - 0.5 * (epoch / total_epochs))
        self.model.set_temperature(temp)
        
        # Loss Weighting:
        # Classification weight: 50.0 (Increased from 20.0 for >95% acc)
        # VAE weight: 0.0 -> 1.0 (Linear Warm-up over first 8 epochs)
        vae_weight = min(1.0, epoch / 8.0)
        cls_weight = 50.0
        
        self.optimizer.zero_grad()
        
        recon, logits, loss_dict = self.model(images)
        
        # Combine VAE loss with Classification loss
        cls_loss = self.cls_criterion(logits, labels)
        
        # Normalize VAE loss by batch size
        vae_loss = loss_dict['total_loss'] / images.size(0)
        
        total_loss = (vae_weight * vae_loss) + (cls_weight * cls_loss)
        
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
                
                _, logits, _ = self.model(images)
                _, predicted = torch.max(logits.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def generate_samples(self):
        """Generate samples for each digit class."""
        print("Generating samples...")
        self.model.eval()
        samples = []
        for i in range(10):
            img = self.model.generate(i, self.device)
            samples.append(img)
        
        # Concatenate and save
        # Each img is (1, 1, 28, 28)
        grid = torch.cat(samples, dim=0)
        save_image(grid, "outputs/mnist_generation.png", nrow=10)
        print("Saved generated samples to outputs/mnist_generation.png")

    def run(self):
        train_loader, test_loader = self.get_loaders()
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                loss = self.train_step(images, labels, epoch, self.epochs)
                total_loss += loss
                
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss:.4f}")
            
            avg_loss = total_loss / len(train_loader)
            accuracy = self.evaluate(test_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] Finished. Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            
        print("Training finished.")
        self.generate_samples()

if __name__ == "__main__":
    experiment = MNISTExperiment()
    experiment.run()
