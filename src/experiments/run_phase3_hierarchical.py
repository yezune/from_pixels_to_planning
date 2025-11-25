import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from src.l_fep.hierarchical import HierarchicalBlock
from src.l_fep.loss import LogicalDivergenceLoss

def run_hierarchical_experiment():
    print("Starting Phase 3: Hierarchical Distinction (Clustering)")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Model: Map 784 pixels to 10 abstract concepts (clusters)
    input_dim = 784
    output_dim = 10
    model = HierarchicalBlock(input_dim, output_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # Reconstruction loss for autoencoder-like training
    
    # Training
    for epoch in range(1, 4):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            optimizer.zero_grad()
            
            # Bottom-up
            z = model(data)
            
            # Top-down
            recon = model.predict(z)
            
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Reconstruction Loss = {total_loss / len(loader):.4f}")
        
    print("Phase 3 Completed.")

if __name__ == '__main__':
    run_hierarchical_experiment()
