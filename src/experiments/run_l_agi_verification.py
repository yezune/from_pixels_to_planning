import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import time
from src.models.logical_spatial_rgm import LogicalSpatialRGM

def run_l_agi_verification():
    print("Starting L-AGI Paper Verification (Logical Spatial RGM on MNIST)")
    
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        
    print(f"Device: {DEVICE}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = LogicalSpatialRGM(latent_dim=32, num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_consist = 0
        total_cls = 0
        correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            recon, z1, z2, z1_prior = model(data)
            
            loss, l_recon, l_consist, l_cls = model.get_loss(recon, data, z1, z1_prior, z2, target)
            
            loss.backward()
            
            # Accuracy
            # Move to CPU to avoid potential MPS argmax issues
            z2_cpu = z2.detach().cpu()
            pred = (z2_cpu ** 2).argmax(dim=1)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += l_recon.item()
            total_consist += l_consist.item()
            total_cls += l_cls.item()
            
            correct += (pred == target.cpu()).sum().item()
            total_samples += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total_samples
        duration = time.time() - start_time
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} (Recon={total_recon/len(train_loader):.2f}, Consist={total_consist/len(train_loader):.2f}, Cls={total_cls/len(train_loader):.2f}) | Acc={accuracy:.2f}% | Time={duration:.2f}s")
        
    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            _, _, z2, _ = model(data)
            z2_cpu = z2.detach().cpu()
            pred = (z2_cpu ** 2).argmax(dim=1)
            test_correct += (pred == target.cpu()).sum().item()
            test_total += target.size(0)
            
    print(f"Final Test Accuracy: {100.0 * test_correct / test_total:.2f}%")
    print("L-AGI Verification Completed.")

if __name__ == '__main__':
    run_l_agi_verification()
