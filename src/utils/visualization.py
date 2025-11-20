import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_reconstruction(original, reconstruction, n=4, save_path=None):
    """
    Plots original and reconstructed images side by side.
    
    Args:
        original (torch.Tensor): Batch of original images (B, C, H, W)
        reconstruction (torch.Tensor): Batch of reconstructed images (B, C, H, W)
        n (int): Number of images to display
        save_path (str): Path to save the figure. If None, shows the figure.
    """
    original = original.cpu().detach().numpy()
    reconstruction = reconstruction.cpu().detach().numpy()
    
    # Limit to batch size
    n = min(n, original.shape[0])
    
    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        img = original[i].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
        if img.shape[2] == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
        plt.title("Original")
        plt.axis("off")
        
        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        recon_img = reconstruction[i].transpose(1, 2, 0)
        if recon_img.shape[2] == 1:
            plt.imshow(recon_img.squeeze(), cmap='gray')
        else:
            plt.imshow(recon_img)
        plt.title("Reconstructed")
        plt.axis("off")
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_free_energy_history(history, save_path=None):
    """
    Plots the history of Free Energy (Loss) over time.
    
    Args:
        history (list): List of loss values.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Variational Free Energy (Loss)')
    plt.xlabel('Iterations / Episodes')
    plt.ylabel('Free Energy')
    plt.title('Free Energy Minimization over Time')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
