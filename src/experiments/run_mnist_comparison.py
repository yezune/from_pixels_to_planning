import torch
import matplotlib.pyplot as plt
import json
import os
from src.experiments.mnist_experiment import MNISTExperiment

def run_comparison():
    print("Running MNIST Comparison Experiment (10 Epochs)...")
    
    # Setup
    epochs = 15
    experiment = MNISTExperiment(epochs=epochs, batch_size=64)
    
    # Tracking
    history = {
        "train_loss": [],
        "test_accuracy": []
    }
    
    # Run Loop
    train_loader, test_loader = experiment.get_loaders()
    
    print(f"Starting training on {experiment.device}...")
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            loss = experiment.train_step(images, labels, epoch, epochs)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        accuracy = experiment.evaluate(test_loader)
        
        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save Results
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Save Metrics
    with open("outputs/mnist_comparison_results.json", "w") as f:
        json.dump(history, f, indent=4)
        
    # 2. Plot Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()
    
    plt.savefig("outputs/mnist_learning_curves.png")
    print("Saved learning curves to outputs/mnist_learning_curves.png")
    
    # 3. Generate Samples
    experiment.generate_samples()
    print("Comparison Experiment Completed.")

if __name__ == "__main__":
    run_comparison()
