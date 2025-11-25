from src.experiments.mnist_logical_experiment import MNISTLogicalExperiment

if __name__ == "__main__":
    experiment = MNISTLogicalExperiment(epochs=3) # Reduced epochs for quick check
    final_acc = experiment.run()
    print(f"Phase 1 Experiment Completed. Final Accuracy: {final_acc:.2f}%")
