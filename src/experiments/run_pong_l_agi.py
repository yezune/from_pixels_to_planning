import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pong_trainer import PongTrainer

def run_optimization_experiment():
    # Hyperparameter sets to try
    # 1. Baseline
    # 2. Higher LR
    # 3. Higher Intrinsic Weight
    params_grid = [
        {'lr': 1e-4, 'gamma': 0.99, 'intrinsic_weight': 0.01},
        {'lr': 5e-4, 'gamma': 0.99, 'intrinsic_weight': 0.02},
        {'lr': 1e-3, 'gamma': 0.95, 'intrinsic_weight': 0.05},
    ]
    
    for i, params in enumerate(params_grid):
        print(f"\n==================================================")
        print(f"Starting Experiment {i+1} with params: {params}")
        print(f"==================================================")
        
        # Force CPU for stability
        trainer = PongTrainer(**params, device='cpu')
        success = trainer.train(num_episodes=100, checkpoint_interval=20)
        
        if success:
            print(f"Experiment {i+1} completed successfully!")
            break
        else:
            print(f"Experiment {i+1} failed to improve. Adjusting parameters...")

if __name__ == '__main__':
    run_optimization_experiment()
