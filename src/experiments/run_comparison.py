import torch
import argparse
import os
import json
import numpy as np
from src.envs.atari_env import AtariPixelEnv
from src.models.agent import ActiveInferenceAgent
from src.models.hierarchical_agent import HierarchicalAgent
from src.models.vae import VAE
from src.models.mlp_vae import MlpVAE
from src.models.transition import TransitionModel
from src.experiments.comparison_runner import ComparisonRunner

def load_flat_agent(env, device, checkpoint_path=None):
    input_dim = (3, 64, 64)
    action_dim = env.action_space.n
    latent_dim = 64
    hidden_dim = 128
    
    vae = VAE(input_shape=input_dim, latent_dim=latent_dim).to(device)
    trans = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading Flat Agent from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Assuming checkpoint structure matches
        if 'vae' in checkpoint:
            vae.load_state_dict(checkpoint['vae'])
        if 'trans' in checkpoint:
            trans.load_state_dict(checkpoint['trans'])
            
    return ActiveInferenceAgent(vae, trans, action_dim, device)

def load_hierarchical_agent(env, device, checkpoint_path=None):
    input_dim = (3, 64, 64)
    action_dim = env.action_space.n
    latent_dim = 64
    hidden_dim = 128
    
    # Level 1
    vae1 = VAE(input_shape=input_dim, latent_dim=latent_dim).to(device)
    trans1 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    # Level 2
    vae2 = MlpVAE(input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    trans2 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading Hierarchical Agent from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'vae1' in checkpoint:
            vae1.load_state_dict(checkpoint['vae1'])
        if 'trans1' in checkpoint:
            trans1.load_state_dict(checkpoint['trans1'])
        if 'vae2' in checkpoint:
            vae2.load_state_dict(checkpoint['vae2'])
        if 'trans2' in checkpoint:
            trans2.load_state_dict(checkpoint['trans2'])
            
    return HierarchicalAgent((vae1, trans1), (vae2, trans2), action_dim, device)

def main():
    parser = argparse.ArgumentParser(description="Run comparison between Flat and Hierarchical agents")
    parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--flat_ckpt", type=str, default=None)
    parser.add_argument("--hier_ckpt", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment
    env = AtariPixelEnv(args.env_id, image_size=64)
    
    # Agents
    flat_agent = load_flat_agent(env, device, args.flat_ckpt)
    hier_agent = load_hierarchical_agent(env, device, args.hier_ckpt)
    
    runner = ComparisonRunner(
        env=env,
        agents={
            'Flat': flat_agent,
            'Hierarchical': hier_agent
        }
    )
    
    # Run Evaluation
    results = runner.evaluate(num_episodes=args.episodes)
    
    # Print Results
    print("\n=== Comparison Results ===")
    for name, stats in results.items():
        print(f"{name}: Mean Reward = {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}, Mean Length = {stats['mean_length']:.2f}")
        
    # Save Results
    with open("comparison_results.json", "w") as f:
        # Convert numpy types to python types for json serialization
        serializable_results = {}
        for name, stats in results.items():
            serializable_results[name] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                          for k, v in stats.items()}
            # Handle lists
            serializable_results[name]['rewards'] = [float(x) for x in stats['rewards']]
            serializable_results[name]['lengths'] = [float(x) for x in stats['lengths']]
            
        json.dump(serializable_results, f, indent=4)
    print("Results saved to comparison_results.json")

if __name__ == "__main__":
    main()
