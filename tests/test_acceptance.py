import unittest
import torch
import gymnasium as gym
import os
import sys
import shutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.synthetic_env import BouncingBallEnv
from src.envs.atari_env import AtariPixelEnv
from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.models.agent import ActiveInferenceAgent
from src.models.hierarchical_agent import HierarchicalAgent
from src.models.mlp_vae import MlpVAE
from src.trainer import ActiveInferenceTrainer
from src.hierarchical_trainer import HierarchicalTrainer
from src.experiments.comparison_runner import ComparisonRunner

class TestAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for checkpoints
        cls.test_dir = "test_acceptance_artifacts"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Device
        cls.device = torch.device("cpu") # Force CPU for tests to ensure compatibility

    @classmethod
    def tearDownClass(cls):
        # Clean up
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_phase4_bouncing_ball_pipeline(self):
        """
        Acceptance Test for Phase 4: Bouncing Ball + Flat Active Inference Agent
        Verifies: Env creation, Model init, Training loop execution.
        """
        print("\n[Acceptance] Testing Phase 4: Bouncing Ball Pipeline...")
        
        # 1. Environment
        env = BouncingBallEnv(size=32) # Use smaller size for speed
        obs_shape = (3, 32, 32) # RGB, 32x32
        action_dim = env.action_space.n
        
        # 2. Models
        vae = VAE(input_shape=obs_shape, latent_dim=16).to(self.device)
        trans = TransitionModel(latent_dim=16, action_dim=action_dim, hidden_dim=32).to(self.device)
        agent = ActiveInferenceAgent(vae, trans, action_dim, device=self.device)
        
        # 3. Trainer
        trainer = ActiveInferenceTrainer(
            env=env,
            agent=agent,
            buffer_size=100,
            batch_size=4,
            lr=1e-3,
            device=self.device
        )
        
        # 4. Run Training Loop (Short)
        trainer.collect_data(num_steps=10)
        vae_loss = trainer.train_vae()
        trans_loss = trainer.train_transition()
        losses = {'vae_loss': vae_loss, 'trans_loss': trans_loss}
        
        # 5. Assertions
        self.assertIn('vae_loss', losses)
        self.assertIn('trans_loss', losses)
        self.assertIsInstance(losses['vae_loss'], float)
        print("[Acceptance] Phase 4 Passed.")

    def test_phase5_atari_hierarchical_pipeline(self):
        """
        Acceptance Test for Phase 5: Atari + Hierarchical Agent
        Verifies: Atari Env Wrapper, Hierarchical Model init, Hierarchical Training loop.
        """
        print("\n[Acceptance] Testing Phase 5: Atari Hierarchical Pipeline...")
        
        # 1. Environment
        # Use a simpler environment if possible, or mock if Atari is too heavy. 
        # But for acceptance, we should try the real thing if installed.
        try:
            env = AtariPixelEnv("BreakoutNoFrameskip-v4", image_size=64)
        except Exception as e:
            self.skipTest(f"Atari environment not available: {e}")
            
        input_dim = (3, 64, 64)
        action_dim = env.action_space.n
        latent_dim = 32
        hidden_dim = 64
        
        # 2. Models
        # Level 1
        vae1 = VAE(input_shape=input_dim, latent_dim=latent_dim).to(self.device)
        trans1 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        
        # Level 2
        vae2 = MlpVAE(input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)
        trans2 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        
        agent = HierarchicalAgent(
            level1_models=(vae1, trans1),
            level2_models=(vae2, trans2),
            action_dim=action_dim,
            device=self.device
        )
        
        # 3. Trainer
        trainer = HierarchicalTrainer(
            env=env,
            agent=agent,
            buffer_size=100,
            batch_size=4,
            lr=1e-4,
            device=self.device
        )
        
        # 4. Run Training Loop (Short)
        trainer.collect_data(num_steps=10)
        losses = trainer.train_step()
        
        # 5. Assertions
        self.assertIn('vae1_loss', losses)
        self.assertIn('trans1_loss', losses)
        self.assertIn('vae2_loss', losses)
        self.assertIn('trans2_loss', losses)
        print("[Acceptance] Phase 5 Passed.")

    def test_comparison_runner_integration(self):
        """
        Acceptance Test for Comparison Runner
        Verifies: Running multiple agents and collecting stats.
        """
        print("\n[Acceptance] Testing Comparison Runner Integration...")
        
        # Mock Env for speed
        env = BouncingBallEnv(size=32) # Reuse simple env, wrap if needed for pixel check
        # BouncingBall returns (32, 32, 3), ComparisonRunner expects tensor.
        
        # Mock Agents (using real classes but small config)
        obs_shape = (3, 32, 32)
        action_dim = env.action_space.n
        
        # Flat Agent
        vae = VAE(input_shape=obs_shape, latent_dim=8).to(self.device)
        trans = TransitionModel(latent_dim=8, action_dim=action_dim, hidden_dim=16).to(self.device)
        flat_agent = ActiveInferenceAgent(vae, trans, action_dim, device=self.device)
        
        runner = ComparisonRunner(
            env=env,
            agents={'Flat_Test': flat_agent}
        )
        
        # Run
        results = runner.evaluate(num_episodes=1)
        
        # Assertions
        self.assertIn('Flat_Test', results)
        self.assertIn('mean_reward', results['Flat_Test'])
        print("[Acceptance] Comparison Runner Passed.")

if __name__ == '__main__':
    # When run directly, execute all tests in the tests/ directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)
