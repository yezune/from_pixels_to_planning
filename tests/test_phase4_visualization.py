import unittest
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.visualization import plot_reconstruction, plot_free_energy_history
except ImportError:
    plot_reconstruction = None
    plot_free_energy_history = None

class TestPhase4Visualization(unittest.TestCase):
    def setUp(self):
        if plot_reconstruction is None:
            self.skipTest("Visualization module not yet implemented")
        
        # Create a dummy output directory for plots
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'test_plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test plots
        import shutil
        if os.path.exists(self.plot_dir):
            shutil.rmtree(self.plot_dir)

    def test_plot_reconstruction(self):
        print("\n[Vis] Testing Reconstruction Plot...")
        # Dummy images (Batch, C, H, W)
        original = torch.rand(4, 1, 64, 64)
        recon = torch.rand(4, 1, 64, 64)
        
        save_path = os.path.join(self.plot_dir, 'recon_test.png')
        
        plot_reconstruction(original, recon, save_path=save_path)
        
        self.assertTrue(os.path.exists(save_path))
        print("  - Plot saved successfully")

    def test_plot_free_energy_history(self):
        print("\n[Vis] Testing Free Energy History Plot...")
        history = [10.0, 8.5, 6.0, 4.5, 3.0, 2.5, 2.0]
        
        save_path = os.path.join(self.plot_dir, 'fe_history_test.png')
        
        plot_free_energy_history(history, save_path=save_path)
        
        self.assertTrue(os.path.exists(save_path))
        print("  - History plot saved successfully")

if __name__ == '__main__':
    unittest.main()
