import unittest
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

class TestNotebooks(unittest.TestCase):
    def setUp(self):
        self.notebooks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # List of notebooks to test
        self.notebooks_to_test = [
            '01_rgm_fundamentals.ipynb',
            '02_mnist_classification.ipynb',
            '03_bouncing_ball.ipynb',
            '04_atari_breakout.ipynb',
            '05_performance_comparison.ipynb',
            '06_hierarchical_planning_results.ipynb'
        ]

    def _test_notebook(self, notebook_name):
        notebook_path = os.path.join(self.notebooks_dir, notebook_name)
        if not os.path.exists(notebook_path):
            print(f"Skipping {notebook_name} (not found)")
            return

        print(f"Testing notebook: {notebook_name}")
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Modify notebook content to run faster for testing
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Reduce training loops
                source = source.replace('range(50)', 'range(1)')
                source = source.replace('range(100)', 'range(1)')
                source = source.replace('range(2000)', 'range(10)') # Added for NB 03
                source = source.replace('num_steps=200', 'num_steps=10')
                source = source.replace('num_steps=1000', 'num_steps=10')
                source = source.replace('num_epochs=5', 'num_epochs=1')
                source = source.replace('epochs=3', 'epochs=1')
                source = source.replace('epochs=5', 'epochs=1')
                source = source.replace('epochs = 50', 'epochs = 1') # Added for NB 03
                source = source.replace('num_episodes = 3', 'num_episodes = 1') # Added for NB 04
                source = source.replace('batch_size=64', 'batch_size=4')
                source = source.replace('batch_size=16', 'batch_size=4')
                source = source.replace('batch_size = 32', 'batch_size = 4') # Added for NB 03
                
                # Break long loops in NB 01
                if 'for x, y in loader:' in source:
                    source = source.replace('for x, y in loader:', 'for i, (x, y) in enumerate(loader):\n            if i > 2: break')

                # Handle potential display/render issues in headless env
                if 'plt.show()' in source:
                    # plt.show() usually doesn't block in nbconvert, but good to know
                    pass
                
                cell.source = source

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Set the path to the project root so imports work
        # We also need to ensure the working directory is correct (usually notebook dir)
        resources = {'metadata': {'path': self.notebooks_dir}}
        
        try:
            # Execute the notebook
            ep.preprocess(nb, resources)
        except Exception as e:
            self.fail(f"Notebook {notebook_name} execution failed: {e}")

    def test_01_rgm_fundamentals(self):
        self._test_notebook('01_rgm_fundamentals.ipynb')

    def test_02_mnist_classification(self):
        self._test_notebook('02_mnist_classification.ipynb')

    def test_03_bouncing_ball(self):
        self._test_notebook('03_bouncing_ball.ipynb')

    def test_04_atari_breakout(self):
        self._test_notebook('04_atari_breakout.ipynb')

    def test_05_performance_comparison(self):
        self._test_notebook('05_performance_comparison.ipynb')

    def test_06_hierarchical_planning_results(self):
        self._test_notebook('06_hierarchical_planning_results.ipynb')

if __name__ == '__main__':
    unittest.main()
