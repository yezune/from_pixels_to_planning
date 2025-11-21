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
            '01_bouncing_ball.ipynb',
            '02_atari_breakout.ipynb',
            '03_performance_comparison.ipynb',
            '04_mnist_classification.ipynb'
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
                source = source.replace('num_steps=200', 'num_steps=10')
                source = source.replace('num_steps=1000', 'num_steps=10')
                source = source.replace('num_epochs=5', 'num_epochs=1')
                source = source.replace('epochs=3', 'epochs=1')
                source = source.replace('epochs=5', 'epochs=1')
                source = source.replace('batch_size=64', 'batch_size=4')
                source = source.replace('batch_size=16', 'batch_size=4')
                
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

    def test_01_bouncing_ball(self):
        self._test_notebook('01_bouncing_ball.ipynb')

    def test_02_atari_breakout(self):
        self._test_notebook('02_atari_breakout.ipynb')

    def test_03_performance_comparison(self):
        self._test_notebook('03_performance_comparison.ipynb')

    def test_04_mnist_classification(self):
        self._test_notebook('04_mnist_classification.ipynb')

if __name__ == '__main__':
    unittest.main()
