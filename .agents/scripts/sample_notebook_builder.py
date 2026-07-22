import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

def create_and_execute_sample_notebook(output_path):
    # 1. Create a blank notebook object
    nb = nbf.v4.new_notebook()
    
    # 2. Define cells
    cells = []
    
    # Add a Markdown cell
    cells.append(nbf.v4.new_markdown_cell("""# Sample Study Notebook
This notebook serves as a programmatic template for generating executed assets.
"""))
    
    # Add a Code cell
    cells.append(nbf.v4.new_code_cell("""import numpy as np
# Simulate some matrix calculations
x = np.array([1.0, 2.0, 3.0])
y = x * 2
print("Input Vector: ", x)
print("Scaled Vector:", y)
"""))
    
    # Add an explanation Markdown cell matching the code output
    cells.append(nbf.v4.new_markdown_cell("""### Explanation of Outputs
- **Input Vector**: $[1.0, 2.0, 3.0]$ is our baseline test array.
- **Scaled Vector**: $[2.0, 4.0, 6.0]$ matches the $y = 2x$ scalar multiplication exactly.
"""))
    
    nb['cells'] = cells
    
    # 3. Add default metadata
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    }
    
    # 4. Save the unexecuted notebook to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created notebook draft at: {output_path}")
    
    # 5. Initialize and run the execution preprocessor
    # Uses the local python3 kernel to execute cells in place and save the logs
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Reload from disk
    with open(output_path, 'r', encoding='utf-8') as f:
        nb_loaded = nbf.read(f, as_version=4)
        
    print("Executing cells...")
    ep.preprocess(nb_loaded, {'metadata': {'path': os.path.dirname(output_path)}})
    
    # Save the final executed notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_loaded, f)
    print(f"Executed and saved notebook at: {output_path}")

if __name__ == "__main__":
    # Test script execution locally
    test_path = os.path.abspath("./sample_notebook.ipynb")
    create_and_execute_sample_notebook(test_path)
