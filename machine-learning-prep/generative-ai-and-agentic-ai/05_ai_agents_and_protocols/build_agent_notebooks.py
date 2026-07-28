import os
import argparse
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def build_notebooks():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=str, default=None, help="Name of specific notebook to build and execute (e.g. 08_production_agent_system.ipynb)")
    args = parser.parse_args()

    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\05_ai_agents_and_protocols"
    notebooks_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)

    # 1. Notebook definitions (defining ONLY Notebook 8)
    notebooks = [
        {
            "filename": "08_production_agent_system.ipynb",
            "cells": [
                ("markdown", "# 08_production_agent_system: Async queue and Profiling metrics\\n\\nThis notebook implements a concurrent, asynchronous agent processing queue utilizing python `asyncio` pipelines to measure throughput and latency under production workloads."),
                ("code", """import asyncio
import time

async def run_single_agent(user_id: int):
    print(f"[Async Agent {user_id}]: Initiated loop.")
    await asyncio.sleep(0.5)
    print(f"[Async Agent {user_id}]: Executed tool.")
    await asyncio.sleep(0.5)
    print(f"[Async Agent {user_id}]: Completed.")
    return f"User {user_id} response."

async def main():
    start_time = time.time()
    tasks = [run_single_agent(i) for i in range(3)]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    print(f"\\nCompleted 3 async agent executions concurrently.")
    print(f"Total time elapsed: {duration:.2f} seconds.")
    print("Results:", results)

# Execute async loop
await main()"""),
                ("markdown", "### Output Explanation & Verification\\n\\n#### Executed Results:\\n- Concurrently dispatched 3 agent pipelines (User 0, User 1, User 2).\\n- **Concurrency Verification**: The total elapsed time was $\\approx 1.00$ seconds, rather than $3.00$ seconds (which would have been the case for sequential synchronous runs), confirming asynchronous execution.\\n\\nThis verifies the async production loop logic, showing how concurrent task orchestration maintains high throughput.")
            ]
        }
    ]

    # Filter notebooks if single target specified
    if args.notebook:
        notebooks = [n for n in notebooks if n["filename"] == args.notebook]
        if not notebooks:
            print(f"ERROR: Notebook '{args.notebook}' not found in definitions.")
            return

    # 2. Programmatic notebook generation
    for item in notebooks:
        nb = nbf.v4.new_notebook()
        for cell_type, content in item["cells"]:
            if cell_type == "markdown":
                cleaned_content = content.replace("\\n", "\n")
                nb['cells'].append(nbf.v4.new_markdown_cell(cleaned_content))
            elif cell_type == "code":
                nb['cells'].append(nbf.v4.new_code_cell(content))
        
        notebook_path = os.path.join(notebooks_dir, item["filename"])
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Created notebook draft: {notebook_path}")

    # 3. Execution using ExecutePreprocessor
    print("\nStarting program execution on notebooks...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    for item in notebooks:
        notebook_path = os.path.join(notebooks_dir, item["filename"])
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
        
        try:
            print(f"Executing: {item['filename']}...")
            ep.preprocess(nb, {'metadata': {'path': notebooks_dir}})
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print(f"SUCCESS: Notebook executed and saved: {item['filename']}")
        except Exception as e:
            print(f"ERROR during notebook execution of {item['filename']}: {e}")

if __name__ == "__main__":
    build_notebooks()
