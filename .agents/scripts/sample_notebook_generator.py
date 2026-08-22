import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from dotenv import find_dotenv, load_dotenv


def _extract_stdout(nb, cell_index):
    """Reads back the real printed stdout of an already-executed code cell."""
    text = ""
    for out in nb["cells"][cell_index].get("outputs", []):
        if out.get("output_type") == "stream":
            text += out.get("text", "")
        elif out.get("output_type") == "execute_result":
            text += out.get("data", {}).get("text/plain", "")
    return text


def generate_and_execute_notebook(notebook_out_path):
    """Two-pass build: (1) construct + execute with code-only cells and a
    placeholder explanation cell, (2) re-open the executed notebook, read its
    real printed output, and rewrite the explanation cell to quote it literally.
    Never author explanation prose before Pass 1 has produced real output."""
    os.makedirs(os.path.dirname(notebook_out_path), exist_ok=True)

    # ----- PASS 1: construct heading + code cells only, execute, save -----
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Production Matrix Multiplication & Performance Profiling\n"
        "\n"
        "This notebook outlines raw execution and VRAM/latency tracking of standard matrix operations. "
        "It dynamically loads environment keys, checks GPU contiguity, and profiles memory footprints."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "print(\"Environment keys loaded.\")"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 1. PyTorch Tensor Contiguity and GPU Execution Profile"
    ))

    # Index of this code cell is tracked so Pass 2 can read its real output back.
    profiling_cell_index = len(cells)
    cells.append(nbf.v4.new_code_cell(
        "import time\n"
        "import torch\n"
        "\n"
        "torch.manual_seed(42)\n"
        "\n"
        "B, L, H = 2, 4, 8\n"
        "x = torch.randn(B, L, H, device='cuda' if torch.cuda.is_available() else 'cpu')  # [B, L, H]\n"
        "w = torch.randn(H, H, device=x.device)  # [H, H]\n"
        "\n"
        "start_time = time.perf_counter()\n"
        "out = torch.matmul(x, w)  # [B, L, H]\n"
        "latency = (time.perf_counter() - start_time) * 1000\n"
        "\n"
        "print(f\"Output shape: {list(out.shape)}\")\n"
        "print(f\"Tensor Contiguous check: {out.is_contiguous()}\")\n"
        "print(f\"Execution Latency: {latency:.4f} ms\")\n"
        "\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"Max GPU Memory Allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB\")\n"
        "\n"
        "assert out.shape == (B, L, H), \"Shape mismatch!\"\n"
        "assert not torch.isnan(out).any(), \"NaNs detected!\""
    ))

    # Placeholder only -- real explanation text is written in Pass 2, from real output.
    explanation_cell_index = len(cells)
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: Tensor Profile\n_(pending real output)_"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_out_path) or "."}})

    with open(notebook_out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Pass 1 complete: notebook executed and saved: {notebook_out_path}")

    # ----- PASS 2: read the real output back, then rewrite the explanation -----
    with open(notebook_out_path, "r", encoding="utf-8") as f:
        nb_executed = nbf.read(f, as_version=4)

    real_stdout = _extract_stdout(nb_executed, profiling_cell_index)
    # real_stdout now literally contains lines like:
    #   "Output shape: [2, 4, 8]"
    #   "Tensor Contiguous check: True"
    #   "Execution Latency: 0.1234 ms"
    # Parse/copy the exact values out of it -- never guess or restate them abstractly.
    lines = [ln for ln in real_stdout.splitlines() if ln.strip()]
    quoted_lines = "\n".join(f"  `{ln}`" for ln in lines)

    nb_executed["cells"][explanation_cell_index]["source"] = (
        "### Output Explanation: Tensor Profile\n"
        f"- **Real printed output from this run:**\n{quoted_lines}\n"
        "- **Dimension flow**: the shape line above shows the actual output shape produced by "
        "`torch.matmul(x, w)` on this run's random seed -- quoted directly, not assumed from the code.\n"
        "- **Contiguity and latency are the literal measured values above**, not restated generically -- "
        "if a future run prints different latency or a different device line, this explanation must be "
        "regenerated from that new output, not left as-is."
    )

    with open(notebook_out_path, "w", encoding="utf-8") as f:
        nbf.write(nb_executed, f)
    print(f"Pass 2 complete: explanation cell rewritten from real output: {notebook_out_path}")


if __name__ == "__main__":
    test_ipynb = os.path.abspath("./test_sample_execution.ipynb")
    try:
        generate_and_execute_notebook(test_ipynb)
    finally:
        if os.path.exists(test_ipynb):
            os.remove(test_ipynb)
