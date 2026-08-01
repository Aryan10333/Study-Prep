import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_and_execute_notebook_07():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "07_attention_and_transformer_prerequisites.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Intro
    cells.append(nbf.v4.new_markdown_cell(
        "# 07_attention_and_transformer_prerequisites: Scaled Dot-Product Attention Implementation\n"
        "\n"
        "This notebook validates scaled dot-product attention mechanics. It implements the attention alignment projection loop in PyTorch, verifying it against our step-by-step hand calculations."
    ))
    
    # Cell 2: Heading
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Scaled Dot-Product Attention in PyTorch"
    ))
    
    # Cell 3: Code
    cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "import numpy as np\n"
        "\n"
        "# Define Query, Keys, and Values matching our hand-calculations exactly\n"
        "q = torch.tensor([[1.0, 2.0]]) # shape (1, d_k)\n"
        "k = torch.tensor([[1.0, 0.0],  # k1\n"
        "                  [0.0, 2.0]]) # k2, shape (L, d_k)\n"
        "v = torch.tensor([[10.0, 20.0],\n"
        "                  [30.0, 40.0]]) # shape (L, d_v)\n"
        "\n"
        "d_k = q.size(-1)\n"
        "\n"
        "# 1. Compute scores (q * K^T)\n"
        "scores = torch.matmul(q, k.t())\n"
        "\n"
        "# 2. Scale by sqrt(d_k)\n"
        "scaled_scores = scores / (d_k ** 0.5)\n"
        "\n"
        "# 3. Softmax attention weights\n"
        "weights = F.softmax(scaled_scores, dim=-1)\n"
        "\n"
        "# 4. Weighted sum of values\n"
        "context = torch.matmul(weights, v)\n"
        "\n"
        "print(\"Raw scores:        \", scores.numpy().flatten())\n"
        "print(\"Scaled scores:     \", scaled_scores.numpy().flatten())\n"
        "print(\"Attention weights: \", weights.numpy().flatten())\n"
        "print(\"Context vector:    \", context.numpy().flatten())\n"
        "\n"
        "# Assertions validating consistency with hand-calculations (context matches [27.8592, 37.8592])\n"
        "np.testing.assert_almost_equal(context.numpy()[0], [27.8592, 37.8592], decimal=4)"
    ))
    
    # Cell 4: Explanation
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: Attention Retrieval" + "\n" +
        r"The PyTorch execution matches our hand calculations. The query $\mathbf{q} = [1, 2]^T$ has a much stronger dot-product alignment with $\mathbf{k}_2$ than $\mathbf{k}_1$, leading to scaled scores of $0.7071$ and $2.8284$. The resulting softmax attention weights assign $89.30\%$ of the probability to index 2, projecting the final output context vector to `[27.8592, 37.8592]` (equal to `[27.8600, 37.8600]` if intermediate values are rounded), verifying execution consistency."
    ))
    
    nb.cells = cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook draft: {notebook_path}")

def create_and_execute_notebook_08():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "08_nlp_evaluation.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Intro
    cells.append(nbf.v4.new_markdown_cell(
        "# 08_nlp_evaluation: BLEU and ROUGE Overlap Metrics Verification\n"
        "\n"
        "This notebook validates NLP evaluation metrics. It implements BLEU precision scoring using NLTK and ROUGE recall scoring from scratch, verifying the math against our step-by-step hand calculations."
    ))
    
    # Cell 2: Heading
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. BLEU Score Precision and Brevity Penalty"
    ))
    
    # Cell 3: Code
    cells.append(nbf.v4.new_code_cell(
        "import nltk\n"
        "from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction\n"
        "import numpy as np\n"
        "\n"
        "# Define candidate and reference tokens matching hand-calculation\n"
        "candidate = [\"the\", \"cat\", \"sat\"]\n"
        "reference = [[\"the\", \"cat\", \"sat\", \"on\", \"the\", \"mat\"]]\n"
        "\n"
        "# Calculate BLEU-2 with weights (0.5, 0.5) and no smoothing\n"
        "weights = (0.5, 0.5)\n"
        "bleu_score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=SmoothingFunction().method0)\n"
        "\n"
        "print(f\"NLTK BLEU-2 Score: {bleu_score:.4f}\")\n"
        "\n"
        "# Verify exact match with hand calculation score of 0.3679\n"
        "np.testing.assert_almost_equal(bleu_score, 0.3679, decimal=4)"
    ))
    
    # Cell 4: Explanation
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: BLEU Metric" + "\n" +
        r"The computed BLEU-2 score is exactly `0.3679`. Because the candidate length $c=3$ is shorter than reference length $r=6$, the brevity penalty $\text{BP} = e^{1-6/3} = e^{-1} \approx 0.3679$ dampens the perfect unigram and bigram precisions ($p_1=1.0, p_2=1.0$), ensuring that models cannot cheat the evaluation metric by outputting short snippets."
    ))
    
    # Cell 5: Heading
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. ROUGE-1 Recall and F1 Score from Scratch"
    ))
    
    # Cell 6: Code
    cells.append(nbf.v4.new_code_cell(
        "from collections import Counter\n"
        "\n"
        "# Count unigram overlaps\n"
        "cand_counts = Counter(candidate)\n"
        "ref_counts = Counter(reference[0])\n"
        "\n"
        "overlaps = 0\n"
        "for word, count in cand_counts.items():\n"
        "    overlaps += min(count, ref_counts[word])\n"
        "\n"
        "recall_rouge = overlaps / len(reference[0])\n"
        "precision_rouge = overlaps / len(candidate)\n"
        "f1_rouge = 2 * (precision_rouge * recall_rouge) / (precision_rouge + recall_rouge)\n"
        "\n"
        "print(f\"ROUGE-1 Recall:    {recall_rouge:.4f}\")\n"
        "print(f\"ROUGE-1 Precision: {precision_rouge:.4f}\")\n"
        "print(f\"ROUGE-1 F1 Score:  {f1_rouge:.4f}\")\n"
        "\n"
        "# Verify consistency with hand calculations (Recall = 0.5, F1 = 0.6667)\n"
        "np.testing.assert_almost_equal(recall_rouge, 0.5000, decimal=4)\n"
        "np.testing.assert_almost_equal(f1_rouge, 0.6667, decimal=4)"
    ))
    
    # Cell 7: Explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: ROUGE Metric\n"
        "The ROUGE recall outputs match our hand calculations perfectly (Recall = `0.5000`, F1 = `0.6667`). This confirms that ROUGE focuses on coverage (measuring how much of the reference was captured), complementary to BLEU's precision focus."
    ))
    
    nb.cells = cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook draft: {notebook_path}")

def create_and_execute_notebook_09():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "09_production_nlp.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Intro
    cells.append(nbf.v4.new_markdown_cell(
        "# 09_production_nlp: Population Stability Index (PSI) Drift Monitoring\n"
        "\n"
        "This notebook validates production drift metrics. It implements Population Stability Index (PSI) and Wasserstein Distance using NumPy and SciPy, verifying them against our hand calculations."
    ))
    
    # Cell 2: Heading
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Population Stability Index (PSI) Calculation"
    ))
    
    # Cell 3: Code
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\n"
        "\n"
        "# Proportions matching our hand calculations\n"
        "expected = np.array([0.6, 0.3, 0.1])\n"
        "actual = np.array([0.5, 0.3, 0.2])\n"
        "\n"
        "def calculate_psi(p, q):\n"
        "    p = np.clip(p, 1e-15, 1.0)\n"
        "    q = np.clip(q, 1e-15, 1.0)\n"
        "    return np.sum((p - q) * np.log(p / q))\n"
        "\n"
        "psi_val = calculate_psi(actual, expected)\n"
        "print(f\"Calculated PSI Score: {psi_val:.4f}\")\n"
        "\n"
        "# Verify exact match with hand calculation score of 0.0875\n"
        "np.testing.assert_almost_equal(psi_val, 0.0875, decimal=4)"
    ))
    
    # Cell 4: Explanation
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: PSI Score" + "\n" +
        r"The computed PSI score is exactly `0.0875`. Since the PSI score is less than the standard stability threshold of `0.10`, the difference between expected (training) and actual (production) distribution represents no significant shift, indicating a stable production pipeline."
    ))
    
    # Cell 5: Heading
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Wasserstein Distance Drift Metric"
    ))
    
    # Cell 6: Code
    cells.append(nbf.v4.new_code_cell(
        "from scipy.stats import wasserstein_distance\n"
        "\n"
        "# Calculate Wasserstein Distance (Earth Mover's Distance)\n"
        "w_dist = wasserstein_distance([0, 1, 2], [0, 1, 2], u_weights=actual, v_weights=expected)\n"
        "print(f\"Wasserstein Distance (EMD): {w_dist:.4f}\")\n"
        "\n"
        "assert w_dist > 0.0, \"Wasserstein distance check failed!\""
    ))
    
    # Cell 7: Explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Wasserstein Distance\n"
        "The computed Wasserstein Distance is `0.2000`. This measures the minimum cost of shifting probability mass to match distributions. Tracking both PSI and Wasserstein Distance provides robust multi-dimensional alerts for data shift detection in production."
    ))
    
    nb.cells = cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook draft: {notebook_path}")

def execute_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved notebook: {notebook_path}")

if __name__ == "__main__":
    create_and_execute_notebook_07()
    create_and_execute_notebook_08()
    create_and_execute_notebook_09()
    
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    execute_notebook(os.path.join(base_dir, "notebooks", "07_attention_and_transformer_prerequisites.ipynb"))
    execute_notebook(os.path.join(base_dir, "notebooks", "08_nlp_evaluation.ipynb"))
    execute_notebook(os.path.join(base_dir, "notebooks", "09_production_nlp.ipynb"))
    print("ALL BATCH 4 NOTEBOOKS GENERATED AND EXECUTED SUCCESSFULLY.")
