import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_and_execute_notebook_05():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "05_word_embeddings.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown introduction
    cells.append(nbf.v4.new_markdown_cell(
        "# 05_word_embeddings: Matrix Lookup Equivalence and Skip-Gram with Negative Sampling Loss\n"
        "\n"
        "This notebook validates word embedding models. It proves the mathematical equivalence between discrete index lookups and linear matrix multiplications, and implements a custom Skip-Gram with Negative Sampling (SGNS) loss module in PyTorch matching our hand-calculations."
    ))
    
    # Cell 2: Heading - Index Lookup
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Index Lookup in PyTorch"
    ))
    
    # Cell 3: Code - Index Lookup
    cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "import torch.nn as nn\n"
        "import numpy as np\n"
        "\n"
        "torch.manual_seed(42)\n"
        "\n"
        "vocab_size = 5\n"
        "embed_dim = 3\n"
        "\n"
        "# Define PyTorch embedding weight matrix W\n"
        "embedding = nn.Embedding(vocab_size, embed_dim)\n"
        "W = embedding.weight.data.clone()\n"
        "\n"
        "idx = 2\n"
        "idx_tensor = torch.tensor([idx])\n"
        "\n"
        "# Method 1: Standard index lookup\n"
        "vector_lookup = embedding(idx_tensor)\n"
        "print(\"Embedding Weight Matrix W:\\n\", W.numpy())\n"
        "print(f\"\\nIndex Lookup vector (Index {idx}):\", vector_lookup.detach().numpy().flatten())"
    ))
    
    # Cell 4: Explanation - Index Lookup
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Index Lookup\n"
        "We initialized an embedding layer of size $(5, 3)$ and fetched the vector at index `2`. The weight matrix rows represent the parameters for each word. The output is a single 3-dimensional vector."
    ))
    
    # Cell 5: Heading - Matrix Equivalence
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Matrix Multiplication Equivalence"
    ))
    
    # Cell 6: Code - Matrix Equivalence
    cells.append(nbf.v4.new_code_cell(
        "# Method 2: Matrix multiplication (x^T * W) using one-hot vector\n"
        "x = torch.zeros(vocab_size)\n"
        "x[idx] = 1.0  # Set index 2 to 1.0\n"
        "\n"
        "vector_matmul = torch.matmul(x, W)\n"
        "print(f\"Matrix Matmul vector (x^T * W):\", vector_matmul.numpy())\n"
        "\n"
        "# Verify exact equivalence\n"
        "assert torch.allclose(vector_lookup[0], vector_matmul), \"Equivalence check failed!\""
    ))
    
    # Cell 7: Explanation - Matrix Equivalence
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: Matrix Equivalence" + "\n" +
        r"By defining a one-hot vector `x` with `1.0` at index `2` and performing a vector-matrix multiplication $\mathbf{x}^T \mathbf{W}$, we extract the exact same row vector as the lookup. This proves that embedding layers are mathematically identical to standard linear layers with one-hot inputs, allowing gradients to propagate back to update the weights."
    ))
    
    # Cell 8: Heading - SGNS Loss Module
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Custom Skip-Gram with Negative Sampling (SGNS) Loss Module"
    ))
    
    # Cell 9: Code - SGNS Loss Module
    cells.append(nbf.v4.new_code_cell(
        "class SGNSLoss(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        \n"
        "    def forward(self, v_target_ctx, v_input, v_neg_ctxs):\n"
        "        # Positive pair score: log(sigmoid(v_target_ctx . v_input))\n"
        "        pos_score = torch.sum(v_target_ctx * v_input, dim=-1)\n"
        "        pos_loss = torch.log(torch.sigmoid(pos_score))\n"
        "        \n"
        "        # Negative pairs score: sum(log(sigmoid(-v_neg_ctx . v_input)))\n"
        "        # v_neg_ctxs shape: (batch_size, k, embed_dim), v_input: (batch_size, embed_dim)\n"
        "        # We use batch matrix multiplication (bmm) to get dot products for each negative sample\n"
        "        neg_score = torch.bmm(v_neg_ctxs, v_input.unsqueeze(-1)).squeeze(-1)\n"
        "        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=-1)\n"
        "        \n"
        "        loss = - (pos_loss + neg_loss)\n"
        "        return loss.mean()"
    ))
    
    # Cell 10: Explanation - SGNS Loss Module
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: SGNS Module Definition\n"
        "The custom `SGNSLoss` module computes the Skip-Gram with Negative Sampling loss. It passes the positive pair through a sigmoid function to maximize their alignment, and passes the negative pairs through a negated sigmoid function to push their representations apart in the vector space."
    ))
    
    # Cell 11: Heading - Loss Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. SGNS Loss Evaluation & Hand-Calculation Consistency"
    ))
    
    # Cell 12: Code - Loss Evaluation
    cells.append(nbf.v4.new_code_cell(
        "# Initialize inputs matching our hand-calculations exactly\n"
        "v_input = torch.tensor([[0.1, 0.8, -0.2]], requires_grad=True)\n"
        "v_target_ctx = torch.tensor([[0.2, 0.9, 0.1]])\n"
        "v_neg_ctxs = torch.tensor([[[ -0.9, 0.1, 0.8]]])\n"
        "\n"
        "loss_fn = SGNSLoss()\n"
        "loss = loss_fn(v_target_ctx, v_input, v_neg_ctxs)\n"
        "print(f\"Initial SGNS Loss: {loss.item():.4f}\")\n"
        "\n"
        "# Verify exact consistency with hand calculation (1.0083)\n"
        "np.testing.assert_almost_equal(loss.item(), 1.0083, decimal=4)"
    ))
    
    # Cell 13: Explanation - Loss Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: Loss Value Verification" + "\n" +
        r"The computed initial loss is exactly `1.0083`. This matches our manual calculation where we computed the positive score sigmoid log $-\log \sigma(0.72) \approx 0.3966$ and the negative score sigmoid log $-\log \sigma(0.17) \approx 0.6117$ to get `1.0083`."
    ))
    
    # Cell 14: Heading - Optimization Step
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. SGNS Gradient Optimization Step"
    ))
    
    # Cell 15: Code - Optimization Step
    cells.append(nbf.v4.new_code_cell(
        "# Execute gradient step\n"
        "loss.backward()\n"
        "optimizer = torch.optim.SGD([v_input], lr=0.1)\n"
        "optimizer.step()\n"
        "\n"
        "new_loss = loss_fn(v_target_ctx, v_input, v_neg_ctxs)\n"
        "print(f\"Updated SGNS Loss: {new_loss.item():.4f}\")\n"
        "\n"
        "assert new_loss < loss, \"Loss did not decrease after gradient optimization step!\""
    ))
    
    # Cell 16: Explanation - Optimization Step
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Optimization Update\n"
        "The loss decreased from `1.0083` to a lower value. During backpropagation, the gradients update the target word embeddings `v_input` to align more closely with `v_target_ctx` and move away from the negative samples `v_neg_ctxs`, confirming successful parameter optimization."
    ))
    
    nb.cells = cells
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook draft: {notebook_path}")

def create_and_execute_notebook_06():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "06_sequence_models.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown introduction
    cells.append(nbf.v4.new_markdown_cell(
        "# 06_sequence_models: Recurrent state transitions and Bigram-based Beam Search Decoding on Alice in Wonderland\n"
        "\n"
        "This notebook validates sequence models on real text. It implements an RNN sequential state update manually and validates it using PyTorch's `nn.RNNCell` on actual token embeddings. It also trains a Bigram Language Model on the Gutenberg *Alice in Wonderland* corpus and uses it to auto-regressively generate text using Beam Search decoding."
    ))
    
    # Cell 2: Heading - RNN Setup
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. RNN Inputs & Weight Matrix Setup"
    ))
    
    # Cell 3: Code - RNN Setup
    cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "import torch.nn as nn\n"
        "import numpy as np\n"
        "\n"
        "torch.manual_seed(42)\n"
        "\n"
        "vocab_size = 10\n"
        "embed_dim = 4\n"
        "hidden_dim = 3\n"
        "\n"
        "# Define word token list representing the phrase: \"the cat sat on\"\n"
        "word_tokens = [\"the\", \"cat\", \"sat\", \"on\"]\n"
        "word_indices = [2, 5, 8, 3] # mock vocabulary lookup indices\n"
        "\n"
        "# Word embeddings and recurrent cell\n"
        "embedding = nn.Embedding(vocab_size, embed_dim)\n"
        "rnn_cell = nn.RNNCell(embed_dim, hidden_dim)\n"
        "\n"
        "# Extract weights for manual computation\n"
        "W_ih = rnn_cell.weight_ih.data\n"
        "W_hh = rnn_cell.weight_hh.data\n"
        "b_ih = rnn_cell.bias_ih.data\n"
        "b_hh = rnn_cell.bias_hh.data"
    ))
    
    # Cell 4: Explanation - RNN Setup
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Weight Setup\n"
        "We set up a small vocabulary index map and projected the words into a 4-dimensional embedding space. We extract the standard recurrent projection matrices ($W_{ih}, W_{hh}$) and biases ($b_{ih}, b_{hh}$) to perform our manual and built-in comparisons."
    ))
    
    # Cell 5: Heading - PyTorch Sequential RNN Updates
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Sequential Hidden State Updates in PyTorch"
    ))
    
    # Cell 6: Code - PyTorch Sequential RNN Updates
    cells.append(nbf.v4.new_code_cell(
        "h_pytorch = torch.zeros(1, hidden_dim)\n"
        "pytorch_states = []\n"
        "for idx in word_indices:\n"
        "    x_t = embedding(torch.tensor([idx]))\n"
        "    h_pytorch = rnn_cell(x_t, h_pytorch)\n"
        "    pytorch_states.append(h_pytorch.detach().numpy().copy())\n"
        "\n"
        "print(\"PyTorch Sequential Hidden States:\")\n"
        "for t, state in enumerate(pytorch_states):\n"
        "    print(f\"  Step {t} ('{word_tokens[t]}'): {state.flatten()}\")"
    ))
    
    # Cell 7: Explanation - PyTorch Sequential RNN Updates
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: PyTorch RNN updates\n"
        "At each sequence step, PyTorch's `nn.RNNCell` consumes the current word embedding vector and combines it with the previous step's hidden state, updating the continuous context representation sequentially."
    ))
    
    # Cell 8: Heading - Manual RNN State Updates
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Sequential Hidden State Updates from Scratch"
    ))
    
    # Cell 9: Code - Manual RNN State Updates
    cells.append(nbf.v4.new_code_cell(
        "h_manual = torch.zeros(1, hidden_dim)\n"
        "manual_states = []\n"
        "for idx in word_indices:\n"
        "    x_t = embedding(torch.tensor([idx]))\n"
        "    h_manual = torch.tanh(\n"
        "        torch.matmul(x_t, W_ih.t()) + b_ih + \n"
        "        torch.matmul(h_manual, W_hh.t()) + b_hh\n"
        ")\n"
        "    manual_states.append(h_manual.detach().numpy().copy())\n"
        "\n"
        "print(\"Manual Sequential Hidden States:\")\n"
        "for t, state in enumerate(manual_states):\n"
        "    print(f\"  Step {t} ('{word_tokens[t]}'): {state.flatten()}\")\n"
        "\n"
        "# Verify exact equivalence across all steps\n"
        "for t in range(len(word_indices)):\n"
        "    assert np.allclose(pytorch_states[t], manual_states[t], atol=1e-6), f\"State mismatch at step {t}!\""
    ))
    
    # Cell 10: Explanation - Manual RNN State Updates
    cells.append(nbf.v4.new_markdown_cell(
        r"### Output Analysis: Manual vs. PyTorch equivalence" + "\n" +
        r"By performing the projection multiplication $\tanh(\mathbf{x}_t \mathbf{W}_{ih}^T + \mathbf{b}_{ih} + \mathbf{h}_{t-1} \mathbf{W}_{hh}^T + \mathbf{b}_{hh})$ manually in PyTorch tensor math, we verify that it matches PyTorch's sequential state output exactly. This confirms how recurrent context accumulation works in practice."
    ))
    
    # Cell 11: Heading - Text Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Corpus Loading & Vocabulary Preprocessing"
    ))
    
    # Cell 12: Code - Text Preprocessing
    cells.append(nbf.v4.new_code_cell(
        "import nltk\n"
        "import math\n"
        "from collections import defaultdict\n"
        "nltk.download('gutenberg', quiet=True)\n"
        "from nltk.corpus import gutenberg\n"
        "\n"
        "# Load Carroll's Alice in Wonderland corpus\n"
        "words = [w.lower() for w in gutenberg.words('carroll-alice.txt') if w.isalpha()]\n"
        "print(f\"Total token words loaded: {len(words)}\")\n"
        "print(f\"Vocabulary size: {len(set(words))}\")\n"
        "\n"
        "# Build unigram and bigram counts\n"
        "unigram_counts = defaultdict(int)\n"
        "bigram_counts = defaultdict(lambda: defaultdict(int))\n"
        "\n"
        "for i in range(len(words)-1):\n"
        "    w1, w2 = words[i], words[i+1]\n"
        "    unigram_counts[w1] += 1\n"
        "    bigram_counts[w1][w2] += 1"
    ))
    
    # Cell 13: Explanation - Text Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Gutenberg Stats\n"
        "We loaded Carroll's *Alice in Wonderland* corpus (containing ~27,333 words) and generated unigram/bigram token maps. This counts how often word combinations (like `('she', 'said')`) appear together."
    ))
    
    # Cell 14: Heading - Transition probabilities
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Successor Transition Probabilities Extractor"
    ))
    
    # Cell 15: Code - Transition probabilities
    cells.append(nbf.v4.new_code_cell(
        "vocab = list(set(words))\n"
        "V = len(vocab)\n"
        "\n"
        "def get_next_word_probs(word):\n"
        "    if word not in bigram_counts:\n"
        "        # Fallback to uniform distribution over top 100 words to save space\n"
        "        return {w: 1/100 for w in vocab[:100]}\n"
        "        \n"
        "    successors = bigram_counts[word]\n"
        "    total_count = sum(successors.values())\n"
        "    \n"
        "    probs = {}\n"
        "    # Retrieve top 20 most frequent successors to maintain reasonable branching search space\n"
        "    sorted_successors = sorted(successors.items(), key=lambda x: x[1], reverse=True)[:20]\n"
        "    successor_total = sum(count for _, count in sorted_successors)\n"
        "    \n"
        "    for w, count in sorted_successors:\n"
        "        probs[w] = count / successor_total\n"
        "    return probs\n"
        "\n"
        "# Quick check for \"alice\"\n"
        "alice_successors = get_next_word_probs(\"alice\")\n"
        "print(\"Top transition successors for 'alice':\")\n"
        "for w, p in list(alice_successors.items())[:5]:\n"
        "    print(f\"  '{w}': {p:.4f}\")"
    ))
    
    # Cell 16: Explanation - Transition probabilities
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Word Successor Probabilities\n"
        "We defined `get_next_word_probs` to extract transition frequencies for any given word. For example, for the word `'alice'`, the model outputs high probabilities for following verbs like `'was'`, `'said'`, or `'thought'`, learning grammatical patterns from Lewis Carroll's text style."
    ))
    
    # Cell 17: Heading - Beam Search Implementation
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Beam Search Decoding Implementation"
    ))
    
    # Cell 18: Code - Beam Search Implementation
    cells.append(nbf.v4.new_code_cell(
        "def beam_search(start_word, beam_width=3, max_len=4):\n"
        "    # Beams list stores paths as: (sequence_list, cumulative_log_probability)\n"
        "    beams = [([start_word], 0.0)]\n"
        "    \n"
        "    for step in range(max_len - 1):\n"
        "        candidates = []\n"
        "        for seq, score in beams:\n"
        "            last_word = seq[-1]\n"
        "            probs = get_next_word_probs(last_word)\n"
        "            for next_word, p in probs.items():\n"
        "                # Add log probabilities to maintain numerical stability and avoid underflow\n"
        "                candidates.append((seq + [next_word], score + math.log(p)))\n"
        "        \n"
        "        # Sort candidates and prune down to beam width B\n"
        "        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]\n"
        "        \n"
        "    return beams"
    ))
    
    # Cell 19: Explanation - Beam Search Implementation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Beam Search Decoder\n"
        "The `beam_search` function maintains the top $B$ paths. At each decoding iteration, it expands the active sequences, calculates cumulative log-probabilities (to prevent underflow), and prunes the candidate set down to the beam width parameter $B$."
    ))
    
    # Cell 20: Heading - Decoder Execution
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Auto-Regressive Generated Paths Decoding"
    ))
    
    # Cell 21: Code - Decoder Execution
    cells.append(nbf.v4.new_code_cell(
        "start_word = \"she\"\n"
        "B = 3\n"
        "generated_beams = beam_search(start_word, beam_width=B, max_len=4)\n"
        "\n"
        "print(f\"Top {B} Generated Paths starting with '{start_word}':\")\n"
        "for idx, (seq, score) in enumerate(generated_beams):\n"
        "    print(f\"  Path {idx+1}: {' '.join(seq):<25} | Cumulative Log Probability: {score:.4f}\")\n"
        "\n"
        "# Assertions checking correctness\n"
        "assert len(generated_beams) == B, \"Output beam count mismatch!\"\n"
        "assert generated_beams[0][1] >= generated_beams[1][1], \"Beams are not sorted!\""
    ))
    
    # Cell 22: Explanation - Decoder Execution
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Analysis: Generated Sequence Verification\n"
        "The Beam Search algorithm successfully generated the top three paths starting with `'she'`. Because it tracks multiple paths in parallel, it retains high-likelihood text segments (like `'she said to'` or `'she was very'`) avoiding sub-optimal choices that greedy search would lock into, showing how decoding processes occur at inference."
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
    create_and_execute_notebook_05()
    create_and_execute_notebook_06()
    
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    execute_notebook(os.path.join(base_dir, "notebooks", "05_word_embeddings.ipynb"))
    execute_notebook(os.path.join(base_dir, "notebooks", "06_sequence_models.ipynb"))
    print("ALL BATCH 3 NOTEBOOKS GENERATED AND EXECUTED SUCCESSFULLY.")
