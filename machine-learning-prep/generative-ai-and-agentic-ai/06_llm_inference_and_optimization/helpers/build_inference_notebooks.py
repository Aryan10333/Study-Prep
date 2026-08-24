import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")


def run_and_save(nb, notebook_out_path, timeout=900):
    os.makedirs(os.path.dirname(notebook_out_path), exist_ok=True)
    nb["metadata"] = {
        "kernelspec": {"display_name": "prep-venv", "language": "python", "name": "prep-venv"},
        "language_info": {"name": "python"},
    }
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="prep-venv")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_out_path) or "."}})
    with open(notebook_out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved: {notebook_out_path}")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================================
# Notebook 01: Decode vs. Prefill Fundamentals -- Real TTFT/TPOT Measurement
# ============================================================================

def build_01_decode_vs_prefill_fundamentals():
    cells = []

    cells.append(md(
        "# Notebook 01: Decode vs. Prefill Fundamentals -- Real TTFT/TPOT Measurement\n"
        "\n"
        "`[REAL]` Companion to Module 01. Real experiments on a local RTX 4060 Laptop GPU with `Qwen/Qwen2.5-0.5B-Instruct`.\n"
        "\n"
        "**This notebook tests a hypothesis, not an expected result** (per the signed-off Track 2 plan): Module 01's roofline "
        "hand calc used an illustrative large-model/datacenter-GPU profile. Whether the canonical memory-bandwidth-bound decode "
        "signature (latency roughly flat as sequence length grows) actually reproduces on this real, much smaller model and "
        "consumer-class GPU is an open, real empirical question this notebook answers directly -- not assumed in advance."
    ))

    cells.append(code(
        "import time\n"
        "import statistics\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "print(f\"Device: {DEVICE}\")\n"
        "if DEVICE == \"cuda\":\n"
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")\n"
        "\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)\n"
        "model.eval()\n"
        "print(f\"Loaded {MODEL_NAME} at FP16 on {DEVICE}\")"
    ))

    cells.append(md(
        "## 1. Measurement Methodology (Revised Per Signed-Off Plan)\n"
        "\n"
        "`[REAL]` A single wall-clock sample on a laptop GPU is unreliable (thermal throttling, background processes, async "
        "CUDA queue). For every timed configuration below: (1) one real warm-up pass, discarded, to exclude CUDA-context/"
        "kernel-compilation startup cost; (2) multiple real repeated timed runs, each bracketed by explicit "
        "`torch.cuda.synchronize()` calls so `time.perf_counter()` measures real completed GPU work, not just kernel-launch "
        "time; (3) report **median and p95** across repeats, not a single sample."
    ))

    cells.append(code(
        "N_REPEATS = 8\n"
        "N_WARMUP = 1\n"
        "\n"
        "def timed_repeats(fn, n_repeats=N_REPEATS, n_warmup=N_WARMUP):\n"
        "    \"\"\"Runs fn() n_warmup times (discarded), then n_repeats times, with CUDA sync bracketing each run.\n"
        "    Returns (median_seconds, p95_seconds, all_samples).\"\"\"\n"
        "    for _ in range(n_warmup):\n"
        "        fn()\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "\n"
        "    samples = []\n"
        "    for _ in range(n_repeats):\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "        start = time.perf_counter()\n"
        "        fn()\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "        elapsed = time.perf_counter() - start\n"
        "        samples.append(elapsed)\n"
        "\n"
        "    median = statistics.median(samples)\n"
        "    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]\n"
        "    return median, p95, samples\n"
        "\n"
        "print(f\"Methodology ready: {N_WARMUP} warm-up + {N_REPEATS} timed repeats per configuration, median/p95 reported.\")"
    ))

    cells.append(md(
        "## 2. Real TTFT (Time to First Token) vs. Prompt Length\n"
        "\n"
        "`[REAL]` TTFT is dominated by the real prefill pass. Measured across a range of real prompt lengths by generating "
        "exactly 1 new token (isolating prefill from any decode-loop cost)."
    ))

    ttft_cell_index = len(cells)
    cells.append(code(
        "PROMPT_LENGTHS = [32, 128, 512, 1024]\n"
        "BASE_TEXT = \"The quick brown fox jumps over the lazy dog. \" * 200\n"
        "\n"
        "def make_prompt(n_tokens):\n"
        "    ids = tokenizer(BASE_TEXT, return_tensors=\"pt\").input_ids[0]\n"
        "    ids = ids[:n_tokens] if len(ids) >= n_tokens else ids.repeat((n_tokens // len(ids)) + 1)[:n_tokens]\n"
        "    return ids.unsqueeze(0).to(DEVICE)\n"
        "\n"
        "ttft_results = []\n"
        "for n_tok in PROMPT_LENGTHS:\n"
        "    input_ids = make_prompt(n_tok)\n"
        "    actual_len = input_ids.shape[1]\n"
        "\n"
        "    def run_prefill():\n"
        "        with torch.no_grad():\n"
        "            model.generate(input_ids, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)\n"
        "\n"
        "    median_s, p95_s, samples = timed_repeats(run_prefill)\n"
        "    ttft_results.append({\"prompt_len\": actual_len, \"median_ms\": median_s * 1000, \"p95_ms\": p95_s * 1000})\n"
        "    print(f\"Prompt length {actual_len:5d} tok: median TTFT = {median_s*1000:7.2f} ms, p95 = {p95_s*1000:7.2f} ms\")\n"
        "\n"
        "print(\"\\n(pending real output)\")"
    ))

    cells.append(md(
        "## 3. Real TPOT (Time per Output Token) vs. Generation Length\n"
        "\n"
        "`[REAL]` TPOT is the real per-token decode cost. Measured by generating varying numbers of new tokens from a fixed "
        "real prompt, then dividing total generation time by tokens generated (an average TPOT across the run)."
    ))

    tpot_cell_index = len(cells)
    cells.append(code(
        "GEN_LENGTHS = [16, 64, 128, 256]\n"
        "FIXED_PROMPT_LEN = 64\n"
        "fixed_input_ids = make_prompt(FIXED_PROMPT_LEN)\n"
        "\n"
        "tpot_results = []\n"
        "for n_new in GEN_LENGTHS:\n"
        "    def run_decode():\n"
        "        with torch.no_grad():\n"
        "            model.generate(fixed_input_ids, max_new_tokens=n_new, min_new_tokens=n_new,\n"
        "                           do_sample=False, pad_token_id=tokenizer.eos_token_id)\n"
        "\n"
        "    median_s, p95_s, samples = timed_repeats(run_decode, n_repeats=5)\n"
        "    tpot_per_token_median_ms = (median_s * 1000) / n_new\n"
        "    tpot_per_token_p95_ms = (p95_s * 1000) / n_new\n"
        "    tpot_results.append({\n"
        "        \"n_new_tokens\": n_new,\n"
        "        \"total_median_ms\": median_s * 1000,\n"
        "        \"tpot_median_ms\": tpot_per_token_median_ms,\n"
        "        \"tpot_p95_ms\": tpot_per_token_p95_ms,\n"
        "    })\n"
        "    print(f\"Generated {n_new:4d} tokens: total median = {median_s*1000:8.2f} ms, \"\n"
        "          f\"avg TPOT/token median = {tpot_per_token_median_ms:6.3f} ms, p95 = {tpot_per_token_p95_ms:6.3f} ms\")\n"
        "\n"
        "print(\"\\n(pending real output)\")"
    ))

    cells.append(md(
        "## 4. Hypothesis Check: Does TPOT Stay Roughly Flat as Generation Length Grows?\n"
        "\n"
        "`[REAL]` The memory-bandwidth-bound decode signature predicts per-token TPOT should stay roughly constant regardless "
        "of how many tokens have already been generated (each decode step re-reads the same weights + a growing-but-"
        "proportionally-small KV cache). This cell checks that directly against this notebook's own real measured numbers -- "
        "reported honestly whichever way it comes out, per the signed-off plan's hypothesis framing."
    ))

    hypothesis_cell_index = len(cells)
    cells.append(code(
        "tpot_values = [r[\"tpot_median_ms\"] for r in tpot_results]\n"
        "tpot_min, tpot_max = min(tpot_values), max(tpot_values)\n"
        "relative_spread_pct = (tpot_max - tpot_min) / tpot_min * 100\n"
        "\n"
        "print(f\"TPOT median across generation lengths {GEN_LENGTHS}: {[round(v, 3) for v in tpot_values]} ms/token\")\n"
        "print(f\"Min: {tpot_min:.3f} ms, Max: {tpot_max:.3f} ms, Relative spread: {relative_spread_pct:.1f}%\")\n"
        "\n"
        "ttft_values = [r[\"median_ms\"] for r in ttft_results]\n"
        "print(f\"\\nTTFT median across prompt lengths {PROMPT_LENGTHS}: {[round(v, 2) for v in ttft_values]} ms\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "01_decode_vs_prefill_fundamentals.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "ttft_cell_index": ttft_cell_index,
        "tpot_cell_index": tpot_cell_index,
        "hypothesis_cell_index": hypothesis_cell_index,
    }


# ============================================================================
# Notebook 02: Real KV Cache Memory Measurement vs. Formula
# ============================================================================

def build_02_kv_cache_memory_measurement():
    cells = []

    cells.append(md(
        "# Notebook 02: Real KV Cache Memory Measurement vs. Formula\n"
        "\n"
        "`[REAL]` Companion to Module 02. Real experiments on the RTX 4060 with `Qwen/Qwen2.5-0.5B-Instruct` -- a real model "
        "that itself uses **Grouped-Query Attention** (confirmed via its real config: 14 query heads, only 2 real KV heads), "
        "making this a direct, real test of Module 02's GQA memory formula, not just an MHA illustration.\n"
        "\n"
        "**Revised measurement approach (per signed-off plan):** `torch.cuda.memory_allocated()` alone is *not* used as a "
        "stand-in for KV-cache memory -- it includes weight memory, activations, and other allocations. This notebook instead "
        "(1) computes the real, exact KV-cache tensor memory directly from `past_key_values` tensor shapes (ground truth, not "
        "an allocator-level proxy), and (2) separately reports real `memory_allocated()` vs. `memory_reserved()` deltas across "
        "sequence lengths to directly expose real PyTorch CUDA-allocator overhead."
    ))

    cells.append(code(
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig\n"
        "\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "\n"
        "config = AutoConfig.from_pretrained(MODEL_NAME)\n"
        "N_LAYERS = config.num_hidden_layers\n"
        "N_KV_HEADS = config.num_key_value_heads\n"
        "N_Q_HEADS = config.num_attention_heads\n"
        "D_HEAD = config.hidden_size // config.num_attention_heads\n"
        "print(f\"Real config: n_layers={N_LAYERS}, n_query_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS}, d_head={D_HEAD}\")\n"
        "print(f\"This model IS a real GQA model: {N_KV_HEADS} KV heads share across {N_Q_HEADS} query heads \"\n"
        "      f\"({N_Q_HEADS // N_KV_HEADS}x sharing ratio)\")\n"
        "\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)\n"
        "model.eval()\n"
        "print(f\"\\nLoaded {MODEL_NAME} at FP16 on {DEVICE}\")"
    ))

    cells.append(md(
        "## 1. Module 02's Formula, Applied to This Model's Real Config\n"
        "\n"
        "`[COMPUTED FROM REAL DATA]` Module 02's formula: $\\text{Mem}_{KV} = 2 \\times B \\times L \\times N_{layers} \\times "
        "N_{KV\\_heads} \\times d_{head} \\times \\text{bytes}_{dtype}$, evaluated with this model's own real, extracted config "
        "values (not the illustrative 7B numbers used in the module itself)."
    ))

    formula_cell_index = len(cells)
    cells.append(code(
        "def kv_cache_formula_bytes(batch_size, seq_len, n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS,\n"
        "                            d_head=D_HEAD, bytes_per_elem=2):\n"
        "    return 2 * batch_size * seq_len * n_layers * n_kv_heads * d_head * bytes_per_elem\n"
        "\n"
        "SEQ_LENGTHS = [128, 512, 1024, 2048]\n"
        "BATCH_SIZE = 1\n"
        "\n"
        "for L in SEQ_LENGTHS:\n"
        "    predicted_bytes = kv_cache_formula_bytes(BATCH_SIZE, L)\n"
        "    print(f\"L={L:5d}: formula-predicted KV cache = {predicted_bytes:,} bytes ({predicted_bytes/1024/1024:.3f} MB)\")\n"
        "\n"
        "print(\"\\n(pending real measured comparison)\")"
    ))

    cells.append(md(
        "## 2. Real Ground-Truth Measurement: Exact KV-Cache Tensor Memory from `past_key_values`\n"
        "\n"
        "`[REAL]` Running a real forward pass with `use_cache=True` at each real sequence length and summing the exact real "
        "byte size of every returned key/value tensor (`numel() * element_size()`, summed across all layers) -- a direct, "
        "exact measurement of the real KV-cache tensor memory PyTorch actually allocated for it, not an allocator-level proxy."
    ))

    ground_truth_cell_index = len(cells)
    cells.append(code(
        "def extract_kv_tensors(past_kv):\n"
        "    \"\"\"Version-robust extraction across transformers' evolving Cache API.\"\"\"\n"
        "    if hasattr(past_kv, \"layers\"):  # transformers >=4.5x: Cache.layers[i].keys/.values\n"
        "        tensors = []\n"
        "        for layer in past_kv.layers:\n"
        "            tensors.append(layer.keys)\n"
        "            tensors.append(layer.values)\n"
        "        return tensors\n"
        "    if hasattr(past_kv, \"key_cache\"):  # older Cache API: separate key_cache/value_cache lists\n"
        "        return list(past_kv.key_cache) + list(past_kv.value_cache)\n"
        "    return [t for layer in past_kv for t in layer]  # legacy tuple-of-tuples API\n"
        "\n"
        "def real_kv_cache_tensor_bytes(seq_len, batch_size=1):\n"
        "    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=DEVICE)\n"
        "    with torch.no_grad():\n"
        "        outputs = model(input_ids, use_cache=True)\n"
        "    tensors = extract_kv_tensors(outputs.past_key_values)\n"
        "    total_bytes = sum(t.numel() * t.element_size() for t in tensors)\n"
        "    del outputs\n"
        "    return total_bytes\n"
        "\n"
        "ground_truth_results = []\n"
        "for L in SEQ_LENGTHS:\n"
        "    torch.cuda.empty_cache() if DEVICE == \"cuda\" else None\n"
        "    measured_bytes = real_kv_cache_tensor_bytes(L, BATCH_SIZE)\n"
        "    predicted_bytes = kv_cache_formula_bytes(BATCH_SIZE, L)\n"
        "    ground_truth_results.append({\"seq_len\": L, \"measured_bytes\": measured_bytes, \"predicted_bytes\": predicted_bytes})\n"
        "    match = \"EXACT MATCH\" if measured_bytes == predicted_bytes else f\"diff={measured_bytes - predicted_bytes:+,} bytes\"\n"
        "    print(f\"L={L:5d}: measured={measured_bytes:,} bytes, predicted={predicted_bytes:,} bytes -- {match}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 3. Real CUDA Allocator Overhead: Allocated vs. Reserved Memory\n"
        "\n"
        "`[REAL]` Separately from the exact tensor-level measurement above, this section measures real "
        "`torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` *deltas* across real sequence lengths -- isolating "
        "KV-cache-attributable growth from the fixed real weight/activation baseline, and directly exposing the real, honest "
        "gap between what's allocated and what the CUDA caching allocator has reserved (a real, distinct overhead source from "
        "the exact tensor size measured in Section 2)."
    ))

    allocator_cell_index = len(cells)
    cells.append(code(
        "if DEVICE == \"cuda\":\n"
        "    torch.cuda.empty_cache()\n"
        "    torch.cuda.reset_peak_memory_stats()\n"
        "    baseline_allocated = torch.cuda.memory_allocated()\n"
        "    baseline_reserved = torch.cuda.memory_reserved()\n"
        "    print(f\"Baseline (model loaded, no forward pass yet): allocated={baseline_allocated:,} bytes, \"\n"
        "          f\"reserved={baseline_reserved:,} bytes\")\n"
        "\n"
        "    allocator_results = []\n"
        "    for L in SEQ_LENGTHS:\n"
        "        torch.cuda.empty_cache()\n"
        "        torch.cuda.reset_peak_memory_stats()\n"
        "        pre_allocated = torch.cuda.memory_allocated()\n"
        "        input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, L), device=DEVICE)\n"
        "        with torch.no_grad():\n"
        "            outputs = model(input_ids, use_cache=True)\n"
        "        post_allocated = torch.cuda.memory_allocated()\n"
        "        post_reserved = torch.cuda.memory_reserved()\n"
        "        allocated_delta = post_allocated - pre_allocated\n"
        "        reserved_minus_allocated = post_reserved - post_allocated\n"
        "        allocator_results.append({\n"
        "            \"seq_len\": L, \"allocated_delta\": allocated_delta,\n"
        "            \"reserved\": post_reserved, \"allocated\": post_allocated,\n"
        "            \"reserved_minus_allocated\": reserved_minus_allocated,\n"
        "        })\n"
        "        print(f\"L={L:5d}: allocated_delta={allocated_delta:,} bytes, \"\n"
        "              f\"reserved-allocated gap={reserved_minus_allocated:,} bytes\")\n"
        "        del outputs\n"
        "else:\n"
        "    print(\"CUDA not available -- skipping allocator-level measurement.\")\n"
        "    allocator_results = []\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "02_kv_cache_memory_measurement.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "formula_cell_index": formula_cell_index,
        "ground_truth_cell_index": ground_truth_cell_index,
        "allocator_cell_index": allocator_cell_index,
    }


# ============================================================================
# Notebook 03: FlashAttention/SDPA Backend Comparison -- Real IO-Aware Latency & Memory
# ============================================================================

def build_03_flashattention_sdpa_backend_comparison():
    cells = []

    cells.append(md(
        "# Notebook 03: FlashAttention/SDPA Backend Comparison -- Real IO-Aware Latency & Memory\n"
        "\n"
        "`[REAL]` Companion to Module 04. Uses `torch.nn.functional.scaled_dot_product_attention`'s real, selectable "
        "backends via `torch.nn.attention.sdpa_kernel` on the real RTX 4060.\n"
        "\n"
        "**Scope of the claim (per signed-off plan):** this notebook measures a real **backend performance difference** "
        "(latency, peak memory) between SDPA kernels -- it does **not** claim to directly measure real HBM read/write "
        "traffic. Module 04's IO-complexity argument is the real, established *mechanism*; this notebook's numbers are "
        "consistent with that mechanism's real, expected consequences, but are not themselves an HBM-traffic measurement "
        "(that would require GPU-level profiling tools such as Nsight Compute, out of scope here)."
    ))

    cells.append(code(
        "import time\n"
        "import statistics\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from torch.nn.attention import sdpa_kernel, SDPBackend\n"
        "\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "print(f\"Device: {DEVICE}\")\n"
        "if DEVICE == \"cuda\":\n"
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n"
        "print(f\"torch version: {torch.__version__}\")"
    ))

    cells.append(md(
        "## 1. Real Backend-Availability Check (Honest Fallback, Per Signed-Off Plan)\n"
        "\n"
        "`[REAL]` Checking which real SDPA backends this exact PyTorch build actually supports on this GPU before running "
        "any timed comparison, using this model's real attention dimensions (14 heads, head_dim=64, matching "
        "`Qwen2.5-0.5B-Instruct`'s real query-head config from Notebook 02)."
    ))

    availability_cell_index = len(cells)
    cells.append(code(
        "torch.manual_seed(0)\n"
        "N_HEADS, D_HEAD = 14, 64\n"
        "\n"
        "def make_qkv(batch_size, seq_len, n_heads=N_HEADS, d_head=D_HEAD):\n"
        "    shape = (batch_size, n_heads, seq_len, d_head)\n"
        "    q = torch.randn(shape, device=DEVICE, dtype=torch.float16)\n"
        "    k = torch.randn(shape, device=DEVICE, dtype=torch.float16)\n"
        "    v = torch.randn(shape, device=DEVICE, dtype=torch.float16)\n"
        "    return q, k, v\n"
        "\n"
        "test_q, test_k, test_v = make_qkv(1, 512)\n"
        "backend_availability = {}\n"
        "for name, backend in [(\"FLASH_ATTENTION\", SDPBackend.FLASH_ATTENTION),\n"
        "                      (\"EFFICIENT_ATTENTION\", SDPBackend.EFFICIENT_ATTENTION),\n"
        "                      (\"MATH\", SDPBackend.MATH)]:\n"
        "    try:\n"
        "        with torch.no_grad(), sdpa_kernel(backend):\n"
        "            _ = F.scaled_dot_product_attention(test_q, test_k, test_v, is_causal=True)\n"
        "        backend_availability[name] = True\n"
        "        print(f\"{name}: available on this real hardware/build\")\n"
        "    except RuntimeError as e:\n"
        "        backend_availability[name] = False\n"
        "        print(f\"{name}: NOT available on this real hardware/build -- {e}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation and backend selection for the timed comparison below)\")"
    ))

    cells.append(md(
        "## 2. Real Latency & Peak Memory Comparison\n"
        "\n"
        "`[REAL]` Comparing whichever two real backends Section 1 confirmed are available on this hardware/build, at a few "
        "real sequence lengths, using this model's real attention dimensions. Each measurement uses a real warm-up pass "
        "plus repeated timed runs with `torch.cuda.synchronize()`, matching Notebook 01's methodology."
    ))

    comparison_cell_index = len(cells)
    cells.append(code(
        "def timed_sdpa(backend, q, k, v, n_repeats=8, n_warmup=1):\n"
        "    def run():\n"
        "        with torch.no_grad(), sdpa_kernel(backend):\n"
        "            F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        "\n"
        "    for _ in range(n_warmup):\n"
        "        run()\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "\n"
        "    samples = []\n"
        "    for _ in range(n_repeats):\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "        start = time.perf_counter()\n"
        "        run()\n"
        "        if DEVICE == \"cuda\":\n"
        "            torch.cuda.synchronize()\n"
        "        samples.append(time.perf_counter() - start)\n"
        "    return statistics.median(samples), samples\n"
        "\n"
        "def peak_memory_sdpa(backend, q, k, v):\n"
        "    torch.cuda.empty_cache()\n"
        "    torch.cuda.reset_peak_memory_stats()\n"
        "    with torch.no_grad(), sdpa_kernel(backend):\n"
        "        F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        "    torch.cuda.synchronize()\n"
        "    return torch.cuda.max_memory_allocated()\n"
        "\n"
        "SEQ_LENGTHS = [512, 1024, 2048, 4096]\n"
        "available_backends = [(name, b) for name, b in\n"
        "                      [(\"FLASH_ATTENTION\", SDPBackend.FLASH_ATTENTION),\n"
        "                       (\"EFFICIENT_ATTENTION\", SDPBackend.EFFICIENT_ATTENTION),\n"
        "                       (\"MATH\", SDPBackend.MATH)]\n"
        "                      if backend_availability.get(name)]\n"
        "print(f\"Comparing real backends: {[n for n, _ in available_backends]}\")\n"
        "\n"
        "comparison_results = []\n"
        "for L in SEQ_LENGTHS:\n"
        "    q, k, v = make_qkv(1, L)\n"
        "    row = {\"seq_len\": L}\n"
        "    for name, backend in available_backends:\n"
        "        median_s, _ = timed_sdpa(backend, q, k, v)\n"
        "        peak_bytes = peak_memory_sdpa(backend, q, k, v)\n"
        "        row[f\"{name}_latency_ms\"] = median_s * 1000\n"
        "        row[f\"{name}_peak_mb\"] = peak_bytes / 1024 / 1024\n"
        "    comparison_results.append(row)\n"
        "    parts = [f\"{name}: {row[f'{name}_latency_ms']:.3f}ms / {row[f'{name}_peak_mb']:.2f}MB\" for name, _ in available_backends]\n"
        "    print(f\"L={L:5d}: \" + \", \".join(parts))\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "03_flashattention_sdpa_backend_comparison.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "availability_cell_index": availability_cell_index,
        "comparison_cell_index": comparison_cell_index,
    }


# ============================================================================
# Notebook 04: Real Quantization Benchmark -- FP16 vs. INT8/INT4 Memory & Latency
# ============================================================================

def build_04_quantization_benchmark():
    cells = []

    cells.append(md(
        "# Notebook 04: Real Quantization Benchmark -- FP16 vs. INT8/INT4 Memory & Latency\n"
        "\n"
        "`[REAL]` Companion to Module 05. Real `bitsandbytes` quantization of `Qwen/Qwen2.5-0.5B-Instruct` on the RTX 4060 "
        "at FP16, INT8, and INT4 -- directly testing Module 05's central caveat: does a real memory reduction here "
        "translate into a proportional real latency reduction on this specific real hardware/kernel combination, or not? "
        "Whichever real outcome occurs is reported as-is."
    ))

    cells.append(code(
        "import time\n"
        "import statistics\n"
        "import logging\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
        "\n"
        "# bitsandbytes emits a real, repeated logging.warning() (not a Python warnings.warn(), which is\n"
        "# why filterwarnings doesn't touch it) on every 8-bit matmul call; across many timed repeats this\n"
        "# produces thousands of near-identical notebook output entries and bloats the saved file to\n"
        "# double-digit MB for no informational gain -- suppressed via the logger itself, deliberately.\n"
        "logging.getLogger(\"bitsandbytes\").setLevel(logging.ERROR)\n"
        "\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "print(f\"Device: {DEVICE}\")"
    ))

    cells.append(md(
        "## 1. Real Memory Footprint: FP16 vs. INT8 vs. INT4\n"
        "\n"
        "`[REAL]` Loading the same real model three times at three real precisions via `bitsandbytes`' "
        "`BitsAndBytesConfig`, reading each real model's `get_memory_footprint()`."
    ))

    memory_cell_index = len(cells)
    cells.append(code(
        "memory_results = {}\n"
        "\n"
        "model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)\n"
        "memory_results[\"FP16\"] = model_fp16.get_memory_footprint() / 1024 / 1024\n"
        "print(f\"FP16 memory footprint: {memory_results['FP16']:.2f} MB\")\n"
        "\n"
        "bnb_int8_config = BitsAndBytesConfig(load_in_8bit=True)\n"
        "model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_int8_config, device_map=DEVICE)\n"
        "memory_results[\"INT8\"] = model_int8.get_memory_footprint() / 1024 / 1024\n"
        "print(f\"INT8 memory footprint: {memory_results['INT8']:.2f} MB\")\n"
        "\n"
        "bnb_int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)\n"
        "model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_int4_config, device_map=DEVICE)\n"
        "memory_results[\"INT4\"] = model_int4.get_memory_footprint() / 1024 / 1024\n"
        "print(f\"INT4 memory footprint: {memory_results['INT4']:.2f} MB\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 2. Real Generation Latency: FP16 vs. INT8 vs. INT4\n"
        "\n"
        "`[REAL]` Measuring real generation latency for a fixed real prompt/output length across all three real precisions, "
        "using the same warm-up + repeated-run + `torch.cuda.synchronize()` + median/p95 methodology as Notebook 01."
    ))

    latency_cell_index = len(cells)
    cells.append(code(
        "def timed_generate(model, input_ids, n_new_tokens, n_repeats=6, n_warmup=1):\n"
        "    def run():\n"
        "        with torch.no_grad():\n"
        "            model.generate(input_ids, max_new_tokens=n_new_tokens, min_new_tokens=n_new_tokens,\n"
        "                           do_sample=False, pad_token_id=tokenizer.eos_token_id)\n"
        "\n"
        "    for _ in range(n_warmup):\n"
        "        run()\n"
        "        torch.cuda.synchronize()\n"
        "\n"
        "    samples = []\n"
        "    for _ in range(n_repeats):\n"
        "        torch.cuda.synchronize()\n"
        "        start = time.perf_counter()\n"
        "        run()\n"
        "        torch.cuda.synchronize()\n"
        "        samples.append(time.perf_counter() - start)\n"
        "    return statistics.median(samples), sorted(samples)[int(0.95 * (len(samples) - 1))]\n"
        "\n"
        "FIXED_PROMPT = \"Explain the concept of gravity in simple terms.\"\n"
        "N_NEW_TOKENS = 64\n"
        "input_ids = tokenizer(FIXED_PROMPT, return_tensors=\"pt\").input_ids.to(DEVICE)\n"
        "\n"
        "latency_results = {}\n"
        "for name, model in [(\"FP16\", model_fp16), (\"INT8\", model_int8), (\"INT4\", model_int4)]:\n"
        "    median_s, p95_s = timed_generate(model, input_ids, N_NEW_TOKENS)\n"
        "    latency_results[name] = {\"median_ms\": median_s * 1000, \"p95_ms\": p95_s * 1000,\n"
        "                              \"tpot_median_ms\": median_s * 1000 / N_NEW_TOKENS}\n"
        "    print(f\"{name}: median={median_s*1000:.2f}ms total, p95={p95_s*1000:.2f}ms, \"\n"
        "          f\"TPOT={median_s*1000/N_NEW_TOKENS:.3f}ms/token\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "04_quantization_benchmark.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "memory_cell_index": memory_cell_index,
        "latency_cell_index": latency_cell_index,
    }


# ============================================================================
# Notebook 05: Real Batching Throughput/Latency + Grounded PagedAttention Simulation
# ============================================================================

def build_05_batching_throughput_and_paged_simulation():
    cells = []

    cells.append(md(
        "# Notebook 05: Real Batching Throughput/Latency + Grounded PagedAttention Simulation\n"
        "\n"
        "`[REAL]` + `[SIMULATION]` Companion to Modules 03 and 06. Real batched generation on the RTX 4060 measuring "
        "real throughput/latency across batch sizes (Module 06), then feeding this notebook's own real measured "
        "per-request generation lengths into a Module-03-style paged-vs-contiguous allocation simulation.\n"
        "\n"
        "**Per the signed-off plan:** the PagedAttention piece is explicitly labeled `[SIMULATION]`, not `[REAL]`, even "
        "though it uses genuinely real measured inputs -- it is not an actual PagedAttention implementation or a real "
        "A/B benchmark against contiguous serving, and this notebook does not claim otherwise."
    ))

    cells.append(code(
        "import time\n"
        "import math\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "tokenizer.padding_side = \"left\"\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token\n"
        "\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)\n"
        "model.eval()\n"
        "print(f\"Loaded {MODEL_NAME} at FP16 on {DEVICE}\")"
    ))

    cells.append(md(
        "## 1. Real Batched Throughput/Latency Across Batch Sizes\n"
        "\n"
        "`[REAL]` A real, diverse set of 8 chat-formatted prompts (short, naturally-varied-length real answers), run at "
        "real batch sizes 1/2/4/8, measuring real wall-clock batch completion time and real throughput (useful tokens "
        "generated per second, counting only up to each real sequence's own first EOS token -- not padding)."
    ))

    throughput_cell_index = len(cells)
    cells.append(code(
        "PROMPTS = [\n"
        "    \"What is the capital of France? Answer in one short sentence.\",\n"
        "    \"Say hello in exactly 3 words.\",\n"
        "    \"What is 2+2? Answer with just the number.\",\n"
        "    \"Name one primary color.\",\n"
        "    \"What is the chemical symbol for water?\",\n"
        "    \"Name a planet in our solar system.\",\n"
        "    \"What language is spoken in Japan?\",\n"
        "    \"Give the opposite of 'hot' in one word.\",\n"
        "]\n"
        "CHAT_PROMPTS = [tokenizer.apply_chat_template([{\"role\": \"user\", \"content\": p}], tokenize=False,\n"
        "                                               add_generation_prompt=True) for p in PROMPTS]\n"
        "MAX_NEW_TOKENS = 80\n"
        "\n"
        "def real_generated_lengths(output_ids, prompt_len):\n"
        "    \"\"\"Real per-sequence useful length: position of first EOS token in the generated region,\n"
        "    or the full generated length if no real EOS was emitted within MAX_NEW_TOKENS.\"\"\"\n"
        "    lengths = []\n"
        "    for seq in output_ids:\n"
        "        gen_ids = seq[prompt_len:].tolist()\n"
        "        eos_pos = next((j + 1 for j, tid in enumerate(gen_ids) if tid == tokenizer.eos_token_id), len(gen_ids))\n"
        "        lengths.append(eos_pos)\n"
        "    return lengths\n"
        "\n"
        "BATCH_SIZES = [1, 2, 4, 8]\n"
        "throughput_results = []\n"
        "all_real_lengths = None\n"
        "for bs in BATCH_SIZES:\n"
        "    batch_prompts = CHAT_PROMPTS[:bs]\n"
        "    inputs = tokenizer(batch_prompts, return_tensors=\"pt\", padding=True).to(DEVICE)\n"
        "    prompt_len = inputs[\"input_ids\"].shape[1]\n"
        "\n"
        "    torch.cuda.synchronize()\n"
        "    start = time.perf_counter()\n"
        "    with torch.no_grad():\n"
        "        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,\n"
        "                              pad_token_id=tokenizer.pad_token_id)\n"
        "    torch.cuda.synchronize()\n"
        "    elapsed_s = time.perf_counter() - start\n"
        "\n"
        "    real_lengths = real_generated_lengths(out, prompt_len)\n"
        "    total_useful_tokens = sum(real_lengths)\n"
        "    throughput_tok_per_s = total_useful_tokens / elapsed_s\n"
        "    per_request_latency_ms = elapsed_s * 1000 / bs\n"
        "    throughput_results.append({\n"
        "        \"batch_size\": bs, \"elapsed_s\": elapsed_s, \"total_useful_tokens\": total_useful_tokens,\n"
        "        \"throughput_tok_per_s\": throughput_tok_per_s, \"per_request_latency_ms\": per_request_latency_ms,\n"
        "    })\n"
        "    print(f\"batch_size={bs}: elapsed={elapsed_s:.2f}s, useful_tokens={total_useful_tokens}, \"\n"
        "          f\"throughput={throughput_tok_per_s:.2f} tok/s, per-request latency={per_request_latency_ms:.1f}ms\")\n"
        "    if bs == BATCH_SIZES[-1]:\n"
        "        all_real_lengths = real_lengths\n"
        "\n"
        "print(f\"\\nReal per-request generation lengths at batch_size={BATCH_SIZES[-1]}: {all_real_lengths}\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 2. Grounded PagedAttention Simulation (Real Measured Inputs, Simulated Allocation)\n"
        "\n"
        "`[SIMULATION]` Feeding this notebook's own real measured per-request generation lengths (Section 1's largest "
        "batch) into a Module-03-style contiguous-vs-paged allocation simulation. **This is a simulation grounded in "
        "real data, not an actual PagedAttention implementation or a real A/B serving benchmark** -- stated explicitly, "
        "per the signed-off plan."
    ))

    simulation_cell_index = len(cells)
    cells.append(code(
        "MAX_SUPPORTED_LEN = MAX_NEW_TOKENS  # the real max_new_tokens ceiling this batch was run under\n"
        "BLOCK_SIZE = 8\n"
        "\n"
        "def contiguous_allocation_waste(lengths, max_len):\n"
        "    reserved = max_len * len(lengths)\n"
        "    used = sum(lengths)\n"
        "    waste = reserved - used\n"
        "    return {\"reserved\": reserved, \"used\": used, \"waste\": waste, \"waste_pct\": waste / reserved * 100}\n"
        "\n"
        "def paged_allocation_waste(lengths, block_size):\n"
        "    reserved = sum(math.ceil(L / block_size) * block_size for L in lengths)\n"
        "    used = sum(lengths)\n"
        "    waste = reserved - used\n"
        "    return {\"reserved\": reserved, \"used\": used, \"waste\": waste, \"waste_pct\": waste / reserved * 100}\n"
        "\n"
        "real_lengths = all_real_lengths\n"
        "print(f\"Real measured generation lengths used as simulation input: {real_lengths}\")\n"
        "\n"
        "contiguous_sim = contiguous_allocation_waste(real_lengths, MAX_SUPPORTED_LEN)\n"
        "paged_sim = paged_allocation_waste(real_lengths, BLOCK_SIZE)\n"
        "\n"
        "print(f\"\\n[SIMULATION] Contiguous: reserved={contiguous_sim['reserved']}, used={contiguous_sim['used']}, \"\n"
        "      f\"waste={contiguous_sim['waste']} ({contiguous_sim['waste_pct']:.2f}%)\")\n"
        "print(f\"[SIMULATION] Paged (block={BLOCK_SIZE}): reserved={paged_sim['reserved']}, used={paged_sim['used']}, \"\n"
        "      f\"waste={paged_sim['waste']} ({paged_sim['waste_pct']:.2f}%)\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "05_batching_throughput_and_paged_simulation.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "throughput_cell_index": throughput_cell_index,
        "simulation_cell_index": simulation_cell_index,
    }


# ============================================================================
# Notebook 06: Real Speculative Decoding + Production Capstone
# ============================================================================

def build_06_speculative_decoding_and_production_capstone():
    cells = []

    cells.append(md(
        "# Notebook 06: Real Speculative Decoding + End-to-End Latency Budget & Cost Capstone\n"
        "\n"
        "`[REAL]` + `[COMPUTED FROM REAL DATA]` + `[SIMULATION]` Companion to Modules 07, 08, and 09. Real speculative "
        "decoding on the RTX 4060 using a real INT4-quantized copy of `Qwen2.5-0.5B-Instruct` as the draft model against "
        "the real FP16 model as the target/verifier (a real, legitimate self-speculative-decoding pattern: same weights, "
        "lower precision, genuinely cheaper per forward pass per Notebook 04's memory findings).\n"
        "\n"
        "**Per the signed-off plan:** real acceptance rate and real end-to-end speedup are measured and reported as two "
        "genuinely separate numbers (Sections 1-2), never conflated -- mirroring Module 07's own two-step formula "
        "discipline. If speculative decoding had proven infeasible on this real hardware, that would be reported "
        "honestly instead of substituting an unrelated comparison; it proved feasible, so no such fallback is needed here."
    ))

    cells.append(code(
        "import time\n"
        "import math\n"
        "import logging\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
        "\n"
        "logging.getLogger(\"bitsandbytes\").setLevel(logging.ERROR)\n"
        "\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "EOS_IDS = set(tokenizer.eos_token_id) if isinstance(tokenizer.eos_token_id, list) else {tokenizer.eos_token_id}\n"
        "\n"
        "target_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)\n"
        "target_model.eval()\n"
        "\n"
        "bnb_int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)\n"
        "draft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_int4_config, device_map=DEVICE)\n"
        "draft_model.eval()\n"
        "print(\"Target (FP16) and draft (INT4) models loaded -- real speculative decoding is feasible on this hardware.\")"
    ))

    cells.append(md(
        "## 1. Real Acceptance Rate (Manual, Verified-Correct Speculative Decoding Loop)\n"
        "\n"
        "`[REAL]` Transformers' built-in `assistant_model=` API doesn't directly expose a per-round acceptance count, so "
        "this notebook implements a manual, transparent speculative-decoding loop instead -- draft proposes `k` tokens, "
        "target verifies all `k` in one real forward pass by comparing `argmax` at each position, accepting the longest "
        "real matching prefix. **Correctness was independently verified** before trusting these results: run against a "
        "real prompt, this loop's output was confirmed byte-for-byte identical to standard `model.generate()` greedy "
        "output for the same length (a real bug was caught and fixed in the process -- see the note below)."
    ))

    acceptance_cell_index = len(cells)
    cells.append(code(
        "# Real bug found and fixed during development: the model's default generation_config sets\n"
        "# repetition_penalty=1.1, which generate() applies even under do_sample=False -- so \"greedy\" via\n"
        "# generate() is NOT plain argmax(logits) unless repetition_penalty is explicitly reset to 1.0. This\n"
        "# loop and every generate() call below pass repetition_penalty=1.0 explicitly for a fair, exact-argmax\n"
        "# comparison; the manual loop's output was verified to exactly match generate()'s output only after\n"
        "# this fix (before the fix, they diverged -- a real, caught discrepancy, not assumed away).\n"
        "\n"
        "def manual_speculative_decode(prompt_ids, k, n_target_tokens):\n"
        "    context = prompt_ids.clone()\n"
        "    total_accepted, total_proposed, n_rounds = 0, 0, 0\n"
        "    start_len = context.shape[1]\n"
        "    while context.shape[1] - start_len < n_target_tokens:\n"
        "        with torch.no_grad():\n"
        "            draft_out = draft_model.generate(context, max_new_tokens=k, do_sample=False,\n"
        "                                              repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)\n"
        "        draft_tokens = draft_out[0, context.shape[1]:]\n"
        "        k_actual = draft_tokens.shape[0]\n"
        "        if k_actual == 0:\n"
        "            break\n"
        "        combined = torch.cat([context, draft_tokens.unsqueeze(0)], dim=1)\n"
        "        with torch.no_grad():\n"
        "            target_out = target_model(combined, use_cache=False)\n"
        "        logits = target_out.logits[0]\n"
        "        ctx_len = context.shape[1]\n"
        "        accepted = 0\n"
        "        for i in range(k_actual):\n"
        "            pred = logits[ctx_len - 1 + i].argmax().item()\n"
        "            if pred == draft_tokens[i].item():\n"
        "                accepted += 1\n"
        "            else:\n"
        "                break\n"
        "        bonus_pred = logits[ctx_len - 1 + accepted].argmax().item()\n"
        "        new_tokens = draft_tokens[:accepted].tolist() + [bonus_pred]\n"
        "        context = torch.cat([context, torch.tensor([new_tokens], device=context.device)], dim=1)\n"
        "        total_accepted += accepted\n"
        "        total_proposed += k_actual\n"
        "        n_rounds += 1\n"
        "        if bonus_pred in EOS_IDS:\n"
        "            break\n"
        "    return context, total_accepted, total_proposed, n_rounds\n"
        "\n"
        "PROMPT = tokenizer.apply_chat_template(\n"
        "    [{\"role\": \"user\", \"content\": \"Explain the concept of gravity in simple terms.\"}],\n"
        "    tokenize=False, add_generation_prompt=True)\n"
        "input_ids = tokenizer(PROMPT, return_tensors=\"pt\").input_ids.to(DEVICE)\n"
        "K_DRAFT = 4\n"
        "N_TARGET_TOKENS = 64\n"
        "\n"
        "spec_out_ids, accepted, proposed, rounds = manual_speculative_decode(input_ids, K_DRAFT, N_TARGET_TOKENS)\n"
        "real_alpha = accepted / proposed\n"
        "print(f\"Real accepted: {accepted}, real proposed: {proposed}, rounds: {rounds}\")\n"
        "print(f\"Real empirical acceptance rate (alpha): {real_alpha:.4f}\")\n"
        "\n"
        "n_new = spec_out_ids.shape[1] - input_ids.shape[1]\n"
        "with torch.no_grad():\n"
        "    greedy_check = target_model.generate(input_ids, max_new_tokens=n_new, do_sample=False,\n"
        "                                          repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)\n"
        "match_len = min(greedy_check.shape[1], spec_out_ids.shape[1])\n"
        "outputs_match = torch.equal(greedy_check[0, :match_len], spec_out_ids[0, :match_len])\n"
        "print(f\"Correctness check -- spec-decode output matches standard greedy output: {outputs_match}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 2. Real End-to-End Speedup (Kept Separate From Acceptance Rate, Per Module 07's Discipline)\n"
        "\n"
        "`[REAL]` Measuring real wall-clock time for this manual speculative-decoding loop versus real wall-clock time "
        "for standard greedy decoding of the same real output length -- a **separate, distinct real measurement** from "
        "Section 1's acceptance rate, not derived from it."
    ))

    speedup_cell_index = len(cells)
    cells.append(code(
        "def timed_run(fn, n_repeats=3, n_warmup=1):\n"
        "    for _ in range(n_warmup):\n"
        "        fn()\n"
        "        torch.cuda.synchronize()\n"
        "    samples = []\n"
        "    for _ in range(n_repeats):\n"
        "        torch.cuda.synchronize()\n"
        "        start = time.perf_counter()\n"
        "        fn()\n"
        "        torch.cuda.synchronize()\n"
        "        samples.append(time.perf_counter() - start)\n"
        "    return sorted(samples)[len(samples) // 2]\n"
        "\n"
        "def run_spec_decode():\n"
        "    manual_speculative_decode(input_ids, K_DRAFT, N_TARGET_TOKENS)\n"
        "\n"
        "def run_standard_greedy():\n"
        "    with torch.no_grad():\n"
        "        target_model.generate(input_ids, max_new_tokens=N_TARGET_TOKENS, min_new_tokens=N_TARGET_TOKENS,\n"
        "                               do_sample=False, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)\n"
        "\n"
        "spec_decode_median_s = timed_run(run_spec_decode)\n"
        "standard_greedy_median_s = timed_run(run_standard_greedy)\n"
        "real_speedup = standard_greedy_median_s / spec_decode_median_s\n"
        "\n"
        "print(f\"Manual spec-decode loop: {spec_decode_median_s*1000:.1f}ms median\")\n"
        "print(f\"Standard greedy decode:  {standard_greedy_median_s*1000:.1f}ms median\")\n"
        "print(f\"Real measured speedup: {real_speedup:.3f}x\")\n"
        "\n"
        "formula_expected_accepted = (1 - real_alpha ** (K_DRAFT + 1)) / (1 - real_alpha) if real_alpha < 1 else K_DRAFT + 1\n"
        "print(f\"\\nModule 07's formula, evaluated at this real measured alpha={real_alpha:.4f}, k={K_DRAFT}: \"\n"
        "      f\"E[accepted]={formula_expected_accepted:.3f} tokens/round\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 3. Capstone: Real-Data Latency Budget & Cost (Module 09)\n"
        "\n"
        "`[COMPUTED FROM REAL DATA]` Filling in Module 09's latency-budget worked example with genuinely real measured "
        "prefill/decode numbers from Notebook 01, replacing that module's illustrative 300ms/1600ms figures. Queue-wait "
        "and network/serialization remain the module's original **stated assumptions** (150ms, 50ms) -- there is no real "
        "network stack or request queue to measure in a single-notebook, single-machine context, and this section does "
        "not claim otherwise."
    ))

    capstone_cell_index = len(cells)
    cells.append(code(
        "# Real measured values, quoted directly from Notebook 01's own executed output:\n"
        "REAL_TTFT_512_MS = 156.49   # real median TTFT at 512 real prompt tokens\n"
        "REAL_TPOT_FLOOR_MS = 138.058  # real median TPOT floor (128-token real generation run)\n"
        "ASSUMED_QUEUE_WAIT_MS = 150  # stated assumption, not measurable here\n"
        "ASSUMED_NETWORK_MS = 50      # stated assumption, not measurable here\n"
        "N_OUTPUT_TOKENS = 100\n"
        "\n"
        "real_decode_ms = REAL_TPOT_FLOOR_MS * N_OUTPUT_TOKENS\n"
        "total_budget_ms = ASSUMED_QUEUE_WAIT_MS + REAL_TTFT_512_MS + real_decode_ms + ASSUMED_NETWORK_MS\n"
        "decode_share_pct = real_decode_ms / total_budget_ms * 100\n"
        "\n"
        "print(f\"Queue (assumed): {ASSUMED_QUEUE_WAIT_MS}ms, Prefill (REAL): {REAL_TTFT_512_MS}ms, \"\n"
        "      f\"Decode (REAL, {N_OUTPUT_TOKENS} tok): {real_decode_ms:.2f}ms, Network (assumed): {ASSUMED_NETWORK_MS}ms\")\n"
        "print(f\"Total budget: {total_budget_ms:.2f}ms, Decode share: {decode_share_pct:.2f}%\")\n"
        "\n"
        "GPU_COST_RATE_PER_HOUR = 2.00  # illustrative representative rate, not this laptop's real billed cost\n"
        "gpu_time_s = (REAL_TTFT_512_MS + real_decode_ms) / 1000\n"
        "cost_request = gpu_time_s * GPU_COST_RATE_PER_HOUR / 3600\n"
        "cost_per_token = cost_request / N_OUTPUT_TOKENS\n"
        "print(f\"\\nReal GPU time: {gpu_time_s:.5f}s, Cost/request: ${cost_request:.6f}, Cost/token: ${cost_per_token:.8f}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 4. Simulation: Least-Loaded Routing Using Real-Grounded Service Times (Module 08)\n"
        "\n"
        "`[SIMULATION]` Converting Notebook 05's real measured generation lengths (`[8, 10, 2, 7, 11, 80, 80, 2]`) into "
        "real-grounded service-time estimates (length x Notebook 01's real TPOT floor), then simulating least-loaded "
        "routing across 3 replicas. **This is a simulation, not a real multi-GPU deployment** -- no actual second or "
        "third GPU exists in this environment; only the input service-time estimates are genuinely real-data-grounded."
    ))

    routing_cell_index = len(cells)
    cells.append(code(
        "REAL_GROUNDED_LENGTHS = [8, 10, 2, 7, 11, 80, 80, 2]  # from Notebook 05's real bs=8 run\n"
        "service_times_ms = [L * REAL_TPOT_FLOOR_MS for L in REAL_GROUNDED_LENGTHS]\n"
        "print(f\"Real-grounded per-request service times (ms): {[round(s, 1) for s in service_times_ms]}\")\n"
        "\n"
        "def least_loaded_route(replica_loads, service_time):\n"
        "    target = min(replica_loads, key=lambda r: replica_loads[r])\n"
        "    replica_loads[target] += service_time\n"
        "    return target\n"
        "\n"
        "replicas = {\"replica_0\": 0.0, \"replica_1\": 0.0, \"replica_2\": 0.0}\n"
        "assignments = {}\n"
        "for i, st in enumerate(service_times_ms):\n"
        "    target = least_loaded_route(replicas, st)\n"
        "    assignments[f\"req_{i}\"] = target\n"
        "    rounded_loads = {k: round(v, 1) for k, v in replicas.items()}\n"
        "    print(f\"req_{i} (service_time={st:.1f}ms) -> {target}, loads now: {rounded_loads}\")\n"
        "\n"
        "final_spread = max(replicas.values()) - min(replicas.values())\n"
        "print(f\"\\nFinal simulated load spread across 3 replicas: {final_spread:.1f}ms\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "06_speculative_decoding_and_production_capstone.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "acceptance_cell_index": acceptance_cell_index,
        "speedup_cell_index": speedup_cell_index,
        "capstone_cell_index": capstone_cell_index,
        "routing_cell_index": routing_cell_index,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "01"
    if target == "01":
        path, indices = build_01_decode_vs_prefill_fundamentals()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "02":
        path, indices = build_02_kv_cache_memory_measurement()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "03":
        path, indices = build_03_flashattention_sdpa_backend_comparison()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "04":
        path, indices = build_04_quantization_benchmark()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "05":
        path, indices = build_05_batching_throughput_and_paged_simulation()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "06":
        path, indices = build_06_speculative_decoding_and_production_capstone()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
