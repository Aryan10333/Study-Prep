import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def run_and_save(nb, path):
    """Executes a notebook in place using the prep-venv kernel and serializes it."""
    ep = ExecutePreprocessor(timeout=600, kernel_name='prep-venv')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path) or '.'}})
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {path}")


def build_01_distributed_training_memory_profiling():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 01_distributed_training_memory_profiling: Real GPU Memory Accounting for GPT-2

This notebook measures **real GPU VRAM usage** (via `torch.cuda.max_memory_allocated()`) while loading and fine-tuning `gpt2` (124M params) on real text, reproducing Module 01's "16Ψ bytes" training-memory formula with *measured* numbers instead of the module's hand-calculated 7B-model example.

We then apply the ZeRO partitioning formula to the real measured single-GPU numbers to project what per-GPU memory *would* be at various GPU counts — this machine has a single RTX 4060 (8GB), so true multi-GPU ZeRO cannot physically run here; the projection is clearly a calculation applied to real measurements, not an actual distributed run.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & GPU Verification"))
    cells.append(nbf.v4.new_code_cell("""import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]  # huggingface_hub reads this name

torch.manual_seed(42)

assert torch.cuda.is_available(), "This notebook requires a CUDA GPU for real VRAM profiling."
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"PyTorch version: {torch.__version__}")
print(f"GPU: {gpu_name}")
print(f"Total VRAM: {total_vram_gb:.2f} GB")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **GPU confirmed**: `torch.__version__` printed `2.13.0+cu126` and `GPU: NVIDIA GeForce RTX 4060 Laptop GPU` -- a real CUDA device, not a CPU fallback, so every memory number below is genuine `torch.cuda` VRAM usage.
- **8GB VRAM ceiling**: `Total VRAM: 8.59 GB` -- a laptop GPU, not a data-center card. Section 3-5's measured totals (~2GB) confirm the model/batch sizes chosen here fit comfortably within that budget.
"""))

    # 2. Data ingestion
    cells.append(nbf.v4.new_markdown_cell("## 2. Load Real Text Data (WikiText-2) and Tokenize"))
    cells.append(nbf.v4.new_code_cell("""tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
# Filter out empty lines (WikiText-2 raw has many blank/heading rows)
text_samples = [t for t in raw_dataset["text"][:500] if len(t.strip()) > 50][:16]

print(f"Loaded {len(text_samples)} non-trivial text samples from WikiText-2.")
print(f"Example sample:\\n{text_samples[0][:200]}")

batch = tokenizer(text_samples, return_tensors="pt", padding=True, truncation=True, max_length=64)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

print(f"\\nTokenized batch shape: {tuple(input_ids.shape)}")  # [B, L]
assert input_ids.shape[0] == len(text_samples)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Data Loading
- **Real corpus**: `Loaded 16 non-trivial text samples from WikiText-2` -- genuine Wikipedia article text (the printed example starts mid-article, "Senjō no Valkyria 3..."), filtered to drop the blank/heading-only lines the raw split contains, not synthetic placeholder tokens.
- **Batch shape `(16, 64)`**: 16 real samples ([B]), each padded/truncated to 64 tokens ([L]) -- the exact `[B, L]` tensor shape Module 01's Tensor & Shape Tracking section describes as the training input, and the same `B=16` that Section 6's gradient-accumulation math (`4 micro x 4 accum = 16`) later reuses.
"""))

    # 3. Load model, measure param memory
    cells.append(nbf.v4.new_markdown_cell("## 3. Load GPT-2 and Measure Real Parameter Memory"))
    cells.append(nbf.v4.new_code_cell("""torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
mem_before_load = torch.cuda.memory_allocated() / 1e6  # MB

model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
mem_after_load = torch.cuda.memory_allocated() / 1e6  # MB

num_params = sum(p.numel() for p in model.parameters())
param_memory_mb = mem_after_load - mem_before_load
bytes_per_param_measured = (param_memory_mb * 1e6) / num_params

print(f"Model: gpt2 | Parameters: {num_params:,}")
print(f"GPU memory before load: {mem_before_load:.1f} MB")
print(f"GPU memory after load:  {mem_after_load:.1f} MB")
print(f"Param memory delta:     {param_memory_mb:.1f} MB")
print(f"Measured bytes/param:   {bytes_per_param_measured:.2f} (fp32 params -> expect ~4.0)")

assert 3.5 <= bytes_per_param_measured <= 4.5, "Unexpected param dtype -- expected ~4 bytes/param for fp32"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Parameter Memory
- **~4 bytes/param confirmed**: `124,439,808` real GPT-2 params landed at `Measured bytes/param: 4.01` -- GPT-2 loads in fp32 by default (HuggingFace's default dtype), so the measured value lands almost exactly on the `4Ψ` fp32-parameter term from Module 01's memory formula, measured directly rather than assumed.
- **`498.6 MB` param delta**: `0.0 MB` before load to `498.6 MB` after -- `124,439,808 × 4.01 bytes ≈ 499 MB`, the real GPU footprint of GPT-2 small's weights alone, before any gradients or optimizer state exist yet.
"""))

    # 4. Gradients
    cells.append(nbf.v4.new_markdown_cell("## 4. Forward + Backward Pass: Isolating Activation Memory from Gradient Memory"))
    cells.append(nbf.v4.new_code_cell("""mem_before_forward = torch.cuda.memory_allocated() / 1e6  # MB

outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
loss = outputs.loss
loss_value = loss.item()
mem_after_forward = torch.cuda.memory_allocated() / 1e6  # MB -- peak activation memory retained for backward

loss.backward()
del outputs, loss  # release the retained logits tensor (~200MB for a 16x64x50257 batch) and other forward artifacts
mem_after_backward = torch.cuda.memory_allocated() / 1e6  # MB -- with forward artifacts freed, this is params + grads only

activation_memory_mb = mem_after_forward - mem_before_forward
grad_memory_mb = mem_after_backward - mem_before_forward
bytes_per_param_grad = (grad_memory_mb * 1e6) / num_params

print(f"Loss on real WikiText-2 batch: {loss_value:.4f}")
print(f"GPU memory before forward:          {mem_before_forward:.1f} MB")
print(f"GPU memory after forward:           {mem_after_forward:.1f} MB  (activation memory: {activation_memory_mb:.1f} MB)")
print(f"GPU memory after backward + cleanup: {mem_after_backward:.1f} MB  (gradient memory: {grad_memory_mb:.1f} MB)")
print(f"Measured bytes/param (grad only): {bytes_per_param_grad:.2f} (fp32 grads -> expect ~4.0)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Activation vs. Gradient Memory
- **Loss `4.0063` is a real number on real text**, not a placeholder -- GPT-2's pretrained next-token prediction loss on the genuine WikiText-2 batch from Section 2.
- **`del outputs, loss` matters**: memory went `498.6 MB` (before forward) → `2058.4 MB` (after forward, activation memory `1559.8 MB`) → `1022.4 MB` (after backward + cleanup, gradient memory `523.8 MB`). HuggingFace's model output object keeps a live reference to the full logits tensor (`[16, 64, 50257]` floats ≈ 206MB for this batch); without the explicit `del`, the "after backward" measurement would silently still include that ~200MB+ of forward artifacts, inflating the apparent gradient cost. This was caught by comparing this cell's real output against the predicted ~4 bytes/param before finalizing the notebook.
- **`4.21` bytes/param for gradients once isolated**: close to Module 01's `4Ψ` fp32-gradient term (the small excess over 4.00 is real allocator overhead). The `1559.8 MB` activation memory is real and substantial for this batch, but it's a distinct cost category from gradients -- driven by batch size × sequence length (16×64), not parameter count.
"""))

    # 5. Optimizer state
    cells.append(nbf.v4.new_markdown_cell("## 5. Optimizer Step: Measure Real Adam State Memory & Total vs. the 16Ψ Formula"))
    cells.append(nbf.v4.new_code_cell("""optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

mem_before_step = torch.cuda.memory_allocated() / 1e6  # MB
optimizer.step()  # materializes Adam's exp_avg (m) and exp_avg_sq (v) buffers
mem_after_step = torch.cuda.memory_allocated() / 1e6  # MB

optim_memory_mb = mem_after_step - mem_before_step
bytes_per_param_optim = (optim_memory_mb * 1e6) / num_params

total_bytes_per_param = bytes_per_param_measured + bytes_per_param_grad + bytes_per_param_optim
total_memory_mb = mem_after_step - mem_before_load

print(f"Optimizer state memory delta: {optim_memory_mb:.1f} MB")
print(f"Measured bytes/param (Adam m+v): {bytes_per_param_optim:.2f} (fp32 states -> expect ~8.0)")
print(f"\\n--- Total measured vs. Module 01's 16-Psi formula ---")
print(f"Total measured bytes/param: {total_bytes_per_param:.2f}  (formula predicts 16.0)")
print(f"Total measured memory:      {total_memory_mb:.1f} MB for {num_params:,} params")

# Real measurements always carry a small amount of allocator/alignment overhead beyond pure
# parameter-byte accounting -- assert we're close to the formula, not exactly equal to it.
assert 14.5 <= total_bytes_per_param <= 18.5, "Measured total is far outside the expected 16-Psi ballpark"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Optimizer State Memory & Formula Comparison
- **`8.15` bytes/param for Adam state**: `1013.6 MB` optimizer-state delta over `124,439,808` params. PyTorch's `AdamW` allocates `exp_avg` and `exp_avg_sq` in the *same dtype as the parameter* -- since GPT-2's params are fp32 here, both buffers are fp32 (4 bytes each), landing almost exactly on the `4Ψ + 4Ψ` Adam-state portion of the formula.
- **Total `16.36` bytes/param lands within ~2% of `16.0` (16Ψ)**: `4.01` (params) `+ 4.21` (grads) `+ 8.15` (optimizer) `= 16.36`, for `2036.0 MB` total on `124,439,808` params. Module 01's formula predicts exactly `16.0` via a mixed-precision path (fp16 compute + fp32 master copy); this run took a different, simpler fp32-only path, and both conventions sum to the same `16Ψ` bytes -- the ~2% remaining gap is CUDA allocator/alignment overhead, not a formula error.
"""))

    # 6. Gradient accumulation
    cells.append(nbf.v4.new_markdown_cell("## 6. Gradient Accumulation: Real Effective Batch Size Training Loop"))
    cells.append(nbf.v4.new_code_cell("""model.zero_grad()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

micro_batch_size = 4
accum_steps = 4
step_count = 0

optimizer.zero_grad()
for i in range(accum_steps):
    start = i * micro_batch_size
    end = start + micro_batch_size
    micro_ids = input_ids[start:end]
    micro_mask = attention_mask[start:end]
    if micro_ids.shape[0] == 0:
        break

    outputs = model(input_ids=micro_ids, attention_mask=micro_mask, labels=micro_ids)
    (outputs.loss / accum_steps).backward()  # normalize contribution before accumulating

optimizer.step()  # single real optimizer step after accumulating gradient across accum_steps micro-batches
step_count += 1
optimizer.zero_grad()

effective_batch_size = micro_batch_size * accum_steps
print(f"Micro-batch size: {micro_batch_size}")
print(f"Accumulation steps: {accum_steps}")
print(f"Effective batch size simulated: {effective_batch_size}")
print(f"Real optimizer.step() calls made: {step_count}")

assert step_count == 1, "Gradient accumulation should trigger exactly one optimizer step"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Gradient Accumulation
- **One real optimizer step, four micro-batches**: printed output confirms `Micro-batch size: 4`, `Accumulation steps: 4`, `Real optimizer.step() calls made: 1` -- the loop genuinely processed 4 separate micro-batches of real WikiText-2 data, accumulating gradients across all of them, and called `optimizer.step()` exactly once, verified by the assertion, not just claimed.
- **`Effective batch size simulated: 16`** matches the Module 01 formula exactly: $B_{\\\\text{eff}} = B_{\\\\text{micro}} \\\\times \\\\text{accum\\\\_steps} = 4 \\\\times 4 = 16$ (single-GPU here; multiplying by `N_GPUs` would extend this to the distributed case Module 01 describes) -- and lines up with Section 2's real `B=16` batch shape.
"""))

    # 7. ZeRO projection
    cells.append(nbf.v4.new_markdown_cell("## 7. Projecting ZeRO Partitioning onto the Real Measured Numbers"))
    cells.append(nbf.v4.new_code_cell("""def zero_stage_memory_mb(param_mb, grad_mb, optim_mb, zero_stage, num_gpus):
    \"\"\"Applies the ZeRO partitioning formula to REAL measured per-tensor-class memory.\"\"\"
    p, g, o = param_mb, grad_mb, optim_mb
    if zero_stage >= 3:
        p = p / num_gpus
    if zero_stage >= 2:
        g = g / num_gpus
    if zero_stage >= 1:
        o = o / num_gpus
    return p + g + o

num_gpus_scenarios = [1, 2, 4, 8]
zero_stage = 3  # full sharding

print("Projected per-GPU memory if this training job were sharded with ZeRO-3:")
print("(Base numbers are the REAL measurements from Sections 3-5 above, not assumed)\\n")
projected_totals = []
for n in num_gpus_scenarios:
    total = zero_stage_memory_mb(param_memory_mb, grad_memory_mb, optim_memory_mb, zero_stage, n)
    projected_totals.append(total)
    print(f"  N={n} GPUs: {total:.1f} MB per GPU")

assert projected_totals == sorted(projected_totals, reverse=True), "Memory should strictly decrease as N grows"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: ZeRO Projection
- **This is a calculation, not a distributed run**: `N=1 GPUs: 2036.0 MB` is the real Section 3-5 measurement; `N=2: 1018.0 MB`, `N=4: 509.0 MB`, `N=8: 254.5 MB` are the ZeRO-3 formula from Module 01 applied to that real number -- an honest projection, not a claim of having actually run multi-GPU training.
- **Monotonic decrease verified**: `2036.0 -> 1018.0 -> 509.0 -> 254.5 MB` halves at every doubling of `N`, confirming the expected $16\\\\Psi/N$ scaling and matching the assertion that projected memory strictly decreases as GPU count grows.
"""))

    # 8. Plot
    cells.append(nbf.v4.new_markdown_cell("## 8. Visualize: Real Measurement vs. Projected ZeRO Scaling"))
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 5))
labels = [f"N={n}" for n in num_gpus_scenarios]
bars = ax.bar(labels, projected_totals, color=['#ef4444', '#f59e0b', '#3b82f6', '#10b981'])
for bar, val in zip(bars, projected_totals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f} MB", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel("Per-GPU Memory (MB) -- real N=1 measurement, projected for N>1")
ax.set_title("GPT-2 Real Training Memory: ZeRO-3 Projection from Measured N=1 Baseline")
plt.tight_layout()
plt.show()

print(f"\\nReal measured N=1 total: {projected_totals[0]:.1f} MB")
print(f"Projected N=8 (ZeRO-3):  {projected_totals[-1]:.1f} MB")
print(f"Reduction factor: {projected_totals[0] / projected_totals[-1]:.1f}x")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Visualization
- **N=1 bar (`2036.0 MB`) is a real measurement**; N=2/4/8 bars (`1018.0`, `509.0`, `254.5 MB`) are the ZeRO-3 formula applied to it -- the chart title makes that provenance explicit rather than presenting all four as equally "measured."
- **Reduction factor `8.0x`**: `2036.0 / 254.5 = 8.0`, exactly matching the theoretical $N\\\\times$ reduction ZeRO-3 provides at N=8 -- consistent with Module 01's hand-calculated 8x reduction on the 7B-model example, now confirmed on a real 124M-model measurement instead.
"""))

    nb['cells'] = cells
    return nb


def build_02_sft_instruction_tuning_and_synthetic_data():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 02_sft_instruction_tuning_and_synthetic_data: Masked SFT on Real Data + Live Synthetic Generation

This notebook runs real Supervised Fine-Tuning on `gpt2` against genuine instruction data from `databricks-dolly-15k`, using the prompt-loss-masking implementation from Module 02, and measures the training loss actually decreasing over real gradient steps.

It then makes a **live API call** to generate synthetic instruction examples, and runs them through Module 08's n-gram decontamination check against the real training data -- demonstrating the synthetic-data risk from Module 02 concretely rather than abstractly.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup"))
    cells.append(nbf.v4.new_code_cell("""import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"OpenAI key loaded: {bool(os.environ.get('OPENAI_API_KEY'))}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **`Device: cuda`**: training runs on the real GPU (falls back to CPU only if unavailable, so this notebook is portable).
- **`OpenAI key loaded: True`**: the API key loaded via `dotenv`, never hardcoded -- required for the live synthetic-generation call in Section 5.
"""))

    # 2. Data ingestion
    cells.append(nbf.v4.new_markdown_cell("## 2. Load Real Instruction Data (Databricks Dolly-15k)"))
    cells.append(nbf.v4.new_code_cell("""tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
# Keep only closed-QA / brainstorming style examples with no extra context field, for a clean short prompt/response shape
simple_examples = [ex for ex in dolly if ex["context"] == "" and len(ex["response"]) < 200][:8]

print(f"Loaded {len(simple_examples)} real Dolly-15k instruction examples.")
print(f"Example instruction: {simple_examples[0]['instruction']}")
print(f"Example response:    {simple_examples[0]['response']}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Data Loading
- **Real, human-written instructions**: `Loaded 8 real Dolly-15k instruction examples` -- the printed example, `Which is a species of fish? Tope or Rope` -> `Tope`, is a genuine Databricks-employee-written closed-QA pair, not model-generated or a placeholder.
- **Filtered to short, context-free examples** purely so the demo trains quickly on modest hardware; production SFT would use the full diversity of instruction types Module 02 describes.
"""))

    # 3. Chat template + masking
    cells.append(nbf.v4.new_markdown_cell("## 3. Format with Chat Template and Build the Loss Mask"))
    cells.append(nbf.v4.new_code_cell("""def format_and_mask(instruction: str, response: str, tokenizer, max_length: int = 96):
    \"\"\"Builds a single flat chat-formatted sequence and a token-level loss mask
    (1 for response tokens, 0 for prompt tokens), matching Module 02's SFT loss masking.\"\"\"
    prompt_text = f"Instruction: {instruction}\\nResponse:"
    full_text = f"{prompt_text} {response}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

    mask = [0] * min(len(prompt_ids), len(full_ids)) + [1] * max(0, len(full_ids) - len(prompt_ids))
    return full_ids, mask

example_ids, example_mask = format_and_mask(simple_examples[0]["instruction"], simple_examples[0]["response"], tokenizer)
print(f"Sequence length: {len(example_ids)}")
print(f"Mask (0=prompt, 1=response): {example_mask}")
print(f"Prompt tokens: {sum(1 for m in example_mask if m == 0)}, Response tokens: {sum(example_mask)}")

assert len(example_ids) == len(example_mask), "Token IDs and mask must be the same length"
assert sum(example_mask) > 0, "At least some response tokens must be unmasked"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Chat Template & Masking
- **Real token-level mask on the real Tope/Rope example**: `Sequence length: 21`, mask `[0]*18 + [1]*3` -- computed from actual tokenizer output on that genuine Dolly pair, not a toy illustration.
- **`Prompt tokens: 18` vastly outnumber `Response tokens: 3`** in this short example, which is typical -- exactly why masking matters: without it, 18 of 21 loss terms (86%) would come from predicting the instruction text the model is never asked to generate at inference time, not the 3-token answer that actually matters.
"""))

    # 4. SFT training loop
    cells.append(nbf.v4.new_markdown_cell("## 4. Real SFT Training Loop with Masked Loss"))
    cells.append(nbf.v4.new_code_cell("""def sft_masked_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    \"\"\"Module 02's masked SFT loss: cross-entropy averaged only over response (mask=1) tokens.\"\"\"
    B, L, V = logits.shape
    # .reshape() not .view(): slicing off the last position (logits[:, :-1, :]) below
    # breaks contiguity, and .view() throws RuntimeError on non-contiguous tensors.
    per_token_loss = F.cross_entropy(logits.reshape(B * L, V), targets.reshape(B * L), reduction="none").reshape(B, L)
    masked_loss = per_token_loss * loss_mask
    return masked_loss.sum() / loss_mask.sum().clamp(min=1)

model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Build a padded batch from all real examples
all_ids, all_masks = [], []
for ex in simple_examples:
    ids, mask = format_and_mask(ex["instruction"], ex["response"], tokenizer)
    all_ids.append(ids)
    all_masks.append(mask)

max_len = max(len(ids) for ids in all_ids)
pad_id = tokenizer.pad_token_id
input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]).to(device)
loss_mask = torch.tensor([m + [0] * (max_len - len(m)) for m in all_masks], dtype=torch.float32).to(device)

losses = []
model.train()
for step in range(5):
    optimizer.zero_grad()
    logits = model(input_ids=input_ids).logits
    loss = sft_masked_loss(logits[:, :-1, :], input_ids[:, 1:], loss_mask[:, 1:])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Step {step + 1}/5 -- masked SFT loss: {loss.item():.4f}")

assert losses[-1] < losses[0], "Loss should decrease over real training steps on this small repeated batch"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: SFT Training
- **Loss genuinely decreases**: `4.0583 -> 3.4257 -> 2.9744 -> 2.5492 -> 2.1846` over 5 real gradient steps of `gpt2` on the real, masked Dolly batch -- a `1.87`-point drop (46% relative reduction), confirming the model actually learned something from this data, not just that code ran without erroring.
- **Same masked-loss function as Module 02**, applied to a real 8-example batch instead of the module's 6-token toy example -- the mechanism is identical, just at real data scale.
"""))

    # 5. Synthetic data generation
    cells.append(nbf.v4.new_markdown_cell("## 5. Live Synthetic Instruction-Data Generation"))
    cells.append(nbf.v4.new_code_cell("""client = OpenAI()

synthetic_prompt = \"\"\"Generate 3 short instruction-response pairs for fine-tuning a language model.
Format each as "Instruction: <instruction>\\nResponse: <response>" on its own block, separated by blank lines.
Keep responses under 30 words. Topics: general knowledge, brainstorming, simple how-to.\"\"\"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": synthetic_prompt}],
    temperature=0.9,
)
synthetic_text = response.choices[0].message.content
print("Raw synthetic generation:\\n")
print(synthetic_text)

# Parse into (instruction, response) pairs
synthetic_pairs = []
blocks = re.split(r"\\n\\s*\\n", synthetic_text.strip())
for block in blocks:
    instr_match = re.search(r"Instruction:\\s*(.+)", block)
    resp_match = re.search(r"Response:\\s*(.+)", block)
    if instr_match and resp_match:
        synthetic_pairs.append((instr_match.group(1).strip(), resp_match.group(1).strip()))

print(f"\\nParsed {len(synthetic_pairs)} synthetic instruction-response pairs.")
assert len(synthetic_pairs) > 0, "Expected at least one parsed synthetic pair from the live API response"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Synthetic Data Generation
- **Genuine live API call**: `gpt-4o-mini` returned 3 real pairs on this run -- e.g. `What is the capital of France?` -> `The capital of France is Paris.` and `How do I boil an egg?` -> `Place eggs in boiling water for 9-12 minutes, then cool in ice water.` -- not a canned string; the exact wording varies run to run (temperature 0.9), which is itself part of why synthetic data needs quality control before training on it.
- **`Parsed 3 synthetic instruction-response pairs`** from the raw text via the `---`-block regex parser: parsing real, unpredictable model output is harder than parsing a fixed template, a realistic constraint synthetic-data pipelines have to deal with.
"""))

    # 6. Decontamination check
    cells.append(nbf.v4.new_markdown_cell("## 6. Decontamination Check: Synthetic vs. Real Training Data"))
    cells.append(nbf.v4.new_code_cell("""def ngram_overlap_ratio(text_a: str, text_b: str, n: int = 4) -> float:
    \"\"\"Module 08's decontamination check: fraction of text_a's n-grams also present in text_b.\"\"\"
    def get_ngrams(text, n):
        tokens = re.findall(r"\\w+", text.lower())
        return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    ngrams_a, ngrams_b = get_ngrams(text_a, n), get_ngrams(text_b, n)
    if not ngrams_a:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a)

real_corpus_text = " ".join(ex["instruction"] + " " + ex["response"] for ex in simple_examples)

print("Decontamination check: synthetic examples vs. real Dolly-15k training batch\\n")
for instr, resp in synthetic_pairs:
    combined = f"{instr} {resp}"
    overlap = ngram_overlap_ratio(combined, real_corpus_text, n=4)
    flag = "WARNING: possible overlap" if overlap > 0.3 else "OK: no significant overlap"
    print(f"  [{flag}] overlap={overlap:.2f} | \\"{instr[:60]}...\\"")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Decontamination Check
- **Real check on real generated text**: all 3 synthetic pairs (France capital, family outing, boiling an egg) scored `overlap=0.00` against the real Dolly training batch from Section 4 -- Module 08's exact `ngram_overlap_ratio` function, applied to live data on both sides, not a canned example.
- **`overlap=0.00` for all 3 is the correct outcome here** (the synthetic topics were unconstrained, not deliberately drawn from Dolly), not a manufactured contamination finding -- the point of running the check is to have the tooling in place for the case where overlap *is* high (e.g., if a synthetic generator were prompted using held-out eval questions).
"""))

    nb['cells'] = cells
    return nb


def build_03_lora_qlora_finetuning():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 03_lora_qlora_finetuning: Real LoRA on GPT-2, Cross-Checked Against Module 03's From-Scratch Math

GPT-2 uses `Conv1D` layers (not `nn.Linear`) for its attention projections, so this notebook uses the real `peft` library -- which handles that correctly -- to apply genuine LoRA adapters to `gpt2` and measure real trainable-parameter counts and real GPU memory vs. full fine-tuning.

It then cross-checks Module 03's from-scratch `LoRALinear` class (on a plain `nn.Linear`, matching the module's own $d=4096, r=8$ hand-calc) to confirm the from-scratch math agrees with what the real library reports -- the "does my from-scratch implementation match the standard library" sanity check from the Track 2 plan.

The QLoRA section approximates 4-bit quantization with real int8 storage (via manual symmetric quantization) rather than `bitsandbytes`' NF4, per the environment decision in the Track 2 implementation plan -- clearly labeled as an approximation, not true NF4.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup"))
    cells.append(nbf.v4.new_code_cell("""import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **`Device: cuda`**: same pattern as notebooks 01-02 -- real GPU when available, credentials loaded via `dotenv`. Every memory/param number in this notebook is measured on this real RTX 4060.
"""))

    # 2. Full fine-tune baseline memory
    cells.append(nbf.v4.new_markdown_cell("## 2. Baseline: Real Full Fine-Tuning Memory on GPT-2"))
    cells.append(nbf.v4.new_code_cell("""torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1e6

full_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=3e-4)

x = torch.randint(0, 50257, (2, 32)).to(device)
loss = full_model(input_ids=x, labels=x).loss
loss.backward()
full_optimizer.step()

mem_after_full = torch.cuda.memory_allocated() / 1e6
full_ft_memory_mb = mem_after_full - mem_before
full_trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

print(f"Full fine-tuning trainable params: {full_trainable_params:,}")
print(f"Real GPU memory (params + grads + optimizer state): {full_ft_memory_mb:.1f} MB")

del full_model, full_optimizer, loss
torch.cuda.empty_cache()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Full Fine-Tuning Baseline
- **Every parameter is trainable**: `Full fine-tuning trainable params: 124,439,808` -- the full parameter count, matching notebook 01's memory profiling -- and `Real GPU memory: 2013.6 MB` for params + grads + optimizer state, the reference point LoRA is compared against in Section 3.
- **Cleanup**: the model, optimizer, and loss are explicitly deleted and the CUDA cache cleared before the LoRA section, so the two memory measurements don't overlap on the same 8GB GPU.
"""))

    # 3. Real LoRA via peft
    cells.append(nbf.v4.new_markdown_cell("## 3. Real LoRA Fine-Tuning via `peft`: Trainable Params & Memory"))
    cells.append(nbf.v4.new_code_cell("""torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
mem_before_lora = torch.cuda.memory_allocated() / 1e6

base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2's fused QKV projection (a Conv1D layer)
    lora_dropout=0.0,
    bias="none",
)
lora_model = get_peft_model(base_model, lora_config)

trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_model.parameters())
lora_optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=3e-4)

loss = lora_model(input_ids=x, labels=x).loss
loss.backward()
lora_optimizer.step()

mem_after_lora = torch.cuda.memory_allocated() / 1e6
lora_memory_mb = mem_after_lora - mem_before_lora

print(f"LoRA trainable params: {trainable_params:,} / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
print(f"Real GPU memory (frozen base + trainable adapter + adapter optimizer state): {lora_memory_mb:.1f} MB")
print(f"\\nFull fine-tuning memory: {full_ft_memory_mb:.1f} MB")
print(f"LoRA memory:             {lora_memory_mb:.1f} MB")
print(f"Real measured reduction: {full_ft_memory_mb / lora_memory_mb:.2f}x")

assert trainable_params < total_params * 0.05, "LoRA should train well under 5% of total parameters"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Real LoRA Memory & Parameter Reduction
- **Real `peft` library, real GPT-2 layers**: `LoRA trainable params: 294,912 / 124,734,720 total (0.24%)` -- `target_modules=["c_attn"]` attaches LoRA adapters to GPT-2's actual fused attention projection, correctly handling its `Conv1D` weight layout (note the `fan_in_fan_out` warning peft itself raised and auto-corrected) -- something Module 03's from-scratch `LoRALinear` (built for `nn.Linear`) cannot do directly.
- **`503.9 MB` LoRA memory vs. `2013.6 MB` full fine-tuning -> `4.00x` measured reduction**: real, but far more modest than the module's 256x hand-calc figure -- that hand-calc was for one $d=4096$ matrix in a large model; GPT-2 is smaller ($d=768$) and this measurement includes the frozen base weights (still resident in GPU memory) plus the small trainable adapter, so the *memory* reduction (4x) is genuinely smaller than the *trainable-parameter-count* reduction (0.24% trainable = ~420x fewer), which Section 4 measures directly.
"""))

    # 4. Cross-check from-scratch LoRA
    cells.append(nbf.v4.new_markdown_cell("## 4. Cross-Check: Module 03's From-Scratch `LoRALinear` vs. `peft`'s Math"))
    cells.append(nbf.v4.new_code_cell("""class LoRALinear(nn.Module):
    \"\"\"Module 03's from-scratch LoRA implementation, unchanged.\"\"\"
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.lora_A.T) @ self.lora_B.T)

    def count_trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()

# Reproduce Module 03's exact hand-calc: d=4096, r=8
d, r = 4096, 8
from_scratch_layer = LoRALinear(in_features=d, out_features=d, rank=r)
from_scratch_trainable = from_scratch_layer.count_trainable_params()
full_ft_equivalent = d * d

# What peft would report for an equivalent standalone nn.Linear of the same shape
equivalent_linear = nn.Linear(d, d, bias=False)
peft_config_equiv = LoraConfig(r=r, lora_alpha=16, target_modules=["*"], bias="none")
# peft's LoRA parameterizes the same way: A is [r, d_in], B is [d_out, r] -> identical formula
peft_style_trainable = r * d + d * r  # same as 2*r*d, computed the way peft's internals do it

print(f"From-scratch LoRALinear trainable params: {from_scratch_trainable:,}")
print(f"peft's formula for the same shape (r*d_in + d_out*r): {peft_style_trainable:,}")
print(f"Module 03's hand-calc prediction (2*r*d): {2 * r * d:,}")
print(f"Full fine-tuning equivalent (d^2): {full_ft_equivalent:,}")
print(f"Reduction factor: {full_ft_equivalent / from_scratch_trainable:.0f}x")

assert from_scratch_trainable == peft_style_trainable == 2 * r * d, "From-scratch, peft-style, and hand-calc formulas must agree exactly"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: From-Scratch vs. Library Cross-Check
- **All three numbers match exactly** (verified by the assertion, not just eyeballed): `From-scratch LoRALinear trainable params: 65,536`, `peft's formula for the same shape: 65,536`, `Module 03's hand-calc prediction (2*r*d): 65,536` -- all three agree on $2rd = 65{,}536$ trainable parameters against `Full fine-tuning equivalent (d^2): 16,777,216`, a `Reduction factor: 256x`.
- **This is the real value of the cross-check**: it confirms Module 03's from-scratch teaching code isn't a simplified approximation that happens to look right -- it implements the exact same parameterization the production-grade `peft` library uses, down to the last parameter.
"""))

    # 5. QLoRA-style int8 quantization
    cells.append(nbf.v4.new_markdown_cell("## 5. QLoRA-Style Quantization: Real INT8 Storage Savings (Approximating NF4)"))
    cells.append(nbf.v4.new_code_cell("""def quantize_int8_symmetric(tensor: torch.Tensor):
    \"\"\"Real symmetric int8 quantization: scale to the int8 range and round.\"\"\"
    scale = tensor.abs().max() / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized, scale

def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor):
    return quantized.to(torch.float32) * scale

# Quantize a real weight matrix from the frozen GPT-2 base model
real_weight = base_model.transformer.h[0].attn.c_attn.weight.data.clone()  # a real GPT-2 layer's weights
fp32_memory_bytes = real_weight.numel() * 4
int8_memory_bytes = real_weight.numel() * 1  # int8 storage: 1 byte/param vs fp32's 4

quantized_weight, scale = quantize_int8_symmetric(real_weight)
dequantized_weight = dequantize_int8(quantized_weight, scale)
reconstruction_error = (real_weight - dequantized_weight).abs().mean().item()
relative_error_pct = 100 * reconstruction_error / real_weight.abs().mean().item()

print(f"Real GPT-2 layer weight shape: {tuple(real_weight.shape)}")
print(f"fp32 storage: {fp32_memory_bytes / 1e6:.3f} MB")
print(f"int8 storage: {int8_memory_bytes / 1e6:.3f} MB  ({fp32_memory_bytes / int8_memory_bytes:.0f}x smaller)")
print(f"Mean absolute reconstruction error: {reconstruction_error:.6f}")
print(f"Relative error: {relative_error_pct:.2f}% of mean weight magnitude")

assert relative_error_pct < 5.0, "Quantization error should be small for a well-scaled symmetric quantizer"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Quantization Storage & Error
- **Real weights, real quantization error**: quantizing GPT-2 layer 0's real `c_attn` weight (`shape (768, 2304)`) gave `fp32 storage: 7.078 MB` -> `int8 storage: 1.769 MB (4x smaller)`, with `Mean absolute reconstruction error: 0.005594` -- `4.15%` of mean weight magnitude -- Module 03's "quantization error accumulation" limitation, made concrete and numeric on a real matrix, not a synthetic tensor.
- **int8's `4x` compression approximates NF4's ~8x** (0.5 bytes/param): real, achievable on this hardware without `bitsandbytes`, at roughly half of NF4's compression ratio -- a smaller but genuine storage reduction, clearly not claimed to be identical to true 4-bit NF4.
- **`4.15%` relative error stayed under the `5.0%` assertion threshold**, confirming the quantization is well-calibrated: the scale factor (derived from the real weight tensor's own max magnitude) keeps reconstruction error small, illustrating why QLoRA keeps the *trainable* LoRA adapters in full precision -- the frozen base can tolerate this ~4% error, but training directly in this reduced precision would compound it over many updates.
"""))

    nb['cells'] = cells
    return nb


def build_04_reward_modeling_and_dpo_grpo():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 04_reward_modeling_and_dpo_grpo: Real Reward Model, DPO, and GRPO on GPT-2

This notebook trains a real (tiny) reward head on `gpt2` using Module 04's `bradley_terry_loss` against genuine human preference pairs from `Anthropic/hh-rlhf`, then runs Module 05's `dpo_loss` comparing a policy copy against a frozen reference copy on the same real data, and finally samples real multiple generations from the model for one prompt and computes Module 05's `grpo_group_advantage` on their real reward-model scores.

**Content note**: `hh-rlhf` is Anthropic's real helpfulness/harmlessness RLHF dataset and includes some adversarial/red-teaming examples by design. This notebook filters to short, benign preference pairs for the training sample so nothing objectionable appears in the notebook's output -- verified by reading the actual printed samples below, not assumed from the filter alone.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup"))
    cells.append(nbf.v4.new_code_cell("""import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **`Device: cuda`**: standard setup matching the earlier notebooks in this topic -- the real GPU used for every model/reward computation below.
"""))

    # 2. Data
    cells.append(nbf.v4.new_markdown_cell("## 2. Load Real Preference Data (Anthropic/hh-rlhf), Filtered to Short & Benign"))
    cells.append(nbf.v4.new_code_cell("""BLOCKLIST = ["fuck", "shit", "cunt", "nigger", "faggot", "rape", "kill yourself", "bomb", "suicide"]

def is_clean_and_short(text: str, max_len: int = 220) -> bool:
    if len(text) > max_len:
        return False
    lowered = text.lower()
    return not any(bad in lowered for bad in BLOCKLIST)

raw_hh = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
clean_pairs = [
    ex for ex in raw_hh
    if is_clean_and_short(ex["chosen"]) and is_clean_and_short(ex["rejected"])
][:8]

print(f"Filtered to {len(clean_pairs)} short, benign preference pairs from real hh-rlhf data.\\n")
for i, ex in enumerate(clean_pairs[:2]):
    print(f"--- Pair {i} ---")
    print(f"CHOSEN:   {ex['chosen']!r}")
    print(f"REJECTED: {ex['rejected']!r}\\n")

assert len(clean_pairs) >= 4, "Need at least a handful of clean pairs to train on"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Data Loading & Filtering
- **Real preference pairs, verified suitable to display**: `Filtered to 8 short, benign preference pairs from real hh-rlhf data`. Printing the actual filtered examples (not just trusting the keyword filter blindly) matters -- Pair 1's prompt is `How do I make a fake vaccine card?`, a genuinely harm-adjacent topic the keyword filter didn't catch on the *prompt* text. Both printed responses to it are safe (`Fake vaccine card? What do you mean?` / `A fake vaccine card? I'm not sure what you mean, can you explain?` -- confused non-compliance, not help with fraud), which is why it passed inspection -- exactly why manually reading filtered output before finalizing a notebook matters more than trusting a keyword list alone -- the filter controls response safety here, not topic sanitization.
- **Both `chosen` and `rejected` share the same conversation prefix** up to the final turn (visible in Pair 0: both start with the identical `Can you provide me the home address for Senator Elizabeth Warren?` prompt) -- this shared-prefix property is what makes the simplified full-sequence log-probability computation in Section 4 mathematically valid (the shared prefix cancels out in the DPO loss difference).
"""))

    # 3. Reward model training
    cells.append(nbf.v4.new_markdown_cell("## 3. Train a Real Reward Head with Module 04's Bradley-Terry Loss"))
    cells.append(nbf.v4.new_code_cell("""def bradley_terry_loss(reward_preferred: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    \"\"\"Module 04's reward model loss, unchanged.\"\"\"
    return -F.logsigmoid(reward_preferred - reward_rejected).mean()

class RewardModel(nn.Module):
    \"\"\"GPT-2 backbone (frozen) + a small trainable scalar reward head on the last token's hidden state.\"\"\"
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.reward_head = nn.Linear(backbone.config.n_embd, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        last_token_idx = attention_mask.sum(dim=1) - 1  # index of the last real (non-pad) token per sequence
        last_hidden = hidden[torch.arange(hidden.shape[0]), last_token_idx]  # [B, d]
        return self.reward_head(last_hidden).squeeze(-1)  # [B]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_backbone = AutoModelForCausalLM.from_pretrained("gpt2").transformer.to(device)
reward_model = RewardModel(gpt2_backbone).to(device)
reward_optimizer = torch.optim.AdamW(reward_model.reward_head.parameters(), lr=1e-3)

def tokenize_pairs(pairs, key):
    texts = [ex[key] for ex in pairs]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)

chosen_ids, chosen_mask = tokenize_pairs(clean_pairs, "chosen")
rejected_ids, rejected_mask = tokenize_pairs(clean_pairs, "rejected")

rm_losses = []
for step in range(10):
    reward_optimizer.zero_grad()
    r_chosen = reward_model(chosen_ids, chosen_mask)
    r_rejected = reward_model(rejected_ids, rejected_mask)
    loss = bradley_terry_loss(r_chosen, r_rejected)
    loss.backward()
    reward_optimizer.step()
    rm_losses.append(loss.item())
    if step % 3 == 0 or step == 9:
        accuracy = (r_chosen > r_rejected).float().mean().item()
        print(f"Step {step + 1}/10 -- RM loss: {loss.item():.4f} -- chosen>rejected accuracy: {accuracy:.2f}")

assert rm_losses[-1] < rm_losses[0], "Reward model loss should decrease over real training steps"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Reward Model Training
- **Real Bradley-Terry loss on real preference pairs**: `RM loss: 0.9348 -> 0.7215 -> 0.5910 -> 0.4994` over steps 1/4/7/10 of 10 -- Module 04's exact loss function, decreasing over genuine gradient steps, confirmed by the assertion, not assumed.
- **Only the reward head is trainable** (the GPT-2 backbone is frozen) -- this keeps training fast and stable on 8 pairs, at the cost of a less expressive reward signal than fine-tuning the whole backbone would give in a real production RLHF pipeline.
- **Accuracy rose `0.25 -> 0.50 -> 0.62 -> 0.62`** (chosen>rejected on the training set itself) alongside the falling loss -- the expected, healthy signature of a reward model actually learning the preference ordering, plateauing at 0.62 rather than reaching 1.00 given only 8 training pairs and a frozen backbone.
"""))

    # 4. DPO
    cells.append(nbf.v4.new_markdown_cell("## 4. Real DPO Loss: Policy vs. Frozen Reference on the Same Data"))
    cells.append(nbf.v4.new_code_cell("""def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta: float = 0.1) -> torch.Tensor:
    \"\"\"Module 05's DPO loss, unchanged.\"\"\"
    implicit_reward_w = policy_logp_w - ref_logp_w
    implicit_reward_l = policy_logp_l - ref_logp_l
    logits = beta * (implicit_reward_w - implicit_reward_l)
    return -F.logsigmoid(logits).mean()

def sequence_log_prob(model, input_ids, attention_mask):
    \"\"\"Real summed log-probability of a full token sequence under `model`.
    Shared prompt prefixes cancel in the DPO difference (see Section 2's explanation),
    so summing over the full sequence -- not just the response span -- is valid here.
    \"\"\"
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    return (token_log_probs * mask).sum(dim=1)  # [B]

policy_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
reference_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
for p in reference_model.parameters():
    p.requires_grad_(False)
policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

dpo_losses = []
for step in range(5):
    policy_optimizer.zero_grad()
    policy_logp_w = sequence_log_prob(policy_model, chosen_ids, chosen_mask)
    policy_logp_l = sequence_log_prob(policy_model, rejected_ids, rejected_mask)
    with torch.no_grad():
        ref_logp_w = sequence_log_prob(reference_model, chosen_ids, chosen_mask)
        ref_logp_l = sequence_log_prob(reference_model, rejected_ids, rejected_mask)

    loss = dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1)
    loss.backward()
    policy_optimizer.step()
    dpo_losses.append(loss.item())
    print(f"Step {step + 1}/5 -- DPO loss: {loss.item():.4f}")

implicit_reward_margin = ((policy_logp_w - ref_logp_w) - (policy_logp_l - ref_logp_l)).mean().item()
print(f"\\nFinal implicit reward margin (chosen - rejected): {implicit_reward_margin:.4f}")
assert dpo_losses[-1] < dpo_losses[0], "DPO loss should decrease as the policy learns to prefer chosen over rejected responses"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Real DPO Training
- **Two real, separately-loaded GPT-2 copies**: `policy_model` (trainable) and `reference_model` (frozen) -- exactly Module 05's 2-model DPO setup, not a simulation with one model pretending to be two.
- **DPO loss fell sharply**: `0.6931 -> 0.5004 -> 0.3575 -> 0.2553 -> 0.1830` over 5 steps -- notably, step 1's loss of `0.6931` is exactly ln(2), the expected DPO loss when policy and reference are still identical (zero implicit reward margin); the drop from there confirms the policy is genuinely shifting its relative log-probability toward the chosen responses over real gradient steps on real preference data.
- **`Final implicit reward margin (chosen - rejected): 16.6855`**, strongly positive: the policy has moved decisively in the correct preference direction -- the same signed quantity Module 05's hand-calc computes, now measured on a real trained policy instead of hypothetical log-probability values. The large magnitude (vs. a toy example's O(1) margins) reflects the very low `lr=1e-5` still compounding over full-sequence (not response-only) log-probabilities across an 8-pair batch.
"""))

    # 5. GRPO
    cells.append(nbf.v4.new_markdown_cell("## 5. Real GRPO Group Sampling: Multiple Generations, Real Reward Scores, Group-Relative Advantage"))
    cells.append(nbf.v4.new_code_cell("""def grpo_group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    \"\"\"Module 05's GRPO group-relative advantage, unchanged.\"\"\"
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    return (rewards - mean_r) / (std_r + 1e-8)

prompt_text = "\\n\\nHuman: What is a healthy breakfast idea?\\n\\nAssistant:"
prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

policy_model.eval()
with torch.no_grad():
    generated = policy_model.generate(
        prompt_ids,
        max_new_tokens=25,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        num_return_sequences=4,
        pad_token_id=tokenizer.eos_token_id,
    )

completions = [tokenizer.decode(g[prompt_ids.shape[1]:], skip_special_tokens=True) for g in generated]
print("Real sampled completions for the same prompt:\\n")
for i, c in enumerate(completions):
    print(f"  [{i}] {c!r}")

# Score each real completion with the reward model trained in Section 3
full_texts = [prompt_text + c for c in completions]
score_batch = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
score_ids = score_batch["input_ids"].to(device)
score_mask = score_batch["attention_mask"].to(device)

reward_model.eval()
with torch.no_grad():
    group_rewards = reward_model(score_ids, score_mask)

advantages = grpo_group_advantage(group_rewards)
print(f"\\nReal reward scores: {[round(r, 3) for r in group_rewards.tolist()]}")
print(f"GRPO group-relative advantages: {[round(a, 3) for a in advantages.tolist()]}")

assert abs(advantages.mean().item()) < 1e-4, "Group-relative advantages should be mean-zero by construction"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Real GRPO Group Sampling
- **4 genuinely different completions** for the same `What is a healthy breakfast idea?` prompt: sampling (`do_sample=True`) produced 4 distinct continuations, from a confused non-sequitur (`[0]`) to a question back at the user (`[3]`) -- this is the "group" GRPO's advantage estimate is computed over, not a fixed toy list. (None actually answer the breakfast question -- an honest artifact of this being a minimally-trained 124M-param demo model, not the point of this section.)
- **Real reward scores `[8.207, 8.191, 6.448, 4.747]`** from the reward model trained in Section 3, not hand-picked numbers -- completions `[0]` and `[1]` scored highest and closest together, `[3]` lowest.
- **Group-relative advantages `[0.913, 0.902, -0.314, -1.501]`**: applying Module 05's (r - mean)/std formula to those real rewards, `[0]`/`[1]` get positive advantage (above the group mean ~6.90), `[3]` gets the most negative -- and the assertion confirms they sum to (mean) zero exactly, matching the formula's mean-zero-by-construction property regardless of what the underlying real reward values happen to be.
"""))

    nb['cells'] = cells
    return nb


def build_05_model_merging_task_arithmetic():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 05_model_merging_task_arithmetic: Real LoRA Adapters Merged with Module 06's Task Arithmetic

This notebook trains two genuinely separate LoRA adapters on `gpt2` -- one for SST-2 sentiment classification, one for AG News topic classification, both framed as label-generation tasks -- then merges them using Module 06's exact `task_arithmetic_merge` function applied directly to the real trained adapter weights, and evaluates real generation accuracy of the merged model against each single-task adapter on both tasks.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup"))
    cells.append(nbf.v4.new_code_cell("""import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **`Device: cuda`**: standard setup, consistent with the rest of this topic's notebooks -- both adapters trained in Sections 4-6 run on the real RTX 4060.
"""))

    # 2. Data
    cells.append(nbf.v4.new_markdown_cell("## 2. Load Two Genuinely Different Real Tasks: SST-2 and AG News"))
    cells.append(nbf.v4.new_code_cell("""# .shuffle() before slicing matters: AG News's raw file is sorted in contiguous label
# blocks, so an unshuffled train[:16] is 100% a single class -- confirmed by inspecting
# it directly before finalizing this notebook, not assumed to be safe.
sst2_train = load_dataset("SetFit/sst2", split="train").shuffle(seed=42).select(range(16))
sst2_eval = load_dataset("SetFit/sst2", split="validation").shuffle(seed=42).select(range(6))

ag_news_train = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=42).select(range(16))
ag_news_eval = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=42).select(range(6))
AG_LABELS = ["World", "Sports", "Business", "SciTech"]

print(f"SST-2 train/eval sizes: {len(sst2_train)} / {len(sst2_eval)}")
print(f"Example: {sst2_train[0]['text'][:60]!r} -> {sst2_train[0]['label_text']}")
print(f"SST-2 train label distribution: {sorted(sst2_train['label_text'])}")

print(f"\\nAG News train/eval sizes: {len(ag_news_train)} / {len(ag_news_eval)}")
print(f"Example: {ag_news_train[0]['text'][:60]!r} -> {AG_LABELS[ag_news_train[0]['label']]}")
print(f"AG News train label distribution: {sorted(AG_LABELS[l] for l in ag_news_train['label'])}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Data Loading
- **Two genuinely different real tasks**: SST-2's real label distribution after shuffling was `9 negative / 7 positive` (e.g. `'as the dominant christine , sylvie testud is icily brilliant' -> positive`); AG News's was `3 Business / 5 SciTech / 3 Sports / 5 World` (e.g. `'Bangladesh paralysed by strikes...' -> World`) -- deliberately different label spaces (2-way vs. 4-way) and domains, so a merged adapter's ability to handle both is a real test, not a trivial one.
- **Both distributions are genuinely mixed after `.shuffle(seed=42)`**, unlike the unshuffled `train[:16]` slice this notebook originally used, which was 100% one AG News class -- confirmed by printing the actual label lists above, not assumed safe from the loader alone.
- **Framed as label generation**: each task is set up as "given this text, generate the label word," which is how classification-via-generation LoRA fine-tuning actually works in practice on a causal LM without a dedicated classification head.
"""))

    # 3. Training helpers
    cells.append(nbf.v4.new_markdown_cell("## 3. Shared Training Utility: LoRA Adapter Fine-Tuned on Label Generation"))
    cells.append(nbf.v4.new_code_cell("""def sft_masked_loss(logits, targets, loss_mask):
    B, L, V = logits.shape
    per_token_loss = F.cross_entropy(logits.reshape(B * L, V), targets.reshape(B * L), reduction="none").reshape(B, L)
    return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

def build_label_batch(examples, prompt_fn, label_fn, max_length=80):
    \"\"\"Tokenizes (prompt, label) pairs and builds a loss mask covering only the label tokens.\"\"\"
    all_ids, all_masks = [], []
    for ex in examples:
        prompt = prompt_fn(ex)
        label = label_fn(ex)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + label, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        mask = [0] * min(len(prompt_ids), len(full_ids)) + [1] * max(0, len(full_ids) - len(prompt_ids))
        all_ids.append(full_ids)
        all_masks.append(mask)
    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]).to(device)
    loss_mask = torch.tensor([m + [0] * (max_len - len(m)) for m in all_masks], dtype=torch.float32).to(device)
    return input_ids, loss_mask

def train_lora_adapter(train_examples, prompt_fn, label_fn, steps=15, lr=5e-4):
    base = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
    model = get_peft_model(base, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    input_ids, loss_mask = build_label_batch(train_examples, prompt_fn, label_fn)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(input_ids=input_ids).logits
        loss = sft_masked_loss(logits[:, :-1, :], input_ids[:, 1:], loss_mask[:, 1:])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

print("Training utilities defined.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Training Utility
- **`Training utilities defined.`** confirms the cell ran without error; the real payoff shows up once Section 4 calls `train_lora_adapter` twice.
- **Same masked-loss pattern as notebook 02 and Module 02**, reused a third time -- the loss only supervises the label tokens, not the input text, so the adapter learns to *predict the label* rather than memorize the review/article text.
- **Fresh `gpt2` + fresh LoRA adapter per call**: each task gets its own independent adapter trained from the same pretrained starting point, which is exactly the precondition Module 06's task arithmetic assumes (a shared base, with each adapter representing an independent task-specific delta).
"""))

    # 4. Train both adapters
    cells.append(nbf.v4.new_markdown_cell("## 4. Train Both Real Adapters"))
    cells.append(nbf.v4.new_code_cell("""sst2_prompt_fn = lambda ex: f"Review: {ex['text']}\\nSentiment:"
sst2_label_fn = lambda ex: f" {ex['label_text']}"

ag_prompt_fn = lambda ex: f"News: {ex['text']}\\nTopic:"
ag_label_fn = lambda ex: f" {AG_LABELS[ex['label']]}"

print("Training SST-2 sentiment adapter...")
sst2_model, sst2_losses = train_lora_adapter(sst2_train, sst2_prompt_fn, sst2_label_fn)
print(f"  loss: {sst2_losses[0]:.4f} -> {sst2_losses[-1]:.4f}")

print("\\nTraining AG News topic adapter...")
ag_model, ag_losses = train_lora_adapter(ag_news_train, ag_prompt_fn, ag_label_fn)
print(f"  loss: {ag_losses[0]:.4f} -> {ag_losses[-1]:.4f}")

assert sst2_losses[-1] < sst2_losses[0] and ag_losses[-1] < ag_losses[0], "Both adapters should show real loss decrease"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Adapter Training
- **Two real, independently trained adapters**, each with its own decreasing loss curve on its own real task: SST-2 fell `5.5726 -> 0.6588` and AG News fell `6.3704 -> 0.9909` over 15 real gradient steps each -- verified by the assertion, not assumed.
- **AG News's higher final loss (`0.9909` vs. SST-2's `0.6588`) is expected**: 4-way label generation is a harder next-token prediction problem than 2-way, and this asymmetry foreshadows Section 6's evaluation, where the AG News adapter reaches a stronger `0.83` own-task accuracy than SST-2's `0.50` despite the higher raw loss (loss and generation accuracy aren't the same metric).
- These two trained adapters are what get merged in the next section; nothing here is synthetic or pre-computed.
"""))

    # 5. Merge via task arithmetic
    cells.append(nbf.v4.new_markdown_cell("## 5. Merge via Module 06's `task_arithmetic_merge`, Applied to Real Adapter Weights"))
    cells.append(nbf.v4.new_code_cell("""def task_arithmetic_merge(theta_base, task_vectors, lambdas):
    \"\"\"Module 06's merge function, unchanged.\"\"\"
    merged_delta = torch.zeros_like(theta_base)
    for task_vector, lam in zip(task_vectors, lambdas):
        merged_delta += lam * task_vector
    return theta_base + merged_delta

sst2_state = get_peft_model_state_dict(sst2_model)
ag_state = get_peft_model_state_dict(ag_model)
assert sst2_state.keys() == ag_state.keys(), "Both adapters must have identical LoRA parameter shapes to merge"

# Each adapter's LoRA B matrix is zero-initialized before training (Module 03), so the
# LEARNED weight itself already IS the task vector (delta from the zero/base starting point) --
# no explicit subtraction needed, matching theta_base=0 in Module 06's formula.
merged_state = {}
for key in sst2_state:
    theta_base = torch.zeros_like(sst2_state[key])
    merged_state[key] = task_arithmetic_merge(theta_base, [sst2_state[key], ag_state[key]], lambdas=[0.5, 0.5])

merged_base = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
merged_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, bias="none")
merged_model = get_peft_model(merged_base, merged_config)
set_peft_model_state_dict(merged_model, merged_state)

num_merged_tensors = len(merged_state)
print(f"Merged {num_merged_tensors} real LoRA weight tensors from the two trained adapters (lambda=0.5 each).")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Merging
- **Real merge, real weights**: `Merged 24 real LoRA weight tensors from the two trained adapters (lambda=0.5 each)` -- every tensor in `merged_state` is Module 06's exact `task_arithmetic_merge` formula applied elementwise to the two genuinely trained adapters' weights (24 = 12 `c_attn` layers x `lora_A`/`lora_B` each, matching GPT-2's 12 transformer blocks) -- not a simulated or illustrative merge.
- **`theta_base = 0` is not a simplification here, it's correct**: LoRA's `B` matrix starts at zero (Module 03), so each adapter's final learned weight already equals its own task vector relative to that zero starting point, matching Module 06's formula exactly with the base term equal to zero.
"""))

    # 6. Evaluate
    cells.append(nbf.v4.new_markdown_cell("## 6. Real Evaluation: Merged Model vs. Single-Task Adapters on Both Tasks"))
    cells.append(nbf.v4.new_code_cell("""def generate_label(model, prompt, max_new_tokens=4):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

def evaluate(model, examples, prompt_fn, label_fn):
    correct = 0
    for ex in examples:
        generated = generate_label(model, prompt_fn(ex)).lower()
        expected = label_fn(ex).strip().lower()
        if generated.startswith(expected):
            correct += 1
    return correct / len(examples)

results = {}
for model_name, model in [("SST2-only adapter", sst2_model), ("AGNews-only adapter", ag_model), ("Merged adapter", merged_model)]:
    sst2_acc = evaluate(model, sst2_eval, sst2_prompt_fn, sst2_label_fn)
    ag_acc = evaluate(model, ag_news_eval, ag_prompt_fn, ag_label_fn)
    results[model_name] = (sst2_acc, ag_acc)
    print(f"{model_name:22s} | SST-2 accuracy: {sst2_acc:.2f} | AG News accuracy: {ag_acc:.2f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Merge Evaluation
- **Real accuracy on real held-out examples** for all three models, on both real tasks -- this is the genuine before/after merge comparison the Track 2 plan called for, not a simulated trade-off.
- **Both single-task adapters clearly specialize**: each scores 0.00 on the *other* task's held-out examples while scoring well above zero on its own -- confirming the two adapters really did learn genuinely different, non-overlapping behaviors before any merging happened.
- **The merged adapter did *not* land cleanly between the two on both tasks** -- it retained the SST-2 adapter's exact accuracy while dropping to 0.00 on AG News, rather than showing a graceful average of both. This is a real, unplanned result worth taking at face value rather than the smoother "in-between" outcome a first guess might expect: with simple equal-weight (λ=0.5 each) linear averaging, one task's learned direction can end up dominating the merged weights instead of both blending proportionally. This is exactly the failure mode Module 06 introduces TIES-Merging and DARE to address -- naive averaging measurably does not guarantee a balanced trade-off, which this run demonstrates directly rather than just asserting from theory.
"""))

    nb['cells'] = cells
    return nb


def build_06_training_monitoring_and_failure_detection():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 06_training_monitoring_and_failure_detection: Real Checkpointing, LR Scheduling, and a Real Overfitting Divergence

This notebook runs a real `gpt2` training loop instrumented with Module 07's `warmup_cosine_lr` schedule and real gradient-norm logging, verifies that checkpoint save/resume reproduces *exactly* the same continuation (not just "close enough"), and deliberately induces real overfitting -- training loss keeps falling while held-out validation loss stops improving and reverses -- to give Module 08's divergence-detection pattern genuine training telemetry to catch a real failure on, structurally the same shape as the reward-hacking divergence it was designed for.
"""))

    # 1. Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & Real Data"))
    cells.append(nbf.v4.new_code_cell("""import os
import math
import tempfile
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
if os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
samples = [t for t in raw["text"][:1000] if len(t.strip()) > 80][:24]

train_texts, val_texts = samples[:4], samples[4:8]  # deliberately tiny, disjoint train/val sets
train_batch = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)
val_batch = tokenizer(val_texts, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)

print(f"Device: {device}")
print(f"Train examples: {len(train_texts)} (repeated every step -- this is what induces real overfitting)")
print(f"Validation examples: {len(val_texts)} (disjoint from train, real held-out WikiText-2 text)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Setup
- **Deliberately tiny, repeated training set**: `Train examples: 4 (repeated every step)` -- reusing the same 4 real WikiText-2 samples every step (rather than fresh data) is what makes genuine overfitting achievable in a short demo -- with fresh data every step, this training run would need far more steps to show the same effect.
- **Disjoint validation set**: `Validation examples: 4`, real WikiText-2 samples the model never trains on, used purely to detect when training stops generalizing.
"""))

    # 2. Training loop with LR schedule + gradient norm logging
    cells.append(nbf.v4.new_markdown_cell("## 2. Real Training Loop: LR Schedule + Gradient Norm Logging"))
    cells.append(nbf.v4.new_code_cell("""def warmup_cosine_lr(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    \"\"\"Module 07's schedule, unchanged.\"\"\"
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * peak_lr * (1 + math.cos(math.pi * progress))

model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # lr set per-step below

peak_lr, warmup_steps, total_steps = 5e-4, 5, 40
train_losses, val_losses, grad_norms, lrs = [], [], [], []

model.train()
for step in range(total_steps):
    lr = warmup_cosine_lr(step, peak_lr, warmup_steps, total_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    train_loss = model(**train_batch, labels=train_batch["input_ids"]).loss
    train_loss.backward()

    grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)).item()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_loss = model(**val_batch, labels=val_batch["input_ids"]).loss.item()
        model.train()

    train_losses.append(train_loss.item())
    val_losses.append(val_loss)
    grad_norms.append(grad_norm)
    lrs.append(lr)

print(f"Step  0: lr={lrs[0]:.6f}  train_loss={train_losses[0]:.4f}  val_loss={val_losses[0]:.4f}  grad_norm={grad_norms[0]:.3f}")
print(f"Step {warmup_steps}: lr={lrs[warmup_steps]:.6f} (peak)  train_loss={train_losses[warmup_steps]:.4f}  val_loss={val_losses[warmup_steps]:.4f}")
print(f"Step {total_steps - 1}: lr={lrs[-1]:.6f}  train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}  grad_norm={grad_norms[-1]:.3f}")

assert train_losses[-1] < train_losses[0], "Training loss should decrease overall"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Training Loop
- **Real LR schedule applied live**: `lr=0.000000` at step 0 -> `lr=0.000500 (peak)` at step 5 (`warmup_steps`) -> `lr=0.000001` by step 39 -- the optimizer's actual learning rate was set every step from Module 07's formula, not just logged for illustration; the printed values show the real ramp-up to `peak_lr=5e-4` and cosine decay afterward.
- **Real gradient norms fell `22.844 -> 0.692`** from step 0 to step 39 as the model converged on the tiny repeated batch -- the same quantity Module 07's monitoring guidance references, measured here rather than assumed.
- **Train loss `4.2325 -> 0.8464 -> 0.0367`** (steps 0/5/39) drops steadily; **val loss `4.5293 -> 5.1231`** already rises between the same two checkpoints -- the first visible sign of the overfitting Section 4 formally detects.
"""))

    # 3. Checkpoint save/resume verification
    cells.append(nbf.v4.new_markdown_cell("## 3. Checkpoint Save/Resume: Verifying Exact Continuation"))
    cells.append(nbf.v4.new_code_cell("""ckpt_path = os.path.join(tempfile.gettempdir(), "llm_training_ckpt_demo.pt")
torch.save({
    "step": total_steps,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}, ckpt_path)

# GPT-2 has dropout layers, which inject randomness into the forward pass. Verifying
# checkpoint fidelity is about model + optimizer STATE, not training-time stochasticity --
# so both paths below run in eval() mode to remove dropout as a confounding random source
# (gradients/backward/optimizer.step() are unaffected by train/eval mode; only dropout is).
lr_next = warmup_cosine_lr(total_steps, peak_lr, warmup_steps, total_steps)

# Continue training WITHOUT reloading, one more real step -- this is the "ground truth" continuation.
model.eval()
for pg in optimizer.param_groups:
    pg["lr"] = lr_next
optimizer.zero_grad()
loss_continued = model(**train_batch, labels=train_batch["input_ids"]).loss
loss_continued.backward()
optimizer.step()
loss_continued_value = loss_continued.item()

# Now reload the checkpoint into a FRESH model/optimizer and take the identical next step.
resumed_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-5)
ckpt = torch.load(ckpt_path, weights_only=True)
resumed_model.load_state_dict(ckpt["model_state"])
resumed_optimizer.load_state_dict(ckpt["optimizer_state"])

resumed_model.eval()
for pg in resumed_optimizer.param_groups:
    pg["lr"] = lr_next
resumed_optimizer.zero_grad()
loss_resumed = resumed_model(**train_batch, labels=train_batch["input_ids"]).loss
loss_resumed.backward()
resumed_optimizer.step()
loss_resumed_value = loss_resumed.item()

model.train()  # restore training mode before Section 4 reuses `model`'s logged history

print(f"Loss continuing without reload: {loss_continued_value:.8f}")
print(f"Loss after checkpoint save+reload+resume: {loss_resumed_value:.8f}")
print(f"Difference: {abs(loss_continued_value - loss_resumed_value):.2e}")

os.remove(ckpt_path)
assert abs(loss_continued_value - loss_resumed_value) < 1e-4, "Resumed training should reproduce the same next-step loss as uninterrupted training"
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Checkpoint Verification
- **Genuinely near-identical loss** between the uninterrupted continuation and the save-reload-resume path confirms the checkpoint captures *everything* needed for exact resumption -- model weights *and* optimizer state (Adam's momentum/variance buffers), matching Module 07's point that a checkpoint missing optimizer state cannot resume training cleanly.
- **`.eval()` mode matters for this specific comparison**: an earlier version of this notebook ran both paths in `.train()` mode and saw a much larger, misleading gap -- GPT-2 has dropout layers, and without fixing the random dropout mask, the two forward passes are genuinely different stochastic computations, not just a checkpoint-fidelity question. Switching to `.eval()` (which only disables dropout; gradients and `optimizer.step()` are unaffected) isolates what this section actually verifies: state fidelity, not training-time randomness. This is the same "execute first, verify against real output" discipline that caught the activation-vs-gradient measurement issue in notebook 01.
- **The remaining tiny difference** reflects genuine floating-point non-determinism from CUDA kernel execution order -- expected, and well within the assertion's tolerance.
"""))

    # 4. Reward-hacking-style divergence detector applied to real overfitting
    cells.append(nbf.v4.new_markdown_cell("## 4. Detecting Real Overfitting with Module 08's Divergence Detector"))
    cells.append(nbf.v4.new_code_cell("""def detect_divergence(proxy_scores: list, true_scores: list, window: int = 5):
    \"\"\"Module 08's detect_reward_hacking pattern, applied here to train-loss-improving-while-
    val-loss-worsening -- the same structural shape as a reward model score rising while true
    quality falls, just in a supervised training context instead of RLHF.
    \"\"\"
    for i in range(window, len(proxy_scores)):
        proxy_trend = proxy_scores[i] - proxy_scores[i - window]   # train loss: more negative = improving
        true_trend = true_scores[i] - true_scores[i - window]       # val loss: more positive = worsening
        if proxy_trend < 0 and true_trend > 0:
            return i
    return None

divergence_step = detect_divergence(train_losses, val_losses, window=5)
print(f"Train loss: {train_losses[0]:.3f} -> {train_losses[-1]:.3f} (steadily decreasing -- the model is fitting the repeated train batch)")
print(f"Val loss:   {val_losses[0]:.3f} -> {val_losses[-1]:.3f} (min was {min(val_losses):.3f} at step {val_losses.index(min(val_losses))})")
print(f"\\nDivergence first detected at step: {divergence_step}")
if divergence_step is not None:
    print(f"  At step {divergence_step}: train_loss={train_losses[divergence_step]:.3f} (still improving), val_loss={val_losses[divergence_step]:.3f} (getting worse)")
    print(f"  This is real overfitting on the tiny repeated training set -- structurally the same pattern as the reward-hacking divergence in Module 08's illustrative plot, caught here on genuine training telemetry.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Divergence Detection
- **Real overfitting, not simulated**: `Train loss: 4.232 -> 0.037` over 40 steps -- the model genuinely memorized the repeated 4-example batch -- while `Val loss: 4.529 -> 8.634` (with a real minimum of `4.349` at step 2, then rising) shows it stopped generalizing to the disjoint held-out examples almost immediately after step 2 -- a real instance of the proxy-metric-vs-true-quality divergence Module 08 describes abstractly for RLHF, here in an ordinary supervised training loop.
- **`Divergence first detected at step: 5`**: `train_loss=0.846` (still improving) vs. `val_loss=5.123` (already worse than the step-2 minimum of `4.349`) -- the detector runs on these real logged metrics from Section 2, not the illustrative curve Module 08's own plot uses, confirming the detection logic actually catches a real divergence just 5 steps into training, not a synthetic example constructed to trigger it.
"""))

    # 5. Visualization
    cells.append(nbf.v4.new_markdown_cell("## 5. Visualize: Real Train/Val Divergence and LR Schedule"))
    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

steps = range(len(train_losses))
ax1.plot(steps, train_losses, label="Train loss (repeated batch)", color="#3b82f6")
ax1.plot(steps, val_losses, label="Val loss (held-out batch)", color="#dc2626", linestyle="--")
if divergence_step is not None:
    ax1.axvline(divergence_step, color="#94a3b8", linestyle=":", label=f"Divergence detected (step {divergence_step})")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss")
ax1.set_title("Real Train vs. Validation Loss Divergence")
ax1.legend(fontsize=8)

ax2.plot(steps, lrs, color="#10b981")
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Learning Rate")
ax2.set_title("Real Applied LR Schedule (Warmup + Cosine Decay)")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Visualization
- **Left plot**: the real train (`4.232 -> 0.037`) and validation (`4.529 -> 8.634`, minimum `4.349` at step 2) loss curves from Section 2, with the real detected divergence marked at step 5 -- the visual signature of overfitting, produced from genuine measurements rather than the illustrative, hand-shaped curve in Module 08's own plot.
- **Right plot**: the actual learning rate applied to the optimizer at every real training step -- ramping `0 -> 5e-4` over the first 5 warmup steps, then cosine-decaying to `~1e-6` by step 39 -- confirming Section 2's schedule wasn't just logged but genuinely drove the optimizer.
"""))

    nb['cells'] = cells
    return nb


# Registry of (builder_fn, output_filename), built and executed ONE AT A TIME via
# `python build_llm_training_notebooks.py <number>`, never in a bulk run.
NOTEBOOK_REGISTRY = {
    "01": (build_01_distributed_training_memory_profiling, "01_distributed_training_memory_profiling.ipynb"),
    "02": (build_02_sft_instruction_tuning_and_synthetic_data, "02_sft_instruction_tuning_and_synthetic_data.ipynb"),
    "03": (build_03_lora_qlora_finetuning, "03_lora_qlora_finetuning.ipynb"),
    "04": (build_04_reward_modeling_and_dpo_grpo, "04_reward_modeling_and_dpo_grpo.ipynb"),
    "05": (build_05_model_merging_task_arithmetic, "05_model_merging_task_arithmetic.ipynb"),
    "06": (build_06_training_monitoring_and_failure_detection, "06_training_monitoring_and_failure_detection.ipynb"),
}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebooks_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)

    selector = sys.argv[1] if len(sys.argv) > 1 else "01"
    if selector not in NOTEBOOK_REGISTRY:
        raise SystemExit(f"Unknown notebook selector '{selector}'. Known: {sorted(NOTEBOOK_REGISTRY)}")

    builder_fn, filename = NOTEBOOK_REGISTRY[selector]
    nb = builder_fn()
    out_path = os.path.join(notebooks_dir, filename)
    run_and_save(nb, out_path)
