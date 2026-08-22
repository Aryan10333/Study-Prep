# Implementation Plan: Track 2 — Companion Notebooks (`02_llm_training_foundations`)

Scope: this plan covers only the `notebooks/` companion notebooks, per `notebook_generator/SKILL.md`. Track 1 (8 study-guide modules) is complete and signed off; Track 3 (Interview Q&A) is separate and not covered here.

---

## 0. Environment Reality Check (affects every notebook below)

| Check | Result |
|---|---|
| GPU present | **Yes** — NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM, driver 581.86, CUDA 13.0 |
| PyTorch build installed | **CPU-only** (`2.13.0+cpu`) — `torch.cuda.is_available()` returns `False` despite the GPU existing |
| `transformers` | Installed (5.14.1) |
| `datasets` | Installed (5.0.0) |
| `peft` | Not installed |
| `trl` | Not installed |
| `accelerate` | Not installed |
| `bitsandbytes` | Not installed |
| `HF_TOKEN` / `OPENAI_API_KEY` / `GROQ_API_KEY` | All set in `.env` |

This topic's entire pedagogical focus is memory footprints and hardware-aware trade-offs (per the syllabus's Profile Focus) — so whether these notebooks profile *real VRAM* on the actual GPU or only CPU RAM as an approximation materially changes how valuable they are. This is flagged as Open Question 1 below; I'm not installing anything without sign-off first.

---

## 1. Notebook List & Target File Paths

| # | File path | Maps to Module(s) |
|---|---|---|
| 01 | `notebooks/01_distributed_training_memory_profiling.ipynb` | 01 (Fine-Tuning Fundamentals & Distributed Training) |
| 02 | `notebooks/02_sft_instruction_tuning_and_synthetic_data.ipynb` | 02 (SFT & Instruction Tuning) |
| 03 | `notebooks/03_lora_qlora_finetuning.ipynb` | 03 (PEFT: LoRA, QLoRA) |
| 04 | `notebooks/04_reward_modeling_and_dpo_grpo.ipynb` | 04 + 05 (Reward Modeling/RLHF, DPO/GRPO) |
| 05 | `notebooks/05_model_merging_task_arithmetic.ipynb` | 06 (Model Merging & Adapters) |
| 06 | `notebooks/06_training_monitoring_and_failure_detection.ipynb` | 07 + 08 (Production Monitoring, Failure Modes) |

Each notebook is built and executed **one at a time, sequentially**, per `notebook_generator/SKILL.md` — not in a batch run — so I can inspect real output before writing the paired explanation cells.

---

## 2. Real-World Datasets & APIs Per Notebook

| # | Datasets / APIs | Why this one |
|---|---|---|
| 01 | `wikitext-2-raw-v1` (HF Hub, `wikitext` dataset) | Small, standard, no gating — enough real text to run genuine training steps and measure real memory deltas |
| 02 | `databricks/databricks-dolly-15k` (HF Hub, real human-written instructions) + a **live GROQ or OpenAI API call** to generate a handful of synthetic instruction examples | Lets the notebook demonstrate the real synthetic-data pipeline (generate → filter → dedupe → contamination-check) from Module 02 against genuine human-written data as the baseline, using the API keys already in `.env` |
| 03 | Same `wikitext-2-raw-v1` slice as notebook 01, so the notebook can directly compare full-FT vs. LoRA vs. QLoRA memory/param counts on an identical task | Controlled comparison, matching the hand-calc structure in Module 03 |
| 04 | `Anthropic/hh-rlhf` (HF Hub, real human preference pairs), small sampled subset | Standard, real, appropriately sized preference dataset for training an actual (tiny) reward model and running real DPO |
| 05 | `sst2` (real sentiment classification) + `ag_news` (real topic classification) — two genuinely different small HF datasets | Fine-tune two separate small LoRA adapters on genuinely different tasks, so the merge in Module 06 has real (not synthetic) task vectors to combine and a real before/after eval on both tasks |
| 06 | Reuses training runs/logs from notebooks 01-03 (no new external data) | Module 07/08's monitoring and failure-detection code operates on training telemetry, which this notebook generates itself via a real training loop |

Base model across all notebooks: **`gpt2`** (124M params, ungated, small enough to fine-tune quickly even on CPU for a few steps, and large enough that memory/parameter-count comparisons are meaningful). Using one consistent base model lets numbers be compared notebook-to-notebook.

---

## 3. Engineering Pipelines, Hardware Assertions & Profiling Steps

- **Notebook 01**: Load `gpt2`, measure real process memory (RSS via `psutil`, and `torch.cuda.max_memory_allocated()` if GPU is enabled per Open Question 1) before/after: (a) loading params, (b) running one backward pass (gradients), (c) initializing AdamW (optimizer states) — reproducing Module 01's "16Ψ bytes" breakdown with *real measured numbers* instead of the hand-calc's assumed 7B model. Then a real gradient-accumulation training loop with logged effective batch size. Since true multi-GPU ZeRO/FSDP cannot physically run on one GPU, the notebook applies the ZeRO partitioning *formula* to the real measured single-replica numbers to project what N-GPU memory would be — explicitly labeled as a projection, not an actual distributed run.
- **Notebook 02**: Real SFT training loop on `gpt2` with prompt-loss masking (reusing Module 02's `sft_masked_loss`) against `dolly-15k` examples; a live API call generating synthetic instruction pairs, then running them through the n-gram overlap/decontamination check from Module 08 against the real eval set to demonstrate the risk concretely.
- **Notebook 03**: Reuses Module 03's from-scratch `LoRALinear` (no `peft` dependency needed), attached to `gpt2`'s attention projections; measures real trainable-parameter counts and real memory before/after vs. full fine-tuning. QLoRA's 4-bit step uses PyTorch-native quantization (`torch.ao.quantization`, CPU-compatible) as an honest, clearly-labeled substitute for `bitsandbytes` NF4 (see Open Question 2).
- **Notebook 04**: Trains a tiny scalar reward head on top of `gpt2` using Module 04's `bradley_terry_loss` against real `hh-rlhf` preference pairs; then runs Module 05's `dpo_loss` on the same data comparing policy vs. a frozen reference copy; then samples real multiple generations from the model for one prompt and computes Module 05's `grpo_group_advantage` on their real (reward-model-scored) rewards.
- **Notebook 05**: Trains two separate small LoRA adapters (via notebook 03's `LoRALinear`) on `sst2` and `ag_news` respectively; merges them via Module 06's `task_arithmetic_merge`; evaluates the merged model's real accuracy on both tasks vs. each single-task adapter, to show a genuine (not simulated) merge trade-off.
- **Notebook 06**: Real training loop instrumented with Module 07's `warmup_cosine_lr` schedule and checkpoint save/resume (verifying exact resumption of optimizer state); logs real gradient norms; feeds the logged metrics through Module 08's `detect_reward_hacking`-style divergence check.

All notebooks include explicit `assert` statements on tensor shapes and numeric bounds, and headless execution uses `matplotlib.use('Agg')` with `%matplotlib inline` + `plt.show()` for any inline plots, per the skill.

---

## 4. Open Design Questions (Require Sign-Off)

1. **Install CUDA-enabled PyTorch to unlock real GPU VRAM profiling?** The RTX 4060 (8GB) is physically present but unused because the installed `torch` is a CPU-only build. Reinstalling with a CUDA wheel (~2-3GB download) would let every notebook measure genuine `torch.cuda.max_memory_allocated()` VRAM numbers instead of CPU RAM approximations — directly serving this topic's memory-footprint focus. This changes the shared `.venv` used by other topics too, so I want explicit confirmation before doing it.
2. **Install `peft`, `trl`, `accelerate` (lightweight, CPU/GPU-compatible)?** These would let notebooks 03-04 optionally cross-check the from-scratch LoRA/DPO implementations against the standard library implementations (a valuable "does my from-scratch version match the real library" sanity check), without replacing the from-scratch code the study guide already teaches. `bitsandbytes` is proposed to be **skipped** — true NF4 quantization needs a CUDA-specific build that's often unreliable outside Linux/WSL, so QLoRA's 4-bit step uses PyTorch-native quantization instead regardless of the GPU decision above.
3. **`hh-rlhf` and `dolly-15k` download size**: both are standard, small HF datasets (low tens of MB for the slices needed), no gating, should download in seconds with the existing `HF_TOKEN`.

## Status: Complete

All 6 notebooks built via `helpers/build_llm_training_notebooks.py`, executed end-to-end on the real RTX 4060 GPU, and verified cell-by-cell against their actual output before finalizing (per the "execute first, explain second" discipline) -- every explanation cell describes what genuinely happened, not a prediction.

| # | Notebook | Real bugs found & fixed via execution |
|---|---|---|
| 01 | Distributed training memory profiling | `memory_allocated()` delta silently conflated retained forward activations with gradient memory (fixed via explicit `del outputs, loss`); result now matches the 16&Psi; formula within ~2% |
| 02 | SFT + synthetic data | None beyond the shared Module 02 fix below |
| 03 | LoRA/QLoRA + peft cross-check | Confirmed from-scratch `LoRALinear` matches `peft`'s internal formula exactly (65,536 params, both ways) |
| 04 | Reward modeling + DPO/GRPO | `hh-rlhf` content-safety filtering required manual verification of filtered output, not just trusting the keyword list |
| 05 | Model merging | `train[:16]` on `fancyzhx/ag_news` was 100% one class (dataset not pre-shuffled) -- fixed with `.shuffle(seed=42)`; merge result didn't land "in between" as predicted, corrected explanation to describe the real naive-averaging dominance observed instead |
| 06 | Training monitoring + failure detection | GPT-2's dropout made the checkpoint-resume comparison non-deterministic for unrelated reasons; fixed by running the verification in `.eval()` mode, which then matched exactly (0.00e+00 difference) |

**Shared fix**: `sft_masked_loss`'s `.view()` calls broke on non-contiguous tensor slices (a realistic usage pattern this module's own hand-written demo never exercised) -- fixed to `.reshape()` in both Module 02's study-guide code and the notebook builder.

Environment: CUDA-enabled PyTorch (`torch==2.13.0+cu126`) and `peft`/`trl`/`accelerate` installed per the sign-off above; `bitsandbytes` skipped, QLoRA approximated with real int8 quantization instead of NF4.
