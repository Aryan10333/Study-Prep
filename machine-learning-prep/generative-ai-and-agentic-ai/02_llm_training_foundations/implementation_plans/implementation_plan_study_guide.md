# Implementation Plan: Track 1 — Study Notes Modules (`02_llm_training_foundations`)

Scope: this plan covers **only** the 8 theory modules in `modules/` and their compiled `02_llm_training_foundations_master_study_guide.{html,pdf}`, per `study_guide_generator/SKILL.md`. Notebooks (Track 2) and the Interview Q&A module (Track 3) are separate tracks with their own implementation-plan checkpoints, planned after this track is signed off and written.

---

## 1. Module List & Target File Paths

| # | File path | Title |
|---|---|---|
| 01 | `modules/01_finetuning_fundamentals_and_distributed_training.md` | Fine-Tuning Fundamentals & Distributed Training Infrastructure |
| 02 | `modules/02_sft_and_instruction_tuning.md` | Supervised Fine-Tuning (SFT) & Instruction Tuning |
| 03 | `modules/03_peft_lora_qlora.md` | Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters) |
| 04 | `modules/04_reward_modeling_and_rlhf.md` | Reward Modeling & RLHF |
| 05 | `modules/05_dpo_grpo_and_alignment.md` | DPO, GRPO & Modern Alignment Methods |
| 06 | `modules/06_model_merging_and_adapters.md` | Model Merging & Adapter Composition |
| 07 | `modules/07_training_production_and_monitoring.md` | Training Production Considerations & Monitoring |
| 08 | `modules/08_failure_modes_and_best_practices.md` | Common Failure Modes & Best Practices |

Compiled deliverables (Track 1 only, via a new `helpers/compile_llm_training.py` built from the hardened `pdf_compiler` pattern already used for the other two topics — portable `BASE_DIR`, base64 image embedding, `check=True` + retry): `llm_training_foundations_master_study_guide.html` / `.pdf`.

---

## 2. Formulas to Retain (with Hand-Calc Plan) vs. Prose-Only

Per `study_guide_generator/SKILL.md`'s Formula Selection Constraint: only core, frequently-asked concepts get a formula block + step-by-step hand calculation on small numbers; everything else is explained in prose/intuition only, with comparison tables where useful.

### Module 01 — Fine-Tuning Fundamentals & Distributed Training
- **Core (formula + hand calc):**
  - Mixed-precision training memory accounting (params + gradients + Adam optimizer states, the "12–20 bytes/param" wall). Hand calc: a concrete 7B-parameter model, breaking down GB for params (bf16), gradients (bf16), optimizer states (fp32 m/v, ± fp32 master weights).
  - Gradient accumulation → effective batch size: $B_{\text{eff}} = B_{\text{micro}} \times \text{accum\_steps} \times N_{\text{GPUs}}$. Hand calc with concrete small numbers (e.g., micro-batch 4, 8 accumulation steps, 4 GPUs).
  - ZeRO stage memory reduction per GPU (partitioning optimizer states / gradients / parameters across $N$ ranks). Hand calc: same 7B model, per-GPU memory at ZeRO-1 vs. ZeRO-2 vs. ZeRO-3 for $N=8$.
- **Prose-only (no formula block):** FSDP mechanics and how it differs from ZeRO (both shard state, FSDP is PyTorch-native full-parameter sharding with all-gather/reduce-scatter — described conceptually, not derived), Tensor Parallelism / Pipeline Parallelism (shape-splitting description + a tensor-shape table, not a formula), communication-compute overlap (bandwidth-bound intuition only).

### Module 02 — SFT & Instruction Tuning
- **Core (formula + hand calc):** SFT loss with prompt-token masking — cross-entropy computed only over completion tokens. Hand calc: a tiny 6-token sequence with a binary mask vector, computing the masked average loss by hand.
- **Prose-only:** Catastrophic forgetting (intuition only), chat template structure (code example, not math), synthetic instruction-data generation pipeline (generate → filter → dedupe → contamination-check, described procedurally with a comparison table of quality-control techniques).

### Module 03 — PEFT (LoRA, QLoRA, Adapters)
- **Core (formula + hand calc):**
  - LoRA trainable parameter count vs. full fine-tuning: $\Delta W = BA$, $\text{params}_{\text{LoRA}} = 2 \times r \times d$ vs. $d^2$ for a full dense layer. Hand calc: a concrete $d=4096$, $r=8$ attention projection, comparing trainable parameter counts and the resulting VRAM ratio.
  - QLoRA 4-bit (NF4) vs. 16-bit storage memory comparison. Hand calc: same model size in 16-bit vs. NF4-quantized base weights, showing the VRAM reduction in GB.
- **Prose-only:** Adapter layers / Prefix / Prompt Tuning (conceptual comparison table against LoRA), double quantization (secondary optimization, described in prose), target module selection ($q/k/v/o\_proj$ vs. also FFN layers — a trade-off table, not a formula).

### Module 04 — Reward Modeling & RLHF
- **Core (formula + hand calc):**
  - Bradley-Terry pairwise preference loss for reward model training. Hand calc: two candidate responses with toy reward scores, computing the loss for one preference pair.
  - PPO clipped surrogate objective (formula + KL penalty term). Hand calc: one token's probability ratio and advantage estimate plugged in to show the clipping effect numerically — kept to the formula and a single illustrative numeric example, not a full policy-gradient derivation.
- **Prose-only:** The 4-model pipeline memory bookkeeping (policy/reference/reward/value — a memory-footprint table, not a formula), reward hacking (intuition only).

### Module 05 — DPO, GRPO & Modern Alignment
- **Core (formula + hand calc):**
  - DPO loss (implicit reward reparameterization). Hand calc: a toy preferred/dispreferred pair with small log-probability values, computing the DPO loss by hand.
  - GRPO group-relative advantage: $A_i = (r_i - \text{mean}(r)) / \text{std}(r)$ over a sampled group. Hand calc: 4 sampled completions with toy reward values, computing normalized advantages by hand.
- **Prose-only:** IPO/KTO/ORPO/SimPO — comparative survey in a GFM table (method vs. core idea vs. data requirement vs. when to prefer it), no individual formula blocks for each (consistent with how the NLP module treated secondary tokenizer variants).

### Module 06 — Model Merging & Adapter Composition
- **Core (formula + hand calc):** Task arithmetic / weight averaging, $\theta_{\text{merged}} = \theta_{\text{base}} + \sum_i \lambda_i (\theta_i - \theta_{\text{base}})$. Hand calc: two tiny toy weight vectors merged with given $\lambda$ values.
- **Prose-only:** TIES-Merging (trim/elect-sign/merge procedure described step-by-step in prose, not as a single formula), DARE (conceptual), merge conflict/task interference (intuition only).

### Module 07 — Training Production Considerations & Monitoring
- **Core (formula + hand calc):** Learning-rate schedule — linear warmup + cosine decay. Hand calc: LR value at 3 checkpoints (start of warmup, end of warmup, mid-decay) for concrete step counts.
- **Prose-only:** Checkpointing/fault-tolerance strategy, training telemetry (loss spikes, gradient-norm monitoring — referencing the gradient-clipping formula already covered in `00_nlp_fundamentals`, not re-deriving it), continued pretraining/domain adaptation, and the broader evaluation-strategy survey (training loss vs. benchmark vs. human/LLM-judge vs. regression/safety/contamination checks) as a comparison table.

### Module 08 — Common Failure Modes & Best Practices
- **No core formulas** — this module is a diagnostic/catalog module (catastrophic forgetting, reward hacking, mode collapse, data contamination, alignment tax, evaluation pitfalls), structured primarily as a failure-mode-vs-symptom-vs-fix comparison table, matching the pattern of interview "common mistakes" content but at study-guide depth.

---

## 3. Visual Diagrams & Plots

Matplotlib PNGs go in `plots/`, generated via a new `helpers/generate_llm_training_plots.py` (Agg backend, `.venv` execution, following the label-clearance rules in `study_guide_generator/SKILL.md` Section 6 point 7 for any box-and-arrow diagrams). Flowcharts/architecture diagrams use inline responsive SVG directly in the module markdown per the same skill's Section 3 (no Mermaid/ASCII).

| Module | Visual | Type |
|---|---|---|
| 01 | Memory breakdown stacked bar: params / gradients / optimizer states / activations, full FT vs. mixed precision, for a fixed model size | Matplotlib PNG |
| 01 | Per-GPU memory vs. ZeRO stage (0/1/2/3) line or bar chart, $N=8$ | Matplotlib PNG |
| 01 | Data / Tensor / Pipeline Parallelism box diagram showing how model and batch dimensions split across GPUs | Inline SVG |
| 02 | Prompt-loss-masking visualization: token sequence with masked (prompt) vs. unmasked (completion) positions | Inline SVG |
| 02 | Synthetic data pipeline flow: generate → filter → dedupe → contamination-check → train | Inline SVG |
| 03 | LoRA decomposition diagram: $W$ (frozen) + $B \times A$ (trainable low-rank update) with shapes labeled | Inline SVG |
| 03 | VRAM comparison bar chart: Full FT vs. LoRA vs. QLoRA for a fixed model size | Matplotlib PNG |
| 04 | RLHF pipeline flow: SFT model → Reward Model → PPO loop (policy/reference/reward/value boxes) | Inline SVG |
| 04 | Reward vs. KL-divergence trade-off illustrative curve | Matplotlib PNG |
| 05 | Model-count comparison diagram: DPO (2 models) vs. PPO (4 models) vs. GRPO (policy + reference + group samples, no value model) | Inline SVG |
| 06 | Model merging / task-arithmetic vector illustration (base + weighted deltas → merged model) | Inline SVG |
| 07 | LR schedule curve: linear warmup + cosine decay over training steps | Matplotlib PNG |
| 07 | Checkpoint/fault-tolerance recovery flow | Inline SVG |
| 08 | Illustrative reward-hacking chart: proxy reward score climbing while true quality diverges downward (clearly labeled as illustrative, not real benchmark data) | Matplotlib PNG |

All plots are illustrative of a mathematical relationship or architecture (consistent with existing modules like the Chinchilla scaling chart) — none present fabricated numbers as if they were real benchmark results from a specific named model.

---

## 4. Open Design Questions / Dependencies

1. **Cross-referencing, not duplicating:** Modules will link back to `01_llm_foundations` where relevant (e.g., referencing its KV cache / attention modules) rather than re-explaining them — flagging this so cross-references read naturally rather than as dead links.
2. **PPO depth check:** Per your "no deep PPO implementation" guidance, Module 04's PPO hand calc will stay at the level of one clipped-objective numeric example, not a full advantage-estimation/GAE walkthrough — flagging this boundary explicitly so it's easy to push back on if you want even less (or slightly more) PPO math.
3. **Compiler script:** Plan assumes a new `helpers/compile_llm_training.py`, written directly from the already-hardened pattern (portable paths, base64 images, retry logic) rather than the older pattern that had to be retrofitted twice — no open question here, just noting it avoids repeating known mistakes.
4. **Notebooks and Q&A are out of scope for this plan** and will each get their own implementation-plan checkpoint once Track 1 is complete, per `notebook_generator/SKILL.md` and `interview_qa_generator/SKILL.md`.

---

Awaiting sign-off before writing any of the 8 raw module files.
