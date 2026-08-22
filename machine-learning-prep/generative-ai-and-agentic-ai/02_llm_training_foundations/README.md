# 02_llm_training_foundations: Fine-Tuning, Alignment & Training Infrastructure Syllabus

## 1. Context & Alignment
* **Profile Focus:** AI Engineer / Applied AI SDE-2/SDE-3 (Aryan Chandra). Emphasizes production engineering, memory footprints of optimizer/RL pipelines, and hardware-aware training trade-offs over purely academic RL theory or proofs.
* **Interview Frequency:** Extremely High. This is the core of "how do you actually turn a pretrained model into a usable assistant" — a standard deep-dive in LLM Engineer, Applied AI Engineer, and MLE screens at frontier labs, AI infra companies, and high-growth AI startups.
* **Core Goal:** Master the full post-pretraining pipeline — distributed training infrastructure, supervised fine-tuning, parameter-efficient fine-tuning (LoRA/QLoRA), reward modeling, RLHF/GRPO, DPO and modern alignment methods, model merging, and the production engineering discipline (monitoring, failure modes) needed to run these training jobs reliably. This topic deliberately starts from an already-pretrained model — Transformer architecture, pretraining objectives, and scaling laws are owned by `01_llm_foundations` and are not re-covered here.

## 2. Module Chapters & Conceptual Scope
These chapters outline the sequential topics to be covered:

- **Module 01: Fine-Tuning Fundamentals & Distributed Training Infrastructure**
  - *Key Concepts:* Full fine-tuning vs. parameter-efficient tuning, optimizer state memory cost (Adam's momentum/variance buffers), mixed precision training (FP16/BF16/FP8), gradient checkpointing, Data/Tensor/Pipeline Parallelism, ZeRO stages 1–3, **FSDP (and how it differs from ZeRO-style sharding, and when to reach for each)**, and **gradient accumulation** (micro-batch size, accumulation steps, GPU count, and effective batch size, worked through a simple numerical calculation).
  - *System Bottlenecks & Focus:* The "12–20 bytes per parameter" optimizer-state VRAM wall, GPU interconnect bandwidth bounds (NVLink/InfiniBand), and communication-compute overlap during distributed training.

- **Module 02: Supervised Fine-Tuning (SFT) & Instruction Tuning**
  - *Key Concepts:* SFT objective and prompt-loss masking, chat template formatting, instruction dataset construction and curation (quality vs. quantity), multi-turn conversation handling, catastrophic forgetting, and **synthetic instruction-data generation** (data quality, diversity, deduplication, contamination, and the specific risks of training on synthetic data).
  - *System Bottlenecks & Focus:* Full-parameter fine-tuning VRAM cost, and the catastrophic-forgetting vs. dataset-diversity trade-off.

- **Module 03: Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters)**
  - *Key Concepts:* LoRA low-rank weight decomposition, rank ($r$) and alpha hyperparameter selection, QLoRA (NF4 quantization, double quantization, paged optimizers), Adapter layers, Prefix/Prompt Tuning, and **LoRA target module selection** (`q_proj`/`k_proj`/`v_proj`/`o_proj` vs. also including FFN layers, and the resulting quality-vs-trainable-parameter trade-off).
  - *System Bottlenecks & Focus:* Quantified VRAM savings vs. full fine-tuning, rank-vs-expressiveness trade-off, and quantization error accumulation.

- **Module 04: Reward Modeling & Reinforcement Learning from Human Feedback (RLHF)**
  - *Key Concepts:* Bradley-Terry preference loss for reward model training, the PPO objective, KL-divergence penalty against a frozen reference policy, the full 4-model RLHF pipeline (policy, reference, reward, value). Scoped to understanding the pipeline, objective, KL constraint, and failure modes — not a from-scratch PPO implementation or general RL theory.
  - *System Bottlenecks & Focus:* RL training instability, reward hacking, and the VRAM cost of holding 4 models simultaneously during PPO.

- **Module 05: Direct Preference Optimization (DPO), GRPO & Modern Alignment Methods**
  - *Key Concepts:* DPO's implicit-reward reparameterization (eliminating the explicit reward model), DPO vs. PPO trade-offs, **GRPO** (group-relative advantage estimation and why removing the critic/value model reduces training overhead vs. PPO), and a brief comparative survey of IPO/KTO/ORPO/SimPO.
  - *System Bottlenecks & Focus:* Reduced memory footprint vs. RLHF's 4-model footprint (DPO: 2 models; GRPO: no value model), and the paired-preference-data requirement.

- **Module 06: Model Merging & Adapter Composition**
  - *Key Concepts:* Weight averaging ("model soups"), TIES-Merging, DARE, task arithmetic, multi-adapter composition and routing.
  - *System Bottlenecks & Focus:* Merge conflict/task interference, and the zero-additional-inference-cost benefit of merging vs. serving multiple models.

- **Module 07: Training Production Considerations & Monitoring**
  - *Key Concepts:* Learning-rate schedules and warmup, checkpointing and fault-tolerance for long training runs, training telemetry (loss spikes, gradient-norm monitoring), continued pretraining / domain adaptation, and **broader evaluation strategy** (training loss vs. benchmark evaluation vs. human/LLM-as-judge evaluation, plus regression, safety, and contamination evaluation).
  - *System Bottlenecks & Focus:* Cost of training-run failure and recovery, and monitoring signal design at scale.

- **Module 08: Common Failure Modes & Best Practices**
  - *Key Concepts:* Catastrophic forgetting, reward hacking and mode collapse, data contamination, the "alignment tax," and common evaluation pitfalls during training.
  - *System Bottlenecks & Focus:* Silent quality regressions and designing a tight eval-training feedback loop to catch them early.

Module 09 (Interview Q&A track, generated separately per `interview_qa_generator/SKILL.md`) will cover standalone screening questions across all eight modules above.

---

## Status: Approved

Syllabus reviewed and signed off. Scope locked to: no re-coverage of `01_llm_foundations` content (pretraining objectives, architecture, scaling laws); Module 01 stays standalone; 8 theory modules + 1 Q&A module, no further expansion. Mandatory additions (FSDP, gradient accumulation, synthetic SFT data, GRPO) and optional additions (LoRA target modules, broader evaluation strategy) are folded into the module scopes above. No deep PPO implementation or extensive RL theory in scope.
