# 01_llm_foundations: Transformer & Modern LLM Architectures Syllabus

## 1. Context & Alignment
* **Profile Focus:** AI Engineer / Applied AI SDE-2/SDE-3 (Aryan Chandra). Emphasizes production engineering, memory footprints, hardware walls, and gradient dynamics over purely academic proofs.
* **Interview Frequency:** Extremely High. A core component of modern system design and model architecture screens at top tech companies and high-growth AI startups.
* **Core Goal:** Master the architectural layout, mathematical foundations, and hardware-aware optimizations of modern Transformer models (attention variants, normalization layers, KV Cache, FlashAttention, and Mixture of Experts architectures).

## 2. Module Chapters & Conceptual Scope
These chapters outline the sequential topics to be covered:

- **Module 01: The Transformer Architecture (Encoder & Decoder Core)**
  - *Key Concepts:* The original Vaswani et al. architecture, Scaled Dot-Product Attention, Multi-Head Self-Attention (MHA) mechanics, Query/Key/Value projection matrices, residual connections, LayerNorm placement (Pre-LN vs Post-LN vs RMSNorm), and Feed-Forward Networks (FFN).
  - *System Bottlenecks & Focus:* Vanishing/exploding gradients in Deep Transformers, LayerNorm scaling effects, projection matrix parameter footprint, and basic quadratic attention complexity $O(L^2)$ in sequence length.

- **Module 02: Positional Encodings & Embeddings**
  - *Key Concepts:* Absolute (Learned & Sinusoidal) vs. Relative positional encodings, Rotary Position Embeddings (RoPE) complex plane rotations and dot-product preservation, Attention with Linear Biases (ALiBi) linear decay, and context window scaling interpolation techniques (YaRN, NTK-aware scaling, RoPE base frequency scaling).
  - *System Bottlenecks & Focus:* Generalization limits on unseen context lengths, memory consumption of large context windows, and compute overhead of relative/rotary transformations.

- **Module 03: Modern Activation Functions & Normalization Layer Variants**
  - *Key Concepts:* Gated Linear Units (GLU) variants, SwiGLU (Swish Gated Linear Unit) mathematical formulation and parameter gating, RMSNorm (Root Mean Square Normalization) efficiency gains over standard LayerNorm, GELU, and gradient dynamics.
  - *System Bottlenecks & Focus:* FLOPS vs memory overhead of activations, computational saving of RMSNorm (removing mean tracking and variance subtraction), and impact on training stability.

- **Module 04: Tokenization & Subword Processing for LLMs**
  - *Key Concepts:* Subword tokenization motivation, Byte-Pair Encoding (BPE) merge loop mechanics, SentencePiece (unigram-based model), Tiktoken (GPT-4) vs. Llama tokenizers, Byte-fallback, and Out-of-Vocabulary (OOV) elimination.
  - *System Bottlenecks & Focus:* Token-to-character ratio, vocabulary size vs. embedding table size trade-off, and tokenization/detokenization CPU latency in production pipelines.

- **Module 05: Modern Attention Variants (MQA, GQA, and Context Windows)**
  - *Key Concepts:* Multi-Query Attention (MQA) & Grouped-Query Attention (GQA) architectures, Query/Key/Value group dimension shapes, memory footprint comparison, and Sliding Window Attention (SWA).
  - *System Bottlenecks & Focus:* Memory-bandwidth bounds during autoregressive decoding, saving KV cache size, and performance degradation of MQA vs. capacity restoration of GQA.

- **Module 06: KV Cache Mechanics & Memory Bottlenecks**
  - *Key Concepts:* Causal masking, KV cache tensor layout and shape tracking, Memory bandwidth bounds during token generation (autoregressive decoding bottleneck), and VRAM allocation formulas for KV cache.
  - *System Bottlenecks & Focus:* High memory bandwidth requirement (roofline model limit), VRAM capacity exhaustion with batch size scaling, and memory-bound vs. compute-bound decoding execution phases.

- **Module 07: GPU Memory Bounds & Inference Optimizations (FlashAttention & PagedAttention)**
  - *Key Concepts:* High Bandwidth Memory (HBM) vs SRAM cache, FlashAttention-1 & -2 tiling, online softmax, and local SRAM cache limits. PagedAttention (vLLM virtual memory concept, memory fragmentation reduction, sharing KV cache for parallel decoding).
  - *System Bottlenecks & Focus:* SRAM bandwidth limits, global HBM read/write bottlenecks, page fragmentation, and GPU utilization limits during serving.

- **Module 08: Pre-training: Architecture Styles, Objectives & Data Engineering**
  - *Key Concepts:* Causal Decoder-Only (GPT, Llama), Non-causal Encoder-Only (BERT), Encoder-Decoder (T5, BART) architectures. Pre-training objectives (Causal LM, Masked LM, Span Corruption). Pre-training data scaling laws (Chinchilla), data cleaning (MinHash deduplication, quality filtering).
  - *System Bottlenecks & Focus:* Scaling laws (parameter size vs token count), compute budgets (FLOPs estimation), and pre-training dataset size constraints.

- **Module 09: Architectural Dissection of Frontier Models, MoE, & Deep Thinking Models**
  - *Key Concepts:* Deep dive into architectures of Llama 3, Mistral 7B, Gemma 2; Mixture of Experts (MoE) routing, sparse gating, load balancing loss; **Deep Thinking/Reasoning Models (e.g., OpenAI o1/o3, DeepSeek-R1), Test-Time Compute (TTC) scaling laws, Reinforcement Learning for reasoning trajectories, Process-supervised Reward Models (PRMs) vs. Outcome-supervised Reward Models (ORMs), and Chain-of-Thought (CoT) search/backtracking.**
  - *System Bottlenecks & Focus:* Memory-routing overhead, parameter storage size vs active parameter size, router collapse, routing latency, **test-time compute cost scaling, latency bounds of multi-step generation, and validation of intermediate reasoning steps.**

- **Module 10: Model Evaluation, Benchmark Metrics, & Alignment Evaluation**
  - *Key Concepts:* Task-specific benchmarks (MMLU, GSM8k, HumanEval, ARC, **GPQA for PhD-level reasoning, AIME/MATH for mathematical reasoning**), Perplexity (PPL) mathematical derivation and link to Causal Loss, Chatbot Arena (Elo system), and evaluation setups (LLM-as-a-judge vs human evaluations).
  - *System Bottlenecks & Focus:* Contamination/data leakage, high variance in LLM evaluation, prompt sensitivity, bias in LLM-as-a-judge approaches, and **evaluating stochastic reasoning traces.**
