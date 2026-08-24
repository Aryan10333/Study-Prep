# LLM Inference & Optimization Syllabus

## 1. Context & Alignment
* **Profile Focus:** Directly targets the "Production Engineering & Inference" (15%) weighting in the candidate's curriculum — vLLM, FlashAttention, PagedAttention, quantization, KV cache, model serving, deployment, monitoring, latency optimization, scalability. Where prior topics (`03_advanced_rag`, `04_ai_agents_and_protocols`, `05_prompt_engineering_and_structured_generation`) treated the model as a black box behind an API, this topic opens that box: how a request actually gets served, what makes it fast or slow, and what a production inference stack looks like underneath the API surface.
* **Interview Frequency:** High, and rising — "explain KV cache," "why is decoding slower than prefill," "how does PagedAttention reduce memory waste," and "how would you reduce p99 latency for an LLM endpoint" are now standard screens for AI Infrastructure, AI Platform Engineer, and Applied/GenAI Engineer roles at both foundation-model companies and infra-focused companies (matches the candidate's stated interest in NVIDIA/AMD/Groq/Cerebras-class infrastructure companies specifically).
* **Core Goal:** Build a precise, engineering-grounded model of the autoregressive decoding loop's real system bottlenecks — memory bandwidth, KV cache growth, batching efficiency — and how production inference engines (vLLM, TensorRT-LLM, TGI) address each one, so a candidate can reason about real latency/throughput/cost trade-offs rather than reciting library names.

## 2. Module Chapters & Conceptual Scope

- **Module 01: Inference Fundamentals & the Autoregressive Decoding Loop**
  - *Key Concepts:* Prefill vs. decode phases, why decoding is inherently sequential (each token depends on the last), compute-bound vs. memory-bandwidth-bound phases, real latency components — time-to-first-token (TTFT) and time-per-output-token (TPOT/inter-token latency) — and why they're driven by genuinely different bottlenecks.
  - *System Bottlenecks & Focus:* Establishing the roofline-model intuition (compute-bound vs. memory-bandwidth-bound) that every later module's optimization technique targets one side of.

- **Module 02: KV Cache Mechanics & Memory Management**
  - *Key Concepts:* What the KV cache stores and why (avoiding recomputing attention over the full prefix at every decode step), its real memory footprint formula and why it grows linearly with sequence length, batch size, and number of layers/heads. Framed explicitly as **workload-dependent**, not a universal claim: KV cache can become the dominant real memory consumer — exceeding model weights — specifically under long contexts, long generations, or high concurrency; for short contexts/low concurrency, model weights can still dominate.
  - *System Bottlenecks & Focus:* The concrete arithmetic of real GPU memory budgets, and why naive KV cache allocation wastes real, substantial memory through fragmentation and over-provisioning.

- **Module 03: PagedAttention & Memory-Efficient Serving**
  - *Key Concepts:* The KV cache fragmentation problem naive contiguous allocation causes, the virtual-memory/paging analogy, block-based KV cache allocation. Stated precisely: **PagedAttention improves KV-cache memory allocation, utilization, and sharing (e.g., for beam search or shared prefixes) — it does not inherently compress the KV cache or reduce its per-token size.** The real payoff is that better utilization (less waste from fragmentation/over-provisioning) lets more real concurrent sequences fit in the same GPU memory budget, directly enabling a higher effective batch size.
  - *System Bottlenecks & Focus:* Why memory *utilization* efficiency, distinct from memory *compression*, directly translates to real throughput via higher effective batch size.

- **Module 04: FlashAttention & IO-Aware Attention Computation**
  - *Key Concepts:* The real memory-bandwidth cost of naive attention (materializing the full $O(n^2)$ attention matrix), tiling and kernel fusion as an IO-aware (not just "faster math") optimization, why FlashAttention reduces real HBM reads/writes rather than FLOPs, the real evolution across FlashAttention versions. Explicit contrast with Module 03: **FlashAttention reduces attention IO/HBM traffic during the attention computation itself; PagedAttention manages KV-cache memory allocation/utilization during serving** — genuinely complementary mechanisms operating at different layers (compute-kernel vs. memory-management), not competing or overlapping techniques.
  - *System Bottlenecks & Focus:* The precise distinction between reducing compute and reducing memory-bandwidth traffic — the actual bottleneck FlashAttention targets, and how that differs from what PagedAttention targets.

- **Module 05: Quantization for Inference**
  - *Key Concepts:* INT8/INT4/FP8 quantization across three distinct targets — **weight quantization, activation quantization, and KV-cache quantization** — each with its own real accuracy/speed/memory trade-off profile. KV-cache quantization specifically as a real, direct lever for long-context/high-concurrency serving, where Module 02 already established KV cache can dominate real memory usage. The practical landscape (GPTQ, AWQ, bitsandbytes, native FP8) and when each earns its real cost.
  - *System Bottlenecks & Focus:* Quantization as a real, direct lever on the memory-bandwidth bottleneck established in Module 01 — smaller weights, activations, or KV-cache entries mean less real data movement per token.

- **Module 06: Batching Strategies — Static, Dynamic & Continuous Batching**
  - *Key Concepts:* Why naive static batching wastes real compute on padding for variable-length sequences, dynamic batching, and continuous (in-flight) batching — the real innovation behind vLLM/TGI's throughput gains — where new requests join a batch mid-generation rather than waiting for the whole batch to finish.
  - *System Bottlenecks & Focus:* The real throughput-vs-latency trade-off batching introduces. Continuous batching generally improves real GPU utilization and throughput, and *can* improve latency — but the actual real latency/throughput outcome is workload- and scheduler-dependent, not an absolute "improves both simultaneously" guarantee.

- **Module 07: Speculative Decoding & Other Decoding-Time Optimizations**
  - *Key Concepts:* Draft-model-plus-verification speculative decoding — the real mechanism behind its speedup, when it genuinely helps vs. adds pure overhead, and other real decoding-time techniques (Medusa-style multi-head prediction, lookahead decoding) at a conceptual level.
  - *System Bottlenecks & Focus:* Why speculative decoding specifically targets the memory-bandwidth-bound decode phase established in Module 01, not the prefill phase.

- **Module 08: Inference Serving Engines & Production Architecture**
  - *Key Concepts:* The real serving-engine landscape (vLLM, TensorRT-LLM, TGI, llama.cpp) compared along stable dimensions, request scheduling and admission control, multi-GPU tensor-parallel and pipeline-parallel serving for models that don't fit on one GPU. The full real **replica-level serving architecture**: request routing/load balancing → inference replicas → per-replica scheduler → GPU — with multi-replica horizontal scaling covered alongside (not instead of) tensor/pipeline parallelism, since they solve different real scaling problems (more concurrent capacity vs. fitting one large model). **Prefill/decode disaggregation** introduced conceptually — running prefill and decode on separate, independently-scaled resources, a real modern production pattern given the two phases' genuinely different compute/memory-bandwidth profiles (Module 01).
  - *System Bottlenecks & Focus:* Comparing engines by real architectural trade-offs (scheduling strategy, kernel optimization approach, hardware support) rather than a feature checklist that changes release to release; understanding replica-level architecture as the layer tensor/pipeline-parallel serving sits inside, not a competing concept.

- **Module 09: Production Monitoring, Cost Modeling & Latency Optimization**
  - *Key Concepts:* Real cost/latency budgeting for an inference deployment; explicit **p95/p99 latency decomposition** — queue wait time + prefill time + decode time + network/serialization overhead — since GPU-level optimization alone doesn't address queueing or network contributions to real tail latency. A deliberately broad, distinct real observability metric set: **GPU compute utilization, memory/HBM utilization, KV-cache utilization, queue depth, TTFT, TPOT, throughput, and p95/p99 latency** — each catching a different real failure mode. Autoscaling considerations for bursty real traffic, prefill/decode disaggregation's monitoring implications (referenced from Module 08), and common real production failure modes (OOM from KV cache growth, head-of-line blocking, cold-start latency).
  - *System Bottlenecks & Focus:* Turning every earlier module's mechanism into a concrete, monitorable production metric and a real cost-per-token/cost-per-request figure; treating p99 latency as a real, decomposable sum where GPU optimization is necessary but often not sufficient on its own.

Module 10 (Interview Q&A track — Track 3) will follow once Tracks 1 and 2 are complete and signed off, per the established 3-track pipeline.

---

### Cross-Module Boundary Discipline
This topic deliberately avoids re-deriving content already owned by prior topics:
- **Transformer attention mechanics and positional encoding fundamentals** → `01_llm_foundations`.
- **Training-time optimization (LoRA/QLoRA, PEFT, training-time mixed precision)** → `02_llm_training_foundations` — this topic covers quantization and precision strictly at *inference* time.
- **Prompt-level cost/latency budgeting and prompt caching** → `05_prompt_engineering_and_structured_generation` — this topic covers the *serving-infrastructure* side (KV cache, batching, kernels) the prompt-level budget ultimately runs on top of.
- **Structured/constrained decoding's grammar-masking mechanism** → `05_prompt_engineering_and_structured_generation` Module 04 — referenced here only where it interacts with real decode-time throughput (e.g., Module 07), not re-derived.

This topic owns: what actually happens between a request arriving and a token being returned — the real memory, compute, and scheduling mechanics of production LLM inference.

## Status: Approved
