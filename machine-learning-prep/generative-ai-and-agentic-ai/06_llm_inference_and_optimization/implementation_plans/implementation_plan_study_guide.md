# Implementation Plan: Track 1 — Study Guide (`06_llm_inference_and_optimization`)

Per `study_guide_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 1 (the 9 theory modules); Track 2 (notebooks) and Track 3 (interview Q&A) are separate, later sign-off gates.

---

## 1. Modules & Target File Paths

| # | Module | File Path |
|---|---|---|
| 01 | Inference Fundamentals & the Autoregressive Decoding Loop | `modules/01_inference_fundamentals_and_decoding_loop.md` |
| 02 | KV Cache Mechanics & Memory Management | `modules/02_kv_cache_mechanics_and_memory_management.md` |
| 03 | PagedAttention & Memory-Efficient Serving | `modules/03_pagedattention_and_memory_efficient_serving.md` |
| 04 | FlashAttention & IO-Aware Attention Computation | `modules/04_flashattention_and_io_aware_computation.md` |
| 05 | Quantization for Inference | `modules/05_quantization_for_inference.md` |
| 06 | Batching Strategies — Static, Dynamic & Continuous | `modules/06_batching_strategies.md` |
| 07 | Speculative Decoding & Decoding-Time Optimizations | `modules/07_speculative_decoding_and_decoding_time_optimizations.md` |
| 08 | Inference Serving Engines & Production Architecture | `modules/08_inference_serving_engines_and_production_architecture.md` |
| 09 | Production Monitoring, Cost Modeling & Latency Optimization | `modules/09_production_monitoring_cost_and_latency.md` |

---

## 2. Per-Module Formulas & Hand Calculations

**Module 01 — Inference Fundamentals & the Autoregressive Decoding Loop**
- **Formula (core):** Arithmetic intensity, $I = \dfrac{\text{FLOPs}}{\text{Bytes moved}}$, compared against a GPU's ridge point $I_{\text{ridge}} = \dfrac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth (Bytes/s)}}$ — the roofline-model concept every later module's optimization targets one side of.
- **Hand calc:** Compute a small, illustrative arithmetic intensity for a decode step (low $I$: one new token, full weight/KV read) vs. a prefill step (higher $I$: many tokens processed per weight read), showing decode sits left of the ridge point and prefill sits right of it. **Framing, revised:** "decode is memory-bandwidth-bound / prefill is compute-bound" is presented explicitly as the *typical* pattern this arithmetic-intensity comparison produces for common serving configurations — not a universal rule; the module states plainly that the real regime depends on real batch size, sequence length, and hardware (e.g., a large enough batch can shift decode's real arithmetic intensity meaningfully).

**Module 02 — KV Cache Mechanics & Memory Management**
- **Formula (core):** KV cache memory footprint, using $N_{\text{KV\_heads}}$ (the number of real key/value heads) rather than the full attention-head count $N_{\text{heads}}$: $\text{Mem}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times N_{\text{KV\_heads}} \times d_{\text{head}} \times \text{bytes}_{\text{dtype}}$. Explicitly covers **Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)** — where multiple query heads share a smaller number of real KV heads ($N_{\text{KV\_heads}} < N_{\text{heads}}$, down to $N_{\text{KV\_heads}}=1$ for MQA) — and why this directly, multiplicatively shrinks the real KV-cache memory footprint versus standard multi-head attention at the identical model size.
- **Hand calc:** A small illustrative model config (e.g., 32 layers, 32 query heads, $d_{\text{head}}=128$) at two real sequence lengths/batch sizes, computed once under standard MHA ($N_{\text{KV\_heads}}=32$) and once under a real GQA configuration ($N_{\text{KV\_heads}}=8$), showing the real, direct memory-footprint reduction GQA provides — and showing KV cache memory crossing over model-weight memory at long context/high concurrency under the MHA case specifically, demonstrating the plan's workload-dependent framing, not asserting a universal claim.

**Module 03 — PagedAttention & Memory-Efficient Serving**
- No new closed-form formula — architectural/procedural, consistent with how prior topics treated protocol/memory-management modules.
- **Hand calc (no formula, worked example):** Contiguous max-length allocation vs. block-based allocation for a small set of variable-length sequences, computing the real percentage of wasted/fragmented memory each approach leaves — the concrete number behind "improves utilization, doesn't compress."

**Module 04 — FlashAttention & IO-Aware Attention Computation**
- **Formula (core):** HBM access count comparison — naive attention's $O(N^2)$ materialization vs. tiled/blocked computation's reduced HBM traffic, following the FlashAttention paper's own IO-complexity framing at an intuitive, non-derivation level (per the Concept Simplification rule — no full derivation, only the final comparison and its intuition).
- **Hand calc:** A tiny sequence length and block size, computing real HBM read/write counts for naive vs. tiled attention, showing the real reduction ratio.

**Module 05 — Quantization for Inference**
- **Formula (core):** Bytes-per-parameter at a given bit-width, and the resulting real model memory footprint, $\text{Mem} = N_{\text{params}} \times \text{bits}/8$, applied separately to weights, activations, and KV cache (three distinct targets per the signed-off syllabus).
- **Hand calc:** A small illustrative parameter count at FP16 vs. INT8 vs. INT4, plus a separate real KV-cache-quantization hand calc reusing Module 02's memory formula at reduced precision, showing the real, distinct memory savings each target contributes.
- **Explicit caveat (new):** The module states plainly that **lower precision does not guarantee proportional real latency/speed gains** — the memory-footprint reduction above is real and direct, but the actual real speedup depends on whether the serving hardware and kernel implementation genuinely support fast execution at that precision, and on whether the workload's real bottleneck (per Module 01's roofline framing) is even memory-bandwidth-bound in the first place; a compute-bound workload on hardware without real low-precision kernel support can see the memory win with little to no real speed win.

**Module 06 — Batching Strategies**
- No new closed-form formula — batching's real trade-offs are demonstrated via a worked example, not derived.
- **Hand calc (no formula, worked example):** A small batch of variable-length sequences under static batching (real, computed padding waste) vs. continuous batching (real, computed reduction in wasted compute), explicitly not asserting a universal latency improvement, per the signed-off syllabus's workload/scheduler-dependent framing.

**Module 07 — Speculative Decoding & Decoding-Time Optimizations**
- **Formula (core, reworded per feedback):** Expected accepted tokens per verification step under a **simplified acceptance model** — a real, constant per-token acceptance probability $\alpha$ and draft length $k$: $E[\text{accepted}] = \dfrac{1-\alpha^{k+1}}{1-\alpha}$. Explicitly labeled as giving **expected accepted tokens under this simplified model, not the complete expected speedup** — actual real speedup additionally depends on the real relative cost of a draft-model forward pass vs. a verification forward pass, which the formula alone doesn't capture.
- **Hand calc, two-step (revised):** Step 1 — a concrete acceptance rate and draft length, computing real expected accepted tokens per verification step from the formula above. Step 2 — only *after* stating explicit draft/verification cost assumptions (e.g., draft-model forward pass costs a real fraction of a full verification forward pass), compute the resulting real expected speedup over standard autoregressive decoding — kept as a clearly separate step from Step 1, not folded into one combined claim.

**Module 08 — Inference Serving Engines & Production Architecture**
- No formula — comparative/architectural survey, consistent with `04_ai_agents_and_protocols` Module 07's framework-landscape treatment. Comparison table across stable dimensions (scheduling strategy, kernel optimization approach, hardware support, ease of multi-GPU scaling) for vLLM/TensorRT-LLM/TGI/llama.cpp.

**Module 09 — Production Monitoring, Cost Modeling & Latency Optimization**
- **Formula (core, reworded per feedback):** An **approximate latency-budget decomposition** — explicitly not an exact mathematical p99 decomposition, since p99 of a sum is not generally the sum of each term's own p99 (the components aren't independent, and tail behavior doesn't add linearly) — presented instead as a practical, additive *budgeting* tool: $\text{Latency}_{\text{budget}} \approx \text{Queue}_{\text{wait}} + \text{Prefill}_{\text{time}} + \text{Decode}_{\text{time}} + \text{Network/Serialization}$, useful for reasoning about which real component to optimize first, not for deriving an exact tail-latency figure.
- **Formula (core, GPU-time-based cost model, per feedback):** $\text{Cost}_{\text{request}} = \text{GPU\_time}_{\text{request}} \times \text{GPU\_cost\_rate}$ (real GPU-seconds consumed by a request, times a real per-second/per-hour GPU cost rate) — explicitly *not* the API-token-price pattern used in prior topics, since this module is about the underlying infrastructure cost, not a hosted API's price sheet. Cost-per-token and cost-per-request are then derived from this GPU-time base: $\text{Cost}_{\text{per\_token}} = \text{Cost}_{\text{request}} / N_{\text{tokens\_generated}}$.
- **Hand calc:** A real, worked latency budget breaking down a target latency SLO across its four additive components (framed as budgeting, not exact p99 math), showing which component dominates and why GPU-level optimization alone doesn't address the queue-wait or network terms; a separate real hand calc deriving cost-per-request and cost-per-token from a concrete real GPU-time-and-rate example.

---

## 3. Diagrams & Plots

**Inline responsive SVG (architectural, no formula-plot):**
- Module 01: Prefill vs. decode phase timeline, annotated with the roofline-model compute-bound/memory-bound distinction.
- Module 03: Contiguous max-length allocation vs. paged block-based allocation, visualizing the real fragmentation-waste difference.
- Module 04: FlashAttention tiling/blocking schematic — SRAM tile vs. full HBM materialization.
- Module 06: Static batching (wait for slowest sequence) vs. continuous batching (rolling request replacement) timeline.
- Module 08: Full replica-level serving architecture (request routing/load balancing → inference replicas → per-replica scheduler → GPU) plus a conceptual prefill/decode disaggregation diagram.
- Module 09: Approximate latency-budget decomposition as a stacked diagram (queue wait + prefill + decode + network/serialization) — captioned explicitly as a budgeting tool, not an exact p99 derivation.

**Matplotlib plots (saved to `plots/`, generated via `helpers/generate_inference_plots.py`, all explicitly labeled illustrative unless computed directly from this module's own hand-calc formula):**
1. Module 01: Roofline plot — compute-bound vs. memory-bound regions, with prefill and decode plotted as real points computed from the module's own arithmetic-intensity formula (real, computed).
2. Module 02: KV cache memory footprint vs. sequence length, computed directly from the module's own memory-footprint formula across a range of real sequence lengths (real, computed).
3. Module 06: Illustrative throughput/latency vs. batch size for static vs. continuous batching (labeled illustrative — no real notebook measurement exists yet at Track 1 time, and the syllabus explicitly avoids an absolute universal claim here).
4. Module 07: Expected speedup vs. acceptance rate, computed directly from the module's own speculative-decoding formula across a range of real $\alpha$ values (real, computed).

This totals 4 plots and 6 diagrams, matching prior topics' established precedent for a 9-module topic.

---

## 4. Open Design Questions / Dependencies

1. All core formulas (arithmetic intensity, KV-cache memory, quantization bytes-per-param, speculative-decoding expected-acceptance, p99 decomposition) are standard, closed-form, and computable directly — no external dependency.
2. Track 2's notebooks (not this track) will need real GPU access for KV-cache/memory measurements and possibly a small local model for real quantization/batching benchmarks — will confirm feasibility at Track 2 planning time using the same RTX 4060 GPU already used in `05_prompt_engineering_and_structured_generation`'s Notebook 04, not blocking Track 1.
3. No open questions block starting Track 1 module writing.

## Status: Complete

All 9 modules written and verified (hand calcs/reference code executed with real, passing assertions; all 7 inline SVG diagrams rendered clean via headless Edge screenshot; all 4 plots computed and generated). Master study guide compiled to HTML/PDF via `helpers/compile_inference.py`, verified with 0 leaks and correct structural div counts. See `token_usage_log.md` entries 1-15 for the full per-step build log, including two real bugs caught and fixed during execution (Module 02's GB/GiB unit mismatch; Module 06's honest partial-rebalancing simulation result).
