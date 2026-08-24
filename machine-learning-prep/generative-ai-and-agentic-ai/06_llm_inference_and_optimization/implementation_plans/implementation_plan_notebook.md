# Implementation Plan: Track 2 — Notebooks (`06_llm_inference_and_optimization`)

Per `notebook_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 2 (6 notebooks); Track 3 (interview Q&A) is a separate, later sign-off gate. Track 1 (all 9 study-guide modules, 7 SVG diagrams, 4 plots, PDF/HTML compilation) is complete — see `implementation_plan_study_guide.md`'s Status: Complete.

---

## 1. Real Hardware/Environment

Real local RTX 4060 Laptop GPU (8.59 GB VRAM), the same hardware used in `05_prompt_engineering_and_structured_generation`'s Notebook 04. Real local model: **`Qwen/Qwen2.5-0.5B-Instruct`**, reused from that same prior notebook — a real, small, fast model that fits comfortably in 8GB VRAM at FP16 with room left for KV cache growth and quantized-variant comparisons, and is already confirmed working on this exact hardware.

**Feasibility note on vLLM/PagedAttention:** A real vLLM install and a real PagedAttention-vs-contiguous-allocation A/B comparison was considered for Notebook 05, but vLLM's own memory/config overhead is a poor fit for an 8GB laptop GPU already hosting the target model — rather than block on that feasibility risk, Notebook 05's PagedAttention angle is a real Python-level allocation simulation (matching Module 03's own worked-example style), but grounded in **this notebook's own real measured per-request completion times** from its batching experiment, not synthetic numbers — a genuine, real-data-grounded substitute, not a downgrade to purely illustrative.

---

## 2. Notebooks & Target File Paths

| # | Notebook | Module(s) Covered | File Path |
|---|---|---|---|
| 01 | Decode vs. Prefill Fundamentals — Real TTFT/TPOT Measurement | Module 01 | `notebooks/01_decode_vs_prefill_fundamentals.ipynb` |
| 02 | Real KV Cache Memory Measurement vs. Formula | Module 02 | `notebooks/02_kv_cache_memory_measurement.ipynb` |
| 03 | FlashAttention/SDPA Backend Comparison — Real IO-Aware Latency & Memory | Module 04 | `notebooks/03_flashattention_sdpa_backend_comparison.ipynb` |
| 04 | Real Quantization Benchmark — FP16 vs. INT8/INT4 Memory & Latency | Module 05 | `notebooks/04_quantization_benchmark.ipynb` |
| 05 | Real Batching Throughput/Latency + Grounded PagedAttention Simulation | Modules 03, 06 | `notebooks/05_batching_throughput_and_paged_simulation.ipynb` |
| 06 | Real Speculative Decoding + End-to-End Latency Budget & Cost Capstone | Modules 07, 08, 09 | `notebooks/06_speculative_decoding_and_production_capstone.ipynb` |

This groups the 9 Track 1 modules thematically across 6 notebooks (not 1:1), matching the established precedent from `05_prompt_engineering_and_structured_generation`'s own 9-module/6-notebook split.

---

## 3. Per-Notebook Real Experiments

**Notebook 01 — Decode vs. Prefill Fundamentals** `[REAL]`
- Load `Qwen2.5-0.5B-Instruct` at FP16 on the real RTX 4060.
- **Measurement methodology (revised):** for every timed configuration, run a real warm-up pass (discarded, to exclude CUDA-context/kernel-compilation startup cost), then multiple real repeated runs with explicit `torch.cuda.synchronize()` calls bracketing each timed region (`time.perf_counter()` alone is not trustworthy against an async CUDA queue). Report **median and p95** across repeats, not a single wall-clock sample — a laptop GPU's real timing is genuinely noisy (thermal throttling, background processes), and a single measurement can produce a misleading TTFT/TPOT conclusion.
- **Framed as a hypothesis, not an expected result (revised):** the notebook tests *whether* measured decode-step latency stays roughly flat as sequence length grows (the memory-bandwidth-bound signature) while prefill's real per-token cost is lower — it does **not** assume this will hold. `Qwen2.5-0.5B` and the RTX 4060 are real, but small/consumer-class relative to the illustrative large-model/datacenter-GPU profile Module 01's roofline hand calc used; the canonical memory-bound decode behavior may or may not reproduce cleanly at this real, different scale.
- Honest reporting either way: if real measured behavior doesn't match the hypothesis, report that mismatch directly as a genuine finding about small-model/consumer-GPU behavior, not a failure to fix — consistent with Module 01's own "typical, not universal" framing.

**Notebook 02 — Real KV Cache Memory Measurement** `[REAL]`
- **Revised measurement approach:** `torch.cuda.memory_allocated()` alone is **not** used as a stand-in for "KV-cache memory" — it includes weight memory, activation memory, and other real allocations, not just the KV cache. Instead: (a) estimate real KV-cache-specific growth as the real *delta* in allocated memory between two generations that differ only in sequence length (holding batch size and everything else fixed), isolating the KV-cache-growth component from the fixed weight/activation baseline; (b) additionally report `torch.cuda.memory_reserved()` (peak reserved by the allocator) alongside `memory_allocated()`, since the real, honest gap between reserved and allocated is itself a direct, measurable illustration of real allocator overhead — a finding worth reporting explicitly, not collapsing into one number.
- Compare the real measured KV-cache-growth delta against Module 02's formula's real predicted values (adjusted for this specific real model's actual `n_layers`, `n_kv_heads`, `d_head` — Qwen2.5-0.5B's real config, not the illustrative 7B numbers used in the module).
- Report the real, honest gap (if any) between predicted and measured — framework/allocator overhead (now directly measured via reserved-vs-allocated, per above), activation memory, etc. are real, expected sources of deviation worth naming explicitly, not hiding.

**Notebook 03 — FlashAttention/SDPA Backend Comparison** `[REAL]`
- Use `torch.nn.functional.scaled_dot_product_attention`'s real, selectable backends (`torch.backends.cuda.sdp_kernel` context manager: flash vs. math/efficient backends) on the real GPU.
- Measure real latency and real peak memory for both backends at a few real sequence lengths.
- **Scope of the claim (revised, narrower):** this notebook demonstrates a real, measured **backend performance difference** (latency and peak memory) between the flash and math/efficient SDPA kernels — it does **not** claim to directly measure real HBM read/write traffic. Module 04's IO-complexity argument (fewer HBM passes) is the real, established *mechanism* behind FlashAttention's design; this notebook's latency/memory numbers are consistent with that mechanism's real, expected consequences, but are not themselves an HBM-traffic measurement (that would require GPU-level profiling tools like Nsight Compute, out of scope here). The notebook's writeup states this distinction explicitly rather than conflating "faster and leaner" with "directly measured less HBM IO."
- Honest fallback: if this exact PyTorch/CUDA version doesn't expose a clean flash-vs-math backend switch on this GPU, report that real constraint directly and adapt to whatever real comparison the installed stack does support, rather than fabricating results.

**Notebook 04 — Real Quantization Benchmark** `[REAL]`
- Load `Qwen2.5-0.5B-Instruct` at FP16 and at a real quantized precision via `bitsandbytes` (INT8, and INT4 if real support allows on this GPU/model).
- Measure real memory footprint and real generation latency for each, directly testing Module 05's central caveat: does a real memory reduction here translate into a proportional real latency reduction on this specific real hardware/kernel combination, or not?
- Report whichever real outcome actually occurs — a real "memory dropped, latency barely moved" result is exactly as valuable to report honestly as a clean proportional win, per this topic's established precedent (Topic 05's real, honestly-reported prompt-injection-detection false-positive finding).

**Notebook 05 — Real Batching Throughput/Latency + Grounded PagedAttention Simulation** `[REAL]` + `[SIMULATION]`
- `[REAL]`: real batched generation at a few real batch sizes, measuring real throughput (tokens/sec) and real per-request latency — a real, measured counterpart to Module 06's illustrative plot.
- `[SIMULATION]` (explicitly labeled as such, not `[REAL]`, per revised discipline): feed this notebook's own real per-request completion times into a Python allocation simulation (Module 03's contiguous-vs-paged style), replacing that module's synthetic sequence-length example with **real measured generation lengths from this notebook's own runs**. Stated plainly in the notebook's own writeup: using real measured inputs makes this a **simulation grounded in real data**, not an actual PagedAttention implementation, and not a real A/B benchmark of PagedAttention-vs-contiguous serving — the notebook does not claim otherwise.

**Notebook 06 — Real Speculative Decoding + Production Capstone** `[REAL]` + `[COMPUTED FROM REAL DATA]`
- `[REAL]`: use Hugging Face `transformers`' built-in assisted/speculative generation (`model.generate(assistant_model=...)`) with a real, smaller draft model (e.g., a smaller Qwen variant, if one fits alongside the target on this GPU's real remaining VRAM) against the target `Qwen2.5-0.5B-Instruct`.
- **Acceptance and speedup kept as separate, distinct measurements (revised, per Module 07's own two-step discipline):** (1) measure the real, empirical token-acceptance rate directly from assisted-generation output; (2) *separately*, measure real end-to-end wall-clock speedup versus standard greedy decoding. These are reported and discussed as two distinct real numbers, not conflated — Module 07's formula predicts expected accepted tokens under a simplified constant-$\alpha$ model, and real end-to-end speedup additionally depends on real draft/verification cost and implementation overhead the formula doesn't capture, exactly as Module 07 itself states. The notebook compares its own real measured acceptance rate against the formula's prediction at that same real $\alpha$, and separately compares real measured speedup against the formula's Step-2 speedup calculation under stated real cost assumptions.
- **No fallback substitution if infeasible (revised — removes the prior greedy-vs-assisted fallback):** if no real draft model fits in this GPU's remaining VRAM, or if `transformers`' assisted-generation path is not usable in this real environment, the notebook reports **speculative decoding as infeasible on this real hardware/environment** and states why — it does not substitute an unrelated greedy-vs-assisted timing comparison in its place, since that would measure something materially different and risk being mistaken for a speculative-decoding result.
- `[COMPUTED FROM REAL DATA]` **Capstone piece:** combine this notebook's own real measured prefill/decode timings (from Notebooks 01-02) into a filled-in version of Module 09's latency-budget worked example, using genuinely measured numbers instead of the module's illustrative 150/300/1600/50ms figures — explicitly labeled as computed from real data (a real-input arithmetic exercise), not a live production measurement. Also includes a small `[SIMULATION]` least-loaded routing exercise (Module 08's diagram) using this notebook's own real measured per-request latencies as the real backing data — labeled as a simulation, not a real multi-replica deployment.

---

## 4. Real vs. Illustrative Labeling Discipline (Revised: Four-Tier Labeling)

A blanket **[REAL]** label across every cell was too coarse — a simulation or a formula-derived computation that merely *uses* real measured inputs is not itself real hardware execution, and collapsing that distinction risks overstating what a given result actually demonstrates. Every notebook section/cell is instead labeled with one of four explicit tags:

- **`[REAL]`** — genuine local GPU execution producing genuinely measured numbers (timings, memory readings, generation outputs) directly from real hardware in this session.
- **`[COMPUTED FROM REAL DATA]`** — a real formula or arithmetic exercise (e.g., Module 09's latency-budget decomposition) evaluated using genuinely measured real inputs from this topic's own notebooks, but not itself a live hardware measurement.
- **`[SIMULATION]`** — a Python-level model of a real mechanism (e.g., Notebook 05's paged-allocation simulation, Notebook 06's routing simulation) that may be grounded in real measured inputs, but is explicitly not an actual implementation of, or real A/B benchmark against, the real system it models.
- **`[EXPLANATION]`** — prose/markdown cells providing conceptual context, referencing Track 1 module content, with no new computation of its own.

Per-notebook tags are noted in Section 3 above; a notebook with multiple tags (e.g., Notebook 05: `[REAL]` + `[SIMULATION]`) applies each tag to its own specific cells/sections, not uniformly to the whole notebook. Any step that turns out to be infeasible on this specific real hardware (VRAM limits, missing kernel/library support) is honestly reported as a real, encountered constraint — with either a genuinely equivalent real fallback (e.g., Notebook 04's FP16-vs-INT8-only fallback if INT4 isn't supported) or, where no equivalent real fallback exists (Notebook 06's speculative decoding specifically, per the revised no-substitution rule above), an honest infeasibility report instead of a substituted, materially different experiment.

## 5. Two-Pass Authoring Discipline

Per established practice: Pass 1 builds and executes all code cells with `_(pending real output)_` placeholders; Pass 2 reads the real executed output and writes explanation cells quoting literal real values in backticks, verified via the Literal-Quote Requirement check, before either pass is considered complete.

## 6. Open Design Questions / Dependencies

1. Exact real VRAM headroom for Notebook 06's draft+target model pair won't be confirmed until Notebook 06 is actually built — if no real draft model fits (or assisted generation isn't otherwise usable in this environment), per the revised no-substitution rule, the notebook reports speculative decoding as infeasible on this real hardware rather than substituting an unrelated comparison.
2. `bitsandbytes` INT4 support on this specific real GPU/driver combination (Notebook 04) will be confirmed at build time; if INT4 real support isn't available, the notebook proceeds with a real FP16-vs-INT8 comparison only, reported as such.
3. No open questions block starting Notebook 01.

## Status: Complete

All 6 notebooks built, executed (real GPU on the RTX 4060 throughout, no illustrative-only notebook), and verified via the two-pass discipline (Pass 1 real execution, Pass 2 literal-quoted real-value explanations). Real bugs found and fixed during execution, documented explicitly in-notebook rather than silently corrected: a transformers Cache-API version mismatch (Notebook 02), a GB/GiB hand-calc unit inconsistency caught by the reference code's own assertion (carried over from Track 1, not this track), a `bitsandbytes` logging-based (not `warnings`-based) output-bloat bug (Notebook 04), and a `repetition_penalty` default plus an f-string formatting bug (Notebook 06). Genuine real findings, several stronger or more surprising than anticipated at plan time: a ≈897x real prefill-vs-decode per-token cost gap (N01); an exact byte-for-byte formula match for KV-cache memory alongside a ≈26x `memory_allocated()` overstatement traced to the real logits tensor (N02); a real ≈33x MATH-vs-EFFICIENT_ATTENTION latency gap with FlashAttention itself confirmed genuinely unavailable on this build (N03); real quantization making generation *slower*, not faster (N04); a genuine, unstaged batching straggler event reproducing Module 06's core claim (N05); and a real, verified-correct α=0.4 speculative-decoding acceptance rate alongside a real, honestly-attributed net slowdown from this reference implementation's unoptimized KV-cache handling (N06). See `token_usage_log.md` entries 18-23 for the full per-notebook build log.
