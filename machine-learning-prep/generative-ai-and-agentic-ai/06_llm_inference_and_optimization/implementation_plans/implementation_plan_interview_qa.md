# Implementation Plan: Track 3 — Interview Q&A (`06_llm_inference_and_optimization`)

Per `interview_qa_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 3 (the standalone Interview Q&A cheatsheet); Tracks 1 (9 study-guide modules, 7 SVG diagrams, 4 plots, compiled PDF/HTML) and 2 (6 notebooks, all real GPU execution on the RTX 4060) are both complete and pushed.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_llm_inference_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `llm_inference_optimization_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `llm_inference_optimization_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_inference.py` — add a second `compile_document()` call in `main()`, producing a separate standalone cheatsheet |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules and their real hand calcs (roofline arithmetic intensity, KV-cache/GQA memory footprint, contiguous-vs-paged allocation waste, simplified HBM-access-count comparison, bytes-per-parameter quantization, static-vs-continuous batching waste, speculative-decoding two-step formula, approximate latency-budget decomposition). Following prior topics' established differentiator: where a question's real Track 2 notebook result adds genuine interview value, the **Production Perspective** or **Common Mistakes** sections cite it explicitly, framed as an observation from a specific real experiment on this specific real hardware (RTX 4060 + Qwen2.5-0.5B-Instruct), not a universal law.

This topic's Track 2 produced an unusually high density of genuinely surprising, real findings worth citing this way: a real ≈897x prefill-vs-decode per-token cost gap despite a near-flat raw TTFT curve (Notebook 01); an exact byte-for-byte KV-cache formula match alongside a real ≈26x `memory_allocated()` overstatement traced to the logits tensor (Notebook 02); FlashAttention itself confirmed genuinely unavailable on this build, with a real ≈33x MATH-vs-EFFICIENT_ATTENTION latency gap at 4096 tokens (Notebook 03); real quantization making generation *slower*, not faster, despite genuine memory savings (Notebook 04); an unstaged real batching straggler event that concretely reproduced the static-batching problem (Notebook 05); and a real, verified-correct α=0.4 speculative-decoding acceptance rate that closely matched the formula's prediction, alongside a real, honestly-attributed net *slowdown* from the reference implementation's unoptimized KV-cache handling (Notebook 06) — a clean, real illustration of why acceptance rate and end-to-end speedup are genuinely separate questions.

## 3. Proposed Question List (54 questions, grouped by module)

**Module 01 — Inference Fundamentals & the Autoregressive Decoding Loop (6)**
1. Why is autoregressive decoding inherently sequential, and how does that sequential dependency shape inference's batching structure and prefill/decode behavior differently from training?
2. Walk through the roofline model — what does arithmetic intensity measure, and what does the ridge point represent?
3. Given peak FLOPs/s and peak memory bandwidth, compute a GPU's ridge point, then compute and classify decode's and prefill's arithmetic intensity at a given model size and prompt length.
4. Why is "decode is memory-bandwidth-bound, prefill is compute-bound" described as a typical pattern rather than a universal rule? What real workload change can shift decode's regime?
5. Precisely distinguish TTFT and TPOT — what does each one's cost actually scale with?
6. A real notebook measured TTFT staying nearly flat across a 32x range of prompt lengths, while TPOT stayed flat across generation length — yet converting TTFT to a per-prompt-token cost revealed a real ≈897x gap versus TPOT. Why does the per-token conversion matter for interpreting this result correctly?

**Module 02 — KV Cache Mechanics & Memory Management (6)**
7. What does the KV cache store, and why does caching it mean only the new token's key/value needs to be computed at each step, while attention still reads the full cached prefix?
8. Walk through the KV-cache memory footprint formula — why does it depend on $N_{\text{KV\_heads}}$ specifically, not the total query-head count?
9. Given a model's real layer count, KV-head count, and head dimension, compute its KV-cache memory footprint at a given batch size and sequence length, once under MHA and once under GQA.
10. Why is "KV cache dominates memory usage" workload-dependent rather than universally true — under what conditions does KV cache overtake model weights?
11. A real notebook measured a model's exact KV-cache tensor size (matching the formula byte-for-byte) but found `memory_allocated()`'s delta was ≈26x larger. What real, distinct source accounted for nearly all of that gap, and why does this matter for capacity planning?
12. How do GQA and MQA reduce KV-cache memory without proportionally reducing model quality?

**Module 03 — PagedAttention & Memory-Efficient Serving (6)**
13. Precisely state what PagedAttention does and does not do — why is "it compresses the KV cache" a common, incorrect claim?
14. Walk through the contiguous-allocation waste problem — why does reserving for the worst case waste real memory even when most real sequences finish early?
15. Given a set of real variable-length sequences, a maximum supported length, and a block size, compute the real wasted-memory percentage under contiguous vs. paged allocation.
16. Under a simple block-based allocation model, why is paged allocation's per-sequence internal-fragmentation waste bounded by the block size, while contiguous allocation's waste is not similarly bounded regardless of the maximum supported sequence length?
17. How does block-based allocation directly enable KV-cache sharing (e.g., for beam search or shared prefixes) in a way contiguous allocation cannot?
18. A real notebook fed its own measured, straggler-containing generation lengths (not synthetic numbers) into a paged-vs-contiguous simulation and found a ≈6.4x waste reduction. Why is it important that this stayed labeled a simulation even though its inputs were genuinely real?

**Module 04 — FlashAttention & IO-Aware Attention Computation (6)**
19. Why is FlashAttention described as an IO-aware optimization rather than a FLOP-reduction technique?
20. Walk through why tiled attention avoids materializing the full attention score matrix in HBM, and how that reduces real HBM↔SRAM traffic compared to naive attention — without relying on a single universal asymptotic formula to make the point.
21. Given a tiny sequence length and head dimension, compute the simplified HBM-access-count comparison between naive and tiled attention, and explain why the reduction ratio grows with sequence length.
22. Precisely distinguish what FlashAttention optimizes from what PagedAttention optimizes — why are they complementary rather than competing?
23. Why can two attention implementations have identical FLOPs but meaningfully different real wall-clock latency?
24. A real notebook found FlashAttention itself genuinely unavailable on the installed hardware/build, and compared a different tiled kernel (EFFICIENT_ATTENTION) against the naive MATH backend instead — with the real latency gap growing from ≈5x to ≈33x as sequence length grew 512→4096. Why does reporting the unavailability honestly matter more than silently substituting a workaround?

**Module 05 — Quantization for Inference (6)**
25. Name the three distinct real quantization targets this topic covers — weights, activations, and KV cache — and explain why each has its own real accuracy/speed/memory trade-off profile.
26. Walk through the bytes-per-parameter formula — why does it apply identically to weights, activations, and KV cache?
27. Given a parameter count and a KV-cache configuration, compute real memory footprint at FP16, INT8, and INT4 for both targets.
28. Why does a real memory-footprint reduction from quantization not guarantee a proportional real latency reduction? Name the two separate real conditions that determine whether it does.
29. Why might KV-cache quantization carry more real accuracy risk than weight quantization?
30. A real notebook measured genuine 36.2% (INT8) and 54.3% (INT4) memory reductions on a small model/consumer GPU, but found generation became slower, not faster — INT8 by 3.37x, INT4 by 1.34x. What real, plausible mechanism explains a memory win producing a net latency loss?

**Module 06 — Batching Strategies — Static, Dynamic & Continuous Batching (6)**
31. Why does static batching waste real GPU compute even when the hardware itself has spare capacity?
32. Walk through how continuous batching removes the fixed-group constraint — what specifically happens the moment one sequence finishes?
33. Given a set of real sequences with different lengths and a real backfill queue, compute the real idle-compute waste under static vs. continuous batching.
34. Why is "continuous batching always improves both throughput and latency" an overclaim? What real, concrete gap can remain even under correct continuous-batching scheduling?
35. How does continuous batching's constant slot-freeing/refilling directly motivate pairing it with PagedAttention in production serving engines?
36. A real notebook's batching experiment showed throughput scaling well from batch size 1 to 4, then real per-request latency jumping 278% at batch size 8 — caused by two real sequences that never emitted a natural stop token. What production lesson does this straggler event illustrate that a synthetic example might not?

**Module 07 — Speculative Decoding & Decoding-Time Optimizations (6)**
37. Walk through the draft-then-verify mechanism — why can the target model verify $k$ candidate tokens in a single parallel forward pass, making verification far cheaper than generating those same $k$ tokens one at a time?
38. Derive the expected-accepted-tokens formula under the simplified constant-$\alpha$ acceptance model, and explain precisely what it does *not* capture about real speedup.
39. Given a real acceptance rate and draft length, compute expected accepted tokens per round; then, given a separate real draft/verification cost assumption, compute expected speedup as a distinct second step.
40. Why does speculative decoding specifically target the decode phase and not prefill?
41. Under what real conditions can speculative decoding produce a *negative* real speedup rather than a positive one?
42. A real notebook implemented and verified a correct speculative-decoding loop (α=0.4000, matching the formula's predicted acceptance closely) but measured a genuine 0.365x end-to-end speedup — a real slowdown. Why don't these two real findings contradict each other, and why is the slowdown specifically attributable to that notebook's own unoptimized implementation and hardware, not a general property of speculative decoding?

**Module 08 — Inference Serving Engines & Production Architecture (6)**
43. Along which stable architectural dimensions should real serving engines (vLLM, TensorRT-LLM, TGI, llama.cpp) be compared, and why is a feature checklist a weaker basis for comparison?
44. Walk through the full real replica-level serving architecture — what does each layer (router, replica, per-replica scheduler, GPU) actually do?
45. Precisely distinguish replica-level scaling from tensor/pipeline-parallel scaling — what real problem does each solve?
46. What is prefill/decode disaggregation, and what real trade-off does it introduce in exchange for reducing (not eliminating) interference between prefill and decode workloads sharing a pool?
47. A least-loaded routing simulation, started from an uneven queue-depth spread and run for 6 new requests, narrowed but did not eliminate that spread. Since this was a simulation and not a production routing experiment, what does the result still reveal about the limits of correct load-balancing logic alone?
48. When might tensor parallelism and replica-level scaling both be needed simultaneously for the same real deployment?

**Module 09 — Production Monitoring, Cost Modeling & Latency Optimization (6)**
49. Why is summing each pipeline stage's own p99 latency not a mathematically valid way to derive true end-to-end p99 latency?
50. Walk through the approximate latency-budget decomposition — what is it actually useful for, given it isn't an exact derivation?
51. Given real or assumed values for queue wait, prefill, decode, and network/serialization, compute a full latency budget and identify the dominant term.
52. Derive the GPU-time-based cost model and compute cost-per-request and cost-per-token from a real GPU-time and cost-rate example — why does this differ from an API token-price model?
53. Name this topic's full observability metric set — TTFT, TPOT, throughput, GPU compute utilization, memory/HBM utilization, KV-cache utilization, queue depth, and p95/p99 latency — and explain what distinct real failure mode each one catches that the others would miss.
54. *(synthesis)* A real notebook filled in the latency-budget decomposition with genuinely measured prefill/decode numbers instead of illustrative ones, finding decode's real share even higher (97.48%) than the illustrative example (76.2%). Design an end-to-end production inference stack for a new LLM feature — KV-cache/batching strategy, quantization target, serving architecture, and monitoring plan — and explain when and why decode-phase optimization should be prioritized first: specifically when real profiling shows it dominates the latency budget, as this topic's own repeated real measurements found, rather than as a universal default.

---

## 4. Batch Plan & Structural Compliance

Written in 5 batches of ~11 questions each (matching prior topics' 10-15-per-batch precedent), each question following the standardized format (`## Question N: Title` → `[ESSENTIAL]` → `[DEEP DIVE]`). Mandatory Section 5 structural compliance check before declaring the track done: all required per-question headings present exactly 54 times each; no derivation chains in any `[DEEP DIVE]` block; Final Revision Sheet present with exactly 3 required subsections (54-row Quick-Recall table, Essential Formula Cheat Sheet, Top Follow-up Q&As); no placeholder markers remain.

## 5. Open Design Questions / Dependencies

1. All 54 questions are derivable directly from already-complete, already-verified Track 1/Track 2 content — no new research or computation required before writing begins.
2. No open questions block starting Batch 1.

## Status: Complete

All 54 questions written to `modules/10_llm_inference_interview_questions.md` across 5 batches, incorporating every revised wording from the 11-point feedback round exactly. Mandatory structural compliance check passed on all points: (1) all 10 required per-question headings occur exactly 54 times each; (2) no derivation chains found in any `[DEEP DIVE]` block; (3) the Final Revision Sheet is present with all 3 required subsections (54-row Quick-Recall table, 9-formula Essential Formula Cheat Sheet, 10-entry Top Follow-up Q&As); (4) no placeholder markers remain. `helpers/compile_inference.py` now compiles a second, standalone `llm_inference_optimization_interview_cheatsheet.html`/`.pdf` (1,115,396 bytes) in addition to the master study guide — verified 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 54 `follow-up-section` divs, and 108 `q-card` divs (2 per question × 54), plus a visual spot-check of the rendered cover page and first question confirming the revised Q1 wording rendered correctly.
