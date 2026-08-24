# Token Usage Log — Topic 06: LLM Inference & Optimization

## Methodology

No token-metering tool is exposed to the agent in this environment (no API-level `usage` object is surfaced per turn), so every number in this log is an **estimate**, not an exact count. Estimation method: `tokens ≈ chars / 4` (a standard rough approximation for English/code text), applied to:
- **Input (read)**: files read via tool calls during that step (source markdown/code the agent had to load into context to do the step's work).
- **Output (written)**: files written/edited during that step (the actual generated deliverable content).

This deliberately excludes the surrounding conversation/instruction overhead (system prompt, tool schemas, prior turns still in context), which this log has no way to measure — so these numbers are a **lower bound** on true total token consumption per step, useful for relative comparison across steps (which steps were expensive) rather than as an absolute cost figure. Logged after each major pipeline step (syllabus draft, each Track 1 module batch, each Track 2 notebook build, each Track 3 Q&A batch, each compilation run), not per individual tool call.

---

## Log

| Step | Input (read) | Output (written) | Est. Input Tokens | Est. Output Tokens | Est. Step Total | Notes |
|---|---|---|---|---|---|---|
| 1. Syllabus draft (README.md) | Prior topic READMEs' established format (held in context from this session) | `README.md` (7,853 chars) | ~1,200 | ~1,963 | **~3,163** | Initial 9-module syllabus draft, pre-sign-off. |

**Running total (est.): ~3,163 tokens**

| 2. Syllabus revision (Pass 1 feedback) | User's 9-point feedback message (~2,600 chars), prior `README.md` (7,853 chars, held in context) | `README.md` diff (7,853 → 10,559 chars, +2,706 chars) | ~2,613 | ~677 | **~3,290** | Qualified the KV-cache-bottleneck claim as workload-dependent, clarified PagedAttention (allocation/utilization, not compression) vs. FlashAttention (IO/HBM traffic) distinction, added KV-cache quantization, softened the batching latency claim, added replica-level architecture + prefill/decode disaggregation, and expanded Module 09's latency decomposition and observability metrics. |

**Running total (est.): ~6,453 tokens**

| 3. Track 1 (study guide) implementation plan draft | Approved `README.md` (10,559 chars, held in context) + `study_guide_generator/SKILL.md` conventions (held in context from this session) | `implementation_plan_study_guide.md` (9,492 chars) | ~2,640 | ~2,373 | **~5,013** | Per-module formula/hand-calc/diagram plan for all 9 modules (roofline model, KV-cache memory, HBM access counts, quantization bytes-per-param, speculative-decoding expected-acceptance, p99 decomposition), pre-sign-off. |

**Running total (est.): ~11,466 tokens**

| 4. Track 1 plan revision (Pass 2 feedback) | User's 6-point feedback message (~1,300 chars), prior plan (9,492 chars, held in context) | Plan diff (9,492→12,699 chars, +3,207 chars) | ~2,698 | ~802 | **~3,500** | Switched Module 02's formula to $N_{\text{KV\_heads}}$ + added GQA/MQA; reworded Module 07's formula as a simplified-acceptance-model expectation with a separate, assumption-stated speedup step; reworded Module 09's p99 formula as an approximate budgeting tool + switched to a GPU-time-based cost model; qualified Module 01's compute/memory-bound claim and Module 05's precision-vs-speed claim as typical patterns, not universal rules. |

**Running total (est.): ~14,966 tokens**

| 5. Track 1, Module 01 write + code/SVG verification | Approved plan's Module 01 section (~1,700 chars, held in context) | `01_inference_fundamentals_and_decoding_loop.md` (17,540 chars) | ~425 | ~4,385 | **~4,810** | Roofline hand calc verified exact (ridge=153.0, decode I=1.0, prefill I=512.0); batched-decode assertion (I=256>ridge) confirmed the "typical, not universal" framing; timeline SVG rendered clean. |

**Running total (est.): ~19,776 tokens**

| 6. Track 1, Module 02 write + code verification (incl. one real fix) | Approved plan's Module 02 section (~1,900 chars, held in context) | `02_kv_cache_mechanics_and_memory_management.md` (13,543 chars) | ~475 | ~3,386 | **~3,861** | Real bug caught by execution: initial hand calc mixed decimal-GB (÷1e9) and binary-GB (÷1024³) units, giving wrong Step 2-4 figures (0.268/68.7/17.2 GB); running the reference code's `assert abs(high_mha_gb - 68.7) < 1.0` failed against the real computed 64.0 GB, exposing the mismatch. Fixed by standardizing the hand calc on binary GB matching the code's own `/1024**3`, giving corrected, code-verified values (weights=13.04 GB, low=0.25 GB, high MHA=64.0 GB exact, high GQA=16.0 GB exact, 4.0x reduction) — re-executed clean, all assertions passed. No planned SVG diagram for this module (per plan, diagrams are M01/03/04/06/08/09 only). |

**Running total (est.): ~23,637 tokens**

| 7. Track 1, Module 03 write + code/SVG verification | Approved plan's Module 03 section (~1,000 chars, held in context) | `03_pagedattention_and_memory_efficient_serving.md` (14,152 chars) | ~250 | ~3,538 | **~3,788** | No-formula worked example (5 real variable-length sequences) computed and verified via direct execution before writing prose: contiguous waste 80.52%, paged waste 1.82%, 44.2x reduction — matched exactly between hand calc and code, no fix needed this time. Contiguous-vs-paged allocation SVG rendered clean (proportional bars, visible waste slivers) via headless Edge screenshot. |

**Running total (est.): ~27,425 tokens**

| 8. Track 1, Module 04 write + code/SVG verification | Approved plan's Module 04 section (~700 chars, held in context) | `04_flashattention_and_io_aware_computation.md` (13,481 chars) | ~175 | ~3,370 | **~3,545** | Simplified HBM-access-count formula (explicitly labeled non-derivation, intuition-level per Concept Simplification rule) computed and verified before writing prose: N=8 ratio=3.0x, N=64 ratio=17.0x, confirming the reduction grows with sequence length — matched exactly, no fix needed. Naive-HBM-materialization vs. tiled-SRAM-streaming schematic SVG rendered clean via headless Edge screenshot. |

**Running total (est.): ~30,970 tokens**

| 9. Track 1, Module 05 write + code verification | Approved plan's Module 05 section (~800 chars, held in context) | `05_quantization_for_inference.md` (10,357 chars) | ~200 | ~2,589 | **~2,789** | Bytes-per-parameter formula applied to both weights (13.04/6.52/3.26 GB across FP16/INT8/INT4) and KV cache (reusing Module 02's high-concurrency MHA scenario: 64.00/32.00/16.00 GB), verified via direct execution before writing prose — matched exactly, no fix needed. No SVG diagram or plot for this module (per plan). |

**Running total (est.): ~33,759 tokens**

| 10. Track 1, Module 06 write + code/SVG verification | Approved plan's Module 06 section (~900 chars, held in context) | `06_batching_strategies.md` (14,857 chars) | ~225 | ~3,714 | **~3,939** | No-formula worked example simulated via real greedy backfill code before writing prose: static waste 54.37%, continuous waste 13.75%, 4.0x reduction — simulation honestly surfaced one real unbackfilled leftover request (a genuine scheduling gap, not zero waste), which became the module's concrete evidence for the workload/scheduler-dependent framing rather than an idealized 100%-utilization claim. Static-vs-continuous timeline SVG rendered clean via headless Edge screenshot. Plot image path referenced for later batch generation (per established pattern). |

**Running total (est.): ~37,698 tokens**

| 11. Track 1, Module 07 write + code verification | Approved plan's Module 07 section (~1,300 chars, held in context) | `07_speculative_decoding_and_decoding_time_optimizations.md` (11,011 chars) | ~325 | ~2,753 | **~3,078** | Two-step formula (simplified acceptance model E[accepted]=3.362 at α=0.8,k=4; separate speedup step with stated 0.2x draft-cost assumption ≈1.868x) verified via direct execution, matched exactly. Added one extra real code check beyond the plan's minimum: a low-α=0.3 case, which real-computed to 0.792x — an actual slowdown, not just "shrinks toward 1x" — a stronger, honestly-discovered confirmation of the module's conditional-win framing. No SVG diagram for this module (per plan); speedup-vs-acceptance-rate plot path referenced for later batch generation. |

**Running total (est.): ~40,776 tokens**

| 12. Track 1, Module 08 write + code/2 SVG verification | Approved plan's Module 08 section (~1,000 chars, held in context) | `08_inference_serving_engines_and_production_architecture.md` (15,017 chars) | ~250 | ~3,754 | **~4,004** | No-formula comparative survey (vLLM/TensorRT-LLM/TGI/llama.cpp across 4 stable dimensions); least-loaded routing simulation verified via direct execution, honestly showing a short 6-request burst narrows real queue-depth spread from 6 to 2 (not fully to 0) — used as-is rather than adjusted to look cleaner. Both replica-architecture and prefill/decode-disaggregation SVG panels rendered clean in one combined diagram via headless Edge screenshot. |

**Running total (est.): ~44,780 tokens**

| 13. Track 1, Module 09 write + code/SVG verification (final Track 1 module) | Approved plan's Module 09 section (~1,600 chars, held in context) | `09_production_monitoring_cost_and_latency.md` (13,154 chars) | ~400 | ~3,289 | **~3,689** | Two real worked examples verified via direct execution before writing prose: latency budget (150+300+1600+50=2100ms, exceeds 2000ms SLO, decode=76.2% of total) and GPU-time cost model ($0.001056/request, $0.00000528/token) — matched exactly, no fix needed. Stacked latency-budget-decomposition SVG (with SLO marker and explicit "budgeting tool, not exact p99" caption) rendered clean via headless Edge screenshot. **All 9 Track 1 modules now complete.** |

**Running total (est.): ~48,469 tokens**

| 14. Plot generation (all 4 planned plots) + fixes | Module 01/02/07's own verified formulas/values (held in context) | `helpers/generate_inference_plots.py` (7,976 chars) + 4 PNGs | ~300 | ~1,994 | **~2,294** | Roofline (M01), KV-cache-memory-vs-length (M02, real), speedup-vs-acceptance-rate (M07, real) plots computed directly from each module's own already-verified formulas/constants (ridge=153.0, MHA/GQA formula, k=4/draft_cost=0.2 — all matched exactly); batching plot (M06) explicitly labeled illustrative. Two title-clipping cosmetic issues (M02, M07 plots) caught via visual inspection and fixed by shortening titles, then re-rendered and re-verified clean. Also fixed an initial Tkinter backend crash (`matplotlib.use("Agg")` added) before any plot could render. |

**Running total (est.): ~50,763 tokens**

| 15. Compilation: master study guide PDF/HTML (adapted compiler, built proactively this time — not forgotten as in Topic 05) | Topic 05's `compile_prompt_eng.py` (~27,300 chars, read as an adaptation base) + all 9 module `.md` files (held in context from writing them) | `helpers/compile_inference.py` (27,336 chars) + `llm_inference_optimization_master_study_guide.html` (687,644 chars) + `.pdf` (1,484,451 bytes) | ~6,825 | ~6,834 | **~13,659** | Compiled successfully on the first real attempt (no retry needed). Verified: 0 `file://` leaks, 0 `MATHPLACEHOLDER` leaks, 9/9 `module-container` divs, 9/9 `follow-up-section` divs, 18/18 `q-card` divs (2 per module × 9), no image-not-found warnings during embedding (all 4 plots + inline SVGs embedded correctly) — plus a visual headless-screenshot spot-check of the cover/Module-01-opening confirming clean rendering. **Track 1 (all 9 modules + 4 plots + compilation) now fully complete.** |

**Running total (est.): ~64,422 tokens**

| 16. Track 2 (notebook) implementation plan draft | All 9 Track 1 modules' real verified formulas/values (held in context) | `implementation_plan_notebook.md` (9,328 chars) | ~500 | ~2,332 | **~2,832** | 6-notebook plan grouping 9 modules thematically (matching Topic 05's precedent), all 6 notebooks planned as real GPU/hardware execution (no illustrative-only notebook this time) using the same RTX 4060 + Qwen2.5-0.5B-Instruct from Topic 05's Notebook 04. Explicit vLLM-infeasibility reasoning documented (8GB VRAM too tight) with a real-data-grounded simulation substitute for Module 03. Pre-sign-off. |

**Running total (est.): ~67,254 tokens**

| 17. Track 2 plan revision (8-point feedback) | User's 8-point feedback message (~1,850 chars), prior plan (9,328 chars, held in context) | Plan diff (9,328→14,612 chars, +5,284 chars) | ~2,795 | ~1,321 | **~4,116** | Notebook 01: reframed decode-vs-length claim as a hypothesis, not expected result; added warm-up/repeated-run/CUDA-sync/median+p95 methodology. Notebook 02: replaced `memory_allocated()`-alone with a delta-based KV-cache-growth estimate plus explicit allocated-vs-reserved reporting to surface allocator overhead. Notebook 03: narrowed the claim to "measured backend performance difference," explicitly not a direct HBM-IO measurement. Notebook 05: explicitly labeled the PagedAttention exercise `[SIMULATION]` even with real inputs. Notebook 06: split acceptance-rate measurement from speedup measurement as two distinct real numbers (mirroring Module 07's own two-step discipline); removed the greedy-vs-assisted fallback in favor of an honest infeasibility report if speculative decoding can't run. Section 4 replaced the blanket `[REAL]` label with a 4-tier `[REAL]`/`[COMPUTED FROM REAL DATA]`/`[SIMULATION]`/`[EXPLANATION]` scheme applied per-cell across Section 3. |

**Running total (est.): ~71,370 tokens**

| 18. Track 2, Notebook 01 build (Pass 1 real GPU execution + Pass 2 explanations) | Signed-off Track 2 plan's Notebook 01 section (~1,400 chars, held in context) | `helpers/build_inference_notebooks.py` (10,138 chars) + `01_decode_vs_prefill_fundamentals.ipynb` (18,344 chars) | ~350 | ~7,121 | **~7,471** | Real GPU execution confirmed (RTX 4060, 8.00GB VRAM, Qwen2.5-0.5B-Instruct loaded FP16). Real measured TTFT stayed flat (143.64→157.61ms across 32→1024 prompt tokens) and TPOT stayed flat (138.058→148.779ms/token, 7.8% spread) — hypothesis-framed per the revised plan, not assumed. Pass 2 converted TTFT to per-prompt-token cost, revealing a real, honestly-discovered ≈897x gap vs. TPOT at 1024 tokens (0.1539ms vs 138.058ms/token) — a genuine, striking confirmation of Module 01's prefill/decode claim not anticipated when the plan was drafted, reported with the literal real numbers quoted in backticks per the Literal-Quote Requirement. |

**Running total (est.): ~78,841 tokens**

| 19. Track 2, Notebook 02 build (Pass 1 real GPU execution, 1 real bug fixed, + Pass 2 explanations) | Signed-off Track 2 plan's Notebook 02 section (~1,700 chars, held in context) | `helpers/build_inference_notebooks.py` diff (+~4,900 chars) + `02_kv_cache_memory_measurement.ipynb` (17,142 chars) | ~425 | ~6,858 | **~7,283** | Real bug caught by execution: `past_key_values.key_cache` doesn't exist in transformers 5.14.1's real Cache API (raised `AttributeError: 'NoneType' object has no attribute 'numel'`); inspected the real object via `dir()` and found `past_kv.layers[i].keys/.values` instead, fixed with a version-robust extractor, re-executed clean. Real GQA config confirmed directly from the model (`n_kv_heads=2`, `n_query_heads=14`, 7x sharing). Exact-tensor-size measurement matched Module 02's formula precisely at all 4 real sequence lengths (0 diff). `memory_allocated()`-alone delta was found ~25.7-26.3x larger than real KV-cache size; root-caused (not just noted) to the real logits tensor (`batch×seq_len×vocab_size`, vocab=151,936) accounting for 93.9-96.1% of that gap — a precise, quantitative validation of the plan's revised methodology, not just a qualitative caveat. |

**Running total (est.): ~86,124 tokens**

| 20. Track 2, Notebook 03 build (real backend-availability probe, honest fallback applied, Pass 1+2) | Signed-off Track 2 plan's Notebook 03 section (~1,650 chars, held in context) | `helpers/build_inference_notebooks.py` diff (+~5,600 chars) + `03_flashattention_sdpa_backend_comparison.ipynb` (14,001 chars) | ~460 | ~6,270 | **~6,730** | Real, honest constraint discovered via execution (probed live, not assumed): `FLASH_ATTENTION` genuinely unavailable on this exact torch 2.13.0+cu126 build ("Torch was not compiled with flash attention"), independent of and in addition to a separate real GQA head-count kernel-compatibility issue found during initial exploration. Applied the plan's fallback discipline: compared `EFFICIENT_ATTENTION` vs. `MATH` instead, reported the constraint honestly rather than silently substituting. Real result: latency ratio grew 5.34x→33.07x and memory ratio grew 3.71x→55.94x as sequence length grew 512→4096; real per-doubling memory scaling (MATH ~3.0-3.8x, EFFICIENT ~1.25-1.57x) closely tracked the expected O(L²) vs O(L) pattern. Explanation cells explicitly reiterated the plan's "consistent with, not a direct measurement of, HBM traffic" scope distinction. |

**Running total (est.): ~92,854 tokens**

| 21. Track 2, Notebook 04 build (bitsandbytes install, 1 real file-bloat bug fixed, Pass 1+2) | Signed-off Track 2 plan's Notebook 04 section (~1,150 chars, held in context) | `helpers/build_inference_notebooks.py` diff (+~4,700 chars) + `04_quantization_benchmark.ipynb` (17,768 chars, after fix; peaked at 14.3MB pre-fix, discarded) | ~350 | ~4,850 | **~5,200** | Installed real `bitsandbytes` (not previously present) and verified real INT8/INT4 loading before building. Real bug caught by inspecting file size: bitsandbytes' `logging.warning()` call (not `warnings.warn()`, confirmed by reading its source) fired once per matmul across all timed repeats, bloating the saved notebook to 14.3MB; fixed via `logging.getLogger("bitsandbytes").setLevel(logging.ERROR)`, verified silent with a small real test before the full rebuild, re-executed clean at 17,768 chars. Real result reported as-is: FP16 942.29MB/97.009ms/token was both smallest-memory-of-none and *fastest*; INT8 (601.04MB, 36.2% smaller) was 3.37x slower; INT4 (430.42MB, 54.3% smaller) was 1.34x slower — a strong, honest, non-adjusted confirmation of Module 05's central caveat that a real memory win here was not just non-proportional but net negative on latency. |

**Running total (est.): ~98,054 tokens**

| 22. Track 2, Notebook 05 build (real batching + grounded simulation, Pass 1+2) | Signed-off Track 2 plan's Notebook 05 section (~1,300 chars, held in context) | `helpers/build_inference_notebooks.py` diff (+~5,400 chars) + `05_batching_throughput_and_paged_simulation.ipynb` (14,394 chars) | ~325 | ~5,943 | **~6,268** | Verified real chat-templated prompts naturally emit varied EOS positions before building (probed live: eos_pos=[7,9,1,6] on a small test) so the simulation's real length inputs weren't contrived. Real throughput scaled 6.36x from bs=1→bs=4, then *dropped* at bs=8 with per-request latency jumping 278% (202.0→764.2ms) — a genuine, unplanned real straggler event (2 of 8 sequences hit the 80-token ceiling with no EOS, min/max spread=40x) that concretely reproduced Module 06's static-batching problem without being staged. `[SIMULATION]` section fed those exact real lengths into a Module-03-style allocator, correctly labeled distinct from `[REAL]` per the revised plan: 68.75% contiguous waste → 10.71% paged waste (6.42x reduction), explicitly traced back to the same two real straggler values. |

**Running total (est.): ~104,322 tokens**

| 23. Track 2, Notebook 06 build (real speculative decoding + capstone, 2 real bugs fixed, Pass 1+2) | Signed-off Track 2 plan's Notebook 06 section (~2,000 chars, held in context) | `helpers/build_inference_notebooks.py` diff (+~10,700 chars) + `06_speculative_decoding_and_production_capstone.ipynb` (27,523 chars) | ~675 | ~9,443 | **~10,118** | Verified real INT4-vs-FP16 assisted generation works before committing to the draft/target pairing (a legitimate self-speculative pattern). Two real bugs caught and fixed via direct verification, not assumed away: (1) the model's default `repetition_penalty=1.1` made `generate()`'s "greedy" output diverge from manual `argmax` comparison — caught by an explicit correctness check that initially failed, fixed by passing `repetition_penalty=1.0` everywhere, re-verified byte-for-byte match; (2) a nested f-string double-brace typo in the routing simulation printed literal unevaluated Python instead of the real dict — caught by reading raw output, fixed, rebuilt clean. Real results, all reported as measured: acceptance rate α=0.4000 (verified-correct), formula-predicted E[accepted]=1.650 closely matched the real observed 40/25=1.6 accepted/round; real end-to-end speedup was 0.365x (a genuine slowdown, honestly attributed to this reference implementation's real, avoidable KV-cache-reuse overhead, kept as a separate finding from acceptance rate per Module 07's two-step discipline). `[COMPUTED FROM REAL DATA]` capstone filled Module 09's budget with real N01 prefill/decode numbers (decode share 97.48%, real GPU cost $0.00007757/token). `[SIMULATION]` routing exercise correctly labeled distinct from `[REAL]`, using N05's real straggler lengths to show workload skew limits even correct least-loaded routing. **All 6 Track 2 notebooks now complete.** |

**Running total (est.): ~114,440 tokens**
