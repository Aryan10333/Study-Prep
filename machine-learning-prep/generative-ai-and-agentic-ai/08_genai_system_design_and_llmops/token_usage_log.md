# Token Usage Log — Topic 08: GenAI System Design & LLMOps

## Methodology

No token-metering tool is exposed to the agent in this environment (no API-level `usage` object is surfaced per turn), so every number in this log is an **estimate**, not an exact count. Estimation method: `tokens ≈ chars / 4` (a standard rough approximation for English/code text), applied to:
- **Input (read)**: files read via tool calls during that step (source markdown/code the agent had to load into context to do the step's work).
- **Output (written)**: files written/edited during that step (the actual generated deliverable content).

This deliberately excludes the surrounding conversation/instruction overhead (system prompt, tool schemas, prior turns still in context), which this log has no way to measure — so these numbers are a **lower bound** on true total token consumption per step, useful for relative comparison across steps (which steps were expensive) rather than as an absolute cost figure. Logged after each major pipeline step (syllabus draft, each Track 1 module batch, each Track 2 notebook build, each Track 3 Q&A batch, each compilation run), not per individual tool call. Note: compiled HTML/PDF output is not counted token-for-token toward output tokens (it's a deterministic transformation of already-counted markdown source) — only newly authored markdown/code is counted, per the convention established in Topic 07's log.

---

## Log

| Step | Input (read) | Output (written) | Est. Input Tokens | Est. Output Tokens | Est. Step Total | Notes |
|---|---|---|---|---|---|---|
| 1. Syllabus draft (README.md) | `USER_PROFILE.md` (~5,900 chars, read) + `syllabus_generator/SKILL.md` (~2,050 chars, read) + Topic 07's own README.md (~6,900 chars, read as a structural/tone reference) | `README.md` (9,650 chars) | ~3,713 | ~2,413 | **~6,126** | Initial 9-module syllabus draft, pre-sign-off. Explicit "composition, not new component" framing — Cross-Module Boundary Discipline section references all 5 prior GenAI-track topics (03-07) by name, since this topic is structurally a synthesis/system-design layer built on top of them rather than a new standalone component topic. |

**Running total (est.): ~6,126 tokens**

| 2. Syllabus revision (12-point feedback) | User's 12-point feedback message (~4,300 chars), prior `README.md` (9,650 chars, held in context) | `README.md` diff (9,650 → 22,107 chars, +12,457 chars) | ~1,075 | ~3,114 | **~4,189** | Module 02: tightened from 5 to 4 core archetypes, on-device/cloud reframed as a variant not a 5th archetype. Module 03: explicit multi-assumption GPU-count estimation (rate/tokens/concurrency/utilization/SLO) and semantic-vs-retrieval caching split. Module 04: full knowledge-base lifecycle (deletion propagation, re-indexing/rollback, multi-tenant isolation). Module 05: real artifact-lineage tracking + Topic 07 boundary reworded to the operational-pipeline framing. Module 06: explicit promotion/rollback/abort criteria. Module 07: retry/idempotency/timeout policy + explicit fallback capability trade-offs. Module 08: data/tool-level authorization + named leakage/injection/unsafe-tool-access control points. Module 09: explicit interview time-boxing. |

**Running total (est.): ~10,315 tokens**

| 3. Track 1 (study guide) implementation plan draft | Approved `README.md` (22,107 chars, held in context) + `study_guide_generator/SKILL.md` (~7,700 chars, read) + Topic 07's own Track 1 plan (~6,500 chars, read as a structural reference) | `implementation_plan_study_guide.md` (15,629 chars) | ~9,077 | ~3,907 | **~12,984** | Per-module formula/hand-calc/diagram plan for all 9 modules (multi-assumption GPU-count estimation, two-layer cache-savings formula, vector storage sizing, canary multi-signal promotion rule, exponential-backoff retry math), correctly leaving several process/architecture modules (01, 02, 05, 08, 09) formula-free per the Formula Selection Constraint. Plan totals 2 plots + 7 diagrams — fewer plots than prior metric-heavy topics, reflecting this topic's architecture-heavy character. Pre-sign-off. |

**Running total (est.): ~23,299 tokens**

| 4. Track 1 plan revision (8-point feedback + 1 formula correction) | User's feedback message (~3,600 chars), prior plan (15,629 chars, held in context) | Plan diff (15,629→22,548 chars, +6,919 chars) | ~900 | ~1,730 | **~2,630** | **Formula correction**: Module 03's GPU-count formula fixed from a single ratio that double-counted service time/concurrency to an explicit two-step Little's-Law derivation. Module 03's cache-savings split into two differently-cost-based formulas. Module 04's storage formula separated replication from index overhead as two sequential steps. Module 06's canary rule gained a minimum-sample-size/observation-duration requirement. Module 07's backoff gained jitter + a retry-eligibility taxonomy. Module 08's injection-defense wording strengthened to trust-boundary/least-privilege framing. Module 09 reframed around real prioritization, not mechanical module coverage. Pre-sign-off. |

**Running total (est.): ~25,929 tokens**

| 5. Track 1, Module 01 write + code/SVG verification | Approved plan's Module 01 section (~1,300 chars, held in context) | `01_the_genai_system_design_interview_framework.md` (17,227 chars) | ~325 | ~4,307 | **~4,632** | No-formula process module. Reference code (`SystemDesignAnswer`/`framework_completeness_check`) verified via direct execution — matched exactly, no fix needed. Framework-flow SVG (5 steps + explicit NFR→architecture feedback path) rendered clean via headless Edge screenshot. Worked example: identical functional requirement under two different real NFR sets (200ms vs. 2s p99) correctly producing two different architectures. |

**Running total (est.): ~30,561 tokens**

| 6. Track 1, Module 02 write + code/SVG verification | Approved plan's Module 02 section (~1,150 chars, held in context) | `02_reference_architectures_for_genai_systems.md` (19,203 chars) | ~288 | ~4,801 | **~5,089** | No-formula composition module. Reference code (`classify_archetype`) verified via direct execution on 3 deliberately-ambiguous prompts — matched exactly, no fix needed. Consolidated 4-archetype comparison SVG (2x2 grid, each citing its real composed prior topic, on-device/cloud variant shown as a dashed overlay on Archetype 3, not a 5th box) rendered clean via headless Edge screenshot. |

**Running total (est.): ~35,650 tokens**

| 7. Track 1, Module 03 write + code verification | Approved (revised) plan's Module 03 section (~2,700 chars, held in context) | `03_capacity_estimation_traffic_modeling_and_cost_engineering.md` (12,946 chars) | ~675 | ~3,237 | **~3,912** | Corrected two-step Little's Law GPU-count derivation (L=120, N_GPU=22) and two-cost-basis cache-savings example ($300/day semantic vs. $80/day retrieval, despite retrieval's higher hit rate) both verified via direct execution — matched exactly, no fix needed. No SVG diagram for this module per plan (only a plot, generated later in the batch step); build-vs-buy cost plot path referenced. |

**Running total (est.): ~39,562 tokens**

| 8. Track 1, Module 04 write + code/SVG verification (incl. 1 real diagram self-catch) | Approved plan's Module 04 section (~1,750 chars, held in context) | `04_data_and_knowledge_infrastructure_at_scale.md` (16,502 chars) | ~438 | ~4,126 | **~4,564** | 3-step storage-sizing formula (raw=61.44GB, replicated=184.32GB, total=221.184GB) and deletion-propagation reaching every replica both verified via direct execution — matched exactly, no fix needed. SVG diagram initial render caught 2 real defects on visual inspection (a deletion-propagation label positioned too close to its own connector arrow, and a rollback-path label clipped by the adjacent isolation-boundary box) — both fixed and re-rendered clean before insertion, per the mandatory visual-inspection discipline. |

**Running total (est.): ~44,126 tokens**

| 9. Track 1, Module 05 write + code/SVG verification | Approved plan's Module 05 section (~1,850 chars, held in context) | `05_llmops_foundations_versioning_registries_and_cicd.md` (14,514 chars) | ~463 | ~3,629 | **~4,092** | Real 5-component lineage diff (`diff_lineage`) correctly isolating a config-only regression, plus quality-gate threshold logic, both verified via direct execution. CI/CD + lineage pipeline SVG (6 stages + explicit FAIL branch) rendered clean via headless Edge screenshot on the first attempt — no defects found this time. |

**Running total (est.): ~48,218 tokens**

| 10. Track 1, Module 06 write + code/SVG verification | Approved (revised) plan's Module 06 section (~2,200 chars, held in context) | `06_deployment_strategies_progressive_rollout_and_experimentation.md` (14,902 chars) | ~550 | ~3,726 | **~4,276** | Real 3-stage canary decision logic verified via direct execution: an early small-sample snapshot correctly held at NOT_YET_DECIDABLE despite looking "all green," and a quality-only regression correctly triggering ROLLBACK despite 3/4 signals passing — both matched exactly. 3-pattern rollout diagram (blue-green/canary/shadow + shared promotion-rule panel) rendered clean via headless Edge screenshot on the first attempt. |

**Running total (est.): ~52,494 tokens**

| 11. Track 1, Module 07 write + code/SVG verification | Approved (revised) plan's Module 07 section (~2,850 chars, held in context) | `07_reliability_engineering_redundancy_fallbacks_and_multi_provider.md` (17,573 chars) | ~713 | ~4,393 | **~5,106** | Jittered-backoff total (1900ms across 3 attempts) and the 5-scenario retry-eligibility taxonomy both verified via direct execution — matched exactly. Circuit-breaker state machine (closed/open/half-open) + 3-tier fallback chain SVG rendered clean via headless Edge screenshot on the first attempt. |

**Running total (est.): ~57,600 tokens**

| 12. Track 1, Module 08 write + code/SVG verification | Approved (revised) plan's Module 08 section (~2,300 chars, held in context) | `08_security_privacy_and_compliance_architecture.md` (15,492 chars) | ~575 | ~3,873 | **~4,448** | Real retrieval-time tenant filtering (correctly excluding all Tenant B docs) and least-privilege tool-authorization check (correctly blocking an injected tool request never in the permitted set) both verified via direct execution. Authentication-vs-authorization architecture SVG (2 stacked layers + 2 named control points) rendered clean via headless Edge screenshot on the first attempt. |

**Running total (est.): ~62,048 tokens**

| 13. Track 1, Module 09 write + code verification (final Track 1 module) | Approved (revised) plan's Module 09 section (~2,150 chars, held in context) | `09_end_to_end_genai_system_design_case_studies.md` (11,666 chars) | ~538 | ~2,917 | **~3,455** | Two real, time-boxed (~35-min) case-study walkthroughs (RAG enterprise assistant, agentic coding copilot), each explicitly citing Modules 02-08 and naming exactly 1 real prioritized deep-dive component. `prioritization_check` reference code verified via direct execution — both case studies correctly register STRONG, and a deliberate 8-component weak counter-example correctly registers WEAK. No diagram for this module per plan (synthesis-only). **All 9 Track 1 modules now complete.** |

**Running total (est.): ~65,503 tokens**

| 14. Plot generation (both planned plots) | Module 03/06's own verified hand-calc data (held in context) | `helpers/generate_system_design_plots.py` (4,827 chars) + 2 PNGs | ~200 | ~1,207 | **~1,407** | Both plots computed directly from each module's own already-verified data (build-vs-buy break-even at 770,000 req/mo from Module 03's N_GPU=22 worked example; canary ramp with the real PROMOTE/ROLLBACK stages from Module 06's own worked example) and generated cleanly on the first attempt — both verified via direct visual inspection, no clipping or overlap issues. |

**Running total (est.): ~66,910 tokens**

| 15. Compilation: master study guide PDF/HTML (adapted from Topic 07's compiler) | Topic 07's `compile_evaluation.py` (~27,500 chars, held in context from this session, read as an adaptation base) + all 9 module `.md` files (held in context from writing them) | `helpers/compile_system_design.py` (27,483 chars, ~95% reused machinery + a new topic-specific `main()`) + `genai_system_design_and_llmops_master_study_guide.html` (465,349 chars) + `.pdf` (1,286,887 bytes) | ~6,870 | ~2,000 | **~8,870** | Compiled successfully on the first real attempt. Verified: 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 9/9 `module-container` opening tags, 9/9 `follow-up-section` divs, 18/18 `q-card` divs — plus a visual headless-screenshot spot-check of the cover page and Module 01 (including the Framework Flow SVG) confirming clean rendering. **Track 1 (all 9 modules + 7 SVG diagrams + 2 plots + compilation) now fully complete.** |

**Running total (est.): ~75,780 tokens**

| 16. Track 2 (notebook) implementation plan draft | All 9 Track 1 modules' real content (held in context) | `implementation_plan_notebook.md` (11,076 chars) | ~500 | ~2,769 | **~3,269** | 6-notebook plan, all real, deterministic Python execution — no GPU/LLM/embedding API needed for any notebook, since this topic's algorithms (queueing simulation, lineage diff, canary decision, retry/backoff, authorization) are orchestration logic, not model behavior. Notebook 01 introduces a genuinely new real-execution pattern for this session: a live discrete-event queueing simulation empirically testing Little's Law, not just recomputing it. Notebook 02 deliberately uses no real embedding model, staying out of Topic 03's retrieval-quality scope. Pre-sign-off. |

**Running total (est.): ~79,049 tokens**

| 17. Track 2 plan revision (9-point feedback) | User's 9-point feedback message (~3,300 chars), prior plan (11,076 chars, held in context) | Plan diff (11,076→16,989 chars, +5,913 chars) | ~825 | ~1,478 | **~2,303** | N01: Little's Law framing corrected to an identity expected to hold under real steady-state measurement, not diverge at high utilization; T_req defined explicitly as service time, distinct from response time including queue wait. N02: explicitly scoped to this topic's own logic, not real vector-DB engine semantics. N03: regression scenarios made a controlled single-variable mutation test, multi-change moved to a separate limitation test. N05: added real success-rate/attempt-count/request-amplification metrics and repeated-trial statistical robustness. N06: capstone now requires an explicit failure-path run, not only the happy path. Labeling section: explicit real-execution-vs-real-observed-production-behavior distinction added. Pre-sign-off. |

**Running total (est.): ~81,352 tokens**

| 18. Track 2, Notebook 01 build (real discrete-event queueing simulation + cost sweep, Pass 1+2) | Signed-off Track 2 plan's Notebook 01 section (~2,900 chars, held in context) | `helpers/build_system_design_notebooks.py` (13,154 chars) + `01_capacity_queueing_simulation_and_cost_sweep.ipynb` (20,648 chars) | ~725 | ~8,451 | **~9,176** | Real FCFS multi-slot discrete-event simulation with a real steady-state measurement window. Experiment A (unbounded capacity): real L=120.52 via time-integration vs. 120.47 via λW, both matching Module 03's theoretical L=120 — Little's Law confirmed as an identity, not a divergence, per the corrected plan framing. Experiment B (ρ=0.857, c=140): an honest, real, somewhat unexpected finding — mean wait time only 0.0047s (0.16% overhead) due to real server-pooling effects at high server count, reported as-is rather than re-tuned to force a more dramatic queuing effect. Cost sweep exactly reproduced Module 03's $15,400 break-even at 770,000 req/mo; cache-savings sweep confirmed the real 10:1 cost-basis ratio holds at every swept hit rate. |

**Running total (est.): ~90,528 tokens**

| 19. Track 2, Notebook 02 build (real vector-DB lifecycle at scale, Pass 1+2) | Signed-off Track 2 plan's Notebook 02 section (~2,500 chars, held in context) | `helpers/build_system_design_notebooks.py` diff (+9,115 chars) + `02_knowledge_infrastructure_lifecycle.ipynb` (13,359 chars) | ~625 | ~5,619 | **~6,244** | No embedding model/API used, per the revised plan's explicit scope statement. Real 5-replica, 1,000-doc deletion propagation converged cleanly (150 deletions, 0 survivors). Real bad-reindex integrity check correctly flagged a 4.71% shrink and rollback exactly restored all 40 erroneously-dropped documents. Real 10-tenant, 2,000-doc isolation stress test found 0 cross-tenant leakage across all 10 combinations. All real results matched expectations cleanly, no fix needed. |

**Running total (est.): ~96,772 tokens**

| 20. Track 2, Notebook 03 build (controlled lineage mutation test + quality-gate sweep, Pass 1+2) | Signed-off Track 2 plan's Notebook 03 section (~2,600 chars, held in context) | `helpers/build_system_design_notebooks.py` diff (+7,707 chars) + `03_llmops_lineage_and_quality_gate.ipynb` (13,141 chars) | ~650 | ~5,212 | **~5,862** | All 5 real single-variable mutation scenarios correctly and uniquely localized. Real multi-component limitation test honestly demonstrated diff_lineage's ambiguity when 2 components change together, then resolved it via a real controlled-bisection function correctly isolating the true cause. Real boundary sweep confirmed inclusive `>=` threshold behavior at the exact 0.85 boundary. All real results matched expectations cleanly. |

**Running total (est.): ~102,634 tokens**

| 21. Track 2, Notebook 04 build (canary decision engine edge-case scenarios, Pass 1+2) | Signed-off Track 2 plan's Notebook 04 section (~1,600 chars, held in context) | `helpers/build_system_design_notebooks.py` diff (+7,237 chars) + `04_canary_decision_engine_scenarios.ipynb` (11,449 chars) | ~400 | ~4,672 | **~5,072** | Real exact-threshold boundary (0.01 promotes, 0.0101 rolls back), real double-signal-failure rollback, and real monitoring-window OR/AND boundary scenarios (N_min-without-T_min and T_min-without-N_min both correctly NOT_YET_DECIDABLE) all verified via direct execution — matched exactly, no fix needed. Consistently `[SIMULATION]`-labeled per the revised plan's real-execution-vs-real-observed-behavior distinction. |

**Running total (est.): ~107,706 tokens**

| 22. Track 2, Notebook 05 build (real flaky-service retry/backoff timing, Pass 1+2) | Signed-off Track 2 plan's Notebook 05 section (~3,000 chars, held in context) | `helpers/build_system_design_notebooks.py` diff (+10,337 chars) + `05_retry_backoff_reliability_timing.ipynb` (16,208 chars) | ~750 | ~6,636 | **~7,386** | Real live flaky mock service (genuine random failure injection) called by real jittered and naive clients across 300 real trials each. Honest real result: success rate/attempts/request-amplification came out statistically similar between strategies (0.9867/1.677/503 vs. 0.9967/1.617/485) — jitter's real cost is purely added per-task latency (37.97ms vs 10.50ms), not a success-rate improvement. Real 15-trial concurrent burst measurement found jitter consistently spreads retry timestamps more than naive (median 26.90ms vs 9.81ms) with an honestly-reported real confound: OS thread-scheduling noise itself contributes non-trivial spread even to the naive case. Real 8-scenario retry-eligibility taxonomy test passed exactly. |

**Running total (est.): ~115,092 tokens**

| 23. Track 2, Notebook 06 build (authorization stress test + full framework capstone, Pass 1+2, final Track 2 notebook) | Signed-off Track 2 plan's Notebook 06 section (~3,200 chars, held in context) | `helpers/build_system_design_notebooks.py` diff (+13,278 chars) + `06_authorization_and_capstone.ipynb` (20,198 chars) | ~800 | ~8,369 | **~9,169** | Real 90-combination authorization stress test found 0 breaches. Real prioritization-check boundary confirmed exactly at 2-vs-3 deep-dive components. Real capstone happy path chained all 8 Modules 01-08 functions successfully on a new HR-chatbot scenario. Real required failure-path run correctly halted the composed pipeline at a real injected canary-quality regression (0.79<0.85, ROLLBACK), verified Modules 07-08 were never reached. All real results matched expectations, no fix needed. **All 6 Track 2 notebooks now complete; Track 2 implementation plan marked Status: Complete.** |

**Running total (est.): ~124,261 tokens**
