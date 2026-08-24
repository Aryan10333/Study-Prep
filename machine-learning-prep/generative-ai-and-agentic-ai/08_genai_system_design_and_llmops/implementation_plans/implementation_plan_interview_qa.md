# Implementation Plan: Track 3 — Interview Q&A (`08_genai_system_design_and_llmops`)

Per `interview_qa_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 3 (the standalone Interview Q&A cheatsheet); Tracks 1 (9 study-guide modules, 7 SVG diagrams, 2 plots, PDF/HTML compilation) and 2 (6 notebooks, all real deterministic Python execution — no GPU/LLM API needed) are both complete and pushed.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_genai_system_design_and_llmops_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `genai_system_design_and_llmops_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `genai_system_design_and_llmops_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_system_design.py` — add a second `compile_document()` call in `main()`, producing a separate standalone cheatsheet |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules and their real hand calcs (the 5-step interview framework, 4 core archetypes, the corrected two-step Little's Law GPU-count derivation, the two-cost-basis caching formulas, the 3-step storage-sizing formula, real artifact lineage, the canary promotion rule with its monitoring-window requirement, jittered backoff + retry eligibility, data/tool-level authorization, and the prioritization-focused case studies). Following prior topics' established differentiator: where a question's real Track 2 notebook result adds genuine interview value, the **Production Perspective** or **Common Mistakes** sections cite it explicitly, framed as an observation from a specific real experiment, not a universal law.

This topic's Track 2 produced a genuinely rich set of real, sometimes-counterintuitive findings worth citing this way: a real, live discrete-event simulation confirming Little's Law's identity holds under real steady-state measurement, alongside an honest real finding that server pooling kept queuing delay negligible even at 85.7% utilization (Notebook 01); real deletion propagation and bad-reindex rollback verified at 5x/10x the module's own worked scale (Notebook 02); a real controlled single-variable lineage-mutation test that correctly localized every one of 5 components individually, paired with an honestly-surfaced real limitation (lineage-diffing alone cannot attribute cause when multiple components change together) resolved via real controlled bisection (Notebook 03); real canary-engine edge-case verification at the exact threshold boundary and at the monitoring-window's AND/OR boundary (Notebook 04); a real, live retry/backoff timing experiment against a genuinely flaky mock service finding that jitter's real value is entirely about added latency traded for measured timestamp desynchronization — not an improvement in real success rate or request amplification, both of which came out statistically similar between strategies (Notebook 05); and a real, consolidated capstone demonstrating both a happy-path and a required failure-path run through the full composed pipeline (Notebook 06).

## 3. Proposed Question List (54 questions, grouped by module)

**Module 01 — The GenAI System Design Interview Framework (6)**
1. Why does an unstructured system-design answer tend to fail even when the candidate knows the underlying technology well?
2. Walk through the module's own 5-step framework — why must non-functional requirements (Step 2) be gathered before the architecture is proposed (Step 4), not after?
3. Given a single functional requirement under two different real, stated latency budgets, explain how the same framework can correctly produce two different architectures for each.
4. Why is back-of-envelope capacity estimation treated as a first-class Step 3, not an afterthought squeezed in after the architecture is drawn?
5. How should a candidate handle an interviewer who deliberately withholds non-functional requirements?
6. A real reference-code check verified that skipping Step 2 or Step 3 is correctly flagged as an incomplete framework application, and that the identical functional requirement under two different real NFR sets produced two genuinely different, both-valid architectures. Why does this real result matter more than simply asserting the framework "sounds reasonable"?

**Module 02 — Reference Architectures for Common GenAI System Design Questions (6)**
7. Why are the archetypes deliberately limited to 4, rather than a longer, more exhaustive taxonomy?
8. Walk through why the on-device/cloud-hybrid pattern is treated as a variant overlay on one archetype, not a 5th standalone archetype.
9. Given a system-design prompt whose surface wording sounds agentic but whose actual requirement is a single retrieval-then-answer flow, explain how to correctly classify it and why the surface wording is misleading.
10. What is each of the 4 archetypes' real dominant bottleneck, and why does correctly naming it matter for where Step 5's deep-dive should go?
11. How should a candidate handle a prompt that doesn't cleanly fit any single archetype?
12. A real reference-code classifier correctly re-derived each of the module's own 3 deliberately-ambiguous prompt classifications from their real control-flow/synchronicity signals, not their surface vocabulary. Why is control-flow/synchronicity a more reliable classification signal than keyword matching?

**Module 03 — Capacity Estimation, Traffic Modeling & Cost Engineering (6)**
13. Why does a single "QPS ÷ per-GPU throughput" shortcut risk double-counting service time and per-GPU concurrency?
14. Walk through the module's own corrected two-step derivation — what does Step 1 (Little's Law) establish, and what does Step 2 add that Step 1 alone cannot?
15. Given a real stated QPS, mean service time, per-GPU concurrent capacity, and utilization target, compute real required concurrency and real GPU count using the two-step method.
16. Why must semantic-cache savings and retrieval-cache savings be computed against two different real cost bases, never summed against one shared baseline?
17. Why is a utilization target strictly less than 1 a real, deliberate design choice, not a conservative afterthought?
18. A real, live discrete-event simulation empirically confirmed Little's Law's identity under real steady-state measurement (two independently-measured quantities agreeing closely), and separately found that at 85.7% real utilization with many parallel server slots, real queuing delay stayed under 0.2% overhead due to a real server-pooling effect. Why is "utilization alone determines queuing delay" the wrong takeaway from this real result — what other real factors (server count, arrival/service variability, scheduling discipline) also shape real queuing delay at a given utilization level?

**Module 04 — Data & Knowledge Infrastructure at Scale (6)**
19. Why does this module's real scope stop at infrastructure operations (deletion, versioning, tenant isolation) rather than extending into retrieval-ranking quality?
20. Walk through the module's own 3-step storage formula, in its explicit real order (base vector storage → replication → index/metadata overhead) — why must index overhead be applied AFTER replication, not before, and why are the two kept as separate, sequential real factors instead of one blended term?
21. Given a real corpus size, embedding dimension, and stated replication/index-overhead factors, compute real total storage in the module's own 3 explicit steps.
22. Why must a real deletion event propagate to every index replica, not just the primary, and what real compliance risk does a lagging replica create?
23. Why is a bad re-index required to have a real rollback path rather than being treated as a one-way commit?
24. A real notebook validated deletion propagation and bad-reindex rollback at 5x/10x the module's own worked scale, and explicitly stated it validates this topic's own lifecycle logic, not any specific vector-database engine's real internal consistency guarantees. Why is that distinction important to state explicitly in a system-design interview answer?

**Module 05 — LLMOps Foundations: Versioning, Registries & CI/CD Pipeline Architecture (6)**
25. Precisely state the real, required operational flow this module owns — why is "versioned inputs → evaluation execution → quality gate → approval → deployment → recorded lineage" a more accurate description than "running tests"?
26. Walk through why a production result is only fully explainable when all 5 real lineage components are jointly recorded, not just the model version.
27. Given two real lineage snapshots that differ in exactly one component, explain how to correctly localize a real regression's cause from that comparison alone.
28. Why can lineage-diffing alone not resolve which of several simultaneously-changed components caused an observed regression, and what real, further step is needed to isolate it?
29. Why should a change to the quality gate's own threshold itself be versioned, rather than silently adjusted?
30. A real notebook ran a controlled single-variable mutation test correctly localizing all 5 real lineage components individually, then honestly demonstrated a real multi-component limitation and resolved it via a real controlled-bisection function. Why does deliberately testing and reporting a tool's own limitation make it a more, not less, credible interview answer?

**Module 06 — Deployment Strategies, Progressive Rollout & Experimentation Infrastructure (6)**
31. Compare blue-green, canary, and shadow deployment along their real risk/speed trade-off — when would each be the right real choice?
32. Walk through the module's own canary promotion rule — why is it a conjunction of independently-monitored signals rather than one blended average score?
33. Why must a real minimum sample size and minimum observation duration both be satisfied before any promote/rollback decision, and what real risk does skipping that requirement introduce? Frame $N_{\text{min}}$/$T_{\text{min}}$ as real, stated minimum decision-quality requirements whose actual values depend on real traffic volume, SLOs, and statistical-confidence needs — not as universal constants that apply identically to every real rollout.
34. Given a canary stage's real per-signal metric values and the module's own thresholds, apply the promotion rule and correctly classify the real outcome.
35. Why is a GenAI system's real regression risk described as more asymmetric than a typical stateless microservice's?
36. A real reference-code engine correctly registered an exact-threshold value as PROMOTE, correctly rolled back on a double-signal failure, and correctly returned NOT_YET_DECIDABLE when only one of the two real monitoring-window conditions was satisfied. Why does testing exact boundary cases, not just comfortably-passing/failing ones, matter for trusting a decision engine's real correctness?

**Module 07 — Reliability Engineering: Redundancy, Fallbacks & Multi-Provider Architecture (6)**
37. Precisely distinguish what exponential growth and real jitter each solve in the backoff formula — why does exponential growth alone reduce sustained retry pressure on a degraded dependency over time, while jitter alone specifically prevents many real clients from retrying in synchronized lockstep, and why does a real production backoff policy need both, not just one?
38. Walk through the real retry-eligibility taxonomy — why must a timed-out, non-idempotent request be treated differently from a timed-out, idempotent one?
39. Why does a circuit breaker need a real HALF-OPEN state rather than transitioning directly from OPEN back to CLOSED after a cooldown?
40. Why must a fallback model's real capability difference (context window, quality, safety behavior) be stated explicitly rather than treated as a transparent substitute?
41. Given a bounded retry budget with real jittered backoff, compute the real total delay across the attempts. A naive immediate-retry baseline remains a useful real comparison point — but explain why jitter's real primary benefit should be evaluated through real synchronization/load-spreading behavior across many concurrent clients, not through per-task latency alone, which the naive baseline will often show as "faster."
42. A real, live experiment against a genuinely flaky mock service found jittered and naive retry produced statistically similar real success rates and real request-amplification, while jitter's real cost was purely added per-task latency; a separate real concurrent-burst measurement found jitter consistently increased real retry-timestamp spread, with an honestly-reported real confound that thread-scheduling noise itself contributes non-trivial spread even to the naive case. What does this real pair of findings correctly attribute jitter's real benefit to, and what does it correctly NOT attribute jitter's benefit to?

**Module 08 — Security, Privacy & Compliance Architecture for GenAI Systems (6)**
43. Why is passing API authentication insufficient to answer the real question "what may this caller's request access"?
44. Walk through why retrieved and tool-returned content is treated as untrusted the moment it enters the model's context. Precisely: this content has different real provenance and trust characteristics from raw user input, but the real, shared security principle is that neither should be automatically treated as trusted instructions — state what genuinely differs between the two sources and what real architectural conclusion nonetheless applies to both.
45. Given a multi-tenant RAG system, explain why the real primary access boundary must be enforced at retrieval time via metadata filtering, not as a post-hoc check on the generated output — and why real defense-in-depth (output-side checks, audit logging, leakage detection) still has a genuine, additional real role even once retrieval-time filtering is in place, rather than being made redundant by it.
46. Why is least-privilege tool/data authorization described as the real backstop against indirect injection, distinct from sanitization? Precisely explain the mechanism: authorization doesn't need to detect or block a malicious payload at all — it real, structurally limits the maximum damage any resulting action could cause, which is why it remains protective even in the real case where detection/sanitization fails to catch the payload.
47. Precisely distinguish this module's real system-level data-governance scope from `07_llm_evaluation_observability_and_guardrails`'s own content-level PII/toxicity detection scope.
48. A real authorization-engine stress test across 90 real (tenant, tool) combinations found 0 unauthorized-access breaches, and a real indirect-injection walkthrough showed a request being correctly denied by the real permitted-tool set even without assuming the sanitization layer caught the payload. Why does bounding real damage via authorization scope matter even when the upstream defense layer is imperfect?

**Module 09 — End-to-End GenAI System Design Case Studies (6)**
49. Why is spending equal, shallow time on all 8 prior modules described as a weaker real interview answer than a correctly prioritized one?
50. Walk through how a candidate should decide which 1-2 components deserve the real deep-dive for a system they haven't seen before.
51. Given a multi-tenant enterprise RAG assistant's stated requirements, explain why data infrastructure/authorization, not raw latency optimization, is the correct real deep-dive priority.
52. Given an agentic coding copilot's stated requirements, walk through how a candidate should determine and justify the correct real deep-dive priority — the module's own worked example prioritizes reliability engineering given ITS specific stated requirements (a multi-step tool-chaining loop with continuous developer reliance), but explain why a different real requirement set for the same system type (e.g., execution against proprietary repositories, or autonomous code execution) could just as legitimately make security the dominant real concern instead — the correct priority follows from the specific stated requirements, not from the system type alone.
53. What should a candidate do if an interviewer explicitly asks to go deeper on a component the candidate didn't prioritize?
54. *(synthesis)* A real, consolidated capstone chained real functions from Modules 01-08 into one working pipeline on a fresh scenario, then re-ran the identical scenario with one real injected failure (a canary quality regression) and verified the composed pipeline correctly halted before reaching the remaining stages, rather than silently propagating a bad state through to deployment. Design a full GenAI system end-to-end — framework, architecture, capacity/cost, data infrastructure, LLMOps, deployment, reliability, and security — and explain why demonstrating a correctly-handled failure path is as interview-relevant as demonstrating a correct happy path.

---

## 4. Batch Plan & Structural Compliance

Written in 5 batches of ~11 questions each (matching prior topics' 10-15-per-batch precedent), each question following the standardized format (`## Question N: Title` → `[ESSENTIAL]` → `[DEEP DIVE]`). Mandatory Section 5 structural compliance check before declaring the track done: all required per-question headings present exactly 54 times each; no derivation chains in any `[DEEP DIVE]` block; Final Revision Sheet present with exactly 3 required subsections (54-row Quick-Recall table, Essential Formula Cheat Sheet, Top Follow-up Q&As); no placeholder markers remain.

## 5. Cross-Module Boundary Discipline (Carried from README)

Per this topic's own README, this Q&A track does not re-derive: `03_advanced_rag`'s retrieval-pipeline construction; `04_ai_agents_and_protocols`'s agent architecture/orchestration; `05_prompt_engineering_and_structured_generation`'s prompt-testing methodology and injection-defense techniques; `06_llm_inference_and_optimization`'s inference-serving mechanics and cost model; or `07_llm_evaluation_observability_and_guardrails`'s evaluation methodology, tracing, and content-safety classification. Question 47 above states the Module 08 boundary explicitly, mirroring the module's own stated scope.

## 6. Open Design Questions / Dependencies

1. All 54 questions are derivable directly from already-complete, already-verified Track 1/Track 2 content — no new research or computation required before writing begins.
2. No open questions block starting Batch 1.

## Status: Complete

All 54 questions written to `modules/10_genai_system_design_and_llmops_interview_questions.md` across 5 batches, incorporating every revised wording from the 9-point feedback round exactly. Mandatory structural compliance check passed on all points: (1) all 10 required per-question headings occur exactly 54 times each; (2) no derivation chains found in any `[DEEP DIVE]` block (0 display-math blocks, no derivation-language matches); (3) the Final Revision Sheet is present with all 3 required subsections (54-row Quick-Recall table, 6-formula Essential Formula Cheat Sheet, 10-entry Top Follow-up Q&As); (4) no placeholder markers remain. One stray heading typo (a literal "#500" instead of "#### Common Follow-up Questions" in Q36) was caught during writing and fixed immediately, before the compliance check confirmed the corrected count. `helpers/compile_system_design.py` now compiles a second, standalone `genai_system_design_and_llmops_interview_cheatsheet.html`/`.pdf` (1,035,095 bytes) in addition to the master study guide — verified 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 1 `module-container` div (single source file), 54 `follow-up-section` divs, 108 `q-card` divs (2 per question × 54), 54/54 Question headings, plus a visual spot-check of the rendered cover page and Question 1 confirming clean rendering.

Revised per a 9-point feedback round: Q18 reframed to avoid implying the real result contradicts "high utilization means large delay," asking instead what other real factors (server count, variability, scheduling) shape queuing delay at a given utilization. Q20 made explicit about the real formula's order (replication before index overhead, not the reverse). Q33 reframed $N_{\text{min}}$/$T_{\text{min}}$ as real, context-dependent minimum decision-quality requirements, not universal constants. Q37 reframed to distinguish exponential growth's real role (sustained retry-pressure reduction) from jitter's real role (synchronization prevention) rather than declaring one more critical. Q41 softened to keep the naive-retry comparison useful while redirecting jitter's real evaluation toward synchronization/load-spreading, not latency alone. Q44 softened "architecturally no different from raw input" to acknowledge real provenance differences while keeping the shared untrusted-by-default principle. Q45 clarified retrieval-time authorization as the real primary boundary without implying output-side defense-in-depth becomes redundant. Q46 clarified authorization's real mechanism (bounding damage, not detecting payloads) as the reason it remains protective under imperfect sanitization. Q52 reframed to avoid a predetermined reliability-over-security answer, asking the candidate to justify priority from the system's specific stated requirements instead. Q54 confirmed unchanged per explicit positive feedback.
