# Implementation Plan: Track 1 — Study Guide (`07_llm_evaluation_observability_and_guardrails`)

Per `study_guide_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 1 (the 9 theory modules); Track 2 (notebooks) and Track 3 (interview Q&A) are separate, later sign-off gates.

---

## 1. Modules & Target File Paths

| # | Module | File Path |
|---|---|---|
| 01 | Why LLM Evaluation Is Fundamentally Hard | `modules/01_why_llm_evaluation_is_fundamentally_hard.md` |
| 02 | Automated Reference-Based & Reference-Free Metrics | `modules/02_automated_reference_based_and_reference_free_metrics.md` |
| 03 | LLM-as-a-Judge — Design, Biases & Calibration | `modules/03_llm_as_a_judge_design_biases_and_calibration.md` |
| 04 | Human Evaluation & Preference Data Collection | `modules/04_human_evaluation_and_preference_data_collection.md` |
| 05 | RAG & Agent-Specific Evaluation | `modules/05_rag_and_agent_specific_evaluation.md` |
| 06 | Hallucination Detection & Factuality Evaluation | `modules/06_hallucination_detection_and_factuality_evaluation.md` |
| 07 | Observability — Tracing, Logging & Structured Monitoring | `modules/07_observability_tracing_logging_and_monitoring.md` |
| 08 | Guardrails & Content Safety Classification | `modules/08_guardrails_and_content_safety_classification.md` |
| 09 | Production Evaluation Pipelines — Continuous Evaluation, Regression Testing & Drift Detection | `modules/09_production_evaluation_pipelines_and_drift_detection.md` |

---

## 2. Per-Module Formulas & Hand Calculations

**Module 01 — Why LLM Evaluation Is Fundamentally Hard**
- No closed-form formula — the module is epistemic/conceptual.
- **Worked example (no formula):** A small set of real, valid paraphrase answers to the same question, scored under exact-match — showing exact-match scoring several genuinely correct paraphrases as "wrong" purely due to surface-form mismatch, computed and verified directly (real match/no-match count), concretely motivating why reference-based, surface-level metrics (Module 02) are an incomplete answer on their own.

**Module 02 — Automated Reference-Based & Reference-Free Metrics**
- **Formula (core):** Simplified n-gram precision (BLEU-style) and n-gram overlap (ROUGE-style) formulas, and perplexity, $\text{PPL} = \exp\left(-\frac{1}{N}\sum \log p(x_i)\right)$ — presented at an intuitive, non-derivation level per the Concept Simplification rule. The term "reference-free" is used carefully and scoped per-metric, not as a single undifferentiated category: perplexity and self-consistency are each tied explicitly to the one specific real property each measures (real fluency-under-the-model's-own-distribution for perplexity; real answer stability across samples for self-consistency) — the module states plainly that neither is a general-purpose quality metric, and the two are never presented as interchangeable or substitutable for each other.
- **Hand calc:** Compute real BLEU-1/ROUGE-1 precision on a tiny sentence pair, and compute perplexity on a tiny toy probability sequence; then a second, real computed example showing a fluent-but-semantically-wrong sentence scoring comparably to a correct one under n-gram overlap — a concrete, computed counterexample to "high n-gram overlap implies correctness."

**Module 03 — LLM-as-a-Judge: Design, Biases & Calibration**
- No closed-form formula for bias itself — architectural/procedural, matching prior topics' treatment of comparative/survey modules.
- **Hand calc (worked example):** A small, real set of mock judge scores vs. mock human scores, computing a real **correlation (e.g., Spearman rank correlation)** between judge scores and human scores as the primary calibration metric — a simple raw agreement percentage is explicitly avoided as the main example, since it can be misleading for continuous/ordinal judge scores where what actually matters is whether judge *rankings* track real human judgments, not whether individual scores match exactly. A second real computed example demonstrating position bias directly — the same pairwise comparison scored twice with response order swapped, showing a real, computed score-flip rate — and a real computed illustration of instability (score variance) across two slightly reworded rubric variants for the same underlying judgment task.

**Module 04 — Human Evaluation & Preference Data Collection**
- **Formula (core):** Cohen's kappa for inter-annotator agreement, $\kappa = \dfrac{p_o - p_e}{1 - p_e}$, where $p_o$ is real observed agreement and $p_e$ is real expected agreement by chance. **Scope stated explicitly:** kappa is appropriate specifically for categorical annotations (e.g., pass/fail, a discrete label set) — the module explicitly notes it is not the right tool for ordinal or continuous human judgments (e.g., a 1-10 quality scale), where a rank-correlation or weighted-agreement measure is the more appropriate real choice instead, rather than implying kappa is a universally applicable agreement statistic.
- **Hand calc:** A small, real 2-annotator confusion table (e.g., labeling a set of responses pass/fail — a categorical task, matching kappa's real stated scope), computing real $p_o$, $p_e$, and $\kappa$ directly — showing how a deceptively high raw agreement percentage can still correspond to a real, low $\kappa$ once chance agreement is accounted for.

**Module 05 — RAG & Agent-Specific Evaluation**
- **Formula (core):** Faithfulness/groundedness as a real fraction of answer claims supported by retrieved context, $\text{Faithfulness} = \dfrac{\text{Supported claims}}{\text{Total claims}}$. **Context precision and recall defined explicitly and precisely before computing them**, since the RAG-evaluation ecosystem has multiple real, differing definitions in practice: this module fixes one concrete real definition — context precision $= \dfrac{\text{Retrieved chunks judged relevant}}{\text{Total retrieved chunks}}$ (precision's denominator is *retrieved* chunks), context recall $= \dfrac{\text{Retrieved chunks judged relevant}}{\text{Total relevant chunks in the corpus for that query}}$ (recall's denominator is *all real relevant* chunks, retrieved or not) — with "relevant" explicitly meaning a chunk judged (by a stated real rubric, e.g., human or LLM-judge label) as containing information necessary to support the real correct answer, stated plainly as one specific, chosen convention rather than presented as the only possible definition. Agent efficiency metrics — tool-call count, token usage, wall-clock latency, cost per completed task — presented as a real, separate axis tracked *alongside* (not instead of) task-success rate.
- **Hand calc:** A small, real worked example computing faithfulness and context precision/recall (under the module's own explicitly-stated definitions above) on a toy claim/context set; a second real computed example comparing two hypothetical agent runs that both reach the same successful task outcome but differ in real tool-call count/tokens/cost — computing the real efficiency delta directly, making concrete why success-rate-only evaluation is incomplete.

**Module 06 — Hallucination Detection & Factuality Evaluation**
- **Formula (core):** A **constructed self-consistency agreement measure used for this module's own illustrative purposes**, $\text{Agreement} = \dfrac{\text{Majority-answer count}}{k}$ across $k$ real samples — explicitly labeled as a real, module-defined signal for building intuition, not a standardized or universally-recognized factuality metric in the literature, and explicitly framed as a real signal, never a factuality detector.
- **Hand calc:** A real, constructed worked example where $k$ real samples agree with high consistency on an answer that is stipulated to be factually wrong (a real, direct numeric illustration that consistency alone cannot exceed some agreement rate while still being wrong), contrasted against a second real computed example of a retrieval-grounded/NLI-style entailment check correctly flagging that same wrong answer as unsupported by real source content — making the signal-vs-detector distinction concrete with real, contrasting computed outcomes.

**Module 07 — Observability: Tracing, Logging & Structured Monitoring**
- No formula — architectural, consistent with how prior topics treated observability/monitoring modules (e.g., `06_llm_inference_and_optimization` Module 09's own metric-set treatment, referenced but not re-derived here).
- **Worked example (no formula):** A real, constructed multi-step pipeline trace (retrieval span → model-call span → tool-call span) for one request, annotated with where a real, specific failure (e.g., a hallucinated tool argument) becomes visible only via the trace's per-span detail, not via an aggregate quality metric alone.

**Module 08 — Guardrails & Content Safety Classification**
- **Formula (core):** Precision/recall/F1 for a safety classifier from a real confusion matrix, $\text{Precision} = \dfrac{TP}{TP+FP}$, $\text{Recall} = \dfrac{TP}{TP+FN}$; real added latency from a guardrail stack, $\text{Latency}_{\text{added}} = \sum \text{Latency}_{\text{check}_i}$ (sequential) vs. $\max_i(\text{Latency}_{\text{check}_i})$ (parallel). **The parallel formulation's assumption stated explicitly**: $\max_i(\text{Latency}_{\text{check}_i})$ holds only when the checks genuinely execute independently and concurrently (no real data dependency between them) *and* downstream generation/response waits for every required check to complete before proceeding — real orchestration overhead, a genuine dependency between checks (e.g., one check's real output gating another), or a partial-wait policy would all change the real result, and the module states this as a simplification, not a universal guarantee.
- **Hand calc:** A small, real confusion matrix for a toy safety classifier, computing real precision/recall/F1 and the real false-positive/false-negative trade-off at two different decision thresholds; a second real computed example comparing sequential vs. parallel guardrail-check latency for a stack of 3 illustrative checks, showing the real, direct latency cost difference between the two architectures.

**Module 09 — Production Evaluation Pipelines: Continuous Evaluation, Regression Testing & Drift Detection**
- No closed-form formula for the pipeline itself — architectural/procedural.
- **Worked example (no formula):** A real, constructed scenario where an evaluation-set or evaluator-version change alone (not a real underlying system change) produces an apparent metric shift — computed directly from two real, different toy evaluation-set compositions scored against the identical model output, showing a real, misleading delta purely from unversioned inputs. **This is explicitly labeled and discussed as *evaluation-pipeline drift/versioning failure* — a distinct real failure mode from genuine system-quality drift**, not folded into the same category. A second, separate real worked example distinguishes the three genuine system-quality drift types — input/data drift, output/quality drift, and model/behavior drift — using three small, distinct toy distributional-shift scenarios, each requiring a genuinely different real remediation, with the module explicit that this second example's drift types are conceptually different from the first example's evaluation-pipeline-versioning failure mode.

---

## 3. Diagrams & Plots

**Inline responsive SVG (architectural, no formula-plot):**
- Module 03: LLM-as-judge evaluation pipeline — input/rubric → judge call → bias-mitigation steps (randomized ordering, calibration check) → score, annotated with where each specific bias-mitigation technique intervenes.
- Module 05: Agent trajectory diagram — a multi-step real task execution annotated with per-step correctness *and* efficiency metrics (tool calls, tokens) side by side.
- Module 06: Hallucination-detection decision flow — self-consistency check (signal) feeding into, not replacing, a retrieval/NLI-grounded verification step (detector), visualizing the signal-vs-detector distinction directly.
- Module 07: Multi-step pipeline tracing waterfall — spans for retrieval/model-call/tool-call on one request timeline, with a specific real failure's root cause visibly localized to one span.
- Module 08: Detection → decision → enforcement guardrail pipeline — three real, distinct stages visualized end-to-end, including the allow/block/rewrite/fallback branch point.
- Module 09: Continuous-evaluation CI/CD pipeline — versioned dataset/model/config/evaluator inputs → automated eval run → gate/canary/rollback decision, visualizing where versioning mismatches (Module 09's own worked example) would enter the pipeline.

**Matplotlib plots (saved to `plots/`, generated via `helpers/generate_evaluation_plots.py`, all explicitly labeled illustrative unless computed directly from a module's own hand-calc formula):**
1. Module 02: N-gram-overlap score vs. the module's own **constructed/annotated correctness label** (a manually stipulated label for this small worked sentence set, not an objectively measured semantic ground truth) — real, computed from the module's own hand-calc data, visualizing the fluent-but-wrong counterexample directly.
2. Module 04: Cohen's kappa vs. raw observed agreement across a swept range of chance-agreement rates (real, computed directly from the kappa formula) — visualizing why raw agreement alone overstates real reliability.
3. Module 06: Self-consistency agreement rate vs. the module's own **constructed/annotated correctness label** (manually stipulated per worked scenario, not an objectively measured ground truth) across the module's constructed worked scenarios — real, computed from the module's own hand-calc data, visualizing the signal-vs-detector gap.
4. Module 08: Real computed guardrail-stack latency (sequential vs. parallel) vs. number of checks (real, computed directly from the module's own latency formula).

This totals 4 plots and 6 diagrams, matching prior topics' established precedent for a 9-module topic.

---

## 4. Open Design Questions / Dependencies

1. All core formulas (n-gram overlap/perplexity, Cohen's kappa, faithfulness/context precision-recall, self-consistency agreement rate, classifier precision/recall/F1, guardrail latency) are standard, closed-form, and computable directly from small, real, constructed worked examples — no external dataset or model dependency required for Track 1.
2. Track 2's notebooks (not this track) will likely need either a real small local/API model for real LLM-as-judge and hallucination-detection experiments, or a real small human-labeled toy dataset for real agreement-rate/classifier experiments — feasibility to be confirmed at Track 2 planning time, plausibly reusing the same RTX 4060 GPU and/or OpenAI API access already used in prior topics' own notebooks.
3. No open questions block starting Track 1 module writing.

## Status: Complete

All 9 modules written and verified (hand calcs/reference code executed with real, passing assertions; all 6 inline SVG diagrams rendered clean via headless Edge screenshot; all 4 plots computed and generated). Master study guide compiled to HTML/PDF via `helpers/compile_evaluation.py`, verified with 0 leaks and correct structural div counts. Two real bugs caught and fixed during execution, documented explicitly in-module rather than silently corrected: Module 07's initial draft omitted the required "Implementation & Reference Code" section (caught on self-review, fixed with a real root-cause-localization function); Module 09's kappa/accuracy-delta assertion hit a real floating-point precision issue (0.8-0.6 ≠ exactly 0.2 in Python), fixed with an epsilon-tolerance assertion. See `token_usage_log.md` entries 1-15 for the full per-step build log.
