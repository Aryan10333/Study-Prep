# Implementation Plan: Track 3 — Interview Q&A (`07_llm_evaluation_observability_and_guardrails`)

Per `interview_qa_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 3 (the standalone Interview Q&A cheatsheet); Tracks 1 (9 study-guide modules, 6 SVG diagrams, 4 plots, compiled PDF/HTML) and 2 (6 notebooks, all real execution — OpenAI API + a real local RTX 4060 toxicity classifier) are both complete and pushed.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_llm_evaluation_observability_guardrails_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `llm_evaluation_observability_guardrails_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `llm_evaluation_observability_guardrails_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_evaluation.py` — add a second `compile_document()` call in `main()`, producing a separate standalone cheatsheet |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules and their real hand calcs (exact-match vs. paraphrase, BLEU-1/perplexity counterexample, Spearman judge-calibration + position-bias flip rate, Cohen's kappa, faithfulness/context precision-recall + agent efficiency, self-consistency agreement rate as signal-not-detector, root-cause localization, guardrail precision/recall/F1 + sequential-vs-parallel latency, eval-versioning vs. genuine drift). Following prior topics' established differentiator: where a question's real Track 2 notebook result adds genuine interview value, the **Production Perspective** or **Common Mistakes** sections cite it explicitly, framed as an observation from a specific real experiment on this specific real model/hardware (`gpt-4o-mini` via the OpenAI API; `unitary/toxic-bert` on the RTX 4060), not a universal law.

This topic's Track 2 produced a genuinely useful mix of confirming and disconfirming real findings worth citing this way: real judge calibration (ρ≈0.8531) alongside a real 33.3% position-bias flip rate concentrated specifically on subtler-quality-gap pairs (Notebook 02); a real perfect κ=1.0000 LLM-rater agreement including on deliberately-ambiguous items, explicitly never called "inter-annotator agreement" (Notebook 03); a real honest retrieval false-positive (precision=0.667) alongside a real faithfulness checker correctly catching one genuine unsupported claim (Notebook 04); a real 0/5 null result for the "wrong but self-consistent" hallucination pattern, including on a deliberately-chosen trick question that the model answered correctly and consistently (Notebook 05); and Notebook 06's own pair of honest surprises — a real threshold sweep with perfect separation (too easy a test set to expose a genuine precision/recall trade-off) and a real **negative** parallel-guardrail-latency result (-21.8%, real `ThreadPoolExecutor` orchestration overhead exceeding a fast check's own cost) alongside a real capstone that did not reproduce Module 09's constructed eval-versioning drop at small scale. Several questions below cite these real results specifically because they complicate, not just confirm, this topic's own module-level formulas — genuine interview material about where clean theory meets messier real behavior.

## 3. Proposed Question List (54 questions, grouped by module)

**Module 01 — Why LLM Evaluation Is Fundamentally Hard (6)**
1. Why is there no single "the accuracy" metric for LLM evaluation the way there is for classification, and what does that structural difference imply for how eval suites must be designed?
2. Walk through the exact-match vs. paraphrase problem — why can a completely correct answer score 0 under exact-match scoring?
3. Given a real set of 5 genuinely correct paraphrases scored under strict exact-match, compute the real exact-match rate and explain what it does and doesn't tell you.
4. Why is "just use a bigger/better reference set" not a full fix for the exact-match problem?
5. Precisely distinguish reference-based from reference-free evaluation — what does each assume, and when does that assumption break?
6. A real notebook found `gpt-4o-mini` scored 16/16 = 100% correct under a real, defensible correctness protocol, yet its BLEU-1 dropped from 0.920 (direct phrasing) to 0.639 (varied phrasing) on the SAME correctness. What does this real gap reveal about what n-gram-overlap metrics actually measure?

**Module 02 — Automated Reference-Based & Reference-Free Metrics (6)**
7. Walk through why real n-gram overlap can score a fluent-but-wrong answer HIGHER than a correct-but-differently-worded one, using this topic's own hand-verified counterexample.
8. Given the module's own worked BLEU-1 numbers (wrong_fluent ≈ 0.8571 vs. correct_rephrased ≈ 0.7143), explain precisely why this is not a metric bug but an expected consequence of what n-gram overlap measures.
9. What does perplexity actually measure, and why is "lower perplexity = more correct" a common but false inference?
10. Precisely distinguish perplexity and self-consistency as evaluation signals — why are they not interchangeable, despite both being "reference-free"?
11. Why can a real evaluation pipeline that only reports aggregate BLEU/ROUGE hide the exact failure mode this topic's own worked example demonstrates?
12. A real notebook found direct-phrasing answers scoring meaningfully higher BLEU-1 than varied-phrasing answers despite 100% real correctness in both groups. Design a real evaluation protocol that would catch this gap instead of masking it behind an aggregate score.

**Module 03 — LLM-as-a-Judge — Design, Biases & Calibration (6)**
13. Why is LLM-as-a-judge attractive compared to purely automated metrics, and what real failure mode does it introduce in exchange?
14. Walk through position bias — precisely what does swapping response order reveal about a judge, and why doesn't a single comparison expose it?
15. Given the module's own worked position-bias experiment (7/10 = 70% flip rate), explain why this number alone doesn't tell you the judge is "bad," only that its raw preference isn't order-invariant.
16. Why is Spearman rank correlation a more defensible way to check judge calibration than raw agreement rate for continuous judge scores?
17. What does "rubric-wording instability" mean, and why can it undermine judge reliability even when position bias is controlled for?
18. A real notebook built a genuinely judge-independent ground truth (a manually-constructed fact-count, verified via assertion, never another LLM's ranking) and found real ρ ≈ 0.8531 rank-correlation calibration alongside a real 33.3% position-bias flip rate concentrated specifically on the subtler-quality-gap pairs, never the obvious ones. Why does strong rank-order calibration NOT imply the judge is order-invariant — precisely what different dimension of judge quality does each of these two real numbers measure — and why does the flip rate's concentration pattern matter more for production judge design than its raw value alone?

**Module 04 — Human Evaluation & Preference Data Collection (6)**
19. Why is human evaluation still considered necessary even once automated metrics and LLM-judges are both available?
20. Walk through Cohen's kappa — why does it correct raw percent agreement for chance, and why does that correction matter?
21. Given the module's own worked example (p_o = 0.75, p_e = 0.60, κ = 0.375), explain what this real κ value implies about agreement once chance is corrected for — and why attaching a fixed qualitative label (e.g., "fair") to a raw κ number is itself contested, since interpretation bands depend on the chosen convention and context?
22. Precisely state Cohen's kappa's real scope — why is it appropriate for categorical ratings but not ordinal/continuous ones, and what should be used instead for the latter?
23. Why can real preference-data collection itself introduce systematic bias (e.g., annotator fatigue, rubric ambiguity) independent of the raters' honesty or competence?
24. A real notebook computed Cohen's kappa between two independently-prompted LLM raters (not human raters) and found perfect real agreement (κ = 1.0000), including on deliberately-ambiguous items. Why must this result be reported as "LLM-rater agreement," never "inter-annotator agreement," and what does it actually measure instead?

**Module 05 — RAG & Agent-Specific Evaluation (6)**
25. Why do RAG and agent systems need evaluation dimensions beyond plain output correctness?
26. Walk through faithfulness — precisely what does a faithfulness score measure, and how does it differ from correctness?
27. Given the module's own explicit definitions (precision denominator = retrieved, recall denominator = all relevant), compute context precision and recall from a real worked retrieval example, and explain why both denominators must be stated explicitly to avoid ambiguity.
28. Why can two agent runs have identical task success yet meaningfully different real efficiency (tool calls, tokens, latency, cost), and why does success-only evaluation hide that difference?
29. Precisely distinguish faithfulness from context precision/recall — can an answer be faithful to a bad retrieval, or unfaithful to a good one?
30. A real notebook's retrieval step returned precision = 0.667/recall = 1.000 on both real queries — a real, honest false-positive chunk retrieved both times — while its faithfulness checker caught one genuine unsupported embellishment claim (1.0 vs. 0.8). Why is it important that these two real numbers (retrieval quality and faithfulness) are reported and interpreted separately rather than blended into one RAG "quality score"?

**Module 06 — Hallucination Detection & Factuality Evaluation (6)**
31. Why is "the model was consistent across samples" not sufficient evidence that it was correct?
32. Walk through the module's own constructed Scenario A vs. B (agreement 0.80-WRONG vs. 0.90-CORRECT) — what does this pairing prove about self-consistency as a hallucination signal?
33. Given a real set of k sampled answers, compute the self-consistency agreement rate, and explain precisely why this notebook's own formula is described as a constructed illustrative measure, not a standardized metric.
34. Why must grounded verification use a source independent of the generation model, and what specific real failure mode does using the same model for both generation and verification risk?
35. Why must the "wrong but self-consistent" criterion be defined before running an experiment, not after seeing results — what specific risk does pre-registration guard against here?
36. A real notebook ran k=5-sample self-consistency plus real, independent Wikipedia-grounded verification on 5 real questions, including a deliberately-chosen trick question, and found 0/5 trials met the pre-stated "wrong but self-consistent" criterion, with every real grounded verdict coming back `entailed`. Does this real null result undermine Module 06's own constructed counterexample, or prove that wrong-but-self-consistent hallucinations are rare in general? Why does a 0/5 result at this small real sample size support neither claim — what would be needed to draw a stronger conclusion either way?

**Module 07 — Observability — Tracing, Logging & Structured Monitoring (6)**
37. Why can aggregate metrics (latency, quality score) alone fail to localize a specific real production failure to a specific pipeline step?
38. Walk through the module's own worked trace (4 spans, 1570ms total) — why does the `tool_call` span, not the final `model_call` span, get correctly identified as the root cause?
39. Given a real multi-span trace with per-span status, apply the module's own root-cause-localization logic and explain why "first non-ok span" is a defensible real heuristic for this kind of failure — and under what real distributed-system conditions (e.g., a failure induced by an upstream service or dependency the trace doesn't capture) this heuristic can misattribute the true root cause.
40. Precisely distinguish this topic's pipeline-level tracing/observability scope from `06_llm_inference_and_optimization`'s own infrastructure-level serving metrics (TTFT, TPOT, GPU utilization) — why are they genuinely different layers?
41. Why might a real production system retain full per-span trace detail only for a sampled subset of traffic, plus any trace tied to negative feedback or a guardrail flag, rather than for every request?
42. A real notebook built a genuine 3-step traced pipeline (retrieve → generate → guardrail_check) and reproduced three distinct real root causes across three real requests — none, an upstream retrieval error, and a downstream guardrail flag — using Module 07's own localization function unmodified. Why does correctly attributing the second case's root cause to `retrieve`, not `generate`, matter even though the generate step behaved correctly given its bad input?

**Module 08 — Guardrails & Content Safety Classification (6)**
43. Why do guardrails need a detection→decision→enforcement architecture rather than a single classifier call?
44. Walk through the module's own worked precision/recall/F1 example at two thresholds — first state explicitly whether raising the toxicity-score threshold makes the classifier more or less likely to flag content, then derive from that which threshold trades precision for recall and which real production framing decides which is "better." Why is this direction not safe to assume without stating it explicitly?
45. Given a real swept range of decision thresholds and a stated business cost framing (e.g., a minimum-recall requirement), explain how to select a threshold using a principled method rather than an arbitrary pair of points.
46. Walk through the module's own sequential-vs-parallel guardrail latency formula — what specific real assumption (stated explicitly) does it require to be valid?
47. Why can real orchestration overhead invalidate the "parallel guardrails are always faster" intuition even when the checks themselves are genuinely independent?
48. A real notebook's own threshold sweep found perfect separation (F1 = 1.0000 at every threshold from 0.1-0.9) — an honest sign the test set was too easy to exercise a genuine trade-off — while its real measured parallel-vs-sequential latency came back NEGATIVE (-21.8% savings) due to real `ThreadPoolExecutor` overhead exceeding a fast regex check's own cost. What does each of these two real, unexpected findings teach about designing a genuinely informative guardrail evaluation, separately from what the module's own formulas predict — and why must the `-21.8%` result specifically be stated as an artifact of this notebook's own orchestration mechanism and check-cost profile, never generalized to "parallel guardrails are slower"?

**Module 09 — Production Evaluation Pipelines — Continuous Evaluation, Regression Testing & Drift Detection (6)**
49. Why can an LLM system's evaluation score change even when the underlying model's outputs have not changed at all?
50. Walk through the module's own worked eval-set-versioning example (0.8 vs. 0.6 accuracy, SAME model output) — why is this correctly diagnosed as an evaluation-pipeline failure, not a real system-quality regression?
51. Precisely distinguish input/data drift, output/quality drift, and model/behavior drift — what real signal distinguishes each from the other two, and why can observed output drift alone NOT be attributed to model/behavior drift without first ruling out upstream causes (changed inputs, prompts, retrieval content, or downstream systems)?
52. Why must evaluation-set/config/evaluator versioning be tracked as a prerequisite for trustworthy continuous evaluation, not an afterthought?
53. Design a real continuous-evaluation pipeline (regression testing + drift detection) for a production LLM feature — what would trigger an alert, and how would you first rule out an artifactual (versioning) explanation before treating it as real drift?
54. *(synthesis)* A real notebook capstone kept per-request tracing (Module 07) and aggregate evaluation-set-versioning comparison (Module 09) as two deliberately separate real experiments rather than blending them, and its own real versioning-comparison attempt did not reproduce an artifactual drop at small scale (3/3 = 1.0000 under both reference sets) — an honest negative result, with the module's own hand-verified example remaining the load-bearing proof. Design a full production LLM evaluation/observability/guardrail stack end-to-end — pipeline tracing, guardrail architecture, and a continuous-evaluation/drift process — and explain why per-request diagnosis and aggregate system-evaluation must remain two distinct real practices rather than one blended dashboard metric.

---

## 4. Batch Plan & Structural Compliance

Written in 5 batches of ~11 questions each (matching prior topics' 10-15-per-batch precedent), each question following the standardized format (`## Question N: Title` → `[ESSENTIAL]` → `[DEEP DIVE]`). Mandatory Section 5 structural compliance check before declaring the track done: all required per-question headings present exactly 54 times each; no derivation chains in any `[DEEP DIVE]` block; Final Revision Sheet present with exactly 3 required subsections (54-row Quick-Recall table, Essential Formula Cheat Sheet, Top Follow-up Q&As); no placeholder markers remain.

## 5. Cross-Module Boundary Discipline (Carried from README)

Per this topic's own README, this Q&A track does not re-derive: `05_prompt_engineering_and_structured_generation`'s prompt-level A/B testing or injection-defense content; `03_advanced_rag`'s RAG retrieval-pipeline construction; `04_ai_agents_and_protocols`'s agent architecture; or `06_llm_inference_and_optimization`'s inference-serving infrastructure metrics. Question 40 above states this boundary explicitly for observability, mirroring the module's own stated scope.

## 6. Open Design Questions / Dependencies

1. All 54 questions are derivable directly from already-complete, already-verified Track 1/Track 2 content — no new research or computation required before writing begins.
2. No open questions block starting Batch 1.

## Status: Complete

All 54 questions written to `modules/10_llm_evaluation_observability_guardrails_interview_questions.md` across 5 batches, incorporating every revised wording from the 9-point feedback round exactly. Mandatory structural compliance check passed on all points: (1) all 10 required per-question headings occur exactly 54 times each; (2) no derivation chains found in any `[DEEP DIVE]` block (0 display-math blocks, no derivation-language matches); (3) the Final Revision Sheet is present with all 3 required subsections (54-row Quick-Recall table, 12-formula Essential Formula Cheat Sheet, 10-entry Top Follow-up Q&As); (4) no placeholder markers remain. `helpers/compile_evaluation.py` now compiles a second, standalone `llm_evaluation_observability_guardrails_interview_cheatsheet.html`/`.pdf` (1,100,928 bytes) in addition to the master study guide — verified 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 1 `module-container` div (single source file), 54 `follow-up-section` divs, 108 `q-card` divs (2 per question × 54), 54/54/54 heading counts (accounting for markdown's `toc`-extension duplicate-id suffixes), plus a visual spot-check of the rendered cover page and Question 1 confirming clean formula rendering and card layout.

Revised per a 9-point feedback round: Q18 clarified that rank-correlation calibration and position-bias order-invariance are different dimensions, not implied by each other. Q21 no longer attaches a fixed qualitative label ("fair") to κ=0.375, asking instead what the value implies given contested interpretation bands. Q36 made explicit that a 0/5 result at small sample size supports neither "undermines the counterexample" nor "hallucinations are rare" — it is inconclusive at this scale. Q39 extended to note the "first non-ok span" heuristic can misattribute root cause when an upstream dependency outside the trace is the true source. Q44 requires stating threshold-direction (more/less likely to flag) explicitly before deriving the precision/recall trade-off, rather than assuming a fixed direction. Q48 strengthened to require stating the `-21.8%` result as specific to this notebook's own orchestration mechanism, never generalized. Q51 extended to require ruling out upstream causes (inputs, prompts, retrieval, downstream systems) before attributing output drift to model/behavior drift. Q24 and Q27 kept unchanged per explicit positive feedback. Q54 confirmed to already stay scoped to evaluation/observability/guardrails, not inference-system design.
