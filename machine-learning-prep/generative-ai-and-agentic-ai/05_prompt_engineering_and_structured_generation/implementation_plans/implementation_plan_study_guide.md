# Implementation Plan: Track 1 — Study Guide (`05_prompt_engineering_and_structured_generation`)

Per `study_guide_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 1 (the 9 theory modules); Track 2 (notebooks) and Track 3 (interview Q&A) are separate, later sign-off gates.

---

## 1. Modules & Target File Paths

| # | Module | File Path |
|---|---|---|
| 01 | Prompting Fundamentals, Instruction Hierarchy & Design Patterns | `modules/01_prompting_fundamentals_and_instruction_hierarchy.md` |
| 02 | Reasoning-Elicitation Techniques | `modules/02_reasoning_elicitation_techniques.md` |
| 03 | Structured Output & Schema-Constrained Generation | `modules/03_structured_output_and_schema_constrained_generation.md` |
| 04 | Constrained Decoding & Grammar-Based Generation | `modules/04_constrained_decoding_and_grammar_based_generation.md` |
| 05 | Prompt Optimization & Automatic Prompt Engineering | `modules/05_prompt_optimization_and_automatic_prompt_engineering.md` |
| 06 | Context Assembly & Prompt-Level Retrieval Integration | `modules/06_context_assembly_and_retrieval_integration.md` |
| 07 | Prompt Evaluation, Testing & Versioning | `modules/07_prompt_evaluation_testing_and_versioning.md` |
| 08 | Prompt Injection, Jailbreaking & Defense | `modules/08_prompt_injection_jailbreaking_and_defense.md` |
| 09 | Production Prompt Engineering, Templating & Model Portability | `modules/09_production_prompt_engineering_templating_and_portability.md` |

---

## 2. Per-Module Formulas & Hand Calculations

**Module 01 — Prompting Fundamentals, Instruction Hierarchy & Design Patterns**
- **Formula (core):** Temperature-scaled softmax, $P_i = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$ — presented with an explicit pipeline framing, **prompt → logits → temperature scaling → sampling**, making clear temperature only reshapes the *sampling* distribution over whatever logits the prompt already produced; it is not the mechanism behind prompt sensitivity itself (that's the prompt changing the logits in the first place, upstream of temperature).
- **Hand calc:** A tiny 4-logit example at $T=1.0$ vs $T=0.3$ vs $T=2.0$, showing the distribution sharpen/flatten numerically — framed explicitly as "given these logits (already determined by the prompt), here's how temperature reshapes sampling," not as an explanation of why the logits themselves changed.
- No other formula blocks — instruction hierarchy and design patterns are prose/diagram content per the Concept Simplification rule.

**Module 02 — Reasoning-Elicitation Techniques**
- **Formula (core):** Self-consistency majority-vote reliability, $P(\text{majority correct}) = \sum_{i=\lceil k/2\rceil}^{k} \binom{k}{i} p^i (1-p)^{k-i}$ — explicitly stated alongside its assumption: **independent samples with an identical, constant per-sample correctness probability $p$**. The module will state directly that real LLM samples from one model are correlated (shared model biases, shared blind spots), so this formula is an idealized upper-bound intuition for *why* majority voting can help, not an exact predictor of real-world self-consistency gains.
- **Hand calc:** $p=0.6$ per-sample accuracy, $k=5$ samples — compute the real majority-vote probability and show it exceeds $p$, then show $k=3$ giving a smaller gain, illustrating diminishing returns, framed as the idealized-case intuition per the assumption above.
- **Secondary hand calc (cost, no new formula):** $k$-sample self-consistency cost/latency multiplier, reusing `04_ai_agents_and_protocols` Module 02's per-task cost model directly rather than re-deriving it.

**Module 03 — Structured Output & Schema-Constrained Generation**
- **Formula (core):** Expected retries under geometric retry, $E[\text{attempts}] = 1/p$ where $p$ = per-attempt schema-validity probability — explicitly stated alongside its assumption: **independent attempts with a constant validity probability $p$**. The module will note that real repair/retry attempts often have a *different* (frequently higher, since a repair prompt includes the specific error) probability than the original attempt, so real production retry cost can diverge from this simple geometric model — presented as a starting intuition, not an exact production predictor.
- **Hand calc:** $p=0.85$ → $E[\text{attempts}] \approx 1.18$; contrast with $p=0.5$ → $2.0$, showing why provider-enforced structured output (higher effective $p$) is worth it over prompting-only JSON.
- **Comparison table (no formula, highest-value distinction in this module):** JSON mode (valid JSON syntax, not necessarily schema-conformant) vs. structured outputs (provider/schema-constrained output) vs. function/tool calling (structured arguments intended specifically for tool invocation) — guarantees, typical use, and failure mode for each, kept sharply distinct rather than treated as three names for the same thing.

**Module 04 — Constrained Decoding & Grammar-Based Generation**
- **Formula (core):** Masked softmax, $P_i = \dfrac{\exp(z_i)\cdot \mathbb{1}[i \in V_{\text{valid}}]}{\sum_{j \in V_{\text{valid}}} \exp(z_j)}$.
- **Hand calc:** A tiny 5-token vocabulary, 2 valid tokens under the current grammar state — show logits before/after masking and the renormalized distribution.
- **Secondary discussion (cost, no formula, revised framing):** Rather than asserting a single fixed "per-token mask computation" overhead figure, the module will explain that real constrained-decoding overhead depends on several real, distinct factors — the grammar implementation's efficiency, tokenizer/vocabulary size, how the valid-token set is computed and whether it's cached across steps, and the serving engine's own integration — so the illustrative plot (Section 3, plot 3) will be explicitly captioned as a **conceptual** shape (overhead generally grows with grammar complexity/sequence length), not a claimed universal per-token cost figure.

**Module 05 — Prompt Optimization & Automatic Prompt Engineering**
- No core formula — optimization-loop cost is a direct reuse of Module 02's/`04_ai_agents_and_protocols` Module 02's per-task cost model: $N_{\text{candidates}} \times M_{\text{eval examples}} \times \text{cost}_{\text{call}}$, with a small worked example ($N=8$, $M=20$) showing real optimization-loop cost, compared against a manual-iteration cost estimate.

**Module 06 — Context Assembly & Prompt-Level Retrieval Integration**
- No new formula — reuses `04_ai_agents_and_protocols` Module 04's context-budget arithmetic, applied to prompt-segment allocation (system + few-shot + retrieved + conversation) instead of memory-summarization triggering. Hand calc: a fixed context window split across 4 segments with a worked example of what gets trimmed first when the budget is exceeded.

**Module 07 — Prompt Evaluation, Testing & Versioning**
- No formula blocks (per Non-Goals: no advanced statistical derivations; confirmed no formula added here per feedback point 10). One worked, numeric example (not a new formula) that deliberately does **not** reduce to regression rate alone: given a prompt change, report **quality/accuracy delta, structured-output validity delta, latency delta, token-cost delta, robustness (variance across paraphrased inputs), and regression rate** together on one small worked example — and explicitly conclude that a prompt with better accuracy but meaningfully worse cost/latency is not automatically the better production choice, making the multi-dimensional trade-off itself the point of the example, not just a list of numbers.

**Module 08 — Prompt Injection, Jailbreaking & Defense**
- No formula blocks — architectural/security content, consistent with how `04_ai_agents_and_protocols` Module 09 treated its own security material (confirmed no formula added here per feedback point 10).
- Scope statement, made explicit in the module text: prompt-only defenses (system-prompt hardening, input/output filtering, delimiters) are **not a security boundary** the moment a system has tool access, reads files, calls external APIs, or ingests external content — real containment for that case requires the least-privilege/sandboxing/approval-gate layers `04_ai_agents_and_protocols` Module 09 already owns, referenced explicitly rather than re-argued.

**Module 09 — Production Prompt Engineering, Templating & Model Portability**
- **Formula (core):** Prompt-caching cost model, $\text{Cost}_{\text{call}} = T_{\text{cached}} \times \text{price}_{\text{cached}} + T_{\text{uncached}} \times \text{price}_{\text{full}} + T_{\text{out}} \times \text{price}_{\text{out}}$ — presented **provider-neutral**: the formula's structure (cached tokens priced differently from uncached tokens) is general, but actual cached-token pricing ratios and cache-eligibility semantics differ by provider and change over time, so any concrete price ratio used is explicitly labeled **illustrative**, not a claimed current provider rate.
- **Hand calc:** A 1,000-token stable system prompt + 200-token variable user turn, with an illustrative (explicitly labeled) fractional cache-price discount — compute per-call cost with and without caching across a repeated-call scenario, and the break-even call count.
- **Portability scope, made explicit (beyond formatting):** instruction hierarchy support (does the target model/API even have a distinct system/developer role, per Module 01), chat template differences, special/reserved tokens, tool-call syntax, structured-output support (per Module 03's three mechanisms), context-window limits, and sampling-behavior differences (e.g., default temperature, top-p support) — covered as the concrete, multi-dimensional checklist for "will this prompt still work if we swap models," not formatting alone.

---

## 3. Diagrams & Plots

**Inline responsive SVG (architectural, no formula-plot):**
- Module 01: Instruction hierarchy diagram (system → developer → user → retrieved/tool content) with an explicit conflict-resolution callout.
- Module 03: Structured-output production pipeline (schema → generation → validation → retry/repair → fallback).
- Module 04: Grammar-constrained decoding step-by-step (FSM state + valid-token-set narrowing across 2–3 generation steps).
- Module 05: DSPy-style optimize loop (generate candidates → evaluate against eval set → select best → repeat).
- Module 06: Prompt segment stacking diagram (system/few-shot/retrieved/conversation) with a "lost in the middle" position callout.
- Module 08: Defense-in-depth layers at the prompt layer (system-prompt hardening, input/output filtering, delimiters) — explicitly captioned as risk-reducing, not complete.

**Matplotlib plots (saved to `plots/`, generated via `helpers/generate_prompt_eng_plots.py`, all explicitly labeled illustrative unless computed directly from this module's own hand-calc formula):**
1. Module 02: Self-consistency majority-vote accuracy vs. $k$ (computed directly from the module's own binomial formula across $k=1..9$ at a fixed $p$ — a real, computed curve, not illustrative).
2. Module 03: Expected retry attempts vs. per-attempt validity probability $p$ (computed directly from $E[\text{attempts}]=1/p$ — real, computed curve).
3. Module 04: Conceptual per-token latency overhead of constrained decoding vs. sequence length (labeled explicitly as a **conceptual** shape — general upward trend only, not a claimed universal per-token cost figure, since real overhead depends on grammar implementation, tokenizer, valid-token-set caching, and serving engine, per the revised Module 04 framing above; no real notebook measurement exists yet at Track 1 time either way).
4. Module 09: Prompt-caching cost savings vs. cache-hit-rate (computed directly from the module's own cost-model formula — real, computed curve).

This totals 4 plots, matching `04_ai_agents_and_protocols`'s precedent of 4 plots for a 9-module topic.

---

## 4. Open Design Questions / Dependencies

1. Module 02's self-consistency formula and Module 03's/Module 09's cost formulas are all standard, closed-form, and computable directly — no external dependency.
2. Module 03/04's Track 2 notebooks (not this track) will need a real structured-output library (e.g., `pydantic` + a provider's native structured-output API, or `outlines`) — confirmed available in the repo's existing `.venv` per prior topics' pattern; will verify at Track 2 planning time, not blocking Track 1.
3. No open questions block starting Track 1 module writing.
4. Per Pass 2 sign-off feedback: the 4-plot count stays exactly as proposed (no additions), and no formulas are added to Modules 05, 07, or 08 — the existing formula distribution (Modules 01, 02, 03, 04, 09) is confirmed appropriate as-is.

---

## Status: Complete

All 9 modules written, verified, and cross-checked against the plan:
- **01 (Prompting Fundamentals):** Temperature-softmax hand calc verified exact (0.6095/0.9593/0.4344); instruction-hierarchy SVG rendered clean.
- **02 (Reasoning-Elicitation):** Majority-vote formula, diminishing-returns gap, and cost multiplier verified. **Found and fixed a real math bug**: the initial threshold `⌈k/2⌉` wrongly counted an exact tie as "majority correct" at even k — corrected to `⌊k/2⌋+1`, added a new verified property (even k is provably worse than k−1), and restricted the plot to odd k, matching real self-consistency practice.
- **03 (Structured Output):** Retry-pipeline code and expected-attempts hand calc (1.176/2.0) verified exact; pipeline SVG rendered clean.
- **04 (Constrained Decoding):** Masked-softmax hand calc verified. **Found and fixed a real prose error**: invalid-token probability mass was wrongly summed as 41.1% (mixing up which token indices were valid/invalid); corrected to the actual 50.7%. Also caught a missing plot reference (fixed) and rendered the FSM-narrowing SVG clean.
- **05 (Prompt Optimization):** Optimization-cost hand calc (0.32/0.03/10.67x) and mock candidate-selection loop verified; loop-diagram SVG rendered clean.
- **06 (Context Assembly):** Budget-split (6,700/4,690/2,010) and whole-chunk-dropping logic verified exact; segment-stacking SVG rendered clean.
- **07 (Prompt Evaluation):** No formula (per plan). **Found and fixed a real flaky-test bug**: probabilistic accuracy construction could occasionally produce a candidate with *lower* accuracy than the baseline despite the regression scenario's intent — rewrote to deterministic explicit correct-id sets, verified stable.
- **08 (Injection & Defense):** Known-pattern filter, novel-attack limitation demo (explicitly shows the filter's real, honest gap), and hardened-assembly code verified; defense-in-depth SVG rendered clean.
- **09 (Production & Portability):** Caching cost model (49.5% savings at illustrative pricing) and versioned-template-store rollback logic verified exact.

All 4 planned plots generated via `helpers/generate_prompt_eng_plots.py`, visually verified clean (2 real/computed from module formulas, 1 real/computed, 1 explicitly-labeled conceptual/illustrative — matching the plan exactly). All 6 planned inline SVG diagrams rendered and visually verified with no overlaps. 3 real bugs found and fixed during verification (1 math bug, 1 prose arithmetic error, 1 flaky-test bug) — none shipped unverified.
