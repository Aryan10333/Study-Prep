# Implementation Plan: Track 2 — Companion Notebooks (`05_prompt_engineering_and_structured_generation`)

Per `notebook_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 2 (6 companion notebooks); Track 1 (9 study-guide modules) is complete and signed off; Track 3 (Interview Q&A) is a separate, later gate.

**Environment check performed before drafting this plan:** all 8 documented API keys are set in `.env` (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `OLLAMA_BASE_URL`, `SERPER_API_KEY`, `TAVILY_API_KEY`, `GITHUB_TOKEN`); `openai` (2.47.0), `pydantic` (2.13.4), `transformers` (5.14.1), `torch` (2.13.0+cu126, CUDA available — RTX 4060 Laptop GPU, 8.59GB VRAM), `tiktoken`, and `jsonschema` are installed in `.venv`. `outlines` is **not** installed — Notebook 04's constrained decoding is implemented via a hand-written `transformers` `LogitsProcessor` (the exact mechanism Module 04 teaches), not an external grammar library, avoiding an unnecessary new dependency.

---

## 1. Notebooks & Target File Paths

| # | Notebook | File Path | Modules Covered |
|---|---|---|---|
| 01 | Prompting Fundamentals & Instruction Hierarchy | `notebooks/01_prompting_fundamentals_and_instruction_hierarchy.ipynb` | Module 01 |
| 02 | Reasoning-Elicitation Techniques | `notebooks/02_reasoning_elicitation_techniques.ipynb` | Module 02 |
| 03 | Structured Output & Schema-Constrained Generation | `notebooks/03_structured_output_and_schema_constrained_generation.ipynb` | Module 03 |
| 04 | Constrained Decoding & Grammar-Based Generation | `notebooks/04_constrained_decoding_and_grammar_based_generation.ipynb` | Module 04 |
| 05 | Prompt Optimization & Context Assembly | `notebooks/05_prompt_optimization_and_context_assembly.ipynb` | Modules 05 + 06 |
| 06 | Prompt Evaluation, Injection Defense & Production | `notebooks/06_prompt_evaluation_injection_defense_and_production.ipynb` | Modules 07 + 08 + 09 |

6 notebooks for 9 modules, matching `03_advanced_rag`'s and `04_ai_agents_and_protocols`'s established precedent of combining the more architecturally-adjacent modules (05+06, and 07+08+09) into single notebooks while giving the four foundational/technically-distinct modules (01–04) their own.

---

## 2. Real-World Datasets, APIs & Data Ingestion

- **Primary LLM**: OpenAI `gpt-4o-mini` (live, `OPENAI_API_KEY`) for every notebook except 04 — matching `03_advanced_rag`/`04_ai_agents_and_protocols`'s established primary-model precedent.
- **Local model** (Notebook 04 only): a small, real instruction-tuned Hugging Face model run locally on the RTX 4060 GPU (candidate: `Qwen/Qwen2.5-0.5B-Instruct`, ~1GB, real logit access required for real constrained decoding — no API-hosted model exposes raw logits for this).
- **Real text content**: Notebook 05's context-assembly experiment uses a real, live-fetched public Wikipedia article (via `requests`, no key required) split into real chunks — not synthetic placeholder text — for genuine token-count-driven budget allocation.
- **Real eval sets**: Notebooks 02, 03, 05, 06 use small, hand-curated real task sets (arithmetic/logic word problems, real extraction targets, a real summarization/classification task) — sized 5–15 examples each, consistent with prior topics' real-but-small eval-set precedent (cost-bounded, still genuinely real, not mocked).
- **Token accounting**: real `usage` fields from OpenAI API responses (not estimates) wherever cost/token figures are reported, plus `tiktoken` for pre-call budget planning in Notebook 05.

---

## 3. Per-Notebook Real Engineering Pipelines & Metrics

**Notebook 01 — Prompting Fundamentals & Instruction Hierarchy**
1. Real zero-shot vs. few-shot comparison on a small real classification task — real measured accuracy delta, **plus real token-usage and latency cost** for the few-shot condition (from `usage` fields and `time.perf_counter()`), so the notebook demonstrates the accuracy-vs-cost trade-off, not accuracy alone.
2. Real temperature-reshaping experiment: **the same prompt, the same model**, called with `logprobs=True, top_logprobs=5` at 3 real temperatures, reporting the real top-token probability shift — explicitly framed as isolating *temperature's* effect (holding the prompt fixed) from *prompt sensitivity* (which would require varying the prompt instead) — the two are kept as clearly separate experiments, not conflated into one.
3. Real instruction-hierarchy conflict test: a system instruction plus a block **explicitly labeled and treated as untrusted retrieved content** containing an embedded override attempt, run across multiple real trials — measured real compliance/non-compliance rate. **Framing discipline**: a real trial where the model resists the override is reported as one empirical data point about this specific model/prompt/attack combination, never as proof the system is "secure" — the same non-overclaiming discipline `04_ai_agents_and_protocols` Module 09's own real injection test used.

**Notebook 02 — Reasoning-Elicitation Techniques**
1. Real direct-answer vs. CoT comparison on a real small set of multi-step word problems — reported as **real accuracy, real token usage, and real latency together**, with no assumption CoT wins; the notebook demonstrates whatever the real trade-off turns out to be, including the possibility CoT doesn't improve accuracy on this specific problem set while still costing more tokens/latency.
2. Real self-consistency experiment: sample $k=1,3,5$ real completions at $T>0$ on the same hard problem(s), compute the real empirical majority-vote accuracy, and compare it explicitly against Module 02's theoretical binomial formula at the same measured per-sample $p$ — the comparison cell states directly that the formula's independence assumption is violated by real correlated LLM samples (shared weights/biases), which is the expected, real reason for any observed gap, not treated as an anomaly.
3. Real $k$-sample latency multiplier: real wall-clock timing (`time.perf_counter()`) for $k=1$ vs. $k=5$ parallel real API calls (`ThreadPoolExecutor`), mirroring `04_ai_agents_and_protocols` Module 02's real parallel-call methodology.

**Notebook 03 — Structured Output & Schema-Constrained Generation**
1. Real, fair comparison across all three mechanisms — JSON mode, OpenAI structured outputs (Pydantic schema via `client.chat.completions.parse` or equivalent), and function/tool calling — on the **same** real, deliberately edge-case-heavy extraction task and the same real input set for all three, measuring: real schema-validity rate, real parsing/validation failure count, real retries needed, real latency, and real token usage where available per mechanism — not schema validity alone.
2. Real validation-retry pipeline: a real repair-retry loop (application-level Pydantic validation, real error fed back into a real repair call) against a real strict schema; **real recorded distribution of attempts-to-success (1 attempt / 2 attempts / 3+ attempts)**, not just the average — a distribution is more informative for real production retry-budget planning than a single mean figure.

**Notebook 04 — Constrained Decoding & Grammar-Based Generation**
1. Real local-model generation, unconstrained, against a small JSON-like grammar target — real measured schema-validity rate (expected: real, nonzero failure rate).
2. The identical real local model with a hand-written `LogitsProcessor` that implements a genuine **state machine tracking valid token transitions at each generation step** (matching Module 04's FSM section — e.g. real states for "expect open-brace" → "expect key" → "expect bool value", each computing its own real `V_valid` set from the actual grammar position) — **not** a single hard-coded mask applied uniformly at every step, since that would demonstrate masking without demonstrating grammar-constrained decoding.
3. Real measured schema-validity rate for the constrained condition, **framed precisely**: expected near/100% *structural* validity given this specific grammar implementation is verified correct — confirmed by an explicit, independent parser/validator cell that checks every constrained output against the real schema, not asserted from the masking mechanism alone. The write-up states plainly that the guarantee is conditional on correct grammar/state-transition implementation, never claimed as an unconditional 100% guarantee.
4. Real per-token latency overhead of the constrained vs. unconstrained generation, measured via `time.perf_counter()` on this specific real local setup — reported as a real number for *this* setup, explicitly not generalized as a universal figure, consistent with Module 04's framing.
5. **Hardware assertions**: `torch.cuda.is_available()` asserted at the top; `torch.cuda.max_memory_allocated()` logged after generation; explicit `del` of the model/tokenizer plus `torch.cuda.empty_cache()` in a mandatory cleanup cell at the end.

**Notebook 05 — Prompt Optimization & Context Assembly**
1. Real automatic prompt-optimization loop: an explicit **baseline prompt recorded and scored first**, plus 2–3 additional real candidate variants, **all evaluated against the identical real small eval set with the identical real scoring criteria** — so the optimization gain is quantified as a real delta against a real, recorded baseline, not just a ranking among unlabeled candidates. Real total cost computed directly from real `usage` fields (upgrading Module 05's assumed hand-calc numbers to real measured ones).
2. Real context-budget allocation: a real, live-fetched Wikipedia article chunked into real segments, real `tiktoken` counts driving the fixed system/few-shot allocations and the real whole-chunk-dropping logic when the retrieved-content budget is real-exceeded — with an **explicit, demonstrated trim-priority order** when the total budget is exceeded: preserve system instructions first, then required output/schema instructions, then essential retrieved context (highest-ranked chunks), then optional few-shot examples/conversation history last — made concrete with a real scenario that actually forces the lowest-priority segment to be trimmed, directly useful for system-design-interview framing.

**Notebook 06 — Prompt Evaluation, Injection Defense & Production**
1. Real multi-dimensional A/B comparison of two real prompt variants on a real small eval set — real accuracy (rule-checked), real structured-output validity, real latency, real token cost (from `usage`), and real per-example regression rate, replacing Module 07's simulated worked example with real measured data.
2. Real direct prompt-injection/jailbreak test: a real crafted injection sent to real `gpt-4o-mini` with and without Module 08's delimiter/reminder mitigation — **the attack text and model configuration held identical between the baseline and mitigation runs**, varying only the mitigation itself, across multiple real trials each. Reported primarily as a real measured **attack success rate** (the more precise framing for a security-relevant experiment) rather than "compliance rate," with an honest report if the mitigation's real effect is partial rather than complete (matching the established "don't overclaim" discipline from `04_ai_agents_and_protocols` Module 09's own real injection test).
3. Real prompt-caching check: a real >1,024-token stable-prefix prompt called repeatedly against `gpt-4o-mini`, inspecting the real `usage.prompt_tokens_details.cached_tokens` field OpenAI exposes. **Observed cache behavior is reported strictly separately from any pricing/discount claim**: the notebook states only what was actually observed (whether `cached_tokens` was nonzero, and for how many of the repeated calls), and does not infer or assert a specific dollar discount from that observation unless the corresponding real per-token cached price is independently confirmed, not assumed. **Explicit graceful fallback**: if the field is absent/zero for this account/tier, the notebook reports that honestly as a real negative result (real API checked, real feature not observed) rather than fabricating a hit rate — never silently substituting a mocked number.

---

---

## 3.5 Cross-Notebook Discipline (applies to all 6)

- **Real vs. illustrative, always labeled.** Every reported number states plainly whether it's a real measurement from this specific run/setup or a theoretical/illustrative figure from Track 1 — never blended without a label. Real measurements from a 5–15-example eval set are reported as exactly that (a real result from this specific, small, real set), not generalized into a universal claim about the technique.
- **`gpt-4o-mini` is a concrete implementation choice, not a conceptual conclusion.** It's used because it's this repository's established primary live model (the same choice `03_advanced_rag` and `04_ai_agents_and_protocols` made for their own notebooks), and it keeps real API cost bounded across 6 notebooks. Track 1's study guide stays provider/model-agnostic throughout (no module ties its concepts to this specific model); these notebooks demonstrate one concrete, real implementation of those concepts, and notebook write-ups are careful not to phrase a `gpt-4o-mini`-specific observation as if it were a general property of "LLMs" or "the instruction hierarchy" itself.
- **No 7th notebook.** Six notebooks remain sufficient for the nine modules; the 05+06 and 07+08+09 groupings stay as planned.

## 4. Open Design Questions / Dependencies

1. **Local model choice for Notebook 04** — `Qwen/Qwen2.5-0.5B-Instruct` is the working candidate (small, real, instruction-tuned, fits comfortably in 8.59GB VRAM); will confirm download succeeds and generates coherent-enough baseline output before finalizing at build time, with a fallback to an equally small alternative (e.g. `HuggingFaceTB/SmolLM2-360M-Instruct`) if not.
2. **OpenAI cache-hit field availability (Notebook 06, point 3)** — not confirmed until the real call is made; the plan already specifies the honest-negative-result fallback per the repo's established `[API UNAVAILABLE — FALLBACK]` discipline, so this does not block starting the build.
3. **Real eval-set sizes** — kept small (5–15 examples per experiment) deliberately to bound real API cost across 6 notebooks' worth of live calls, consistent with prior topics' real-but-small precedent; no credential or quota concern identified.
4. No open question blocks starting the build.

## Status: Complete

All 6 notebooks built, executed, and verified via the mandatory two-pass workflow (code + real execution, then Pass 2 explanations authored from real printed output with literal quotes):

- **01 (Prompting Fundamentals):** Real zero-shot/few-shot, temperature-reshaping, and instruction-hierarchy experiments (~30 real OpenAI calls). Few-shot bought 0 accuracy gain for +104.2% tokens; temperature barely reshaped a heavily-peaked logprobs distribution (a genuine API-behavior nuance); instruction hierarchy resisted 5/5 real override attempts.
- **02 (Reasoning-Elicitation):** Direct-answer scored 0/5, CoT scored 5/5 on genuinely multi-step problems; self-consistency empirical estimates (1.000) exceeded the theoretical formula (0.987/0.997), honestly explained as small-sample variance; real k=5 parallel latency was 1.46x, not 5x.
- **03 (Structured Output):** All three mechanisms (JSON mode, structured outputs, function calling) tied at 6/6 real validity — an honest negative result — but structured outputs used +52.0% more tokens than JSON mode despite the strongest guarantee, and function calling was both cheapest and fastest.
- **04 (Constrained Decoding):** Real local `Qwen/Qwen2.5-0.5B-Instruct` on the RTX 4060 GPU with a genuine hand-written state-machine `LogitsProcessor`. Lenient validation showed no gap (15/15 both), but **exact-match rate was 0/15 unconstrained vs. 15/15 constrained** — the real structural guarantee a lenient check concealed. Real GPU cleanup confirmed (966.5MB → 8.1MB).
- **05 (Prompt Optimization & Context Assembly):** Real prompt optimization against a recorded baseline (clarifying wording: +0.10 accuracy; adding few-shot: +0.00 accuracy, higher cost). Real live-fetched Wikipedia article (fixed a real 403 by adding a required `User-Agent` header) drove a real trim-priority chain: dropped few-shot, then 33/40 lowest-ranked chunks, keeping exactly the top-7 by real relevance rank.
- **06 (Evaluation, Injection Defense & Production):** Real multi-dimensional A/B (V2's extra detail cost +124.3% tokens for zero accuracy gain). Real injection-mitigation test — **caught and fixed a real substring-vs-exact-match false positive** (the same failure class `04_ai_agents_and_protocols` encountered previously), revealing the true attack-success reduction from 1.00→0.20. Real OpenAI prompt-caching hits observed on 5/5 calls, reported strictly separately from any pricing claim.

4 real bugs found and fixed across Track 2 (stable-prefix token shortfall, attack-detection false positive, Wikipedia 403, plus Track 1's flaky-test fix) — none shipped unverified. Every notebook released its real resources (API clients / GPU memory) in an explicit cleanup cell, verified runnable from a fresh kernel.
