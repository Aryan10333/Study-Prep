# Implementation Plan: Track 3 — Interview Q&A (`05_prompt_engineering_and_structured_generation`)

Per `interview_qa_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 3 (the standalone Interview Q&A cheatsheet); Tracks 1 (9 study-guide modules) and 2 (6 companion notebooks, all built on real live APIs and a real local GPU model) are both complete and pushed.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_prompt_engineering_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `prompt_engineering_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `prompt_engineering_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_prompt_eng.py` — add a second `compile_document()` call in `main()`, at the existing `# Interview Q&A cheatsheet compilation will be added here once Track 3 is written.` placeholder comment, producing a separate standalone cheatsheet |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules and their real hand-calcs (temperature-softmax, self-consistency majority vote, expected-retries, masked-softmax, prompt-caching cost model). Following `03_advanced_rag`'s and `04_ai_agents_and_protocols`'s established differentiator: where a question's real Track 2 notebook result adds genuine interview value, the **Production Perspective** or **Common Mistakes** sections cite it explicitly, framed as an observation from a specific real experiment, not a universal law — matching this topic's own Track 1 discipline around formula assumptions (self-consistency independence, geometric-retry constant-$p$).

This topic's Track 2 produced an unusually high density of genuinely surprising, real findings worth citing this way: few-shot bought zero accuracy gain for +104.2% tokens (Notebook 01); temperature barely reshaped a heavily-peaked real logprobs distribution (Notebook 01); direct-answer scored 0/5 while CoT scored 5/5 on genuinely multi-step problems (Notebook 02); real empirical self-consistency exceeded the theoretical formula, honestly explained as small-sample variance (Notebook 02); all three structured-output mechanisms tied at real 6/6 validity, but differentiated sharply on cost (Notebook 03); a lenient JSON validator showed no gap between constrained/unconstrained decoding, while the *exact-match* rate was a real 0/15 vs. 15/15 (Notebook 04); clarifying prompt wording beat adding few-shot examples on a real recorded baseline (Notebook 05); a real trim-priority chain fully exercised on live Wikipedia content (Notebook 05); a real substring-vs-exact-match bug was caught and fixed in an injection-detection experiment, revealing the true attack-success reduction from 1.00→0.20 (Notebook 06); and real OpenAI prompt-caching hits were observed and reported strictly separately from any pricing claim (Notebook 06).

## 3. Proposed Question List (59 questions, grouped by module)

**Module 01 — Prompting Fundamentals, Instruction Hierarchy & Design Patterns (6)**
1. What is in-context learning, precisely — and why is it not the same thing as the model "learning" in the training sense?
2. Walk through the instruction hierarchy (system → user → retrieved/tool content) — why is it a trained preference rather than a structurally enforced boundary?
3. Temperature reshapes the sampling distribution; the prompt determines the logits it reshapes. Walk through why conflating these two leads to debugging the wrong stage of the pipeline.
4. Given a tiny logit set, compute how temperature reshapes the resulting probability distribution, and explain what stays invariant across temperatures.
5. What real trade-off does adding few-shot examples introduce, and how would you decide whether it's worth paying?
6. Name three practical prompt design patterns and the specific failure each one prevents.

**Module 02 — Reasoning-Elicitation Techniques (6)**
7. What's the mechanistic difference between Chain-of-Thought and direct-answer prompting, and why does CoT help specifically on multi-step problems?
8. Walk through the self-consistency majority-vote formula and its governing independence assumption — why do real LLM samples violate it?
9. Given a per-sample correctness probability and a sample count, compute the theoretical majority-vote reliability, and explain why production self-consistency implementations use odd $k$.
10. When would Tree-of-Thought's added search cost be justified over plain CoT or self-consistency?
11. What real cost does self-consistency's $k$-sample multiplier impose, and how does parallel execution change latency vs. total cost differently?
12. A real experiment found direct-answer scored 0/5 while CoT scored 5/5 on the same problem set. What does this — and doesn't this — tell you about when to reach for CoT?

**Module 03 — Structured Output & Schema-Constrained Generation (7)**
13. Precisely distinguish JSON mode, structured outputs, and function/tool calling — what does each actually guarantee?
14. Walk through the full production reliability pattern: schema → generation → validation → retry/repair → fallback. Why is validation still necessary even with provider-enforced structured output?
15. What real failure modes beyond "malformed JSON" does a production structured-output pipeline need to handle?
16. Derive the expected-attempts-under-geometric-retry formula and its governing assumption — why do real repair retries often violate it?
17. Given a per-attempt validity probability, compute the expected number of attempts, and explain why a small improvement in validity probability can produce an outsized reduction in expected retries.
18. A real experiment found all three structured-output mechanisms tied at 6/6 validity on a deliberately tricky task, but differed sharply on cost and latency. What does this tell you about how to evaluate structured-output mechanisms in practice?
19. Why might structured outputs cost more tokens than JSON mode despite offering a stronger guarantee?

**Module 04 — Constrained Decoding & Grammar-Based Generation (6)**
20. Why can't prompting alone *guarantee* valid structural output, no matter how precisely worded?
21. Walk through the masked-softmax mechanism — what does it change about the sampling distribution, and what does it leave alone?
22. Given a small vocabulary and a set of valid next tokens, compute the masked, renormalized probability distribution.
23. What's the real difference between FSM/regex-constrained decoding and CFG-constrained decoding, and when does the extra CFG cost become necessary?
24. What does constrained decoding's real per-step cost actually depend on, and why is a single universal "per-token overhead" figure misleading?
25. A real local-model experiment found a lenient JSON validator showed no gap between constrained and unconstrained generation, but the *exact-match* rate was 0/15 vs. 15/15. What does this reveal about validator design itself?

**Module 05 — Prompt Optimization & Automatic Prompt Engineering (6)**
26. What discipline turns informal prompt iteration into something closer to a measurable optimization process, even without automation?
27. Why does few-shot example *selection and ordering* matter beyond simply including examples at all?
28. What is prompt compression, and how do you verify it hasn't silently degraded output quality?
29. How does meta-prompting change *where prompt variants come from* without changing the requirement to validate them?
30. Precisely scope what DSPy-style automatic prompt optimization is and is not — why is it not a competing agent/RAG orchestration framework?
31. A real automatic-optimization experiment found that clarifying prompt wording beat adding few-shot examples, against a recorded baseline. Why does recording the baseline first matter for interpreting a result like this?

**Module 06 — Context Assembly & Prompt-Level Retrieval Integration (6)**
32. Why can a prompt-construction failure waste a retrieval system's correct results just as thoroughly as a bad retrieval would?
33. How does "lost in the middle" apply specifically to *where* you place already-selected retrieved content, as distinct from retrieval ranking itself?
34. Walk through a sane default policy for allocating a fixed context budget across system, few-shot, retrieved, and conversation segments.
35. When a retrieved-content budget is exceeded, why is dropping whole lowest-ranked chunks preferable to truncating every chunk proportionally?
36. Given a context window and segment allocations, compute the remaining budget and determine which chunks survive a real trim.
37. A real experiment fully exercised a trim-priority chain (drop few-shot first, then lowest-ranked chunks) on live Wikipedia content. Why does the *order* of that chain matter for a system-design interview answer?

**Module 07 — Prompt Evaluation, Testing & Versioning (6)**
38. Why does a prompt template deserve the same engineering discipline as a code change?
39. What are the specific pitfalls of using an LLM as a judge for prompt-output quality, beyond the general LLM-as-judge pitfalls?
40. What does A/B testing a prompt variant in production catch that offline eval-set testing structurally can't?
41. Why does prompt versioning matter even if you never need to actually roll back?
42. Walk through a worked example where a prompt shows higher aggregate accuracy but a real, nonzero regression rate. What does the regression-rate figure add that aggregate accuracy alone hides?
43. A real multi-dimensional A/B comparison found a more detailed prompt variant tied on accuracy and validity but cost 124% more tokens. How should this shape the production decision?

**Module 08 — Prompt Injection, Jailbreaking & Defense (7)**
44. Name three direct prompt-injection/jailbreak technique families, and explain what real model capability each one is exploiting.
45. Walk through the three prompt-layer defenses (system-prompt hardening, input/output filtering, delimiters) — why is each one explicitly risk-reducing, not complete?
46. Precisely distinguish direct from indirect prompt injection — why do they require genuinely different defenses despite sharing a root cause?
47. Why are prompt-layer defenses explicitly *not* a sufficient security boundary for a tool-connected system — what layer actually is?
48. What's the real, honest limitation of a pattern-based input filter against a jailbreak attempt?
49. A real experiment initially reported a 1.00 attack-success rate under mitigation, using a substring check — then a corrected exact-match check revealed the true rate was 0.20. What general lesson does this teach about designing a security-relevant detection metric?
50. Given a real attack-success-rate reduction from a mitigation test, how would you decide whether that mitigation is production-ready?

**Module 09 — Production Prompt Engineering, Templating & Model Portability (7)**
51. Why should a production prompt be a versioned template artifact rather than an inline string literal?
52. Walk through the concrete, multi-dimensional portability checklist for "will this prompt still work if we swap models" — name at least four dimensions beyond raw formatting.
53. What is prompt/prefix caching, and what real cost-structure does it exploit?
54. Derive the prompt-caching cost model and compute a hand-calc example — under what condition does caching provide the largest real benefit?
55. A real prompt-caching check observed nonzero `cached_tokens` on live API calls. Why is confirming the field is populated not the same as confirming a specific dollar discount?
56. How should multi-turn conversation state be reconstructed across calls, and what's a common, hard-to-debug failure mode when it's done inconsistently?
57. Why can a prompt validated and passing regression tests on one model still silently degrade after a model/version swap?
58. *(synthesis)* Design the full production prompt-engineering stack end-to-end for a new LLM feature — instruction hierarchy, structured output, context assembly, evaluation/versioning, injection defense, and caching/portability — and identify where you'd deliberately cut scope for an MVP vs. a mature system.
59. *(synthesis)* Across this topic's own real notebook experiments, name two cases where a technique that "sounds" like an improvement (more detail, more examples, a stronger guarantee) measurably did *not* help, or actively hurt. What single discipline explains both?

---

## Status: Complete

All 59 questions written to `modules/10_prompt_engineering_interview_questions.md` across 5 batches. Mandatory Section 5 structural compliance check passed on all four points: (1) all 10 required per-question headings occur exactly 59 times each; (2) no derivation chains found in any `[DEEP DIVE]` block; (3) the Final Revision Sheet is present with all 3 required subsections (59-row Quick-Recall table, 7-formula Essential Formula Cheat Sheet, 10-entry Top Follow-up Q&As); (4) no placeholder markers remain. `helpers/compile_prompt_eng.py` now compiles a second, standalone `prompt_engineering_interview_cheatsheet.html`/`.pdf` (1,164,528 bytes) in addition to the master study guide — verified 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 59 `follow-up-section` divs, and 118 `q-card` divs (2 follow-ups × 59 questions), plus a visual spot-check of the rendered cover page and first question.
