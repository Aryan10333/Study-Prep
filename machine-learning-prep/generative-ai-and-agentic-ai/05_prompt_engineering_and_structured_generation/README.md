# Prompt Engineering & Structured Generation Syllabus

## 1. Context & Alignment
* **Profile Focus:** Directly targets the "Generative AI Systems" (25%) weighting in the candidate's curriculum. Prompt engineering and structured/constrained generation are the layer every production LLM system — RAG (`03_advanced_rag`), agents (`04_ai_agents_and_protocols`), and classic LLM applications — sits on top of; this topic makes that layer explicit rather than leaving it implicit inside prior topics' code.
* **Interview Frequency:** High — "how would you design a prompt for X," "how do you get reliable JSON out of an LLM," and "how do you defend against prompt injection" are standard screens for Applied/Generative AI Engineer roles at both foundation-model companies and product companies building LLM features.
* **Core Goal:** Build a precise, engineering-grounded model of how prompts actually steer a model's output distribution, how to force that output into a reliable structured/schema-constrained form in production, how to evaluate and version prompts as real software artifacts, and how to defend the prompt layer against injection/jailbreak attacks.

## 2. Module Chapters & Conceptual Scope

- **Module 01: Prompting Fundamentals, Instruction Hierarchy & Design Patterns**
  - *Key Concepts:* Zero-shot vs. few-shot prompting, system vs. user vs. assistant roles, in-context learning as implicit gradient-free adaptation, how instruction-tuning/RLHF change a base model's prompt sensitivity, prompt sensitivity and non-determinism (temperature/top-p interaction with prompting); the **instruction hierarchy** — system → developer → user → retrieved/tool content — and how conflicting instructions across that hierarchy are (and aren't) resolved, which is the conceptual root of why prompt injection works; practical, scoped prompt design patterns (clear task specification, explicit constraints, positive instructions, few-shot examples, output contracts, delimiters, task decomposition) — kept as a practical checklist, not an exhaustive catalog.
  - *System Bottlenecks & Focus:* Why few-shot examples cost real context-window and latency budget; why prompt wording changes can silently shift output distributions in production without any code change; why a lower-priority instruction (retrieved/tool content) being treated as equally authoritative is the structural root cause Module 08 covers the exploitation of.

- **Module 02: Reasoning-Elicitation Techniques**
  - *Key Concepts:* Chain-of-Thought (zero-shot and few-shot), Self-Consistency (sampling + majority vote), Tree-of-Thought and other search-augmented prompting — covered at the level of *when and why* each earns its cost, not as a deep research-level algorithm survey.
  - *System Bottlenecks & Focus:* Token cost and latency multiplication from multi-sample/multi-path techniques, evaluated explicitly against the accuracy gain (or lack thereof) each technique buys; explicit boundary — this module covers *prompting patterns that elicit reasoning within generation*, not agentic tool-use loops (ReAct's Action/Observation mechanics and reasoning/agentic workflow depth are `04_ai_agents_and_protocols` Module 01's subject).

- **Module 03: Structured Output & Schema-Constrained Generation**
  - *Key Concepts:* Three explicitly distinguished mechanisms — **JSON mode** (valid JSON syntax, not necessarily schema-conformant), **structured outputs** (schema-constrained generation, e.g. via Pydantic/JSON Schema), and **function/tool calling** (structured arguments intended for tool invocation, building on `04_ai_agents_and_protocols` Module 02's tool-schema coverage rather than re-deriving it); the full production reliability pattern — **schema → generation → validation → retry/repair → fallback** — including partial/truncated responses, model refusals, schema-mismatch handling, and real per-provider structured-output limitations.
  - *System Bottlenecks & Focus:* Reliability of prompting-only JSON vs. provider-enforced structured output; the real validation/retry cost budget, and what a sane fallback looks like when repair attempts are exhausted.

- **Module 04: Constrained Decoding & Grammar-Based Generation**
  - *Key Concepts:* Why prompting alone can't *guarantee* valid structure; the logit/probability-level mechanism — masking invalid tokens' logits to zero probability *before* sampling, not merely "grammar-based generation" as a black box; finite-state-machine/regex-constrained decoding, context-free-grammar-constrained decoding, libraries implementing this (Outlines, Guidance, provider-native grammar support).
  - *System Bottlenecks & Focus:* The real inference-time latency/throughput cost of constrained decoding (per-token mask computation) vs. its reliability guarantee; where this genuinely earns its cost over prompting + validation-retry (Module 03).

- **Module 05: Prompt Optimization & Automatic Prompt Engineering**
  - *Key Concepts:* Manual iterative prompt refinement discipline, few-shot example selection/ordering strategies, prompt compression, meta-prompting (using an LLM to write/refine prompts); automatic prompt/program optimization (DSPy-style) treated specifically as **prompt optimization against a measurable evaluation set**, replacing manual tweaking with a repeatable process — explicitly *not* covered as another full agent/RAG orchestration framework.
  - *System Bottlenecks & Focus:* The real cost of an optimization loop (many real LLM calls to evaluate candidate prompts) vs. its production payoff; when automatic optimization is worth it over manual iteration.

- **Module 06: Context Assembly & Prompt-Level Retrieval Integration**
  - *Key Concepts:* Prompt template structure for RAG/tool contexts, context ordering and position bias ("lost in the middle") from the prompt-construction side, context budget allocation across system/few-shot/retrieved/conversation segments, prompt-level deduplication and truncation strategies.
  - *System Bottlenecks & Focus:* Explicit boundary — this module covers *how retrieved/tool content gets assembled into a prompt*, not retrieval mechanics themselves (chunking, embeddings, ANN indexing are `03_advanced_rag`'s subject) or agent memory/state (`04_ai_agents_and_protocols` Module 04's subject).

- **Module 07: Prompt Evaluation, Testing & Versioning**
  - *Key Concepts:* Prompt regression testing against a fixed eval set, LLM-as-judge for prompt-output quality, A/B testing prompt variants in production, prompt versioning and rollback as a software-engineering discipline, prompt-change observability; evaluation deliberately scoped across **multiple, distinct dimensions** — output accuracy, structured-output validity rate, latency, token usage/cost, robustness to input variation, and regression rate against the prior version — not collapsed into one "quality" number.
  - *System Bottlenecks & Focus:* Treating a prompt change with the same rigor as a code change — a prompt template is production code, not a static string to edit freely.

- **Module 08: Prompt Injection, Jailbreaking & Defense**
  - *Key Concepts:* Direct prompt injection and jailbreak technique families (role-play/persona override, encoding obfuscation, many-shot jailbreaking), defense-in-depth at the prompt layer (system-prompt hardening, input/output filtering, delimiters marking untrusted content) — presented explicitly as *risk-reducing layers*, never as a complete solution on their own.
  - *System Bottlenecks & Focus:* Explicit boundary — this module covers *direct* injection/jailbreaking of the model via the prompt itself; *indirect* prompt injection via tool outputs/retrieved content/files/APIs is `04_ai_agents_and_protocols` Module 09's subject, referenced not duplicated. Explicitly states that prompt-layer defenses alone are insufficient once a system is tool-connected — real containment for a tool-using system requires the least-privilege/sandboxing/approval-gate layers `04_ai_agents_and_protocols` Module 09 already covers, not a stronger prompt.

- **Module 09: Production Prompt Engineering, Templating & Model Portability**
  - *Key Concepts:* Prompt templating systems (Jinja/f-string discipline, versioned template stores), multi-turn prompt/conversation-state construction, model-specific prompt-formatting quirks (chat templates, role handling, system-prompt support, tool-call syntax, context-window limits, structured-output support) and what breaks silently when a prompt built for one model/version is ported to another; prefix/prompt caching — how stable system instructions and few-shot examples can be cached to reduce latency and input-token cost on repeated calls.
  - *System Bottlenecks & Focus:* Real production failure modes — a prompt that worked on one model/version silently degrading on another; treating prompt length as a first-class cost/latency variable, not an afterthought; when prompt caching genuinely pays off vs. when a low cache-hit-rate makes it not worth the added complexity.

Module 10 (Interview Q&A track — Track 3) will follow once Tracks 1 and 2 are complete and signed off, per the established 3-track pipeline.

---

### Cross-Module Boundary Discipline
This topic deliberately avoids re-deriving content already owned by prior topics:
- **ReAct's Thought/Action/Observation mechanics, tool-calling internals, and agent memory/state** → `04_ai_agents_and_protocols`.
- **Chunking, embeddings, ANN indexing, and hybrid/reranking retrieval mechanics** → `03_advanced_rag`.
- **Indirect prompt injection (via tool outputs, retrieved content, files, external APIs)** → `04_ai_agents_and_protocols` Module 09.
- **In-context learning's theoretical basis and transformer attention mechanics** → `01_llm_foundations`.

This topic owns: the prompt as the primary interface surface — its construction, optimization, structured-output enforcement, evaluation, versioning, and defense against direct manipulation.

## Status: Approved
