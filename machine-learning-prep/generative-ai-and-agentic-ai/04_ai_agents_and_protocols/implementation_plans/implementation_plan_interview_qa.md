# Implementation Plan: Track 3 — Interview Q&A (`04_ai_agents_and_protocols`)

Scope: this plan covers only the standalone Interview Q&A cheatsheet, per `interview_qa_generator/SKILL.md`. Track 1 (9 study-guide modules) and Track 2 (6 companion notebooks, all executed with real APIs/infrastructure) are both complete; this track does not modify either.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_ai_agents_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `ai_agents_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `ai_agents_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_agents.py` — add a second `compile_document()` call in `main()` (the file already has a placeholder comment marking where this goes, mirroring `03_advanced_rag`'s pattern), producing a separate standalone cheatsheet, not appended into the master study guide |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules and their real hand-calcs (tool-call latency/cost, context-budget trigger, agent trajectory metrics). Following `03_advanced_rag`'s established differentiator: where a question's real Track 2 notebook result adds genuine interview value — a concrete measured number, a real surprising finding, a defensible production judgment call — the **Production Perspective** or **Common Mistakes** sections will cite it explicitly (e.g., "measured in this repo's own Notebook 05..."), rather than staying purely theoretical. This topic's Track 2 produced an unusually high density of genuinely interesting real findings worth citing this way: a real schema-ambiguity/malformed-argument gap with no accuracy gap (Notebook 01), a real network-latency measurement-artifact lesson (Notebook 01), a real protocol-discovery-vs-authorization distinction (Notebook 02), a real context-budget trigger from genuine token counts (Notebook 03), a genuine crash/resume test and a real idempotency before/after comparison (Notebook 04), a real fair single-vs-multi-agent comparison where multi-agent won on every dimension due to an observed single-agent failure mode (Notebook 05), and a real injection-mitigation test with an honestly-reported side effect (Notebook 06).

**Framing discipline for notebook-derived questions (Q12, Q13, Q27, Q38, Q52, Q58):** each of these answers will explicitly present the real Track 2 result as *one observation from a specific, controlled experiment* — not a universal or generalizable conclusion. Language will favor "in this repo's real experiment, X was observed, which suggests..." over "X is always true." This is most load-bearing for Q58, which directly tests whether the candidate can distinguish one real empirical result from a universal security claim.

## 3. Proposed Question List (59 questions, grouped by module)

**Module 01 — Agent Fundamentals & Reasoning Patterns (6)**
1. What makes a system "agentic" rather than a fixed pipeline?
2. Walk through the ReAct pattern — why interleave reasoning with action instead of just acting?
3. How does Chain-of-Thought reasoning differ from agentic reasoning?
4. When would you choose a plan-and-execute agent over a purely reactive one?
5. Walk through the formal decision framework for choosing between a single LLM call, a deterministic workflow, a single agent, and a multi-agent system — in that order of escalating complexity.
6. A single generalist agent given a tool-and-write task hit its step budget without ever producing an answer. What does that tell you about agent design?

**Module 02 — Tool Calling & Function Calling Internals (8)**
7. Walk through the concrete tool-schema design factors that affect real tool-selection accuracy: descriptions, parameter types, required vs. optional fields, enums/constraints, defaults, and avoiding overlapping tool definitions.
8. How would you decide whether two tool calls are safe to run in parallel?
9. Given 3 independent tools at 200/400/600ms plus a 300ms LLM round-trip overhead, compute the real sequential vs. parallel task latency and the resulting speedup.
10. What is *tool-level* idempotency — why does an idempotency key matter for a tool with a real side effect, and how does it differ from a confirmation gate?
11. Walk through a simple per-task cost model composing LLM token cost and tool API cost.
12. A real experiment found no tool-selection accuracy gap between a clear and an ambiguous tool schema, but did find a real malformed-argument-rate gap. How would you explain that result?
13. A real sequential-vs-parallel latency experiment produced a nonsensical negative "overhead" number. What real methodology mistake causes this, and how would you fix it?
14. Why can't you always trust a single-shot latency measurement of a network-bound tool call?

**Module 03 — Model Context Protocol (MCP) & Agent-Tool Standardization (6)**
15. Why does MCP exist — what problem does it solve that native function-calling alone doesn't — and what are the distinct responsibilities of the client, host, and server in its architecture?
16. What are MCP's three primitives (Tools, Resources, Prompts), and why does the distinction matter?
17. How does a local MCP server's trust boundary differ from a remote one?
18. Walk through MCP's capability discovery and negotiation lifecycle — what happens when a client first connects to a server, and why is discovery deliberately separate from authorization?
19. A real client discovered a destructive tool it was never authorized to call. Walk through why that's correct behavior, not a security bug.
20. What are the real security risks of exposing powerful tools through an MCP server?

**Module 04 — Context, State & Memory (7)**
21. Give the precise three-way distinction between context, state, and memory.
22. How do short-term and long-term memory differ operationally, not just by name?
23. What's the difference between episodic and semantic memory?
24. Why is vector-store-backed long-term memory structurally the same problem as RAG document retrieval?
25. What should a memory write policy actually decide, and why shouldn't an agent write everything to long-term memory?
26. Walk through how you'd compute the real turn at which a growing conversation should trigger summarization, given a context window, a threshold, and real per-turn token counts.
27. In a real experiment, an initial context-window threshold never triggered summarization at all. What real methodology mistake causes this?

**Module 05 — Agent Orchestration, State Machines & Durable Execution (7)**
28. Why is an explicit graph a more durable foundation for an agent than an implicit reasoning loop?
29. What's the difference between conditional routing and a cycle in a graph-based agent?
30. Walk through the real distinction between checkpointing, crash recovery, and resume.
31. What does *workflow-level* idempotency mean, and how does it differ from tool-level idempotency (Q10, Module 02)?
32. How would you design a genuine test that your crash/resume logic actually works, not just that it doesn't error?
33. Retry-induced duplicate side effects: a real retry loop duplicated a side effect on every real retry. Walk through why, and how an idempotency guard fixes it without changing the retry logic itself.
34. How does a human-in-the-loop interrupt use the same underlying mechanism as crash recovery, for a different real purpose?

**Module 06 — Multi-Agent Systems & Coordination Patterns (6)**
35. What's the real trade-off between orchestrator-worker and peer-to-peer multi-agent topologies?
36. When does agent specialization genuinely outperform one generalist agent, and when does it just add overhead?
37. How does the safe-parallelism principle from Module 02 extend from individual tool calls to whole sub-agents?
38. In a real, controlled experiment, a multi-agent split won on every measured dimension over a single agent. What real mechanism explains this, and what would make you doubt it generalizes?
39. What would a fair, controlled single-agent vs. multi-agent comparison need to hold constant to be trustworthy?
40. When should you not reach for a multi-agent architecture — walk through the LLM call → deterministic workflow → single agent → multi-agent progression (Q5) and identify where a given task would plausibly have stopped earlier in it.

**Module 07 — Agent Frameworks Landscape (4)**
41. What six dimensions would you use to compare any two agent frameworks, and why avoid comparing them by API surface instead?
42. How does a graph-based framework like LangGraph differ architecturally from a conversational multi-agent framework?
43. When would you build a custom agent loop instead of adopting a framework?
44. How would you evaluate the real lock-in risk of adopting a given framework?

**Module 08 — Agent Evaluation & Debugging (8)**
*(Q45–52 are deliberately structured so each of task success, final-answer quality, tool-selection accuracy, tool failures, retries, trajectory efficiency, cost, and latency is addressed as its own distinct dimension, not conflated with another.)*

45. Why does evaluating only an agent's final output (task success) miss information a full trajectory evaluation would catch?
46. What's the real difference between tool-selection accuracy and tool failure rate as two separate metrics?
47. Why are retry rate and tool failure rate kept as two distinct metrics instead of one?
48. Walk through computing trajectory efficiency and steps-per-successful-task from one real trajectory — why are both worth tracking, not just one?
49. How would you evaluate final-answer *quality* as distinct from task success or trajectory efficiency, and what are the specific pitfalls of using an LLM as a judge for this?
50. Why track cost-per-successful-task and latency as their own metrics, separate from accuracy-style metrics — what production decisions do they inform that accuracy metrics can't?
51. Name three common agent failure modes and how you'd distinguish them from trajectory logs.
52. A real trajectory batch showed a 0% retry rate even though a real tool failure occurred. Is that a bug or a legitimate outcome — how do you tell?

**Module 09 — Production Agent Systems, Safety & Security (8)**
53. How would you decide which actions an agent can take autonomously vs. which need human approval?
54. What are the four real sources of indirect prompt injection, and why is "user prompt injection" defenses alone insufficient against them?
55. Walk through the five layered mitigations for indirect prompt injection — why isn't any single one sufficient alone?
56. What's the difference between least-privilege tool access and sandboxing as defensive layers?
57. Why does a long-running async agent carry more real risk than a synchronous one, even with identical tools and permissions?
58. *(synthesis)* A real, deterministic prompt-injection test found a live model followed an injected instruction without a mitigation and didn't with one. What does — and doesn't — this one real experiment prove?
59. *(synthesis, flagship)* Design the full production agent stack end-to-end for a new agent with real tool access — reasoning pattern, tool schema design, memory, orchestration/durability, evaluation, security/guardrails, and cost/latency — justify which level of the LLM call → deterministic workflow → single agent → multi-agent progression (Q5, Q40) the task actually warrants, and identify where you'd deliberately cut scope for an MVP vs. a mature system.

---

## Status: Complete

All 59 questions written to `modules/10_ai_agents_interview_questions.md` across 5 batches. Mandatory Section 5 structural compliance check passed on all four points: (1) all 10 required per-question headings occur exactly 59 times each; (2) no derivation chains found in any `[DEEP DIVE]` block; (3) the Final Revision Sheet is present with all 3 required subsections (59-row Quick-Recall table, Essential Formula Cheat Sheet, 10-entry Top Follow-up Q&As); (4) no placeholder markers remain. `helpers/compile_agents.py` now compiles a second, standalone `ai_agents_interview_cheatsheet.html`/`.pdf` (1,226,661 bytes) in addition to the master study guide — verified 0 `file:///` leaks, 0 `MATHPLACEHOLDER` leaks, 59 `follow-up-section` divs, and 118 `q-card` divs (2 follow-ups × 59 questions).
