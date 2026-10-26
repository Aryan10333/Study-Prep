# Implementation Plan: Track 1 — Study Guide (`09_context_engineering_a2a_and_agent_security`)

Per `study_guide_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 1 (the 8 theory modules). Track 2 (notebooks) and Track 3 (interview Q&A) are separate, later sign-off gates.

---

## 1. Modules & Target File Paths

| # | Module | File Path |
|---|---|---|
| 01 | Context Engineering as a Discipline | `modules/01_context_engineering_as_a_discipline.md` |
| 02 | Context Rot & Long-Context Degradation | `modules/02_context_rot_and_long_context_degradation.md` |
| 03 | Prompt Caching & Cache-Aware Context Layout | `modules/03_prompt_caching_and_cache_aware_context_layout.md` |
| 04 | Agent Harness & Long-Running Loop Design | `modules/04_agent_harness_and_long_running_loop_design.md` |
| 05 | A2A Protocol & Agent Interoperability | `modules/05_a2a_protocol_and_agent_interoperability.md` |
| 06 | Agent Supply-Chain Threats | `modules/06_agent_supply_chain_threats.md` |
| 07 | Securing the Agent Tool Supply Chain in Production | `modules/07_securing_the_agent_tool_supply_chain.md` |
| 08 | Common Failure Modes & Best Practices | `modules/08_failure_modes_and_best_practices.md` |

Every module follows the skill's 4-section structure (Introduction & Intuition → Core Concepts → Implementation & Reference Code → Interview Deep-Dive & System Trade-offs), with YAML frontmatter and a "Builds on" line naming the exact prior-topic chapters it references instead of repeating.

---

## 2. Per-Module Formulas & Hand Calculations

Formulas are kept only where they carry standard interview signal: token-cost and caching arithmetic. The protocol, threat and defense modules are formula-free, matching how `08_genai_system_design_and_llmops` treated its architecture and security modules.

**Module 01 — Context Engineering as a Discipline**
- **Formula (core, cumulative input tokens of an agent loop):** with a fixed static context $C_s$ (system prompt + tool definitions) and history growing by $g$ tokens per step, step $t$ re-reads $C_s + g\,t$ tokens, so over $T$ steps
  $$\text{Tokens}_{\text{total}} = T\,C_s + g\,\frac{T(T+1)}{2}$$
  **Intuition:** agent cost grows quadratically with loop length because every step re-reads all prior history. This is the single number that motivates all four operations (write, select, compress, isolate).
- **Hand calc:** $T=20$, $C_s=5{,}000$, $g=1{,}500$ → $100{,}000 + 1{,}500 \times 210 = 415{,}000$ input tokens. Then the same loop with tool definitions trimmed from 5,000 to 1,500 tokens (progressive disclosure) shows the static term's share of the saving (70,000 tokens), separately from the history term.
- **Worked example (no formula):** one agent step's context broken into its segments (system, tools, retrieved, history, memory, scratchpad) with token counts, labeled static vs dynamic and tagged with which of the four operations applies to each.

**Module 02 — Context Rot & Long-Context Degradation**
- **Formula (core, usable context limit):** given measured task accuracy $A(L)$ at context length $L$ and a baseline $A(L_0)$ at a short length, the usable limit is the largest tested length that stays within a tolerated drop $\delta$:
  $$L_{\text{usable}} = \max\{\,L : A(L) \ge A(L_0) - \delta\,\}$$
  **Intuition:** turns "the model supports 1M tokens" into a measured, task-specific ceiling that drives selection and compaction policy.
- **Hand calc:** a small, explicitly illustrative accuracy table (e.g. 8k: 0.92, 16k: 0.90, 32k: 0.86, 64k: 0.78, 128k: 0.66) with $\delta = 0.05$ → $L_{\text{usable}} = 32\text{k}$. Repeated for a single-needle task (flat curve) vs a multi-hop task (steep curve) to show the limit depends on the task, not just the model.
- **No formula** for attention dilution or distractor effects: plain-language intuition plus a worked "distractor" example (two near-duplicate passages, only one correct).

**Module 03 — Prompt Caching & Cache-Aware Context Layout**
- **Formula (core, per-step input cost with caching):** with base input price $p$, cache-read multiplier $r$ (e.g. 0.1) and cache-write multiplier $w$ (e.g. 1.25), a step whose prompt has $P_{\text{hit}}$ cached prefix tokens and $N_{\text{new}}$ newly written tokens costs
  $$\text{Cost}_{\text{step}} = p\,\big(r\,P_{\text{hit}} + w\,N_{\text{new}}\big)$$
  **Intuition:** a stable prefix is re-read at a fraction of the price, but anything that rewrites the prefix (a timestamp, a reordered tool list, a compaction summary) turns cheap reads back into full-price writes.
- **Hand calc (the central trade-off):** the Module 01 loop ($T=20$, $C_s=5{,}000$, $g=1{,}500$) costed three ways with stated, explicitly illustrative multipliers: (a) no caching, (b) keep full history with caching, (c) compact every 8 steps to a 2,000-token summary (each compaction invalidates the cache, forcing a rewrite). The worked numbers show (b) cheapest at this length, then a longer loop where (c) wins, giving a concrete break-even step count.
- **Worked example (no formula):** a "cache-buster audit" of a sample prompt: timestamp in the system prompt, per-request ID, tools serialized in nondeterministic order. Each is fixed and the hit rate before and after is shown.

**Module 04 — Agent Harness & Long-Running Loop Design**
- **No formula.** Memory-summarization triggers are owned by `04` Module 04 and not re-derived.
- **Worked example 1:** the same coding task run as (a) one agent with everything in context vs (b) an orchestrator plus 3 sub-agents, each with a clean window, returning 1,000-token summaries. Token totals for both, showing the isolation quality gain and the token multiplier side by side.
- **Worked example 2:** a long-running session across 3 context resets, showing what the instruction file, the agent-written notes file and the checkpoint each preserve, and what is lost without them.
- **Worked example 3:** progressive disclosure: 40 tool definitions (about 8,000 tokens) replaced by a 600-token index with on-demand loading.

**Module 05 — A2A Protocol & Agent Interoperability**
- **No formula.**
- **Worked example 1:** a full task trace between a client agent and a remote agent, from discovery through the agent card to completion: submitted → working → input-required (the remote agent asks a clarifying question) → working → completed with an artifact. Includes a cancellation branch.
- **Worked example 2:** a decision exercise on 4 scenarios (internal database lookup, third-party travel-booking agent, a deterministic pricing API, a partner company's claims agent), each classified as MCP tool, A2A agent or plain HTTP, with the reason.

**Module 06 — Agent Supply-Chain Threats**
- **No formula.**
- **Worked example 1:** threat model of a realistic agent with 4 MCP servers (filesystem, GitHub, Slack, a community "web-tools" server), marking each point where a supply-chain attack enters.
- **Worked example 2:** a defanged tool-poisoning example: a tool description with a hidden instruction the user never sees, and a step-by-step trace of how it steers a legitimate, credentialed tool call. A rug-pull timeline (approved on day 1, definition changed on day 9) and a tool-shadowing example (two servers exposing `send_message`).
- **Mapping table:** each threat mapped to the relevant OWASP Top 10 for LLM Applications (2025) entry (LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM03 Supply Chain, LLM06 Excessive Agency, LLM10 Unbounded Consumption, and so on) and to the defense in Module 07.

**Module 07 — Securing the Agent Tool Supply Chain in Production**
- **No formula.**
- **Worked example 1:** one tool call traced through the reference architecture: registry allowlist check → definition hash check → gateway policy and description scan → OAuth token exchange for a short-lived, narrowly scoped token → sandboxed execution with egress limits → audit-log entry.
- **Worked example 2:** the Module 06 rug pull replayed against this architecture: the changed definition's hash mismatches the pinned hash, the tool is quarantined, and re-approval is required. Each Module 06 threat is ticked off against the layer that stops it.
- **Worked example 3:** an approval-fatigue budget: which actions need human approval (irreversible, external, high-privilege) and which can be auto-approved within scope, so a reviewer sees a handful of prompts per session, not dozens.

**Module 08 — Common Failure Modes & Best Practices**
- **No formula.**
- **Worked example:** a failure-mode table (symptom → detection signal → root cause → fix) covering context, interop and security failures, plus a pre-launch checklist for a production agent.

---

## 3. Implementation & Reference Code (Section 3 of each module)

All code is self-contained Python, deterministic and runnable with no API keys, using mock LLM responses where a model would be called. Each snippet is executed with assertions before the module is marked done.

| Module | Reference code |
|---|---|
| 01 | Token-budget calculator for an agent loop (the formula above plus per-segment breakdown) |
| 02 | Usable-context-limit finder over a measured accuracy table, plus a tiny distractor-retrieval demo |
| 03 | Prompt-cache cost simulator comparing no-cache, keep-all and compaction policies across loop lengths |
| 04 | Minimal agent harness: loop, tool index with on-demand loading, notes file, checkpoint/resume (mock LLM) |
| 05 | Agent card as JSON plus an A2A-style task state machine enforcing legal transitions, with cancellation |
| 06 | Heuristic tool-description scanner flagging hidden-instruction patterns, with its false-negative limits stated |
| 07 | Tool-definition pinning (SHA-256 of canonical JSON), allowlist check and a gateway policy function |
| 08 | Prompt cache-buster linter and an agent config audit against the pre-launch checklist |

---

## 4. Diagrams & Plots

**Inline responsive SVG** (rendered and visually checked before each module is marked done, per the skill):
- Module 01: the four operations (write, select, compress, isolate) around a single context window, with each segment labeled static or dynamic.
- Module 03: cache-aware prompt layout: stable prefix (system, tools, documents) → cache breakpoint → volatile suffix, with cache-buster positions marked.
- Module 04: harness architecture: loop, tool registry, permission layer, context manager, state store, stop conditions, and the human outer loop.
- Module 05: MCP vs A2A side by side (agent → tools vs agent ↔ agent across a trust boundary), plus the A2A task state machine.
- Module 06: supply-chain attack surface: where malicious servers, poisoned descriptions, rug pulls and shadowing enter an agent.
- Module 07: reference defense architecture: registry → gateway → scoped credentials → sandbox → audit log.

**Matplotlib plots** (saved to `plots/`, generated by `helpers/generate_context_security_plots.py`, each labeled as computed from the module's hand calculation or as illustrative):
1. Module 01: cumulative input tokens vs agent step (quadratic growth), with and without tool-definition trimming.
2. Module 02: illustrative accuracy vs context length for a single-needle task and a multi-hop task, with $L_{\text{usable}}$ marked for each.
3. Module 03: cumulative cost vs step for no-cache, keep-all-cached and compaction policies, with the break-even step marked.

Total: 3 plots and 6 diagrams.

---

## 5. Open Design Questions / Dependencies

1. **Fact-check before writing Modules 03, 05 and 07.** Caching prices and rules, the A2A spec (agent card fields, task states) and the MCP authorization spec (OAuth 2.1, token-passthrough rules) change quickly. Each will be checked against current official docs at writing time. The hand calcs use explicitly illustrative multipliers so they stay correct even if prices change.
2. **Module 02's accuracy numbers are illustrative**, labeled as such and consistent in shape with published long-context benchmarks. Track 2 can replace them with a real measurement (see question 3).
3. **Track 2 (not this track):** the notebooks can run real experiments: a real context-rot curve on a small local model or a cheap API, a real prompt-caching cost run, and a hands-on tool-poisoning attack-and-defense lab. API vs local-model choice to be settled at Track 2 planning.
4. **Defensive framing:** the tool-poisoning examples in Modules 06–07 are defanged and shown only to explain the detection and defense.
5. Nothing else blocks starting Track 1.

## Status: Complete

All 8 modules written and verified: every reference-code block executed with passing assertions; all 6 inline SVG diagrams rendered via headless Edge and visually checked; all 3 plots generated by `helpers/generate_context_security_plots.py` and checked. Master study guide compiled to HTML/PDF via `helpers/compile_context_security.py` (adapted from Topic 08's compiler): 8/8 module containers, 8/8 follow-up sections, 24/24 Q&A cards, no markdown leaks.

Fixes made during execution:
- Module 05's lifecycle diagram had two arrows crossing the interrupted-state boxes; boxes moved above and below the arrow paths.
- Module 03's plot label overlapped a curve; moved.
- Module 04's progressive-disclosure demo initially showed no saving (the script used every tool); unused tools added so the saving is visible.
- Module 07 had a misleading code comment about key-order changes; corrected.
- Follow-up Q&As re-indented to 4 spaces so the compiler renders them as cards.
- The compiler no longer force-closes running Edge windows before printing (Topic 08's version did); it uses a fresh profile plus `--headless=new` and waits for the PDF file instead.
- Fact-checked at writing time: A2A v1.0 spec (Agent Card fields, task states, discovery path, operations), MCP authorization and security best practices (revision 2026-07-28), Anthropic prompt-caching multipliers, Chroma "Context Rot" report (July 2025).
