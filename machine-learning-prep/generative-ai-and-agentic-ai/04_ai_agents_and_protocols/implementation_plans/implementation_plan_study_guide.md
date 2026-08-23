# Implementation Plan: Track 1 — Study Guide Modules (`04_ai_agents_and_protocols`)

Scope: this plan covers only the 9 theory study-guide modules, per `study_guide_generator/SKILL.md`. Track 2 (companion notebooks) and Track 3 (Interview Q&A) are separate and not covered here.

---

## 1. Module List & Target File Paths

| # | Module | Target Path |
|---|---|---|
| 01 | Agent Fundamentals & Reasoning Patterns | `modules/01_agent_fundamentals_and_reasoning.md` |
| 02 | Tool Calling & Function Calling Internals | `modules/02_tool_calling_and_function_calling.md` |
| 03 | Model Context Protocol (MCP) & Agent-Tool Standardization | `modules/03_mcp_and_agent_tool_standardization.md` |
| 04 | Context, State & Memory for Agents | `modules/04_context_state_and_memory.md` |
| 05 | Agent Orchestration, State Machines & Durable Execution | `modules/05_orchestration_and_durable_execution.md` |
| 06 | Multi-Agent Systems & Coordination Patterns | `modules/06_multi_agent_systems.md` |
| 07 | Agent Frameworks Landscape | `modules/07_agent_frameworks_landscape.md` |
| 08 | Agent Evaluation & Debugging | `modules/08_agent_evaluation_and_debugging.md` |
| 09 | Production Agent Systems, Safety & Security | `modules/09_production_agents_safety_and_security.md` |

Each module follows the standard 4-section structure (Introduction & Intuition → Core Concepts & Mathematical Formulation → Implementation & Reference Code → Interview Deep-Dive & System Trade-offs).

---

## 2. Formulas & Hand Calculations (Formula Selection Constraint applied)

Per the skill's formula-selection rule, most of this topic is architectural/procedural (agent loops, protocol design, orchestration patterns, security policy) and stays prose-only with no formula blocks — matching how `03_advanced_rag` kept its own architectural modules (Query Transformation, GraphRAG, Agentic RAG) formula-free. Three modules have genuinely quantifiable, interview-core content worth a real hand calculation:

**Module 02 — Sequential vs. Parallel Tool-Call Latency + Per-Task Cost Model**
- Formula 1 (latency composition):
  $$T_{\text{sequential}} = \sum_{i=1}^{n} t_{\text{tool},i}, \qquad T_{\text{parallel}} = \max_{i}(t_{\text{tool},i})$$
  (both plus the fixed LLM round-trip overhead either side)
- Hand calc: 3 independent tools with real, different latencies (e.g., 200ms/400ms/600ms) plus a fixed LLM call overhead — compute total sequential vs. parallel task time and the resulting speedup, directly motivating *why* safe parallelization (Module 02's own "safe parallel vs. sequential" content) matters.
- Formula 2 (per-task cost model):
  $$\text{Cost}_{\text{task}} = \sum_{i=1}^{n_{\text{turns}}} (\text{tokens}_{\text{in},i} + \text{tokens}_{\text{out},i}) \times \text{price}_{\text{token}} + \sum_{j=1}^{n_{\text{tools}}} \text{cost}_{\text{tool},j}$$
- Hand calc: a small multi-turn tool-calling task (e.g., 3 LLM turns + 2 tool calls with a per-call API cost) — compute total task cost, mirroring `03_advanced_rag` Module 01's RAG-cost-model hand-calc style.

**Module 04 — Context Budget & Memory Summarization Trigger**
- Formula: a simple running-token-count-vs-budget threshold check, framed explicitly in terms of the module's own Context vs. State vs. Memory distinction (the left-hand side is *context* — what's actually about to be sent to the model — assembled from state and retrieved/carried memory) —
  $$\text{trigger\_summarization} = (\text{tokens}_{\text{system}} + \text{tokens}_{\text{history}} + \text{tokens}_{\text{next\_turn\_budget}}) > \theta \times \text{context\_window}$$
- Hand calc: a concrete context window size, system-prompt/tool-schema overhead, and per-turn growth rate — compute the turn number at which summarization must trigger to stay under budget, giving the abstract "context budget vs. memory-recall completeness" trade-off (already in the README) one concrete worked number.

**Module 08 — Agent Trajectory Evaluation Metrics**
- Formulas (the "core evaluation statistics" the skill explicitly allows), now covering seven distinct, complementary metrics per the candidate's Pass 2 feedback (not collapsing retry rate into tool failure rate, and adding a raw-count companion to the efficiency ratio):
  $$\text{Task Success Rate} = \frac{N_{\text{successful tasks}}}{N_{\text{total tasks}}}, \qquad \text{Trajectory Efficiency} = \frac{\text{Steps}_{\text{minimal}}}{\text{Steps}_{\text{actual}}}, \qquad \text{Steps per Successful Task} = \frac{\sum \text{Steps}_{\text{actual}}}{N_{\text{successful tasks}}}$$
  $$\text{Tool-Selection Accuracy} = \frac{N_{\text{correct tool chosen}}}{N_{\text{tool calls}}}, \qquad \text{Tool Failure Rate} = \frac{N_{\text{tool calls erroring}}}{N_{\text{tool calls}}}, \qquad \text{Retry Rate} = \frac{N_{\text{steps retried}}}{N_{\text{total steps}}}$$
  $$\text{Cost per Successful Task} = \frac{\sum \text{Cost}_{\text{task}}}{N_{\text{successful tasks}}}$$
- Hand calc: one toy 5-step agent trajectory (mirroring `03_advanced_rag` Module 09's Recall@k/MRR/NDCG "one toy query" hand-calc pattern) — some steps correct, one wrong tool call, one tool that errors independent of tool choice, one retried step — compute all seven metrics together from the same worked trajectory, showing how they can tell different, complementary stories about the same run (e.g., distinguishing "the agent picked the wrong tool" from "the agent picked the right tool but it failed" from "the agent needed a retry").

All other modules (01, 03, 05, 06, 07, 09) stay formula-free per the skill's "Concept Simplification" rule, using tables, decision checklists, and plain-language architecture descriptions instead — consistent with how `03_advanced_rag` treated its own purely-architectural modules.

---

## 3. Diagrams & Plots

**Inline SVG diagrams** (responsive, `viewBox`-based, no LaTeX inside `<text>` nodes, per the skill's SVG rules):
1. Module 01 — ReAct loop (Thought → Action → Observation, cyclic).
2. Module 02 — Sequential vs. parallel tool-call timeline (visualizing the Section 2 hand calc directly).
3. Module 03 — MCP client-host-server architecture (Tools/Resources/Prompts primitives).
4. Module 04 — Context vs. State vs. Memory (three-way diagram: state and memory both feed into the assembled context sent to the model on a given turn).
5. Module 05 — Graph-based orchestration with checkpoint markers (state machine + durable-execution checkpoints on a cyclic graph).
6. Module 06 — Orchestrator-worker vs. peer-to-peer multi-agent topology comparison.
7. Module 08 — Minimal vs. actual trajectory path (visualizing trajectory efficiency).
8. Module 09 — Permission-boundary diagram (autonomous actions vs. human-approval-gated actions).

**Matplotlib plots** (saved to `plots/`, referenced directly under their relevant subsection, real-vs-illustrative labeling applied honestly — same discipline as `03_advanced_rag`):
1. `02_sequential_vs_parallel_latency.png` — task completion time (sequential vs. parallel) as tool count grows; the underlying formula is real (Section 2's latency-composition formula), plotted over illustrative example tool-latency values, labeled as such in the plot title.
2. `04_context_budget_over_turns.png` — running token count vs. context-window budget across conversation turns, marking the summarization-trigger point computed directly from the Module 04 hand calc's real numbers.
3. `06_multi_agent_cost_scaling.png` — task cost vs. number of agents in a multi-agent system, illustrating Module 06's coordination-overhead/cost-multiplication argument. **Per the candidate's explicit Pass 2 feedback, this plot's title/caption must clearly label it as an illustrative cost model (a qualitative shape the coordination-overhead argument predicts), not a universal or measured production curve** — no notebook in this topic will measure a real multi-agent system's cost scaling, so this plot must not visually imply it's derived from real measurements the way, e.g., `03_advanced_rag`'s Track 2 plots were.
4. `08_trajectory_metrics.png` — bar chart of the seven Module 08 hand-calc metrics for the one toy trajectory (mirrors `03_advanced_rag` Module 09's retrieval-metrics plot).

---

## 4. Open Design Questions / Dependencies

1. **Cross-references to `03_advanced_rag`**: Module 04's vector-store-backed long-term memory and Module 08's evaluation content will cross-reference (not re-derive) `03_advanced_rag`'s embedding/indexing and evaluation-metric modules, per the README's cross-module boundary note. No new dependency — this repo's existing topic.
2. **Framework version currency (Module 07)**: LangGraph/AutoGen/CrewAI/OpenAI Agents SDK are fast-moving; per the candidate's Pass 2 feedback, the module will compare frameworks along six explicit, stable dimensions (architecture, control, state, observability, extensibility, lock-in) rather than version-specific API details, to avoid the guide going stale quickly.
3. **No open blockers** — proceeding module-by-module (01 → 09) once this plan is approved, matching `03_advanced_rag`'s Track 1 execution pattern.

---

## Status: Complete

All 9 modules written, verified, and structurally checked:

1. `01_agent_fundamentals_and_reasoning.md` — ReAct loop, CoT vs. agentic reasoning, plan-and-execute vs. reactive, formal LLM-call/workflow/single-agent/multi-agent decision framework.
2. `02_tool_calling_and_function_calling.md` — schema design's effect on selection accuracy, safe parallel/sequential execution, idempotent tool execution; hand-calc'd 2.0x parallel speedup (2,400ms → 1,200ms) and $0.008875 per-task cost.
3. `03_mcp_and_agent_tool_standardization.md` — client-host-server architecture, local vs. remote trust boundaries, capability discovery, versioning, auth/authz, tool-level permissions, security risks.
4. `04_context_state_and_memory.md` — explicit three-way Context/State/Memory distinction; hand-calc'd summarization trigger at turn 17 (8,000-token window, θ=0.8).
5. `05_orchestration_and_durable_execution.md` — graph-based orchestration, checkpointing, crash recovery, resume, workflow-level idempotency, pause/resume, long-running workflows.
6. `06_multi_agent_systems.md` — orchestrator-worker vs. peer-to-peer, agent-level safe parallelism, Module 01's decision framework applied to the multi-agent boundary.
7. `07_agent_frameworks_landscape.md` — LangGraph/AutoGen/CrewAI/OpenAI Agents SDK compared along 6 stable dimensions (architecture, control, state, observability, extensibility, lock-in).
8. `08_agent_evaluation_and_debugging.md` — 7 quantitative trajectory/batch metrics hand-calc'd from one toy 5-step trajectory + a toy 5-task batch (efficiency 0.6, tool-selection accuracy 0.8, tool failure rate 0.2, retry rate 0.2, task success rate 0.8, 3.75 steps/success, $0.0085/success).
9. `09_production_agents_safety_and_security.md` — guardrails/permission tiers, rate/cost budgets, indirect prompt injection (4 sources) with 5 layered mitigations, authorization boundaries, sandboxing.

Verification: all 8 Python reference-code blocks execute cleanly (exit 0, all assertions pass); all 8 inline SVG diagrams rendered via headless Edge and visually inspected for overlap/clipping defects (all clean, one fixed during Module 02's build); all 4 matplotlib plots generated, visually verified, one label-overlap fixed (Module 02's latency plot) and one value corrected for pedagogical honesty (Module 08's Task Success Rate, changed from a trivial 1.0 to a real batch-derived 0.8); the illustrative multi-agent cost-scaling plot (Module 06) is explicitly labeled as such in its own title, per the candidate's Pass 2 feedback. Structural check confirmed: 9/9 files present, both required section headers present exactly once per file, no stray unescaped `%` in math blocks, no Mermaid/ASCII diagrams, no LaTeX inside SVG `<text>` nodes.

`helpers/compile_agents.py` created (adapted from `03_advanced_rag/helpers/compile_advanced_rag.py`'s reusable `compile_document()`), compiling `ai_agents_master_study_guide.html`/`.pdf` (58 pages, 2,015,514 bytes). Compilation succeeded on the first attempt. Verified: 0 `file:///` leaks, 0 unresolved `MATHPLACEHOLDER` leaks, 9/9 `module-container` divs, 18 `q-card` divs (2 follow-up Q&As × 9 modules), 9 `follow-up-section` divs, 4/4 plot images correctly embedded as base64 with no missing-image warnings during compilation.

Track 1 complete. Awaiting the user's review before proceeding to Track 2 (companion notebooks).

**Pass 2 revisions** (mirroring the README's own Pass 2): Module 04 renamed/reframed around the explicit Context vs. State vs. Memory three-way distinction, with its hand-calc formula now explicitly framed in those terms; Module 05's durable-execution scope note (covered in the README, no plan-level formula change needed since Module 05 stays formula-free); Module 08's metric formulas expanded from 4 to 7 (adding tool failure rate, retry rate as distinct from each other, and steps-per-successful-task alongside trajectory efficiency); the Module 04 SVG diagram updated to a three-way Context/State/Memory diagram; the multi-agent cost-scaling plot's illustrative-labeling requirement made explicit; Module 07's open design question updated to reference the six explicit comparison dimensions. Mathematical scope confirmed otherwise unchanged (no formulas added to Modules 01, 03, 05, 06, 07, 09).
