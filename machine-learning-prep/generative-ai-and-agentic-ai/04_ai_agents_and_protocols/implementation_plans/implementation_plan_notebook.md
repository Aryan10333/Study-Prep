# Implementation Plan: Track 2 — Companion Notebooks (`04_ai_agents_and_protocols`)

Scope: this plan covers only the `notebooks/` companion notebooks, per `notebook_generator/SKILL.md`. Track 1 (9 study-guide modules) is complete and signed off; Track 3 (Interview Q&A) is separate and not covered here.

---

## 0. Environment Reality Check (affects every notebook below)

| Check | Result |
|---|---|
| `langgraph` | **Installed** — real graph-based orchestration with real checkpointing (Notebook 04) |
| `langchain` / `langchain_openai` / `langchain_core` | **Installed** (1.3.14 / 1.4.0 / 1.5.0) |
| `mcp` (official Model Context Protocol SDK) | **Installed** — real local MCP server + client (Notebook 02), not a simulation |
| `openai` | **Installed** (2.47.0) — real live function-calling for Notebooks 01, 05, 06 |
| `tavily` (Tavily search client) | **Installed** — real live web search tool for Notebooks 01, 05 |
| `sentence_transformers` / `faiss` | **Installed** — real embedding + vector-store-backed memory for Notebook 03, reusing `03_advanced_rag`'s verified `nomic-embed-text-v1.5` pipeline |
| `chromadb` | **Missing** — consistent with `03_advanced_rag`'s prior decision to skip it; `faiss` covers the same real vector-store need |
| `groq`, `langfuse` | **Missing** — not required: OpenAI covers all live-LLM needs; Module 08's "observability/tracing" content is demonstrated with a real, self-contained structured trace logger (already prototyped in the Track 1 module's own reference code) rather than requiring an external Langfuse cloud account with credentials not present in `.env` |
| `tiktoken` | To be verified at Notebook 03 build time — needed for real token counting in the context-budget experiment; installed via pip if missing (matching the `einops` precedent from `03_advanced_rag`) |
| GPU | NVIDIA RTX 4060 Laptop, 8.6GB VRAM — available for Notebook 03's embedding model |
| Real API keys available in `.env` | `OPENAI_API_KEY`, `TAVILY_API_KEY` confirmed present and working (verified live in `03_advanced_rag`'s own Track 2 session) |

**Graceful API-failure handling**: every live OpenAI/Tavily call across all 6 notebooks is wrapped in the same `call_llm`/`call_tool` try/except pattern established in `03_advanced_rag`'s Notebooks 05/06 — a labeled `[API UNAVAILABLE — FALLBACK]` message plus a deterministic fallback, so a transient API outage doesn't halt notebook execution. Per the candidate's explicit Pass 2 feedback: every measurement derived from a fallback path must be visibly marked as a fallback result in both the printed output and the explanation cell, and must never be included in the same aggregate/average as real measured results — a metric computed from a mix of real and fallback values is not a real benchmark number, and this notebook set will not present one as if it were.

**Resource discipline**: explicit `del model_var` + `torch.cuda.empty_cache()` between notebook sections that load the embedding model (Notebook 03 only — this is the sole notebook using a local GPU model; the rest are API-only and need no GPU cleanup). Per the candidate's Pass 2 feedback, this is extended beyond GPU memory: every notebook also explicitly releases large objects and long-lived clients it created (MCP client/server connections in Notebook 02, the LangGraph app/checkpointer connection in Notebook 04, HTTP clients in the API-heavy notebooks) in its own final cleanup cell, and every notebook is built to run correctly end-to-end from a fresh kernel restart, not just in the same session it was authored in.

**Real vs. illustrative labeling**: every result is real by construction (live API calls, real local MCP protocol traffic, real LangGraph checkpoints on disk); any deliberately-constructed test case (e.g., the injection test in Notebook 06) is explicitly labeled as such, matching `03_advanced_rag`'s established discipline.

---

## 1. Notebook List & Target File Paths

| # | Notebook | Modules Covered | Target Path |
|---|---|---|---|
| 01 | ReAct Agent & Real Tool Calling | 01, 02 | `notebooks/01_react_agent_and_tool_calling.ipynb` |
| 02 | Real MCP Client & Server | 03 | `notebooks/02_mcp_client_and_server.ipynb` |
| 03 | Context, State & Memory | 04 | `notebooks/03_context_state_and_memory.ipynb` |
| 04 | LangGraph Orchestration & Durable Execution | 05 | `notebooks/04_langgraph_orchestration_and_durability.ipynb` |
| 05 | Multi-Agent Coordination | 06 | `notebooks/05_multi_agent_coordination.ipynb` |
| 06 | Agent Evaluation, Debugging & Guardrails | 08, 09 | `notebooks/06_agent_evaluation_and_guardrails.ipynb` |

**Module 07 (Agent Frameworks Landscape) has no dedicated notebook** — it is a comparative architectural survey with no single hands-on engineering pipeline of its own, the same reasoning `03_advanced_rag` applied when it excluded its own Module 01 (Fundamentals) from Track 2. Framework comparison is instead demonstrated *implicitly*: Notebook 01 uses raw OpenAI function-calling (no framework), Notebook 04 uses LangGraph — giving a real, working point of comparison between "custom loop" and "framework" without a seventh notebook whose only content would be re-stating Module 07's own comparison table.

---

## 2. Real-World Datasets, APIs & Engineering Pipelines

### Notebook 01 — ReAct Agent & Real Tool Calling
- **Real APIs**: OpenAI `gpt-4o-mini` (live function-calling), Tavily (live web search), plus a real (non-mocked) arithmetic-evaluation tool.
- **Pipeline**: a real ReAct loop (Module 01's mechanics) driving live OpenAI function-calling across a small set of real, tool-requiring questions (some needing live search, some needing calculation).
- **The real experiment (schema quality)**: build two versions of the *same* 3-tool schema — one with clear, distinct names/descriptions, one with deliberately ambiguous/overlapping ones — kept deliberately small (3 tools) so the experiment isolates schema quality rather than being confounded by sheer tool count, and run both against a sufficiently varied real query set (a real mix of search-only, calculation-only, and ambiguous-either-could-apply queries, not a handful of near-duplicate ones). Report **two** real metrics per schema version, not accuracy alone: **tool-selection accuracy** (right tool chosen) and **malformed-argument rate** (the tool was called with arguments that don't satisfy the schema — a distinct real failure mode ambiguous descriptions can also cause).
- **The real experiment (parallel tool calls)**: explicitly verify the tools used in this test are genuinely independent (no data dependency, per Module 02's own dependency-analysis principle) before timing them, so the parallel path is actually valid, not just fast. Report four real numbers, not just wall-clock times: real sequential latency, real parallel latency, the real resulting speedup, and any **overhead** the parallel path itself introduces (e.g., async/concurrency setup cost) that a naive latency-only comparison would hide.

### Notebook 02 — Real MCP Client & Server
- **Real protocol traffic**: an actual local MCP server (stdio transport, via the installed `mcp` SDK) exposing 2-3 real tools/resources (e.g., querying a small real local SQLite database, reading a real local config resource), and an actual MCP client connecting to it — not a simulation of the protocol, real client-server messages over a real transport.
- **Pipeline**: real capability discovery at connect time (the client asking the server what it exposes), kept explicitly **distinct** from a separate real application-level authorization layer (a client-side permission check the *application* enforces on top of whatever the server discovered) — per the candidate's Pass 2 feedback, these are two different real mechanisms (protocol-level "what exists" vs. application-level "what am I allowed to call") and the notebook keeps them as two clearly separate steps, not one conflated check. The permission boundary itself is deterministic: a client scoped to a fixed, smaller permission set than the server exposes, with an explicit `assert` that an unauthorized tool invocation attempt is actually rejected (not just "expected" to be rejected) — a real, falsifiable check, not a demonstration that merely looks like it worked.

### Notebook 03 — Context, State & Memory
- **Real APIs/models**: `nomic-embed-text-v1.5` (same verified model as `03_advanced_rag`) for real memory embeddings, `tiktoken` for real token counting, OpenAI `gpt-4o-mini` for real summarization once the budget trigger fires.
- **Pipeline (state vs. memory)**: explicitly demonstrates all three concepts from Module 04's own organizing distinction, not just memory in isolation — **context** (the real assembled prompt sent to the model on a given turn, printed and inspected directly), **state** (real run-scoped data tracked within one simulated conversation session), and **memory** (a real FAISS-backed store written deliberately and intentionally, per a real write policy). The notebook then genuinely starts a **new, separate simulated session** and shows, concretely, which of the two survives into it and which doesn't — state from the first session is gone, while a fact written to long-term memory is still retrievable — the real, observable version of the module's own core distinction, not just an assertion of it.
- **Pipeline (context budget)**: real token counting doesn't rely on the model's nominal context-window limit alone — it separately tracks and sums real token counts for **system prompt + tool schemas + conversation history + retrieved memory + a reserved next-turn budget**, the same five real components Module 04's hand calculation names, each counted for real via `tiktoken` rather than estimated. The notebook shows the exact real turn at which the running total crosses the real configured threshold and triggers summarization, mirroring the hand calc's turn-17 result but from real, measured token counts instead of an assumed per-turn growth rate.

### Notebook 04 — LangGraph Orchestration & Durable Execution
- **Real framework**: `langgraph`, with a real on-disk checkpointer (`SqliteSaver` from the now-installed `langgraph-checkpoint-sqlite` package, not `MemorySaver`) so the durability claim is real, not simulated.
- **Pipeline**: a real graph with conditional routing and a cycle (a retry loop), real checkpointing after each node, and a real human-in-the-loop interrupt using LangGraph's actual interrupt mechanism.
- **The real crash/resume test — one of the highest-value experiments in this notebook set**: per the candidate's explicit Pass 2 emphasis, the "crash" must be genuine, not merely simulated in appearance. The notebook will `del` the original compiled graph object (and any other Python references to it) after checkpointing partway through a run, construct a **brand-new** `StateGraph`/compiled app object from scratch pointing at the same on-disk SQLite checkpoint file, and resume purely from what that fresh object reads off disk — proving the resumed execution genuinely does not depend on any in-memory state surviving from the original run, not just asserting that it doesn't.
- **The real idempotency example**: directly connecting durability to production safety (Module 09), a small, explicit demonstration of a side-effecting tool node that, if naively re-run after a resume, would duplicate its real effect (e.g., incrementing a counter twice) — shown first *without* an idempotency guard (the duplicate genuinely happens) and then *with* one (an idempotency key/guard correctly prevents the duplicate on the resumed replay), a real, falsifiable before/after comparison rather than an assertion that guards "would" help.

### Notebook 05 — Multi-Agent Coordination
- **Real APIs**: OpenAI `gpt-4o-mini` for all agents, Tavily for the researcher agent's real live search.
- **The real experiment — a fair, controlled comparison**: run the exact *same* real task, the exact same underlying model, the exact same available tools, and the exact same evaluation criteria, two ways — a single generalist agent doing everything, and an orchestrator-worker multi-agent split (researcher + writer) — changing only the architecture, per the candidate's explicit Pass 2 requirement that this be a genuinely fair comparison, not two differently-configured setups. Reports real measurements across (at minimum) four dimensions for both conditions: **task success** (did it actually produce a correct/complete result), **latency**, **token usage/cost**, and **number of tool calls/steps taken** — this is what makes the "multi-agent isn't automatically better" conclusion defensible rather than asserted from one convenient metric.
- **Also measured**: real wall-clock speedup from running two genuinely independent sub-agents in parallel vs. sequentially.

### Notebook 06 — Agent Evaluation, Debugging & Guardrails
- **Real APIs**: OpenAI `gpt-4o-mini`, reusing Notebook 01's real tool set.
- **Pipeline**: run several real agent trajectories across a small real task batch, log full real trajectories, and compute all seven Module 08 metrics (task success rate, trajectory efficiency, tool-selection accuracy, tool failure rate, retry rate, steps/cost per successful task) from this real data — not the toy hand-calc example.
- **Real guardrail demo**: the actual `GuardrailPolicy` class from Module 09 enforcing real permission tiers on real tool calls, with a real audit log.
- **The real, deterministic experiment**: a crafted (explicitly labeled as deliberately-constructed) tool-output string containing a hidden instruction, held **byte-for-byte identical** across both conditions per the candidate's Pass 2 requirement — only the mitigation itself changes between the **without-mitigation** and **with-mitigation** (marking untrusted tool output as data, not instructions) runs, isolating the mitigation as the one real variable. For each condition, the notebook records three real, distinct signals: **whether the live model's response actually followed the injected instruction**, **whether the resulting tool action was blocked by the guardrail policy**, and **the resulting audit-log event** — three separate real observations, not one collapsed pass/fail verdict.
- **Explicit scope honesty on the security result**: per the candidate's Pass 2 feedback, the write-up presents this as **one concrete empirical demonstration that this specific mitigation changed this specific model's behavior on this specific crafted input** — not as proof the guardrail provides complete or universal prompt-injection protection. A single real experiment establishes exactly what it measured, nothing broader; the module's own text already carries this caveat, and the notebook's explanation cells will state it explicitly rather than let the real, positive result be read as a stronger claim than it supports.

---

## 3. Open Design Questions / Dependencies — Resolved by Pre-Flight Verification

1. **`tiktoken`**: verified already installed — no action needed at Notebook 03 build time.
2. **MCP server transport**: Notebook 02 uses local stdio transport only (matching Module 03's "local vs. remote" distinction) — no remote/SSE server is stood up, since that would require standing up real internet-facing infrastructure out of scope for a companion notebook. `mcp.server` and `mcp.client` both verified importable.
3. **LangGraph checkpointer**: `langgraph.checkpoint.memory` (in-memory only, verified installed) is **not** sufficient for a real durability demonstration — it wouldn't survive a genuinely fresh process. Real on-disk persistence requires the separate `langgraph-checkpoint-sqlite` package, which was **not** initially installed; installed it live (`pip install langgraph-checkpoint-sqlite`, pulling in `aiosqlite`), then verified `from langgraph.checkpoint.sqlite import SqliteSaver` imports successfully. Notebook 04 will use real `SqliteSaver`-backed on-disk checkpoints.
4. **No open blockers** — proceeding notebook-by-number (01 → 06) once this plan is approved, matching `03_advanced_rag`'s Track 2 execution pattern (one notebook fully built, executed, and explained before the next begins).

---

## Status: Complete

All 6 notebooks built, executed, and verified (0 unexecuted cells, 0 errors, 0 pending explanation placeholders across every notebook; no leftover artifact files — each notebook's own cleanup cell removed everything it created):

1. `01_react_agent_and_tool_calling.ipynb` — real ReAct loop over 3 live tools; real schema-quality experiment found no tool-selection accuracy gap (1.000 both) but a real malformed-argument-rate gap (0.083 ambiguous vs. 0.000 clear); real parallel tool-call speedup measured at 1.62x, with an honestly-explained negative-overhead measurement artifact from single-shot network-latency variance.
2. `02_mcp_client_and_server.ipynb` — a real local MCP server (stdio, official SDK) and client; real capability discovery found `delete_note` regardless of authorization; a real, deterministic application-level authorization boundary correctly blocked an unauthorized `delete_note` call and allowed an authorized one, verified against real database state before and after.
3. `03_context_state_and_memory.ipynb` — real FAISS-backed memory (via `nomic-embed-text-v1.5`) explicitly demonstrated surviving a new session while real run-scoped state did not; a real context-budget trigger found at turn 8 from genuine `tiktoken` counts (rescaled from an initial 8000-token config that never triggered against this notebook's real, modest conversation length); real live summarization achieved a real 25.3% token reduction.
4. `04_langgraph_orchestration_and_durability.ipynb` — a real on-disk `SqliteSaver`-backed graph; a genuine crash/resume test (original objects `del`eted, a fresh object rebuilt from disk continued execution without re-running the completed node); a real idempotency before/after comparison (2 real duplicate charges without a guard vs. 1 with); a real human-in-the-loop interrupt gate.
5. `05_multi_agent_coordination.ipynb` — a real, fair single-agent vs. multi-agent comparison on the identical task/tool/model/criterion found the multi-agent split winning on every real dimension (task success, latency, tokens, tool calls), traced to a real single-agent failure (hit `max_steps` without producing an answer); a real `RunMetrics` timing bug (recomputed elapsed time on a later call, inflating the reported single-agent latency) was found and fixed before the final comparison; real parallel sub-agent speedup measured at 2.28x.
6. `06_agent_evaluation_and_guardrails.ipynb` — all seven Module 08 metrics computed from a real, organically-generated 5-task trajectory batch (a real tool failure from a deliberately-invalid timezone, real retry-rate of 0 reflecting a real graceful-explanation recovery instead of a retry); the real `GuardrailPolicy` correctly blocked an unauthorized real tool call; a real, deterministic prompt-injection mitigation test (byte-identical malicious input across conditions) found the live model genuinely followed the injection without mitigation and did not with it — a real detection-logic bug (exact-string match missing a trailing period) was found and fixed before this result was trusted: explicitly framed as one empirical demonstration, not proof of complete protection.

Two real, load-bearing engineering fixes were required and are documented directly in the affected notebooks' own code comments: `stdio_client()`'s default `errlog=sys.stderr` fails inside a Jupyter kernel (no `.fileno()`) and needed a real opened log file instead; LangGraph's async `await` calls needed top-level `await`, not `asyncio.run()`, since the Jupyter kernel already runs its own event loop.

Track 2 complete.
