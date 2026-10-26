# Implementation Plan: Track 2 — Notebooks (`09_context_engineering_a2a_and_agent_security`)

Per `notebook_generator/SKILL.md`'s Pre-Flight Checkpoint. Covers only Track 2 (4 notebooks). Track 1 (8 modules, compiled HTML/PDF) is complete; Track 3 (interview Q&A) is a separate, later sign-off gate.

---

## 1. Real Hardware, Models and APIs (checked 2026-10-26)

| Resource | Status | Use |
|---|---|---|
| OpenAI API (`OPENAI_API_KEY`) | Works (key updated by user); `gpt-4o-mini` available; 2M tokens/min; reports `cached_tokens` (probe: 1,920 of 1,972 prompt tokens cached on a repeat call) | Long-context sweep up to 128k (Notebook 01), real provider caching (Notebook 02), tool-calling agents (Notebooks 03–04) |
| Local GPU | NVIDIA RTX 4060 Laptop, 8.6 GB VRAM | Local context-rot curves, KV-cache reuse timing, VRAM profiling |
| Local models (Hugging Face) | `Qwen/Qwen2.5-0.5B-Instruct` (cached) and `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB download, approved) | Context-rot curves up to 32k; prefix-cache timing |
| Ollama (local) | Reachable; `llama3.1`, `gemma:2b` | Not used (the lab that needed it was removed) |
| Groq API | Works, but 8k tokens/min on this key | Not needed now that OpenAI works |
| Gemini key | Rejected (401) | Not used |
| `mcp` SDK 1.28.1, `cryptography`, `httpx`, `uvicorn`, `datasets`, `tiktoken` | Installed | MCP lab, Agent Card signing, local HTTP, data, token counting |
| Datasets | SQuAD (cached) | Needle/distractor passages |

**Spend cap:** the builder tracks OpenAI usage from every response and stops at **$5.00** total. Expected spend is about $2–4, almost all in Notebook 01's 128k-token prompts (`gpt-4o-mini`, ~$0.15 per million input tokens, less with caching).

## 2. Notebooks & Target File Paths

| # | Notebook | Modules | File Path |
|---|---|---|---|
| 01 | Measuring Context Rot on a Real Model | 01, 02 | `notebooks/01_measuring_context_rot.ipynb` |
| 02 | Prefix Caching, Cache-Busters & the Keep-vs-Compact Decision | 03, 08 | `notebooks/02_prefix_caching_and_compaction.ipynb` |
| 03 | A Real Agent Harness: Tool Index, Notes, Checkpoints, Sub-Agents | 04 | `notebooks/03_agent_harness_long_running.ipynb` |
| 04 | A2A From the Spec: Agent Cards, Signatures, Task Lifecycle | 05 | `notebooks/04_a2a_from_the_spec.ipynb` |

Built by `helpers/build_context_security_notebooks.py`, one notebook at a time, two-pass (code + placeholders → execute → explanations written from real output).

---

## 3. Per-Notebook Real Experiments

**Notebook 01 — Measuring Context Rot on a Real Model** `[REAL]`
- Build long contexts from SQuAD: one gold paragraph containing the answer, padded to target lengths of 1k, 2k, 4k, 8k, 16k and 32k tokens (counted with the model's own tokenizer).
- Models: `gpt-4o-mini` at 2k, 8k, 16k, 32k, 64k and 128k tokens; `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct` locally at 1k–32k. Three curves show how model size and serving stack change the usable limit.
- Two padding conditions: **unrelated filler** (paragraphs from other articles) vs **close distractors** (other paragraphs from the *same* article, which share topic and vocabulary).
- Needle position randomized (deterministic seed), so the result is about length, not position bias.
- 25 questions per length per condition for `gpt-4o-mini` (300 calls) and 40 per length per condition for each local model, greedy decoding, graded by normalized exact match / answer containment against SQuAD gold answers.
- Outputs: accuracy-vs-length curves for both conditions, $L_{\text{usable}}$ at $\delta = 0.05$ computed with Module 02's function, plus **hardware profiling** per length: prefill latency (`time.perf_counter()`), peak VRAM (`torch.cuda.max_memory_allocated()`), and KV-cache size versus the theoretical $2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{bytes}$.
- Module 01 tie-in: per-segment token breakdown of one real agent-step prompt using `tiktoken`.
- Honest scope: small local models degrade much earlier than hosted models; results describe these models on this task, not frontier models in general.

**Notebook 02 — Prefix Caching, Cache-Busters & the Keep-vs-Compact Decision** `[REAL]` + `[COMPUTED FROM REAL DATA]`
- Build a real 30-step agent transcript (system prompt + tool schemas + growing tool results, using real SQuAD text as tool output).
- For each step, time prefill two ways on the local GPU: **full recompute** vs **reusing the previous step's KV cache** (Hugging Face `DynamicCache`, i.e. what a prefix cache does), and verify both produce identical next-token logits (within float tolerance).
- **Cache-buster experiment:** the same transcript with a timestamp at the top of the system prompt; measure the longest reusable prefix per step (0) and the resulting prefill time.
- **Real provider caching:** replay the same 30-step transcript against `gpt-4o-mini` with (a) a stable layout and (b) a timestamp at the top of the system prompt; record `cached_tokens`, total prompt tokens and latency per step from the API, and compute the real cost of each from the usage numbers.
- **Keep vs compact:** take the transcript's real per-step token counts and apply Module 03's cost formula with stated multipliers; find the real break-even step, and compare it with the illustrative step 29 from Module 03. Also run a real compaction (the local model summarizes the history) and measure what the summary drops (answer accuracy on facts from early steps before vs after compaction).
- Module 08 tie-in: run the cache-buster linter on the real prompts from this notebook.

**Notebook 03 — A Real Agent Harness** `[REAL]` (`gpt-4o-mini`, tool calling)
- Task: a small Python package written to a temp workspace with 3 seeded bugs and a pytest suite; the agent has `list_files`, `read_file`, `write_file`, `run_tests` (sandboxed to the workspace).
- Measure per run: steps, total input tokens, largest window, tests passing at the end.
- Experiments: (a) full tool schemas vs a tool index with on-demand loading (static-token saving measured from real usage); (b) a forced **context reset** mid-task with and without an agent-written notes file (does it redo work or break fixed code?); (c) a simulated crash with **checkpoint/resume**; (d) one orchestrator + per-bug sub-agents vs a single agent: tokens, max window, success.
- Reported across repeated runs (5 per variant), as a distribution, because model runs are non-deterministic.

**Notebook 04 — A2A From the Spec** `[REAL]` (local HTTP + `gpt-4o-mini`-backed remote agent)
- A minimal A2A v1.0-style JSON-RPC server written from the spec (no official SDK installed), serving an Agent Card at `/.well-known/agent-card.json`, implementing `SendMessage`, `GetTask` and `CancelTask`. The remote "booking agent" is LLM-backed and returns `INPUT_REQUIRED` when the request is ambiguous.
- A client agent that discovers the card, **verifies its JWS signature** (Ed25519 via `cryptography`, over a canonicalized card), sends a task, answers the clarification with the same `taskId`, and stores the artifact.
- Tests: tampered card → signature check fails; illegal state transitions rejected; deadline expiry → `CancelTask`; terminal task cannot be resumed; duplicate completion handled idempotently.
- Labeled as a minimal implementation of the spec's semantics, not the official SDK or a conformance test.

**Notebook 05 — removed.** The planned live attack lab is not built (user chose to finish with Notebooks 01–04). Modules 06–07 cover the threats and contain tested defense code.

---

## 4. Real vs Illustrative Labeling

Same four-tier scheme as earlier topics: `[REAL]` (live execution and measurement), `[COMPUTED FROM REAL DATA]` (a formula applied to measured values), `[SIMULATION]` (constructed scenario, real code), `[ILLUSTRATIVE]` (not used here). Every notebook states that "real execution" is not "real production behavior": results describe this model, this hardware and these constructed tasks.

## 5. Two-Pass Authoring

Pass 1 builds and executes with `_(pending real output)_` placeholders; Pass 2 writes each Output Explanation from the executed output, quoting literal values in backticks.

## 6. Open Design Questions / Dependencies

1. Resolved: OpenAI key updated and working; Ollama approved for local agents; `Qwen2.5-1.5B-Instruct` download approved.
2. Long `gpt-4o-mini` calls at 128k take several seconds each; Notebook 01 is the slowest (~20–30 minutes).
3. **Notebook 05:** removed at the user's choice; Track 2 is Notebooks 01–04.

## Status: Approved (with the user's changes: use OpenAI, allow Ollama, add Qwen2.5-1.5B)
