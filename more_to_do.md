Curriculum for the target role (AI Engineer / Applied AI Engineer / GenAI Engineer / LLM Engineer, ~3 YOE).

Updated 2026-10-26 from current hiring and industry signals:

- **Baseline in most postings:** Python, PyTorch, AWS, Kubernetes.
- **Fastest growing:** agent engineering.
- **Scarcest skill:** fine-tuning and post-training (covered in module 02).
- **Expected but no longer a differentiator:** RAG.
- **Where the industry is moving:** from "agents" to the harness around them (context, permissions, evals, persistent state), with coding agents at the center.

---

# Done (on the site)

| # | Module |
|---|---|
| 00 | NLP Fundamentals |
| 01 | LLM Foundations |
| 02 | LLM Fine-tuning, Alignment & Training Infrastructure (folder: `02_llm_finetuning_and_alignment`) |
| 03 | Advanced RAG |
| 04 | AI Agents & Protocols |
| 05 | Prompt Engineering & Structured Generation |
| 06 | LLM Inference & Optimization |
| 07 | LLM Evaluation, Observability & Guardrails |
| 08 | GenAI System Design & LLMOps |
| 09 | Context Engineering, A2A & Agent Supply-Chain Security (study guide done; notebooks and Q&A pending) |

---

# Core Modules To Do (priority order)

## 09. Context Engineering, A2A & Agent Supply-Chain Security

Gap-filler only. Most of the basics already live in 04 (memory, durable execution, HITL, MCP auth, injection, sandboxing, approval gates), 05 (context budgets, lost in the middle, prompt injection) and 08 (trust boundaries, governance). Cross-reference those chapters instead of repeating them. Covers only what is missing:

* **Context engineering as a discipline:** what goes into the window and why, context rot (accuracy dropping as context grows, beyond position bias), prompt caching vs compaction trade-offs (cache-aware context layout), agent instruction and memory files, and harness/loop design for long-running autonomous agents.
* **A2A protocol:** agent cards, task lifecycle, agent-to-agent communication across trust boundaries, and how A2A complements MCP (both now under the Linux Foundation).
* **Agent supply-chain security:** malicious MCP servers and skills, tool poisoning and "rug pulls", MCP registries and gateways, secrets management for tools, and the OWASP Top 10 for LLM applications.
* **Hands-on:** an authenticated MCP server talking to an A2A agent pair, plus a tool-poisoning attack and its defense.

---

## 10. Cloud & Deployment for AI Systems

AWS and Kubernetes are among the most-listed requirements in AI engineer postings. Covers Docker, Kubernetes basics (deployments, autoscaling, GPU node pools), one cloud platform in depth (AWS Bedrock + SageMaker), serving on managed vs self-hosted GPUs, GPU sizing and cost, IaC basics, secrets and networking, and CI/CD for model and prompt changes.

---

## 11. Coding Agents

The main topic at the AI Engineer World's Fair 2026. Two angles: (1) using coding agents well (Claude Code, Codex): task decomposition, repo instructions, review and verification loops; (2) how they're built: repository context and code search, sandboxed execution, tool and permission design, test-driven loops, SWE-bench-style evaluation.

---

## 12. PyTorch Fluency & LLM Coding Rounds

PyTorch is the framework employers name most, and many GenAI loops include a live coding round. Restore the deep-learning notes from `temp/` and turn them into practice: implement from scratch self-attention, multi-head attention, KV cache, BPE tokenizer, top-k/top-p/beam sampling, a LoRA layer, a training loop with mixed precision, and a minimal vector search. Includes a one-page classical ML refresher (metrics, bias-variance, trees, class imbalance) for screening rounds.

---

# Framework Modules

## 13. LangGraph

The agent framework guides and postings name most. Covers graphs, state management, checkpoints, interrupts, human-in-the-loop, persistence, multi-agent workflows, deployment patterns, and production best practices.

---

## 14. OpenAI Agents SDK

Covers tools, handoffs, sessions, memory, tracing, structured outputs, guardrails, streaming, MCP integration, and multi-agent orchestration.

---

## 15. vLLM

Covers PagedAttention, continuous batching, prefix caching, scheduling, quantization support, multi-model and multi-LoRA serving, deployment on Kubernetes, profiling, and performance tuning.

---

## 16. FastAPI for AI Applications

Covers async programming, streaming responses, WebSockets, SSE, authentication, background jobs, deployment, scaling, observability, and integrating LLM applications into backend systems.

---

## 17. Langfuse (Observability)

Covers tracing, prompt management, evaluations, datasets, experiments, cost tracking, user sessions, debugging, and production monitoring of LLM applications.

---

# Optional (low demand signal; only if time allows or a target job asks)

* **Reasoning models & RL with verifiable rewards** — extends module 02 (GRPO) to o1/R1-style training and test-time compute.
* **Voice & Multimodal Agents** — real-time voice pipelines, latency budgets, VLMs, document AI. Rising trend, weak hiring evidence so far.
* **LlamaIndex** — RAG framework; less in demand than LangGraph.
* **DSPy** — automatic prompt optimization; niche.
* **Google ADK / Agno / Haystack / Semantic Kernel** — learn only if a target company uses one.
* **Milvus / Qdrant / Pinecone** — vector database deep dives.
* **SGLang / TensorRT-LLM / llama.cpp** — specialized inference engines.
* **Deep RL** — mostly research roles with few openings; GRPO/RLHF concepts in module 02 are enough.

---

# Interview Readiness (run alongside the modules)

* **2–3 capstone projects** that tie modules together, e.g. a RAG + agent app with evals, guardrails, and an MCP tool, deployed on AWS behind FastAPI with Langfuse tracing. These are your "walk me through a system you built" answers.
* **Paper list** with a one-paragraph summary and likely questions for each: Attention Is All You Need, InstructGPT, LoRA, DPO, RAG, ReAct, FlashAttention, PagedAttention (vLLM), DeepSeek-R1.
* **Behavioral prep:** STAR stories mapped to the resume, including a "confident solution that turned out wrong" story and trade-off decisions you can defend.
