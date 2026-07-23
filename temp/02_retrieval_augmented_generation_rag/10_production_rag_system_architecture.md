# Module 10: Production RAG System Architecture

This guide details FastAPI streaming endpoints, Semantic Vector Caching (Redis / GPTCache), LangSmith / LangFuse latency tracing, rate limiting, and token cost optimization.

> **Notebook Companion**: [10_production_rag_system_architecture.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/10_production_rag_system_architecture.ipynb)

---

## 1. Enterprise Infrastructure Architecture Topology

```text
FastAPI Gateway ──► Semantic Vector Cache ──► [Cache Miss] ──► Hybrid Retriever ──► Cross-Encoder ──► LLM ──► SSE Streaming
```

### Infrastructure Components:
1. **API Layer (FastAPI)**: Asynchronous web app server handling HTTP POST requests and returning Server-Sent Events (SSE) streaming output.
2. **Semantic Vector Cache (Redis)**: Compares query embedding similarity against previous queries. If $\cos(v_q, v_{\text{cached}}) \ge 0.96$, returns cached LLM response instantly ($0\text{ms}$ latency, zero API cost).
3. **Observability & Span Tracing (LangSmith / LangFuse)**: Logs span-level latencies across embedding, retrieval, reranking, and generation.

---

## 2. Latency Budget Breakdown (Andrew Ng Style)

Let a production RAG SLA requirement be P99 Latency $\le 1,200\text{ ms}$.

### Pipeline Stage Budget Breakdown:
1. **Query Embedding Generation**: $T_{\text{embed}} = 50\text{ ms}$
2. **Hybrid Vector & BM25 Retrieval**: $T_{\text{vector}} = 80\text{ ms}$
3. **Cross-Encoder Reranking (Top-20 -> Top-3)**: $T_{\text{rerank}} = 120\text{ ms}$
4. **LLM Time-To-First-Token (TTFT)**: $T_{\text{ttft}} = 450\text{ ms}$
5. **Streaming Output Generation (100 tokens)**: $T_{\text{stream}} = 300\text{ ms}$

$$\text{Total Latency} = 50 + 80 + 120 + 450 + 300 = \mathbf{1,000\text{ ms} \le 1,200\text{ ms SLA}}$$

---

## 3. Production FastAPI & Semantic Cache Implementation

```python
import os
from dotenv import load_dotenv

load_dotenv()

cache_db = {"what is api rate limit?": "API rate limit is 100 requests per minute."}

def check_semantic_cache(query: str) -> str:
    for q, ans in cache_db.items():
        if query.lower() == q.lower():
            return f"[CACHE HIT (0ms Latency)] {ans}"
    return "[CACHE MISS -> Executing Retrieval & LLM Pipeline]"

print("=== Semantic Cache Audit ===")
print(check_semantic_cache("what is api rate limit?"))
print(check_semantic_cache("what is database host?"))
```

---

## 4. Production Failure Modes & Selection Rules

- **Cache Poisoning**: Storing inaccurate or outdated LLM answers in the semantic cache.
  - *Fix:* Set strict Time-To-Live (TTL) expiration on cache keys (e.g. 24 hours) and purge cache on index updates.