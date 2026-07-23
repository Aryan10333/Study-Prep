# Module 12: Production Failure Modes & Debugging Playbook

This guide details 10 production failure modes (Retrieval Failures, Poor Chunking, Missing Context, Duplicate Chunks, Incorrect Reranking, Context Overflow, Stale Index, High Latency, High Costs, Hallucinations) and presents a 4-Step RCA Debugging Playbook with automated Pytest regression suites.

> **Notebook Companion**: [12_production_failure_modes_debugging.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/12_production_failure_modes_debugging.ipynb)

---

## 1. 10 Production Failure Modes & Remediation Matrix

```text
Failure Mode           Root Cause                              Target Remediation Strategy
----------------------------------------------------------------------------------------------------------------------
1. Retrieval Failure   Query keywords absent in vector space   Deploy Hybrid Search (BM25 + Dense Vector)
2. Poor Chunking       Sentences broken across chunk boundaries Switch to Parent-Child or Semantic Chunking
3. Missing Context     Top-k value set too low (k=2)          Increase k=20 and apply Cross-Encoder Reranker
4. Duplicate Chunks    Redundant documents indexed             Run MinHash LSH deduplication during ingestion
5. Incorrect Ranking   Bi-Encoder similarity score distortion  Rerank candidate chunks using Cross-Encoder
6. Context Overflow    Retrieved chunks exceed context limit  Apply LLMLingua perplexity context compression
7. Stale Knowledge     Vector index not updated on file edit  Deploy incremental webhook index upserts
8. High Latency        Un-cached repeated LLM queries         Deploy Redis Semantic Vector Caching
9. High API Costs      Large prompts with un-cached system rules Implement System Prompt KV Cache Reuse
10. Hallucinations     LLM generating ungrounded facts        Enforce strict negative constraints + RAGAS audit
```

---

## 2. Root Cause Analysis (RCA) 4-Step Debugging Playbook

When a production RAG query fails, engineers follow this systematic 4-step triage sequence:

```text
[User Query Failure] ──► 1. Audit Ingestion ──► 2. Audit Vector Search ──► 3. Audit Reranker ──► 4. Audit Prompt Context
```

1. **Step 1: Audit Ingestion**: Check raw document parsing in DB. Did OCR miss the relevant table or page?
2. **Step 2: Audit Vector Search**: Inspect top-k retrieved chunks before reranking. Is the target document chunk returned in top-50 candidates? If not, fix chunking or deploy Hybrid Search.
3. **Step 3: Audit Reranker**: Inspect top-3 chunks after reranking. Did the Cross-Encoder demote the correct chunk? If so, fine-tune Cross-Encoder logits.
4. **Step 4: Audit Prompt Context**: Inspect assembled prompt string. Did the LLM ignore the retrieved chunk due to "Lost in the Middle" attention degradation?

---

## 3. Automated Pytest RAG Regression Assertion Suite

```python
def test_rag_retrieval_integrity():
    retrieved_chunks = ["Microservice A uses gRPC over TLS 1.3 for internal RPC."]
    generated_answer = "Microservice A uses gRPC over TLS 1.3."
    
    # Assertion 1: Candidate chunks non-empty
    assert len(retrieved_chunks) > 0, "RAG Regression Error: Zero chunks retrieved!"
    
    # Assertion 2: Critical factual token present in answer
    assert "TLS 1.3" in generated_answer, "RAG Regression Error: Fact missing in answer!"
    
    print("=== Pytest Production RAG Assertions Passed (2/2) ===")

test_rag_retrieval_integrity()
```