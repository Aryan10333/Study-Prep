# Module 07: Advanced & Agentic RAG Paradigms

This guide details Multi-hop Retrieval, Self-RAG reflection tokens, Corrective RAG (CRAG) web search fallbacks, Adaptive RAG routing, GraphRAG, Fusion-in-Decoder, RAPTOR tree-organized retrieval, and Agentic state machines.

> **Notebook Companion**: [07_advanced_and_agentic_rag_paradigms.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/07_advanced_and_agentic_rag_paradigms.ipynb)

---

## 1. Advanced RAG Paradigms Taxonomy

```text
Paradigm         Mechanism                                    Primary Advantage                Best Suited For
----------------------------------------------------------------------------------------------------------------------
Self-RAG        Generates reflection tokens ([Retrieve], [IsRelevant]) Self-evaluates retrieval need  High-precision QA
Corrective RAG  Evaluates retrieval score; falls back to Web Search Automatically corrects bad context Low-confidence search
Adaptive RAG    Router LLM selects zero-shot vs RAG vs Web Search Optimizes latency & API cost Complex dynamic queries
GraphRAG        Constructs Knowledge Graph (Entities & Relations) Discovers multi-hop relationships Enterprise knowledge
RAPTOR          Recursive tree summarization of document clusters High-level abstract summarization Massive book corpora
Fusion-in-Decoder Concatenates passages in decoder self-attention Handles dozens of documents   Multi-document QA
```

---

## 2. Self-RAG Special Reflection Tokens

Self-RAG fine-tunes language models to output four types of special reflection tokens:
1. `[Retrieve]`: Decides whether to invoke vector retrieval (`[Yes]`, `[No]`, `[Continue]`).
2. `[IsRelevant]`: Evaluates if retrieved passage is relevant (`[Relevant]`, `[Irrelevant]`).
3. `[IsSupported]`: Evaluates if generated answer is grounded in passage (`[Fully supported]`, `[Partially supported]`, `[No support]`).
4. `[IsUseful]`: Evaluates overall answer usefulness (`[5]`, `[4]`, `[3]`, `[2]`, `[1]`).

---

## 3. Corrective RAG (CRAG) Decision Architecture

Corrective RAG (CRAG) evaluates the relevance of retrieved vector chunks before generation. If the retrieval confidence score falls below a threshold $\tau_{\min}$, CRAG triggers an automated fallback to a Web Search API (Tavily/DuckDuckGo).

```text
Query ──► Vector Search ──► Relevance Evaluation Score (S) ──► S >= 0.8 ──► LLM Synthesis
                                   │
                                   ▼ S < 0.5
                      [Tavily Web Search API Fallback]
```

---

## 4. GraphRAG & Knowledge Graph Traversal

GraphRAG combines vector embeddings with Knowledge Graphs:
1. **Entity & Relation Extraction**: Uses LLM to extract subject-predicate-object triples `(Entity A, Relationship, Entity B)` from raw text.
2. **Community Detection (Leiden Algorithm)**: Clusters related nodes into hierarchical communities and generates LLM summaries per community.
3. **Multi-Hop Traversal**: Executes graph traversal queries to answer global questions across disparate documents.

---

## 5. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

RAPTOR constructs a hierarchical tree of document summaries:
1. **Leaf Chunks**: Raw document text chunks at bottom level.
2. **Clustering & Summarization**: Clusters leaf embeddings using Gaussian Mixture Models (GMM) and generates an abstract summary per cluster using an LLM.
3. **Recursive Tree Building**: Repeats clustering up the hierarchy, enabling simultaneous high-level thematic queries and fine-grained vector retrieval.

---

## 6. Production LangChain Code Implementation

```python
def crag_decision_router(query: str, retrieved_docs: list[str]) -> str:
    has_keywords = any("Nvidia" in d for d in retrieved_docs)
    return "GENERATE_ANSWER" if has_keywords else "WEB_SEARCH_FALLBACK"

query = "What is Nvidia Q4 revenue forecast?"
retrieved = ["Intel CPUs released.", "AMD market share growth."]
action = crag_decision_router(query, retrieved)

print("=== Corrective RAG (CRAG) Action Triggered ===")
print(f"Action: {action} (Internal context irrelevant -> Triggering Tavily Web Search)")
```