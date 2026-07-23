# Best Practices, Failure Modes & Decision Frameworks

Designing production RAG systems requires anticipating structural failure patterns and navigating architectural decision trade-offs based on cost, scale, latency, and accuracy.

---

## 1. Common Failure Modes & Debugging Checklists

### 1. The "Lost-in-the-Middle" Phenomenon
- **Definition**: LLMs are best at retrieving information from the beginning and end of long input prompts. Chunks placed in the middle of long contexts are often ignored.
- **Diagnostics**: Graph recall accuracy against chunk position indices.
- **Resolution**: Use Cross-Encoder rerankers to compress candidate pools, keeping context payloads small and relevant.

### 2. Retrieval Drift
- **Definition**: Over time, changes in terminology, naming conventions, or product models cause user query vectors to miss older database documents.
- **Resolution**: Implement hybrid search (BM25 + Dense) and continuous index auditing.

### 3. Context Window Saturation
- **Definition**: Formatting high counts of candidate documents causes token limit saturation, increasing generation costs and causing generation truncation.
- **Resolution**: Enforce dynamic top-K selection based on score distance distributions.

---

## 2. Decision Frameworks for Production GenAI

### Framework A: Adaptation Strategies Selection

```
                         Adaptation Requirement
                                   |
         +-------------------------+-------------------------+
         |                                                   |
      Factual/Grounding                              Behavior/Style/Format
         |                                                   |
   RAG Indexing                                      Supervised Fine-Tuning (SFT)
```

- **RAG**: Choose when you need to ground generation in verifiable, real-time files, retrieve specific codes, or maintain data lineage.
- **Fine-Tuning (SFT)**: Choose when you need to enforce structured output formatting, match a strict persona style, or adapt to hardware VRAM limitations.
- **Prompt Engineering**: Choose for fast prototyping, simple formatting, or low-volume task routing.

---

### Framework B: Long-Context Prompts vs. Vector RAG

With Claude 3.5 and Gemini Pro supporting 1M+ contexts, stuffing whole documentation files directly is a valid approach.

| Adaptability Dimension | Long-Context Prompt (Full stuffing) | Vector-Based RAG |
| :--- | :--- | :--- |
| **Operational Setup** | Low (no database, no indexing pipelines) | High (Vector DB, ingestion sync, chunking) |
| **Retrieval Accuracy** | High (captures cross-document relations) | Variable (depends on search recall) |
| **Query Cost** | Very High (paying for full document text per call) | Low (paying only for matching top-K chunks) |
| **Query Latency** | High (seconds to compile full inputs) | Low (fast execution) |

*Production Choice*: Use **Long-Context** for low-volume complex queries (e.g. legal document synthesis). Use **Vector RAG** for high-volume, low-latency, and cost-constrained production APIs.

---

### Framework C: GraphRAG vs. Traditional Vector RAG
- **GraphRAG**: Choose when queries require global synthesis over entire collections (e.g. "What are the common research trends?").
- **Vector RAG**: Choose for query-centric lookups (e.g. "What was Intel's Q3 revenue?").

---

### Framework D: Passive RAG vs. MCP Active Tool-Calling
- **Passive RAG**: Choose for static documentation search (FAQs, manuals).
- **MCP Active Tool-Calling**: Choose when the agent needs to query live databases, execute remote APIs, or dynamically choose search strategies based on intermediate responses.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Decision frameworks and failure playbooks prevent developers from implementing incorrect architectures, avoiding system rewrites.
- **Why was it introduced?**
  To guide system designers through balancing the trade-offs of cost, latency, complexity, and accuracy.
- **What are its limitations?**
  Heuristic decisions must be validated against continuous offline regression evaluation suites.
- **Production Use Cases**
  - Selecting RAG + Prompt Caching to balance cost-accuracy budgets for enterprise customer support agents.
- **Follow-up questions interviewers ask**
  - *If your system encounters the Lost-in-the-Middle phenomenon, why would increasing context windows fail to resolve it?*
  - *Under what pricing constraints does Long-Context prompt caching become cheaper than Vector RAG?*
