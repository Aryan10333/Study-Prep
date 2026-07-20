# Module 14: Master Interview Preparation & Revision Cheat Sheet

This flashcard guide presents 30 senior-level RAG interview questions, scenario system design challenges, resume discussion points, best practices, and a 1-page Revision Cheat Sheet.

---

### Part 1: Senior RAG Interview Flashcards

#### Q1: Compare Bi-Encoder vs Cross-Encoder reranking.
- **Answer:** Bi-Encoders compute query and document vectors independently, enabling fast $O(1)$ similarity search for candidate generation. Cross-Encoders process query and document together through joint self-attention layers, providing high precision for reranking top-20 candidates.

#### Q2: How does Reciprocal Rank Fusion (RRF) merge BM25 and Vector Search results?
- **Answer:** RRF sums inverse rank positions across rank lists: $RRF(d) = \sum \frac{1}{k + r_m(d)}$ where $k=60$. It avoids score scale incompatibility between BM25 unbounded scores and Cosine bounded similarities.

#### Q3: What is Parent-Child Hierarchical Chunking?
- **Answer:** It splits text into small child chunks (128 tokens) for embedding generation and vector search, but links each child to a larger parent chunk (1024 tokens). When a child is matched, its parent chunk is returned to the LLM prompt.

#### Q4: What is the "Lost in the Middle" phenomenon in RAG prompts?
- **Answer:** Attention mechanisms recall tokens at the top (beginning) and bottom (end) of long prompts best, suffering a $20\%-40\%$ recall drop for chunks placed in the middle. Place most critical context at prompt boundaries.

#### Q5: Explain Self-RAG and reflection tokens.
- **Answer:** Self-RAG fine-tunes models to output special reflection tokens (`[Retrieve]`, `[IsRelevant]`, `[IsSupported]`, `[IsUseful]`) to dynamically decide when to retrieve context and evaluate response quality.

#### Q6: What is Corrective RAG (CRAG)?
- **Answer:** CRAG evaluates retrieval quality. If internal vector retrieval confidence is high, it generates the response; if low, it triggers a fallback web search API (Tavily/DuckDuckGo).

#### Q7: How does Product Quantization (PQ-64) compress vector index memory?
- **Answer:** PQ divides $d$-dimensional FP32 vectors into $m$ sub-vectors and quantizes each sub-vector to its nearest codebook centroid byte ($0-255$), compressing memory footprints by $96\text{x}$.

#### Q8: Explain the RAGAS Triad metrics.
- **Answer:** Context Precision (relevance of retrieved chunks), Context Recall (retrieval completeness), and Faithfulness (answer groundedness in retrieved context).

---

### Part 2: System Design Interview Challenges & Resume Discussion Points

#### System Design Challenge: Design a 100M Document Enterprise RAG Search Engine
- **Architectural Solution**:
  1. **Ingestion & Parsing**: Asynchronous worker pool using LlamaParse for OCR tables, MinHash LSH for deduplication ($J(A, B) \ge 0.85$).
  2. **Storage Layer**: Qdrant / Milvus distributed vector cluster using HNSW index ($M=16, \text{ef\_construction}=128$) sharded by tenant ID.
  3. **Retrieval Layer**: Hybrid Search (Sparse Elasticsearch BM25 + Dense Qdrant Vector) combined via Reciprocal Rank Fusion (RRF).
  4. **Precision Layer**: Cross-Encoder (`BAAI/bge-reranker-base`) reranking top-20 candidates down to top-3 context chunks.
  5. **Caching & API**: FastAPI streaming SSE response with Redis Semantic Vector Cache ($0\text{ms}$ latency cache hits).

#### Resume Pitch Point: Optimizing Enterprise RAG Latency & Costs
- **Interview Response**: *"In my previous role, I optimized an enterprise technical support RAG system by implementing Parent-Child chunking and Redis Semantic Vector Caching. This reduced P99 API latency by 45% (from 1.8s to 980ms) and cut OpenAI embedding costs by 60% through Matryoshka dimension slicing (1536d to 256d) with zero degradation in MRR retrieval quality."*

---

### Revision Cheat Sheet

```text
====================================================================================================
                               REVISION CHEAT SHEET: RETRIEVAL-AUGMENTED GENERATION
====================================================================================================
1. HYBRID SEARCH: Combine Sparse BM25 (keywords/SKUs) + Dense Vector Search via RRF.
2. CHUNKING: Prefer Parent-Child Chunking (retrieve small 128-token child, feed 1024-token parent).
3. RERANKING: Always rerank top-20 vector candidates using a Cross-Encoder for +15% precision.
4. EVALUATION: Use RAGAS Triad (Context Precision, Context Recall, Faithfulness).
5. CACHING: Use Semantic Vector Caching to serve frequent user queries in 0ms.
====================================================================================================
```