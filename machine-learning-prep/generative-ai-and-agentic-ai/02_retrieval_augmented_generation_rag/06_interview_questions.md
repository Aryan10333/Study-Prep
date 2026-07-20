# Top 20 Retrieval-Augmented Generation (RAG) Interview Questions

This flashcard guide presents 20 high-frequency interview questions asked by top tech companies (Meta, Google, OpenAI, Amazon, Cohere) covering RAG architectures, lost-in-the-middle mitigation, chunking, HyDE, RRF rank fusion, Cross-Encoders, Self-RAG, and GraphRAG.

---

### Q1: What is the "Lost in the Middle" problem in RAG, and how do you mitigate it?
- **Answer:** LLMs attend heavily to tokens placed at the very beginning and very end of long input prompts, while ignoring chunks placed in the middle ($20\% - 40\%$ recall drop). Mitigate it by:
  1. Using Cross-Encoder rerankers to select only top-3 or top-5 ultra-relevant chunks.
  2. Sorting retrieved chunks such that the most relevant documents are placed at the top and bottom of the prompt context.

---

### Q2: Why does Naive RAG fail in enterprise production environments?
- **Answer:** Naive RAG uses fixed-size single-embedding vector lookups without pre-retrieval query expansion or post-retrieval reranking. This causes low retrieval precision (injecting noisy irrelevant chunks) and low recall (missing facts due to vocabulary mismatches or short vague queries).

---

### Q3: Explain HyDE (Hypothetical Document Embeddings) and how it improves retrieval recall.
- **Answer:** HyDE prompts an LLM to generate a hypothetical answer passage first. The hypothetical passage is then embedded into vector space. Even if the hypothetical answer contains minor factual errors, its vector embedding resides inside the document embedding manifold, resulting in significantly higher cosine similarity match to true ground-truth document chunks compared to short raw user queries.

---

### Q4: How does Reciprocal Rank Fusion (RRF) work in Hybrid Search?
- **Answer:** RRF aggregates document rankings from multiple retrievers (e.g. Sparse BM25 + Dense Vector) by calculating:
  $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  By operating strictly on document **ordinal rank positions** rather than raw uncalibrated scores, RRF provides robust hybrid rank fusion without needing manual score normalization.

---

### Q5: Compare Bi-Encoders and Cross-Encoders in a Two-Stage Retrieval pipeline.
- **Answer:**
  - **Bi-Encoder:** Encodes query and document independently into fixed vectors. Fast ($O(1)$ vector lookup), used for first-stage candidate retrieval (Top-50).
  - **Cross-Encoder:** Concatenates query and document ($[CLS] \ q \ [SEP] \ d$) and processes them jointly through transformer cross-attention layers. Slow ($O(K)$ forward passes), used for second-stage high-precision reranking (Top-5).

---

### Q6: What is Parent-Child Document Chunking?
- **Answer:** Parent-Child chunking splits documents into small child chunks (e.g. 200 tokens) for high-precision vector search. However, when a child chunk matches, the retriever fetches its larger parent document (e.g. 1000 tokens) or section to pass to the LLM, ensuring rich context during generation.

---

### Q7: What is Corrective RAG (CRAG)?
- **Answer:** CRAG evaluates the relevance of retrieved documents using a lightweight evaluator model. If retrieval confidence is high, local generation proceeds; if retrieval confidence is low or ambiguous, CRAG triggers fallback routing to external web search APIs or Knowledge Graphs.

---

### Q8: What are Self-RAG reflection tokens?
- **Answer:** Self-RAG trains an LLM to output special control reflection tokens (such as `[Retrieve]`, `[IsRelevant]`, `[IsSupported]`, `[IsUseful]`) during text generation, allowing the model to dynamically decide when to fetch external documents and self-evaluate its answer groundedness.

---

### Q9: What is GraphRAG, and when should you prefer it over standard Vector RAG?
- **Answer:** GraphRAG constructs a Knowledge Graph of Entity-Relation-Entity triples from documents and builds community summaries. Prefer GraphRAG over Vector RAG for global corpus-level queries (e.g., *"What are the main themes across all 500 reports?"*) where vector similarity fails to aggregate global insights.

---

### Q10: Write a Python function for Reciprocal Rank Fusion (RRF) scoring.
```python
def rrf_score(bm25_ranks: dict, dense_ranks: dict, k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for doc in set(bm25_ranks).union(dense_ranks):
        score = 0.0
        if doc in bm25_ranks: score += 1.0 / (k + bm25_ranks[doc])
        if doc in dense_ranks: score += 1.0 / (k + dense_ranks[doc])
        scores[doc] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

### Q11: How do you prevent context window blowup when retrieving multiple long document chunks?
- **Answer:** Apply contextual compression (pruning non-essential sentences within chunks), set strict Cross-Encoder score cutoffs ($\tau \ge 0.70$), and use token compression algorithms like LLMLingua to prune low-entropy words.

---

### Q12: What is Multi-Query Expansion?
- **Answer:** Multi-Query Expansion uses an LLM to rephrase a single user query into 3–5 distinct semantic variations. Each variation executes against the retriever, and the union of retrieved documents is aggregated, increasing recall across diverse query phrasings.

---

### Q13: What is Step-Back Prompting in RAG?
- **Answer:** Step-Back Prompting asks the LLM to generate a higher-level, more abstract concept question derived from the user's specific query. Retrieving context for both the step-back abstract query and the specific query prevents retrieval failure on over-specific details.

---

### Q14: Derive the maximum possible RRF score for a document ranked #1 in both Sparse and Dense retrievers with $k=60$.
- **Answer:** 
  $$\text{RRF\_Score}(d) = \frac{1}{60 + 1} + \frac{1}{60 + 1} = \frac{2}{61} \approx \mathbf{0.032787}$$

---

### Q15: How do you evaluate a RAG pipeline in production without ground-truth human labels?
- **Answer:** Use the **RAG Triad** framework:
  1. **Context Relevance:** LLM-as-a-Judge evaluates query vs retrieved context.
  2. **Groundedness / Faithfulness:** LLM checks if generated answer claims are supported by context.
  3. **Answer Relevance:** LLM checks if final answer addresses user query.

---

### Q16: Why are exact product model numbers or legal IDs missed by Dense Vector Search?
- **Answer:** Dense embedding models compress text into dense continuous vectors where rare character strings or specific IDs (e.g. `"A100-80GB-SXM"`) get smoothed into nearby semantic concepts. Sparse BM25 keyword search is required for exact character match retrieval.

---

### Q17: How does chunk overlap prevent context loss in RAG?
- **Answer:** Sliding window overlap (e.g., 512-token chunk size with 100-token overlap) ensures that key sentences spanning across chunk boundaries are not split in half, preserving semantic continuity across adjacent chunks.

---

### Q18: What is Contextual Compression in LangChain RAG?
- **Answer:** Contextual Compression passes retrieved chunks through a fast compressor module (such as a LLM or Cross-Encoder) that extracts and returns only the specific sentences relevant to the query, stripping out irrelevant boilerplate text before prompt assembly.

---

### Q19: How do you optimize Time-To-First-Token (TTFT) in Advanced RAG pipelines?
- **Answer:** Parallelize sparse and dense retrievals using asynchronous threads (`asyncio.gather`), cache frequent query embeddings, cap candidate reranking pools ($K \le 15$), and stream LLM generation tokens immediately upon first context retrieval.

---

### Q20: How do you debug a production RAG system that answers "I do not know" to valid questions?
- **Answer:**
  1. Inspect Vector DB retrieval logs: Check if ground-truth chunks were retrieved in top-$k$.
  2. If missing from top-$k$: Fix chunking size, add HyDE/Multi-Query expansion, or implement Hybrid Search.
  3. If present in top-$k$ but generation failed: Check for "Lost in the Middle" positioning or relax overly strict system prompt instructions.
