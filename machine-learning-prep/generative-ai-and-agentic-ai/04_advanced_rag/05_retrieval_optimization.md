# Retrieval Optimization

Retrieval optimization refines the query representation and compresses retrieved documents to maximize generation accuracy while minimizing latency and cost.

---

## 1. Query Transformations: HyDE, Expansion & Rewriting

A raw user query is often short and semantically different from the corresponding target document's answers.

### 1. HyDE (Hypothetical Document Embeddings)
- **Workflow**: Pass the user query to an LLM to generate a zero-shot *hypothetical* response (which may contain hallucinated facts). Embed this hypothetical response and use it to query the vector database.
- **Why it works**: A hypothetical answer shares the same vocabulary, format, and structure as the target document, yielding higher retrieval recall than the raw question vector.

### 2. Query Rewriting
- **Workflow**: Parse the query (using an LLM) to remove conversational filler, resolve pronouns, and structure search terms before embedding.

### 3. Multi-Query Expansion
- **Workflow**: Ask an LLM to generate $N$ variations of the query from different perspectives. Retrieve documents for all $N$ variations, and merge the results using Reciprocal Rank Fusion (RRF).
- **Trade-off**: Improves recall but increases database query latency.

---

## 2. Bi-Encoders vs. Cross-Encoders (Reranking)

### Bi-Encoders (Retrieval)
Generate query and document embeddings independently. Similarity is measured using fast vector math (dot product).
- **Usage**: Scanning millions of documents quickly to find the top $K$ candidates (e.g. $K=100$).
- **Limitation**: No cross-attention between query and document tokens during embedding generation.

### Cross-Encoders (Reranking)
Pass the query and document *together* through the transformer encoder, allowing self-attention layers to compute token relationships across query-document boundaries.
- **Usage**: Evaluating the top $K$ candidate chunks retrieved by the bi-encoder to determine exact relevance scores.
- **Limitation**: Too slow to scan the entire database, but highly accurate for local reranking.

---

## 3. Contextual Compression & Dynamic Top-K

 stuffing the top 100 documents directly into the prompt context leads to token bloat and recall degradation ("lost-in-the-middle").
- **Contextual Compression**: Splits retrieved documents into sentences and runs a lightweight scorer to extract only the sentences that directly match the query, discarding irrelevant text.
- **Dynamic Top-K Selection**: Adjusts the number of documents retrieved ($K$) dynamically based on relevance score distributions. If only three documents have scores above a threshold, only three are passed to the prompt.

---

## 4. Context Caching Strategies

With the introduction of large context windows (1M+ tokens), RAG systems can store static documents (manuals, source code) directly in the LLM's context window. To make this cost-effective, providers support **Context Caching** (Prompt Caching).

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Prompt Context Caching Flow</div>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Cache Hit -->
        <div style="flex: 1; min-width: 220px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; text-align: center; border-top: 3px solid #059669;">
            <div style="color: #065f46; font-weight: bold; font-size: 11px;">Cache Hit (Fast Path)</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Tokens match cached prefix</div>
            <div style="background-color: #ecfdf5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 8px;">Cost: 90% discount</div>
            <div style="background-color: #ecfdf5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 4px;">TTFT: Sub-100ms</div>
        </div>
        <!-- Cache Miss -->
        <div style="flex: 1; min-width: 220px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; text-align: center; border-top: 3px solid #dc2626;">
            <div style="color: #991b1b; font-weight: bold; font-size: 11px;">Cache Miss (Slow Path)</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">New tokens or expired cache</div>
            <div style="background-color: #fef2f2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 8px;">Cost: 100% full pricing</div>
            <div style="background-color: #fef2f2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 4px;">TTFT: Multi-second compilation</div>
        </div>
    </div>
</div>

### Context Caching Price & Latency Profiles

Prompt caching reduces input token costs for matches matching a cached prefix (typically requiring minimum chunk sizes of 1024 or 32768 tokens):

| LLM Provider | Cache Minimum Block Size | Cache Hit Discount | Latency (Time to First Token) |
| :--- | :--- | :--- | :--- |
| **Anthropic (Claude 3.5)** | 1024 tokens | **$90\%$ off** ($0.30$ vs. $3.00$ per M) | $\approx 200\text{ms}$ |
| **OpenAI (GPT-4o)** | 1024 tokens | **$50\%$ off** ($1.25$ vs. $2.50$ per M) | $\approx 150\text{ms}$ |
| **DeepSeek (V3 / R1)** | 1024 tokens | **$93\%$ off** ($0.014$ vs. $0.14$ per M)| $\approx 100\text{ms}$ |

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Retrieval optimization solves query-document distribution gaps (HyDE), matches precision issues (rerankers), and context input token costs (prompt caching).
- **Why was it introduced?**
  Because raw search embeddings often fail to match document answers directly, and processing large raw context payloads is too slow and expensive.
- **What are its limitations?**
  HyDE adds LLM query latency. Cross-encoders add GPU inference latency. Caching requires structured prompt design prefixes.
- **Computational Complexity (Time & Memory)**
  - **Bi-Encoder Search**: $O(\log M)$ query traversal.
  - **Cross-Encoder Rerank**: $O(K \cdot L^2)$ transformer forward pass, where $K$ is candidates evaluated and $L$ is combined length.
- **Component Variable Denotation Legend**
  - $M$: Number of total documents.
  - $K$: Reranker candidate pool size.
  - $L$: Sequence length of combined query-document tokens.
- **Production Use Cases**
  - Prompt caching enterprise internal manuals using Claude 3.5 or DeepSeek-R1.
- **Follow-up questions interviewers ask**
  - *Why can't Cross-Encoder rerankers be pre-computed and stored in a vector index?*
  - *If a user updates their document, how do you handle prompt cache invalidation?*
