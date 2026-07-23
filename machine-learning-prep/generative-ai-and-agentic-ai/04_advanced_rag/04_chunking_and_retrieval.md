# Chunking & Retrieval Strategies

To populate a vector index, documents must be parsed and split into distinct text blocks (chunks). In production RAG systems, the strategy used to chunk documents determines the semantic density of vectors and the precision of context retrieval.

---

## 1. Traditional vs. Semantic vs. Late Chunking

### Traditional Chunking
- **Fixed-Size Chunking**: Splits text strictly on character or token counts (e.g. 512 tokens with a 10% overlap).
  - *Gotcha*: Frequently cuts sentences in half, causing loss of contextual meaning.
- **Recursive Chunking**: Splits text recursively using a hierarchical list of separators (typically `\n\n`, `\n`, ` `, `""`). It attempts to keep paragraphs and sentences intact within token constraints.

### Semantic Chunking
Calculates the cosine similarity between embeddings of consecutive sentences. A split boundary is triggered when the similarity difference falls below a calculated threshold:
$$\text{Threshold} = \mu_{\text{diff}} - k \cdot \sigma_{\text{diff}}$$
Where $\mu_{\text{diff}}$ is the mean similarity between adjacent sentences, $\sigma_{\text{diff}}$ is the standard deviation, and $k$ is a tuning multiplier (typically $0.5$ to $1.2$). This ensures text blocks contain a single coherent topic.

### Late Chunking (Jina AI / ColBERT)
Traditional chunking splits text first, then embeds each chunk individually.
- **Limitation**: Splitting removes context. A sentence containing "It was launched in 2025" lacks the context of what "It" refers to if the subject was defined in a previous chunk.
- **Late Chunking Solution**: Pass the *entire document* through the embedding model's transformer encoder layers first. Then, perform average pooling over the token embedding vectors matching the chunk boundary boundaries.
- **Benefit**: Each chunk vector contains bidirectional attention context from tokens spanning the entire document, preserving references across chunk boundaries.

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Traditional Chunking vs. Late Chunking</div>
    <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Traditional -->
        <div style="flex: 1; min-width: 220px; max-width: 260px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center; border-top: 3px solid #64748b;">
            <div style="color: #475569; font-weight: bold; font-size: 11px; text-transform: uppercase;">Traditional Chunking</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center;">1. Split text to Chunks A & B</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center;">2. Embed Chunk A [No context of B]</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center;">3. Embed Chunk B [No context of A]</div>
        </div>
        <!-- Late Chunking -->
        <div style="flex: 1; min-width: 220px; max-width: 260px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center; border-top: 3px solid #2563eb;">
            <div style="color: #1e40af; font-weight: bold; font-size: 11px; text-transform: uppercase;">Late Chunking</div>
            <div style="background-color: #eff6ff; color: #1e40af; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center; font-weight: 600;">1. Embed Entire Document</div>
            <div style="background-color: #f5f3ff; color: #5b21b6; border: 1px solid #7c3aed; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center; font-weight: 600;">2. Bidirectional Attention (A <-> B)</div>
            <div style="background-color: #ecfdf5; color: #065f46; border: 1px solid #059669; padding: 4px 8px; border-radius: 4px; font-size: 10px; width: 90%; text-align: center; font-weight: 600;">3. Pool token vectors at boundaries</div>
        </div>
    </div>
</div>

---

## 2. Decoupled Retrieval: Parent-Child & Sentence-Window

During matching, small text blocks (e.g. 100 tokens) yield dense, focused vectors that improve semantic similarity retrieval. However, generation models require broader surrounding context (e.g. 500 tokens) to yield fluent responses. Decoupled retrieval separates the chunk used for matching from the context passed to the LLM.

### Sentence Window Retrieval
- **Indexing**: Segment documents into single sentences, embed them, and index them.
- **Retrieval**: Retrieve the top matches.
- **Assembly**: Expand the context passed to the LLM by pulling $W$ sentences before and after the retrieved sentence.

### Parent-Child Retrieval
- **Indexing**: Split a document into large parent chunks (e.g. 1024 tokens), then subdivide them into small child chunks (e.g. 256 tokens). Maintain relationships in metadata.
- **Retrieval**: Retrieve child vectors.
- **Assembly**: Swap out the retrieved child chunks with their corresponding parent chunks before formatting the prompt context.

---

## 3. Hybrid Search & Reciprocal Rank Fusion (RRF)

**Hybrid Search** queries both Dense and Sparse indexes, merging results to capture semantic context and exact keywords. Since dense cosine similarity scores (range $[-1, 1]$) cannot be directly compared against sparse BM25 scores (range $[0, \infty)$), we use **Reciprocal Rank Fusion (RRF)** to combine rank outputs.

### RRF Formula
$$RRF(D) = \sum_{m \in M} \frac{1}{k + r_m(D)}$$
Where $r_m(D)$ is the rank position of document $D$ in retrieval model $m$, and $k$ is a constant smoothing parameter (typically $60$) that prevents top ranks from dominating scores.

---

### Step-by-Step RRF Score Calculation

Let's calculate the RRF score for a document $D_1$ present in both indexes:
- Dense retrieval rank: $r_{\text{dense}}(D_1) = 2$
- Sparse retrieval rank: $r_{\text{sparse}}(D_1) = 5$
- Smoothing constant: $k = 60$

$$RRF(D_1) = \frac{1}{60 + r_{\text{dense}}(D_1)} + \frac{1}{60 + r_{\text{sparse}}(D_1)}$$
$$RRF(D_1) = \frac{1}{60 + 2} + \frac{1}{60 + 5} = \frac{1}{62} + \frac{1}{65}$$
$$\text{Calculations}: \frac{1}{62} \approx 0.016129, \quad \frac{1}{65} \approx 0.015385$$
$$RRF(D_1) \approx 0.016129 + 0.015385 = 0.031514$$

---

## 4. Chunk Size & Overlap Trade-offs

| Metric | Small Chunk Size ($<256$ tokens) | Large Chunk Size ($>1024$ tokens) |
| :--- | :--- | :--- |
| **Precision@K** | High (retrieved text matches query intent precisely) | Low (retrieved text contains irrelevant information) |
| **Recall@K** | Low (broader context is omitted) | High (surrounding details are captured) |
| **Context Window Efficiency**| High (no token waste) | Low (risk of exceeding context windows) |
| **Overlap Size** | Typically 20–50 tokens (preserves context transition boundaries) | Typically 100–200 tokens (helps connect long text blocks) |

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Advanced chunking and RRF solve the context-loss and score-calibration problems, enabling dense and sparse search rankings to be combined cleanly.
- **Why was it introduced?**
  To prevent documents from being cut in half, and to avoid score mismatches during hybrid searches.
- **What are its limitations?**
  Late chunking increases embedding API latency. Decoupled retrieval patterns require maintaining complex relational metadata caches.
- **Computational Complexity (Time & Memory)**
  - **RRF Merge Complexity**: $O(K_d + K_s)$ where $K_d, K_s$ are candidate rank set sizes.
  - **Semantic Chunking**: $O(S \cdot d)$ cosine similarities between adjacent sentence vectors.
- **Component Variable Denotation Legend**
  - $k$: Smoothing constant for RRF scores.
  - $r_m(D)$: Rank of document $D$ in index model $m$.
  - $S$: Total number of sentences in the source document.
- **Production Use Cases**
  - Querying technical documentations using Hybrid Dense+BM25 with RRF.
- **Follow-up questions interviewers ask**
  - *Why is a smoothing constant $k=60$ used in RRF instead of simple rank sums?*
  - *How does late chunking preserve cross-sentence context in long documents?*
