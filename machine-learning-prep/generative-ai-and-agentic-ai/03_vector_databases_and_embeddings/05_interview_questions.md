# Top 15 Vector Databases & Embeddings Interview Questions

This flashcard guide presents 15 high-frequency interview questions asked by top tech companies (Meta, Google, Pinecone, Qdrant, OpenAI) covering dense vector embeddings, InfoNCE loss, Matryoshka Representation Learning (MRL), HNSW graphs, Product Quantization (PQ), and vector DB architectures.

---

### Q1: What is the difference between exact kNN and Approximate Nearest Neighbor (ANN) search?
- **Answer:** Exact kNN computes pairwise distances between the query and all $N$ database vectors ($O(N \cdot d)$ time complexity), which is too slow for production. ANN search uses index data structures (like HNSW or IVF) to trade a tiny fraction of recall ($1\% - 3\%$) for $100\text{x} - 1000\text{x}$ faster query QPS.

---

### Q2: How does Matryoshka Representation Learning (MRL) allow truncating embedding dimensions without retraining?
- **Answer:** MRL trains dense embedding models by enforcing contrastive loss simultaneously across multiple nested sub-vector dimensions (e.g., $64\text{d}, 128\text{d}, 256\text{d}, 512\text{d}, 1024\text{d}$). The most critical semantic information is concentrated in the early vector dimensions, allowing developers to slice vectors to 128d to reduce storage by $8\text{x}$ while retaining $>95\%$ benchmark accuracy.

---

### Q3: Explain HNSW (Hierarchical Navigable Small World) graph indexing.
- **Answer:** HNSW is a multi-layer graph skip-list. Top layers contain sparse long-range edges for fast macro-routing across vector space. Lower layers contain increasingly dense local graph connections for fine-grained nearest neighbor resolution, achieving $O(\log N)$ search time.

---

### Q4: How does Product Quantization (PQ) compress vector index memory footprint by up to 16x?
- **Answer:** PQ splits a $d$-dimensional vector into $m$ smaller sub-vectors. For each sub-vector subspace, k-means clustering builds a codebook of 256 centroids ($256 = 2^8 = 1\text{ byte}$). Each sub-vector is replaced by its 1-byte centroid index, compressing an FP32 vector from $d \times 4$ bytes down to $m \times 1$ bytes.

---

### Q5: Prove why sorting by Cosine Similarity, Dot Product, and $L_2$ Distance yields identical document rankings under unit normalization.
- **Answer:** For unit-normalized vectors $\|u\| = \|v\| = 1.0$:
  1. $\text{cos\_sim}(u, v) = \frac{u \cdot v}{1 \cdot 1} = u \cdot v$ (Cosine Similarity equals Dot Product).
  2. $\|u - v\|^2 = \|u\|^2 + \|v\|^2 - 2(u \cdot v) = 2 - 2(u \cdot v) = 2 - 2 \text{cos\_sim}(u, v)$.
  Because $L_2^2$ is a strictly monotonic inverse linear function of Cosine Similarity, all three metrics produce identical rank orderings.

---

### Q6: Why is Metadata Pre-Filtering preferred over Post-Filtering in multi-tenant vector databases?
- **Answer:** Post-filtering retrieves global top-$k$ vector neighbors first, then discards items that fail tenant metadata filters; if a tenant owns a small fraction of total vectors, post-filtering returns empty results. Pre-filtering restricts HNSW graph traversal exclusively to matching tenant vectors, guaranteeing $100\%$ precision and recall.

---

### Q7: Explain ColBERT Late Interaction and how it differs from standard Bi-Encoder embeddings.
- **Answer:** Bi-encoders compress an entire document into a single vector. ColBERT outputs a matrix of token-level embeddings (one 128d vector per token). At retrieval time, ColBERT computes a token-level MaxSim score ($S = \sum_{i \in Q} \max_{j \in D} (q_i \cdot d_j^T)$), providing fine-grained token interaction quality without full cross-encoder latency.

---

### Q8: What is the InfoNCE loss function used in embedding model training?
- **Answer:** InfoNCE is a categorical cross-entropy loss over normalized similarity scores:
  $$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum_i \exp(\text{sim}(q, k_i^-) / \tau)}$$
  It maximizes similarity between positive query-document pairs $(q, k^+)$ while pushing negative pairs $(q, k^-)$ apart.

---

### Q9: Write a Python function that normalizes a matrix of PyTorch embeddings to unit length ($L_2$ norm = 1.0).
```python
import torch
import torch.nn.functional as F

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=-1)
```

---

### Q10: What is Scalar Quantization (SQ8), and how does it compare to Product Quantization (PQ)?
- **Answer:** SQ8 maps each 32-bit floating point value (FP32) in a vector independently to an 8-bit integer (INT8) based on absolute min/max range scaling, achieving $4\text{x}$ compression with almost zero recall loss ($<1\%$). PQ achieves higher compression ($16\text{x}+$) by quantizing sub-vectors into centroids, but suffers higher recall loss ($3\% - 8\%$).

---

### Q11: How do you choose between Qdrant, Pinecone, and Pgvector for enterprise deployment?
- **Answer:**
  - **Pgvector:** Best if you already run PostgreSQL and need simple vector search co-located with relational data.
  - **Qdrant:** Best for self-hosted high-QPS production workloads requiring complex payload pre-filtering and memory efficiency (Rust engine).
  - **Pinecone:** Best for fully managed cloud-native serverless pipelines without infrastructure maintenance.

---

### Q12: Calculate the uncompressed FP32 RAM required to store 50,000,000 vectors of dimension 1536.
- **Answer:** 
  $$\text{Bytes} = 50,000,000 \times 1536 \times 4 \text{ bytes} = 307,200,000,000 \text{ bytes} \approx \mathbf{307.2 \text{ GB RAM}}$$

---

### Q13: What is Asymmetric Distance Computation (ADC) in Product Quantization?
- **Answer:** In ADC, the query vector is kept in full floating-point precision (unquantized), while database vectors are stored as quantized PQ codes. Distance is computed between the unquantized query and the PQ codebook centroids, improving search accuracy compared to quantizing both query and database vectors.

---

### Q14: Why can Dot Product search fail if embedding vectors are not $L_2$-normalized?
- **Answer:** Unnormalized dot product incorporates vector magnitude $\|v\|$. Unusually long document chunks with large norm values will produce artificially high dot product scores regardless of semantic relevance, distorting retrieval.

---

### Q15: How do you debug a production vector search index experiencing sudden drops in QPS?
- **Answer:**
  1. Check index build/compaction jobs: HNSW background graph optimization consumes CPU/RAM.
  2. Check memory usage: If the index size exceeds available RAM, OS page swapping to disk severely throttles QPS.
  3. Verify payload filters: Ensure metadata pre-filters are supported by payload indexes.
