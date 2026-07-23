# Embeddings & Vector Search Mechanics

Vector search structures semantic representations into multidimensional spaces, enabling models to retrieve documents based on conceptual overlap rather than keyword matching.

---

## 1. Dense vs. Sparse Retrieval (BM25 / SPLADE)

### Dense Retrieval (Bi-Encoders)
Dense embeddings project sequences to continuous high-dimensional vector spaces (e.g. 768 or 1536 float values).
- **Core Strengths**: Exceptional mapping of synonyms, semantics, and high-level conceptual questions (e.g. matching "medical treatment" with "therapies").
- **Limitations**: Struggles with rare alphanumeric IDs, serial codes, names, or exact SKU strings.

### Sparse Retrieval (BM25 / SPLADE)
Sparse vectors map tokens to large vocabulary indices (size $|V| \approx 30,000+$) containing mostly zero values.
- **BM25**: A bag-of-words keyword frequency-based scoring index. Calculates score matching using term frequency (TF) and inverse document frequency (IDF):
  $$\text{score}_{\text{BM25}}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  Where $f(q_i, D)$ is the term frequency, $|D|$ is the document length, and $\text{avgdl}$ is the average document length.
- **SPLADE**: Uses deep learning to generate sparse query expansions, learning which non-present vocabulary tokens should be activated to represent the document.
- **Comparison**: BM25 and SPLADE outperform dense embeddings on queries requiring exact-match lookups like part numbers or specific codes.

---

## 2. Distance Metrics: Math & Worked Hand-Calculation

During retrieval, vector databases measure the similarity of the query vector $\mathbf{q}$ and candidate document vector $\mathbf{d}$.

### Similarity Formulas

#### 1. Dot Product (Inner Product)
$$\langle\mathbf{q}, \mathbf{d}\rangle = \sum_{i=1}^d q_i d_i$$

#### 2. Cosine Similarity
$$\text{Cosine}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^d q_i d_i}{\sqrt{\sum_{i=1}^d q_i^2} \sqrt{\sum_{i=1}^d d_i^2}}$$

#### 3. Euclidean ($L_2$) Distance
$$L_2(\mathbf{q}, \mathbf{d}) = \sqrt{\sum_{i=1}^d (q_i - d_i)^2}$$

> [!NOTE]
> **Dot Product Identical to Cosine Similarity**:
> If the embedding vectors are **unit-normalized** (i.e. $\|\mathbf{q}\|_2 = 1$ and $\|\mathbf{d}\|_2 = 1$), the denominator of Cosine Similarity is $1 \cdot 1 = 1$, making Cosine Similarity mathematically identical to Dot Product. Dot product is highly preferred in production because it avoids calculating expensive square-root norms.

---

### Step-by-Step Hand-Calculation

Let's calculate distance metrics on a tiny 2D sample space:
- Query vector: $\mathbf{q} = [0.8, 0.6]$ (Unit length: $\sqrt{0.8^2 + 0.6^2} = 1.0$)
- Document vector: $\mathbf{d} = [0.5, 0.866]$ (Unit length: $\sqrt{0.5^2 + 0.866^2} \approx 1.0$)

#### 1. Dot Product
$$\mathbf{q} \cdot \mathbf{d} = (0.8 \times 0.5) + (0.6 \times 0.866) = 0.4 + 0.5196 = 0.9196$$

#### 2. Cosine Similarity
$$\text{Cosine}(\mathbf{q}, \mathbf{d}) = \frac{0.9196}{1.0 \times 1.0} = 0.9196$$

#### 3. Euclidean ($L_2$) Distance
$$L_2^2 = (0.8 - 0.5)^2 + (0.6 - 0.866)^2 = (0.3)^2 + (-0.266)^2 = 0.09 + 0.070756 = 0.160756$$
$$L_2 = \sqrt{0.160756} \approx 0.4009$$

---

## 3. Approximate Nearest Neighbor (ANN) Indexing

Performing exact flat search over $10^7$ high-dimensional vectors requires $O(M \cdot d)$ computations, which is too slow for real-time applications. Instead, databases construct indices to trade recall accuracy for sub-second lookup latency.

### 1. HNSW (Hierarchical Navigable Small World)
HNSW builds a multi-layer graph index. The top layers contain long-range highway links (fast traversal), while the bottom layer contains short-range, dense localized nodes.
- **$M$ (Max Connections)**: Controls the maximum number of bidirectional link connections per node. Larger $M$ values improve recall on high-dimensional vectors but increase index file sizes.
- **$ef\_construction$ (Exploration Depth)**: Number of neighbors evaluated during index build. Larger values yield higher retrieval accuracy but increase build time.

### 2. IVF (Inverted File Index)
IVF divides the vector space into $C$ distinct Voronoi cells using K-Means clustering. During retrieval, the query vector is compared against cluster centroids. Only vectors within the closest $n_{probe}$ clusters are searched.
- **$n_{probe}$**: Trade-off parameter. Higher values query more clusters, improving recall while increasing latency.

### 3. Product Quantization (PQ) & SQ8
PQ splits vectors into $m$ smaller sub-vectors, runs K-Means on each sub-space to generate codebooks, and replaces vectors with quantized cluster centroid byte-IDs.

#### Worked Memory Savings Math:
Let's profile a database containing $M = 10,000,000$ (10M) vectors of hidden dimension $d = 768$.

1. **Standard FP32 Index (No Compression)**:
   $$\text{Size} = 10,000,000 \times 768 \times 4\text{ bytes} = 30,720,000,000\text{ bytes} \approx 30.72\text{ GB}$$
2. **SQ8 Quantization (Scalar Quantization to Int8)**:
   Each FP32 float (4 bytes) is mapped to an 8-bit integer (1 byte).
   $$\text{Size} = 10,000,000 \times 768 \times 1\text{ byte} = 7,680,000,000\text{ bytes} \approx 7.68\text{ GB}$$
   *VRAM Reduction*: $75\%$ savings.
3. **Product Quantization (PQ - $m=96$ sub-vectors, 8-bit centroids)**:
   Each 768-dim vector is divided into $m=96$ sub-vectors of size $8$ ($768/96 = 8$). Each sub-vector is represented by a single centroid byte ID.
   $$\text{Size} = 10,000,000 \times 96\text{ bytes} = 960,000,000\text{ bytes} \approx 0.96\text{ GB}$$
   *VRAM Reduction*: $\approx 96.8\%$ savings (enabling a 10M vector index to easily fit in $<1\text{ GB}$ VRAM).

---

## 4. Metadata Filtering Mechanics

Enterprise RAG queries must incorporate filters (e.g., retrieving only docs where `tenant_id == 'Google'` and `creation_year >= 2025`).

### 1. Pre-filtering
Filters are applied *before* the vector search. The database creates a subset of document IDs matching the query and runs flat vector matches against them.
- **Drawback**: Bypasses ANN index performance. If the filtered subset is large, query execution reverts to slow flat search.

### 2. Post-filtering
ANN search retrieves the top $K$ semantic documents first, then discards records that do not match metadata filters.
- **Drawback**: Risk of **Recall Collapse**. If the filter matches are rare, filtering discards most retrieved records, returning fewer than $K$ (or even $0$) results.

### 3. Single-Stage Payload Filtering (Modern)
Integrated filtering. While traversing the HNSW graph layers, nodes are checked against the metadata condition before being added to the search path.
- **Strength**: Retains optimal ANN search speed while guaranteeing that exactly $K$ valid documents are returned.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  ANN indexes and quantization solve the memory-speed bottleneck, allowing sub-second searches over millions of high-dimensional vectors.
- **Why was it introduced?**
  Because flat searches require scanning every vector in memory, which is too slow for production.
- **What are its limitations?**
  Quantization degrades vector fidelity, introducing semantic retrieval recall losses.
- **Computational Complexity (Time & Memory)**
  - **HNSW Search Time**: $O(\log M)$, scaling logarithmically.
  - **IVF Search Time**: $O(\frac{n_{probe}}{C} \cdot M \cdot d)$, proportional to cluster probes.
- **Component Variable Denotation Legend**
  - $M$: Number of database vectors.
  - $d$: Vector dimension size.
  - $C$: IVF cluster centroids.
  - $n_{probe}$: Closest clusters inspected.
- **Production Use Cases**
  - Storing customer support vectors in pgvector or Qdrant with Single-Stage payload filters.
- **Follow-up questions interviewers ask**
  - *If your database updates frequently, why would HNSW be more difficult to maintain than IVF?*
  - *Under what conditions does Product Quantization yield poor recall scores?*
