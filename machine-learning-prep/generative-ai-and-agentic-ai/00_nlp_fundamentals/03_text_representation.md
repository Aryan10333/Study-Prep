# Module 03: Text Representation & Classical Retrieval Models

Text representation maps natural language tokens into mathematical vector spaces. This module details sparse vector formats, provides step-by-step TF-IDF calculations, explains the Feature Hashing trick, and compares keyword vs. semantic search retrieval systems.

---

## 1. Sparse Vector Formats: One-Hot, Bag of Words, and N-grams

Sparse representations construct high-dimensional vector spaces where each dimension represents a specific vocabulary word or n-gram:

- **One-Hot Encoding**: Represents each word as a binary vector containing a single `1` at the word's index.
  - *Limitation*: Captures zero semantic similarity (orthogonal vectors).
- **Bag of Words (BoW)**: Represents a document as a vector of word counts, ignoring word order.
  - *Limitation*: Long documents have larger counts, biasing classification and retrieval scoring.
- **N-grams**: Extends BoW by tracking sequences of $N$ tokens (e.g. `"not bad"` as a bigram), preserving local word order.
  - *Limitation*: Dimension scales exponentially as $O(|V|^N)$, leading to extreme data sparsity.

---

## 2. TF-IDF Derivation and Step-by-Step Hand-Calculations

Term Frequency-Inverse Document Frequency (TF-IDF) scales word counts by penalizing words that appear frequently across the entire corpus:

### Mathematical Formulation
$$\text{TF}(t, d) = f_{t, d} \quad \text{(raw count of term } t \text{ in document } d\text{)}$$

$$\text{IDF}(t, D) = \log \left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- $N = |D|$ is the total number of documents in corpus $D$.
- $\text{DF}(t)$ is the document frequency (number of documents containing term $t$).

---

### Step-by-Step Hand-Calculation on a Tiny Corpus
Let's calculate the TF-IDF vectors for a 2-document corpus:
- Document 1 ($d_1$): `"cat feline"`
- Document 2 ($d_2$): `"feline rug"`
- Query term set (Vocabulary): `["cat", "feline", "rug"]` ($N = 2$)

#### 1. Calculate Document Frequency (DF) and IDF
- `"cat"`: $\text{DF} = 1 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+1}\right) + 1 = \log(1.5) + 1 \approx 0.405 + 1 = 1.405$
- `"feline"`: $\text{DF} = 2 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+2}\right) + 1 = \log(1) + 1 = 0 + 1 = 1.000$
- `"rug"`: $\text{DF} = 1 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+1}\right) + 1 \approx 1.405$

#### 2. Compute Unnormalized TF-IDF Vectors
- **Document 1 ($d_1$) counts**: `{"cat": 1, "feline": 1, "rug": 0}`
  $$\text{Vector}_{d_1} = [1 \times 1.405, \ 1 \times 1.0, \ 0 \times 1.405] = [1.405, \ 1.0, \ 0.0]$$
- **Document 2 ($d_2$) counts**: `{"cat": 0, "feline": 1, "rug": 1}`
  $$\text{Vector}_{d_2} = [0 \times 1.405, \ 1 \times 1.0, \ 1 \times 1.405] = [0.0, \ 1.0, \ 1.405]$$

#### 3. Apply $L_2$ Normalization (Unit Length)
- Norm for $d_1$: $\|\text{Vector}_{d_1}\|_2 = \sqrt{1.405^2 + 1.0^2} \approx 1.725$
  $$\mathbf{v}_{d_1} = \left[\frac{1.405}{1.725}, \ \frac{1.0}{1.725}, \ 0.0\right] \approx [0.814, \ 0.580, \ 0.0]$$
- Norm for $d_2$: $\|\text{Vector}_{d_2}\|_2 \approx 1.725$
  $$\mathbf{v}_{d_2} \approx [0.0, \ 0.580, \ 0.814]$$

---

## 3. Cosine Similarity

![Cosine Similarity Heatmap](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/tfidf_similarity_heatmap.png)

Cosine similarity measures the angular orientation between two normalized vectors, ignoring magnitude differences:

$$\text{CosineSimilarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$

For our normalized vectors:
$$\mathbf{v}_{d_1} \cdot \mathbf{v}_{d_2} = (0.814 \times 0.0) + (0.580 \times 0.580) + (0.0 \times 0.814) = 0.3364$$
Since the vectors are pre-normalized, the cosine similarity is simply the dot product: $\approx 0.336$.

---

## 4. Feature Hashing (The Hashing Trick)

To avoid storing a large vocabulary dictionary in memory, production pipelines use Feature Hashing:

- **Mechanics**: Maps raw words to a fixed array index size $B$ using a hash function:
  $$\text{Index} = h(w) \pmod B$$
- **Sign Hash**: A second independent hash function $\xi(w) \in \{-1, +1\}$ scales token values to ensure expected collision errors average to zero:
  $$x_i = \sum_{w : h(w) \equiv i} \xi(w) \cdot \text{Count}(w)$$
- **Collision Example**: If `"purchase"` and `"buy"` both hash to index 4, the sign hash might assign $+1$ to `"purchase"` and $-1$ to `"buy"`, canceling out collision bias on average.
- **Advantages**: Zero vocabulary lookup table storage footprint; constant $O(1)$ index lookup speed.
- **Collision Trade-offs**: If $B$ is too small, collisions introduce noise, degrading classification accuracy.

---

## 5. Retrieval Models: Boolean, TF-IDF, and BM25

Retrieval systems query document collections using three primary paradigms:

1. **Boolean Retrieval**: Checks documents for binary term matches using boolean operators (`AND`, `OR`, `NOT`). Offers high precision but does not rank documents.
2. **TF-IDF Retrieval**: Ranks documents by summing the TF-IDF weights of query terms.
3. **Okapi BM25 Retrieval**: A robust retrieval algorithm that scales Term Frequency non-linearly and normalizes by document length:
   $$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{\text{TF}(q_i, D) \cdot (k_1 + 1)}{\text{TF}(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
   - **$k_1$ parameter** (typically $1.2\text{--}2.0$): Controls term frequency saturation. As Term Frequency increases, the score approaches an asymptote, preventing a single term from dominating.
   - **$b$ parameter** (typically $0.75$): Controls document length normalization. It penalizes long documents containing query terms simply due to length.

---

> [!TIP]
> **Production Insight: Sparse (BM25) vs. Dense Retrieval**
> - **Sparse (BM25)**: Excellent for exact matches (e.g. product IDs, serial numbers, specific names). Extremely cheap to host (Lucene/Elasticsearch).
> - **Dense (Embeddings)**: Captures semantic meaning (synonyms like `"cat"` and `"feline"`), but requires higher computational cost (GPU vector search) and does not handle exact matching well.
> - **Hybrid Search**: Combine both models using Reciprocal Rank Fusion (RRF) to get the benefits of both paradigms in production search engines.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Translates text documents into numeric vectors to compute similarities and retrieve relevant records.
- **Why was it introduced?**
  Introduced to score term significance relative to corpus frequencies, avoiding term density bias in documents.
- **What are its limitations?**
  Sparse vectors fail to capture semantic relationships (synonyms like `"cat"` and `"feline"` remain orthogonal).
- **Computational Complexity (Time & Memory)**
  - **TF-IDF Vectorization Time**: $O(N \cdot L)$ where $N$ is document count and $L$ is sequence length.
  - **Feature Hashing Lookup Time**: $O(1)$ index lookup.
- **Component Variable Denotation Legend**
  - $N$: Total document count.
  - $L$: Document token sequence length.
  - $B$: Number of feature hash buckets.
  - $k_1$: BM25 term frequency saturation scaling parameter.
  - $b$: BM25 document length normalization parameter.
- **Production Use Cases**
  - High-speed text retrieval index systems (Elasticsearch, BM25).
  - Memory-constrained text classification pipelines using feature hashing.
- **Follow-up questions interviewers ask**
  - *How does BM25 prevent document length bias?* (Long documents are penalized via the $b \cdot (|D|/\text{avgdl})$ term in the denominator. This reduces the score if query terms appear in long, noisy texts).
  - *Why does the sign hash function $\xi(w)$ prevent collision degradation in Feature Hashing?* (By randomly assigning $+1$ or $-1$ to each word, collisions cancel each other out on average, preventing systematic inflation of collision indices).
