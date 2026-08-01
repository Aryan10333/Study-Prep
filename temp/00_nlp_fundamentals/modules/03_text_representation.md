# Module 03: Text Representation & Classical Retrieval Models

Text representation maps natural language tokens into mathematical vector spaces. This module details sparse vector formats, vector space models, TF-IDF derivations, the Feature Hashing trick, and classical keyword-based retrieval systems (Boolean, BM25).

---

## 1. Vector Spaces & Sparse Representations

In NLP, the Vector Space Model represents text documents as vectors in a high-dimensional continuous space.

### Document-Term Matrix (DTM) from First Principles
A Document-Term Matrix is a mathematical representation of a corpus where:
- **Rows** represent individual documents ($D_1, D_2, \dots, D_N$).
- **Columns** represent unique words in the vocabulary ($w_1, w_2, \dots, w_{|V|}$).
- **Cell Value $X_{i,j}$** represents the weight (e.g. raw count, frequency, or TF-IDF score) of word $w_j$ in document $D_i$.

The matrix shape is $N \times |V|$, where $N$ is the number of documents and $|V|$ is vocabulary size. Because any single document uses only a tiny fraction of the vocabulary, most cell values are 0, making this a **sparse matrix**.

### Vocabulary Building & Indexing
To build a vector representation, we construct a vocabulary lookup dictionary during training. Each unique word in the cleaned corpus is assigned a unique index dimension:
```python
vocab = {"cat": 0, "feline": 1, "rug": 2}
```
At inference time, a new text string is parsed using this dictionary:
- `"cat rug"` $\rightarrow$ Token counts: `{"cat": 1, "rug": 1}` $\rightarrow$ Mapped index vector: `[1, 0, 1]` (length $|V| = 3$).

### Sparse Formats: One-Hot, Bag of Words, and N-grams
- **One-Hot Encoding:** Represents each isolated word as a binary vector of size $|V|$ containing a single `1` at the word's index.
  - *Bottleneck:* The vectors are orthogonal; the dot product between any two different words is 0, capturing zero semantic similarity (e.g., `"cat"` and `"feline"` are treated as unrelated as `"cat"` and `"microchip"`).
- **Bag of Words (BoW):** Represents a document as a vector of word frequencies, ignoring sequence order.
  - *Bottleneck:* Long documents contain higher counts purely due to their length, biasing classification and retrieval models.
- **N-grams:** Preserves local word order by tracking contiguous sequences of $N$ tokens (e.g. `"not bad"` as a bigram).
  - *Bottleneck:* The dimensionality scales exponentially as $O(|V|^N)$ ($100,000^2 = 10^{10}$ columns for bigrams), leading to extreme data sparsity and memory exhaustion.

---

## 2. Vector Normalization: L2 Norm & Cosine Similarity

To prevent document length from biasing modeling and retrieval, vectors must be normalized.

### Geometry of L2 Normalization
The $L_2$ norm (Euclidean norm) of a vector $\mathbf{v}$ is its physical length in Euclidean space:
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{|V|} v_i^2}$$
Dividing a vector by its $L_2$ norm scales its length to exactly 1:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
Geometrically, this projects all document vectors onto a **unit hypersphere** of radius 1. Under this projection, documents with identical word proportions but different lengths map to the exact same coordinate on the hypersphere, eliminating document length bias.

### Cosine Similarity vs. Euclidean Distance Length Bias
- **Euclidean Distance Length Bias:** Euclidean distance measures the straight-line distance between vector coordinates:
  $$d_{\text{Euclidean}}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum (a_i - b_i)^2}$$
  If Document A is `"cat feline"` and Document B is `"cat feline"` repeated 100 times, their raw count vectors are $[1, 1, 0]$ and $[100, 100, 0]$. Even though they share the identical word distribution, their Euclidean distance is massive ($\approx 140$). This makes Euclidean distance unsuitable for variable-length texts.
- **Cosine Similarity:** Measures the cosine of the angle $\theta$ between two vectors, ignoring magnitude:
  $$\text{CosineSimilarity}(\mathbf{a}, \mathbf{b}) = \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$
  Because the vectors are normalized, the angle $\theta = 0 \rightarrow \cos(0) = 1.0$, indicating perfect similarity regardless of word density.

---

## 3. TF-IDF Derivation and Step-by-Step Hand-Calculations

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
- `"cat"`: $\text{DF} = 1 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+1}\right) + 1 = \log(1.5) + 1 \approx 0.4055 + 1 = 1.4055$
- `"feline"`: $\text{DF} = 2 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+2}\right) + 1 = \log(1) + 1 = 0 + 1 = 1.0000$
- `"rug"`: $\text{DF} = 1 \rightarrow \text{IDF} = \log\left(\frac{1+2}{1+1}\right) + 1 \approx 1.4055$

#### 2. Compute Unnormalized TF-IDF Vectors
- **Document 1 ($d_1$) counts**: `{"cat": 1, "feline": 1, "rug": 0}`
  $$\text{Vector}_{d_1} = [1 \times 1.4055, \ 1 \times 1.0, \ 0 \times 1.4055] = [1.4055, \ 1.0, \ 0.0]$$
- **Document 2 ($d_2$) counts**: `{"cat": 0, "feline": 1, "rug": 1}`
  $$\text{Vector}_{d_2} = [0 \times 1.4055, \ 1 \times 1.0, \ 1 \times 1.4055] = [0.0, \ 1.0, \ 1.4055]$$

#### 3. Apply $L_2$ Normalization (Unit Length)
- Norm for $d_1$: $\|\text{Vector}_{d_1}\|_2 = \sqrt{1.4055^2 + 1.0^2} = \sqrt{1.9754 + 1.0} \approx 1.7249$
  $$\mathbf{v}_{d_1} = \left[\frac{1.4055}{1.7249}, \ \frac{1.0}{1.7249}, \ 0.0\right] \approx [0.8148, \ 0.5800, \ 0.0]$$
- Norm for $d_2$: $\|\text{Vector}_{d_2}\|_2 \approx 1.7249$
  $$\mathbf{v}_{d_2} \approx [0.0, \ 0.5800, \ 0.8148]$$

#### 4. Cosine Similarity Calculation
Since the vectors are pre-normalized, the cosine similarity is the dot product:
$$\mathbf{v}_{d_1} \cdot \mathbf{v}_{d_2} = (0.8148 \times 0.0) + (0.5800 \times 0.5800) + (0.0 \times 0.8148) = 0.3364$$

##### Findings & Interpretation:
The cosine similarity score is $0.3364$. This represents moderate similarity, driven entirely by the shared token `"feline"`. The words `"cat"` and `"rug"` remain completely orthogonal, illustrating the string-matching limits of classical sparse representations.

---

### Python Code Integration

The following Python code uses Scikit-Learn's `TfidfVectorizer` (with default settings matching the formula above) to verify the hand calculations:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "cat feline",
    "feline rug"
]

# Initialize vectorizer (smooth_idf=True, sublinear_tf=False matches hand calculations)
vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, sublinear_tf=False)
tfidf_matrix = vectorizer.fit_transform(corpus).toarray()

# Extract vocabulary order
vocab = vectorizer.vocabulary_
print("Vocabulary mapping:", vocab)

print("\nCompiled TF-IDF Vectors:")
for doc, vec in zip(corpus, tfidf_matrix):
    print(f"Doc: '{doc}' -> Vector: {np.round(vec, 4)}")

# Calculate Cosine Similarity via dot product
similarity = np.dot(tfidf_matrix[0], tfidf_matrix[1])
print(f"\nComputed Cosine Similarity: {similarity:.4f}")
```

---

## 4. Okapi BM25 Retrieval Model

Okapi BM25 is a robust classical retrieval algorithm that evaluates term significance by scaling Term Frequency non-linearly and normalizing for document length.

### Mathematical Formulation
$$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{\text{TF}(q_i, D) \cdot (k_1 + 1)}{\text{TF}(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $\text{TF}(q_i, D)$ is the frequency of query term $q_i$ in document $D$.
- $|D|$ is the length of document $D$ in tokens, and $\text{avgdl}$ is the average document length across the corpus.
- $k_1$ is the term frequency saturation scaling parameter (typically $1.2\text{--}2.0$).
- $b$ is the document length normalization scaling parameter (typically $0.75$).

---

### Step-by-Step Hand-Calculation:
Consider a corpus of two documents:
- Document 1 ($D_1$): `"cat feline"` ($|D_1| = 2$)
- Document 2 ($D_2$): `"feline rug garden cat"` ($|D_2| = 4$)
- Average document length: $\text{avgdl} = \frac{2 + 4}{2} = 3$
- Let the query be $Q = \{\text{"cat"}\}$.
- Let the computed IDF value for `"cat"` be $\text{IDF}(\text{"cat"}) = 1.40$.
- Hyperparameters: $k_1 = 1.2$, $b = 0.75$.

#### 1. Calculate Score for Document 1 ($D_1$):
- Term Frequency: $\text{TF}(\text{"cat"}, D_1) = 1$
- Length Ratio: $\frac{|D_1|}{\text{avgdl}} = \frac{2}{3} \approx 0.6667$
- Denominator Factor:
  $$d_{\text{factor}} = 1 + 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot 0.6667\right) = 1 + 1.2 \cdot \left(0.25 + 0.50\right) = 1 + 1.2 \cdot 0.75 = 1 + 0.9 = 1.9$$
- Score:
  $$\text{Score}(D_1, Q) = 1.40 \cdot \frac{1 \cdot (1.2 + 1)}{1 + 1.9} = 1.40 \cdot \frac{2.2}{2.9} \approx 1.40 \cdot 0.7586 \approx 1.0620$$

#### 2. Calculate Score for Document 2 ($D_2$):
- Term Frequency: $\text{TF}(\text{"cat"}, D_2) = 1$
- Length Ratio: $\frac{|D_2|}{\text{avgdl}} = \frac{4}{3} \approx 1.3333$
- Denominator Factor:
  $$d_{\text{factor}} = 1 + 1.2 \cdot \left(1 - 0.75 + 0.75 \cdot 1.3333\right) = 1 + 1.2 \cdot \left(0.25 + 1.0\right) = 1 + 1.2 \cdot 1.25 = 1 + 1.5 = 2.5$$
- Score:
  $$\text{Score}(D_2, Q) = 1.40 \cdot \frac{1 \cdot 2.2}{1 + 2.5} = 1.40 \cdot \frac{2.2}{3.5} \approx 1.40 \cdot 0.6286 \approx 0.8800$$

##### Findings & Interpretation:
- $\text{Score}(D_1, Q) \approx 1.0620$
- $\text{Score}(D_2, Q) \approx 0.8800$

Even though both documents mention the query term `"cat"` exactly once, Document 1 scores higher. This occurs because Document 1 is shorter ($2$ tokens) than the average length ($3$), while Document 2 is longer ($4$ tokens). The length normalization parameter $b = 0.75$ penalizes Document 2, assuming that the presence of the word `"cat"` in a longer document is less significant because it is diluted by surrounding words.

---

## 5. Feature Hashing (The Hashing Trick)

To avoid storing a large vocabulary dictionary in memory, production pipelines use Feature Hashing:

- **Mechanics:** Maps raw words to a fixed array index size $B$ using a hash function:
  $$\text{Index} = h(w) \pmod B$$
- **Sign Hash:** A second independent hash function $\xi(w) \in \{-1, +1\}$ scales token values to ensure expected collision errors average to zero:
  $$x_i = \sum_{w : h(w) \equiv i} \xi(w) \cdot \text{Count}(w)$$
- **Collision Example:** If `"purchase"` and `"buy"` both hash to index 4, the sign hash might assign $+1$ to `"purchase"` and $-1$ to `"buy"`, canceling out collision bias on average.
- **Advantages:** Zero vocabulary lookup table storage footprint; constant $O(1)$ index lookup speed.
- **Collision Trade-offs:** If $B$ is too small, collisions introduce noise, degrading classification accuracy.

---

> [!TIP]
> **Production Insight: Sparse (BM25) vs. Dense Retrieval**
> - **Sparse (BM25):** Excellent for exact matches (e.g. product IDs, serial numbers, specific names). Extremely cheap to host (Lucene/Elasticsearch).
> - **Dense (Embeddings):** Captures semantic meaning (synonyms like `"cat"` and `"feline"`), but requires higher computational cost (GPU vector search) and does not handle exact matching well.
> - **Hybrid Search:** Combine both models using Reciprocal Rank Fusion (RRF) to get the benefits of both paradigms in production search engines.

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
