# Module 03: Traditional Text Representations: One-Hot, BoW, N-Grams & TF-IDF

This study guide covers traditional frequency-based text representations, vector space models, One-Hot Encoding, Bag-of-Words (BoW), N-Gram language modeling, Term Frequency-Inverse Document Frequency (TF-IDF) mathematical equations, step-by-step TF-IDF matrix hand-calculations, Scikit-Learn code, limitations, failure modes, and interview flashcards.

> **Notebook Companion**: [03_traditional_text_representations_tfidf.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/03_traditional_text_representations_tfidf.ipynb)

---

## 1. Vector Space Model & Unstructured Text Projection

To process human text with machine learning algorithms, discrete text tokens must be projected into a numerical **Vector Space** $\mathbb{R}^{|V|}$, where $|V|$ represents the size of the vocabulary.

```text
Document String ──► Tokenization ──► Vocabulary Index Mapping ──► Numerical Feature Vector v_d in R^|V|
```

In traditional frequency-based text representations, each unique word in the vocabulary corresponds to a distinct coordinate axis in vector space.

---

## 2. Discrete Vector Encoding Taxonomy

```text
Representation    Vector Dimensions  Element Value               Preserves Word Order?  Primary Limitation
----------------------------------------------------------------------------------------------------------------------------------
One-Hot           |V| (Per word)     Binary (0 or 1)             No                     Massive memory footprint, orthogonal vectors
Bag-of-Words      |V| (Per doc)      Term Frequency Count        No                     Ignores syntax & penalizes common words
N-Grams           |V|^N (Per doc)    N-gram Frequency Count      Local Phrase Context   Combinatorial vocabulary explosion
TF-IDF            |V| (Per doc)      Normalized TF-IDF Score     No                     Sparse matrix, zero semantic similarity
```

### 1. One-Hot Encoding
Each word $w_i$ is represented as a high-dimensional binary vector of length $|V|$ with a single `1` at the word's vocabulary index and `0` everywhere else.

$$e_{\text{"cat"}} = [1, 0, 0, \dots, 0]^\top, \quad e_{\text{"dog"}} = [0, 1, 0, \dots, 0]^\top$$

- **Fundamental Defect (Orthogonality)**: The dot product between any two distinct one-hot word vectors is zero:

  $$e_i^\top e_j = 0 \quad (\forall i \neq j) \implies \text{CosineSimilarity}(e_{\text{"cat"}}, e_{\text{"kitten"}}) = 0$$

  One-hot vectors fail to capture semantic similarity.

### 2. Bag-of-Words (BoW)
A document vector $v_d$ aggregates individual word occurrences into a frequency count vector over vocabulary $V$:

$$v_d = [\text{Count}(w_1, d), \text{Count}(w_2, d), \dots, \text{Count}(w_{|V|}, d)]^\top$$

- **Limitation**: Ignores word order (`"dog bites man"` and `"man bites dog"` yield identical BoW vectors).

### 3. N-Gram Language Modeling
N-Grams preserve local phrase context by combining contiguous sequences of $N$ tokens:
- **Unigrams ($N=1$)**: `["cloud", "computing"]`
- **Bigrams ($N=2$)**: `["cloud computing", "computing architecture"]`
- **Trigrams ($N=3$)**: `["cloud computing architecture"]`

- **Limitation**: Vocabulary size grows exponentially as $|V|^N$, causing extreme matrix sparsity.

---

## 3. Mathematical Precision: TF-IDF & Normalization

TF-IDF measures how informative a word $t$ is to a specific document $d$ within a corpus $D$.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 1. Term Frequency (TF)
Measures the frequency of term $t$ in document $d$:

- **Raw Term Frequency**: $\text{TF}(t, d) = f_{t,d}$
- **Sublinear Log-Scaled TF** (prevents high-frequency words from dominating):

  $$\text{TF}_{\text{log}}(t, d) = \begin{cases} 1 + \log(f_{t,d}) & \text{if } f_{t,d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2. Inverse Document Frequency (IDF)
Penalizes common words that appear across many documents (e.g., `"the"`, `"system"`) while boosting rare informative terms:

$$\text{IDF}(t, D) = \log \left( \frac{1 + |D|}{1 + \text{DF}(t)} \right) + 1$$

Where:
- $|D|$ is the total number of documents in the corpus.
- $\text{DF}(t) = |\{d \in D : t \in d\}|$ is the Document Frequency (number of documents containing term $t$).
- Addition of $+1$ prevents division-by-zero (smoothing).

### 3. L2 Vector Normalization
To prevent long documents from producing artificially higher TF-IDF vector norms, final document vectors are L2-normalized to unit length ($\|v\|_2 = 1$):

$$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^{|V|} v_i^2}}$$

---

## 4. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we have a corpus $D$ of $|D| = 3$ documents:
- **Doc 1**: `"cat sat mat"`
- **Doc 2**: `"cat sat"`
- **Doc 3**: `"dog sat"`

### 1. Extract Unique Vocabulary ($|V| = 4$):
$$V = [\text{"cat"}, \text{"dog"}, \text{"mat"}, \text{"sat"}]$$

### 2. Calculate Document Frequency (DF) and Smooth IDF for each term:
- $|D| = 3$

$$\text{IDF}(t) = \log \left( \frac{1 + 3}{1 + \text{DF}(t)} \right) + 1 = \log \left( \frac{4}{1 + \text{DF}(t)} \right) + 1$$

- **`"cat"`**: $\text{DF} = 2 \implies \text{IDF} = \log(4 / 3) + 1 = \log(1.3333) + 1 \approx 0.2877 + 1 = \mathbf{1.2877}$
- **`"dog"`**: $\text{DF} = 1 \implies \text{IDF} = \log(4 / 2) + 1 = \log(2.0) + 1 \approx 0.6931 + 1 = \mathbf{1.6931}$
- **`"mat"`**: $\text{DF} = 1 \implies \text{IDF} = \log(4 / 2) + 1 = \log(2.0) + 1 \approx 0.6931 + 1 = \mathbf{1.6931}$
- **`"sat"`**: $\text{DF} = 3 \implies \text{IDF} = \log(4 / 4) + 1 = \log(1.0) + 1 = 0 + 1 = \mathbf{1.0000}$

### 3. Calculate Raw TF-IDF Matrix (TF $\times$ IDF):

```text
Document    TF("cat")    TF("dog")    TF("mat")    TF("sat")  │  Raw TF-IDF("cat")  Raw TF-IDF("dog")  Raw TF-IDF("mat")  Raw TF-IDF("sat")
--------------------------------------------------------------┼───────────────────────────────────────────────────────────────────────────
Doc 1       1            0            1            1          │  1.2877             0.0000             1.6931             1.0000
Doc 2       1            0            0            1          │  1.2877             0.0000             0.0000             1.0000
Doc 3       0            1            0            1          │  0.0000             1.6931             0.0000             1.0000
```

### 4. Apply L2 Normalization to Doc 1 Vector:
- Raw vector for Doc 1: $v_1 = [1.2877, 0.0000, 1.6931, 1.0000]$
- Compute L2 Norm:

  $$\|v_1\|_2 = \sqrt{(1.2877)^2 + (0.0000)^2 + (1.6931)^2 + (1.0000)^2} = \sqrt{1.6582 + 0 + 2.8666 + 1.0000} = \sqrt{5.5248} \approx \mathbf{2.3505}$$

- Normalized Vector $v_{1, \text{norm}}$:
  - `"cat"`: $1.2877 / 2.3505 = \mathbf{0.5478}$
  - `"dog"`: $0.0000 / 2.3505 = \mathbf{0.0000}$
  - `"mat"`: $1.6931 / 2.3505 = \mathbf{0.7203}$
  - `"sat"`: $1.0000 / 2.3505 = \mathbf{0.4254}$

$$\mathbf{v_{1, \text{norm}} = [0.5478, 0.0000, 0.7203, 0.4254]^\top}$$

---

## 5. Production Python Code Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Enterprise Technical Documents
documents = [
    "PostgreSQL primary database failover script failed due to connection timeout.",
    "Microservice API gateway authentication failed with HTTP status 401 unauthorized.",
    "PostgreSQL database backup completed successfully in 45 seconds."
]

# 1. Initialize Production TfidfVectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    sublinear_tf=True,     # Replaces TF with 1 + log(TF)
    ngram_range=(1, 2),    # Extracts Unigrams + Bigrams
    max_features=10        # Restricts vocabulary size to top 10 features
)

# 2. Fit and Transform Documents to Sparse Matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# 3. Represent as Pandas DataFrame
feature_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=["Doc 1", "Doc 2", "Doc 3"])

print("=== 1. Scikit-Learn TF-IDF Feature Matrix ===")
print(df_tfidf.round(4))

# 4. Search Query Cosine Similarity Retrieval
query = "database connection failure"
query_vec = vectorizer.transform([query])
sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

print("\n=== 2. Cosine Similarity Search Results ===")
for doc_idx, score in enumerate(sim_scores, 1):
    print(f"Doc #{doc_idx} Similarity Score: {score:.4f} | {documents[doc_idx-1]}")
```

> [!NOTE]
> **Production TF-IDF Alert:**
> - `sublinear_tf=True` prevents long documents with repeated terms from overwhelming feature importance.
> - Always call `.toarray()` **only** on small sample subsets! In production datasets ($1\text{M}$ docs), keep TF-IDF in `scipy.sparse.csr_matrix` format to prevent memory exhaustion (RAM OOM).

---

## 6. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Vocabulary Out-of-Memory (OOM) Crash**: Converting large sparse TF-IDF matrices to dense NumPy arrays (`matrix.toarray()`) on a 500,000 document corpus will crash server memory.
   - *Remediation*: Keep TF-IDF matrices in SciPy sparse format (`scipy.sparse.csr_matrix`) or restrict `max_features=25000`.
2. **Vocabulary Shift at Test Time**: Test documents containing unseen domain terms (e.g. `"kubernetes"`) are silently ignored by a fixed-vocabulary `TfidfVectorizer`.
   - *Remediation*: Deploy hybrid sparse-dense retrieval (BM25 + Dense Embeddings).

---

## 7. Master Interview Flashcards & Questions

#### Q1: Why does raw term frequency (TF) perform worse than TF-IDF in text retrieval?
- **Answer:** Raw term frequency weights all words equally based on count. Common stop words and generic domain terms (e.g., `"the"`, `"system"`, `"data"`) appear frequently across almost all documents, dominating raw TF vectors. TF-IDF applies Inverse Document Frequency (IDF) scaling to penalize words that appear across many documents, highlighting rare, highly informative terms.

#### Q2: What is the computational limitation of Bag-of-Words and N-Gram representations?
- **Answer:** Bag-of-Words completely ignores word order and syntactic structure. N-Grams preserve local phrase context, but the vocabulary size expands combinatorially as $|V|^N$. For $N=3$ and $|V|=100k$, the potential vocabulary space is $10^{15}$, leading to massive matrix sparsity ($>99.99\%$ zeros) and extreme memory overhead.

#### Q3: Why are One-Hot encoded word vectors mathematically incapable of capturing semantic similarity?
- **Answer:** One-hot vectors treat each word as an independent, orthogonal unit vector in $|V|$-dimensional space ($e_i^\top e_j = 0$ for $i \neq j$). Consequently, the dot product and Cosine similarity between any two distinct one-hot vectors is identically zero, making $\text{CosineSimilarity}(\text{"cat"}, \text{"kitten"}) = 0$.
