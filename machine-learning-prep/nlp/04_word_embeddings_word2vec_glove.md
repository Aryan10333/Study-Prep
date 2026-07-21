# Module 04: Word Embeddings: Word2Vec, GloVe, FastText & Vector Arithmetic

This study guide details the Distributional Hypothesis, Word2Vec architectures (CBOW vs. Skip-Gram), Hierarchical Softmax, Negative Sampling loss derivations, GloVe matrix factorization, FastText subword n-gram embeddings, vector space arithmetic, step-by-step Negative Sampling hand-calculations, Gensim code, failure modes, and interview flashcards.

> **Notebook Companion**: [04_word_embeddings_word2vec_glove.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/04_word_embeddings_word2vec_glove.ipynb)

---

## 1. The Distributional Hypothesis & Continuous Embeddings

The foundation of modern statistical semantics is the **Distributional Hypothesis**:
> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

Words that occur in similar context window environments tend to share similar semantic meaning.

```text
High-Dimensional Sparse Space R^|V| (One-Hot / TF-IDF) ──► Word2Vec / GloVe Projection ──► Low-Dimensional Dense Continuous Space R^d (d ~ 100 - 300)
```

Dense word embeddings map high-dimensional sparse representations ($|V| \ge 100,000$) into a low-dimensional continuous vector space $\mathbb{R}^d$ ($d \in [100, 300]$), where geometric proximity (Cosine similarity) correlates directly with semantic similarity.

---

## 2. Word2Vec Architecture: CBOW vs. Skip-Gram

Word2Vec (Mikolov et al., 2013) uses a 2-layer shallow neural network to learn dense word representations.

```text
Model Paradigm    Input Layer                        Output Layer                       Training Speed  Performance on Rare Words
----------------------------------------------------------------------------------------------------------------------------------
CBOW              Context Words {w_{t-c}, ..., w_{t+c}} Target Center Word w_t          Fast            Moderate
Skip-Gram         Target Center Word w_t             Context Words {w_{t-c}, ..., w_{t+c}} Slower          Superior for rare words & small corpora
```

```text
        CBOW Architecture                              Skip-Gram Architecture

 Context (w_{t-2}) ──┐                                      ┌──► Context (w_{t-2})
 Context (w_{t-1}) ──┼──► [Projection] ──► Target (w_t)  Target (w_t) ──► [Projection] ──┼──► Context (w_{t-1})
 Context (w_{t+1}) ──┤                                      ├──► Context (w_{t+1})
 Context (w_{t+2}) ──┘                                      └──► Context (w_{t+2})
```

---

## 3. Mathematical Optimization: Negative Sampling

### 1. The Full Softmax Computational Bottleneck
In a standard Skip-Gram model, the conditional probability of predicting context word $w_O$ given target center word $w_I$ is:

$$P(w_O | w_I) = \frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w=1}^{|V|} \exp({v'_{w}}^\top v_{w_I})}$$

Where $v_{w}$ is the input embedding and $v'_{w}$ is the output embedding of word $w$.

- **Problem**: Computing the denominator partition function requires summing over all $|V|$ words in the vocabulary ($|V| \ge 100,000$), making gradient updates prohibitively expensive ($O(|V|)$ per token).

### 2. Negative Sampling Loss Objective
Negative Sampling (SGNS) reformulates the multiclass Softmax task into $K + 1$ independent binary logistic regression tasks:
- Predict `1` for the true positive context pair $(w_I, w_O)$.
- Predict `0` for $K$ randomly drawn "negative" noise words $(w_I, w_k) \sim P_n(w)$.

$$\mathcal{L}_{\text{SGNS}} = -\log \sigma({v'_{w_O}}^\top v_{w_I}) - \sum_{k=1}^K \log \sigma(-{v'_{w_k}}^\top v_{w_I})$$

Where $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the sigmoid activation function, and $P_n(w) \propto U(w)^{3/4}$ is the unigram noise distribution raised to the $3/4$ power to boost the sampling probability of rare words.

---

## 4. GloVe: Global Vectors for Word Representation

While Word2Vec scans local context windows, **GloVe** (Pennington et al., 2014) combines local context windows with global matrix factorization of the global word co-occurrence matrix $X$.

- $X_{ij}$ = Number of times word $j$ appears in the context of word $i$.

### GloVe Loss Function:
$$J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

Where $f(X_{ij}) = \min\left(1, \left(\frac{X_{ij}}{x_{\max}}\right)^\alpha\right)$ (with $\alpha = 0.75, x_{\max} = 100$) is a weighting function that caps the influence of extremely frequent co-occurrences (e.g., `"the"`, `"and"`).

---

## 5. FastText: Subword Character N-Gram Embeddings

Developed by Facebook (Bojanowski et al., 2017), **FastText** extends Word2Vec by representing each word as a bag of character n-grams.

- For word `"where"` and $n=3$, character n-grams are: `"<wh"`, `"whe"`, `"her"`, `"ere"`, `"re>"` plus special word `"where"`.

The overall embedding vector $v_w$ for word $w$ is the sum of its character n-gram vectors $z_g$:

$$v_w = \sum_{g \in G_w} z_g$$

- **Major Advantage**: FastText can construct embeddings for **Out-Of-Vocabulary (OOV)** rare or misspelled words (e.g., `"unmicroservice"`) at inference time by summing the learned vectors of its subword n-grams.

---

## 6. Vector Space Arithmetic & Semantic Geometry

Dense embeddings preserve linear regularities and relational analogies in vector space:

$$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$$

$$v_{\text{Paris}} - v_{\text{France}} + v_{\text{Germany}} \approx v_{\text{Berlin}}$$

```text
   Vector Space Geometry
   
    [King] ──────────(-man + woman)──────────► [Queen]
      │                                          │
      │ (+France - Germany)                      │ (+France - Germany)
      ▼                                          ▼
   [Paris] ──────────(-man + woman)──────────► [Rome]
```

---

## 7. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we have a 2-dimensional embedding space ($d=2$) and a target center word $w_I = \text{"server"}$ with vector $v_{w_I} = [1.0, 0.5]^\top$.

We evaluate 1 positive context word $w_O = \text{"database"}$ ($v'_{w_O} = [0.8, 0.6]^\top$) and $K=1$ negative noise word $w_1 = \text{"apple"}$ ($v'_{w_1} = [-0.5, 0.2]^\top$).

### 1. Compute Dot Products:
- Positive pair dot product: $z_+ = {v'_{w_O}}^\top v_{w_I} = (0.8 \times 1.0) + (0.6 \times 0.5) = 0.8 + 0.3 = \mathbf{1.10}$
- Negative pair dot product: $z_- = {v'_{w_1}}^\top v_{w_I} = (-0.5 \times 1.0) + (0.2 \times 0.5) = -0.5 + 0.1 = \mathbf{-0.40}$

### 2. Compute Sigmoid Probabilities:
- $\sigma(z_+) = \frac{1}{1 + \exp(-1.10)} = \frac{1}{1 + 0.3329} \approx \mathbf{0.7503}$
- $\sigma(-z_-) = \frac{1}{1 + \exp(0.40)} = \frac{1}{1 + 1.4918} \approx \mathbf{0.4013}$

### 3. Compute Negative Sampling Loss:
$$\mathcal{L} = -\log \sigma(z_+) - \log \sigma(-z_-)$$

$$\mathcal{L} = -\log(0.7503) - \log(0.4013) \approx 0.2873 + 0.9130 = \mathbf{1.2003}$$

---

## 8. Production Python Code Implementation

```python
import gensim.downloader as api
from gensim.models import Word2Vec

# 1. Real-World Enterprise IT Corpus
corpus = [
    ["microservice", "communicates", "with", "database", "via", "grpc"],
    ["database", "failover", "script", "executed", "successfully"],
    ["grpc", "protocol", "ensures", "low", "latency", "microservice", "communication"],
    ["kubernetes", "cluster", "manages", "microservice", "deployment"]
]

# 2. Train Custom Gensim Skip-Gram Word2Vec Model
model = Word2Vec(
    sentences=corpus,
    vector_size=50,    # Embedding dimension d = 50
    window=3,          # Context window size c = 3
    min_count=1,       # Minimum word frequency threshold
    sg=1,              # sg=1 for Skip-Gram (sg=0 for CBOW)
    negative=5,        # K = 5 negative samples
    epochs=100
)

# 3. Vector Similarity & Arithmetic Execution
print("=== 1. Word Similarity Scores ===")
sim_score = model.wv.similarity("microservice", "database")
print(f"Similarity ('microservice', 'database'): {sim_score:.4f}")

print("\n=== 2. Most Similar Words to 'microservice' ===")
similar_words = model.wv.most_similar("microservice", topn=3)
for word, score in similar_words:
    print(f"Word: {word:<15} | Cosine Similarity: {score:.4f}")
```

> [!NOTE]
> **Production Embeddings Alert:**
> - `sg=1` (Skip-Gram) is preferred for small domain corpora and rare technical jargon terms.
> - Static word embeddings (Word2Vec/GloVe) assign **one static vector per word**, meaning `"apple"` (fruit) and `"apple"` (tech company) receive identical vector representation. Modern applications use contextual embeddings (BERT/OpenAI) to resolve polysemy.

---

## 9. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Static Polysemy Breakdown**: Word2Vec maps a word to a single static vector regardless of sentence context.
   - *Example*: `"river bank"` and `"investment bank"` share the exact same Word2Vec vector for `"bank"`.
   - *Remediation*: Use contextual embeddings (BERT, RoBERTa, OpenAI `text-embedding-3`).
2. **Out-Of-Vocabulary (OOV) Rejection**: Standard Word2Vec and GloVe throw KeyError exceptions when encountering un-indexed test words (e.g. `"fastapi"`).
   - *Remediation*: Deploy FastText (subword character n-grams) or BPE subword tokenizers.

---

## 10. Master Interview Flashcards & Questions

#### Q1: Compare CBOW vs. Skip-Gram in Word2Vec.
- **Answer:** CBOW predicts the center target word given surrounding context words. Skip-Gram predicts context words given a center target word. CBOW trains faster and works well for frequent words. Skip-Gram is slower but performs significantly better on small datasets and rare words.

#### Q2: How does Negative Sampling solve the Softmax computational bottleneck in Word2Vec?
- **Answer:** Full Softmax requires computing a sum over all $|V|$ words in the vocabulary for every gradient update ($O(|V|)$ complexity). Negative Sampling converts the task into $K+1$ binary logistic regression tasks (1 positive pair, $K$ negative noise words), reducing update computational complexity from $O(|V|)$ to $O(K)$, where $K \ll |V|$ (typically $K=5-20$).

#### Q3: Why does FastText handle Out-Of-Vocabulary (OOV) words better than Word2Vec or GloVe?
- **Answer:** Word2Vec and GloVe treat words as atomic entities and assign a single vector per vocabulary word, failing completely on unseen test words. FastText represents words as a bag of character n-grams (e.g. `"<wh"`, `"whe"`, `"her"`, `"ere"`, `"re>"`). For an unseen word, FastText sums the vectors of its constituent character n-grams to construct a meaningful embedding.
