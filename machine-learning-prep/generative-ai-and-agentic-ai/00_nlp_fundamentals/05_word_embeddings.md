# Module 05: Word Embeddings & Semantic Spaces

Word embeddings project discrete vocabulary tokens into low-dimensional, dense continuous vector spaces. This module details embedding projection theory, compares Word2Vec architectures, and explains FastText Out-of-Vocabulary (OOV) resolution.

---

## 1. Embedding Space Projection Theory

Word embeddings translate semantic similarity into spatial proximity. The learning pipeline follows four main steps:

```
Distributional Hypothesis      Prediction Objective             Hidden Layer                Embedding Space
("Words in similar contexts     (Predict context given       (Linear projection maps       (Dense continuous vectors
 have similar meanings")        target word, or vice-versa)   tokens to bottleneck)       capture semantic analogies)
          │                            │                             │                            │
          ▼                            ▼                             ▼                            ▼
   Co-occurrence data           Skip-gram / CBOW           Shared Weight Matrix W        Cosine similarity maps
```

- **Distributional Hypothesis**: Words that occur in similar contexts share semantic meaning.
- **Prediction Objective**: Rather than counting co-occurrences directly, models set up a prediction task (e.g. predicting a word from its neighbors).
- **Hidden Layer**: To solve this task, the model maps inputs through a low-dimensional bottleneck layer.
- **Embedding Space**: The weights of this bottleneck layer form the embedding matrix $\mathbf{W} \in \mathbb{R}^{|V| \times d}$, capturing semantic properties like analogies (e.g., $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$).

### Spatial Analogy Visualization

![Word Embedding Analogy Projection](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/embedding_analogy_projection.png)

```
Vector Dimension y
      ▲
      │       [queen] 
      │       (e.g., coordinate: [1.8, 2.5])
      │      /
      │     / (Shift by vector: "woman" -> "man")
      │    v
      │   [king] (e.g., coordinate: [1.8, 1.2])
      │
      │                                     [woman] (e.g., coordinate: [0.5, 2.5])
      │                                    /
      │                                   / (Shift by vector: "woman" -> "man")
      │                                  v
      │                                 [man] (e.g., coordinate: [0.5, 1.2])
      │
      └──────────────────────────────────────────────────────────▶ Vector Dimension x
```

---

## 2. Word2Vec: CBOW vs. Skip-gram

Word2Vec uses local context window predictions to train embedding vectors:

- **Continuous Bag-of-Words (CBOW)**: Predicts a target word $w_t$ given context words. Context vectors are averaged, making CBOW faster to train but less sensitive to rare words.
- **Skip-gram**: Predicts context words given a target word $w_t$. Skip-gram runs multiple predictions per target word, making it slower to train but yielding better representations for rare tokens.

### Negative Sampling Optimization (SGNS)
To avoid calculating Softmax over the entire vocabulary, Negative Sampling converts the task into binary logistic regression. It updates the target word and a few ($K$) randomly selected "negative" words:
- Negative samples are drawn from a noise distribution $P_n(w)$ raised to the $3/4$ power, which increases the probability of sampling rare words:
  $$P_n(w) \propto U(w)^{0.75}$$

---

## 3. GloVe and FastText

Alternative models address Word2Vec's limitations:

- **GloVe (Global Vectors)**: Factorizes the global word co-occurrence matrix directly rather than scanning local windows, balancing local context and global statistics.
- **FastText (Subword N-grams for OOV)**: Represents each word as a bag of character n-grams.
  - *Example*: For word `"supercomputing"` and $n=3$, character n-grams include: `<su`, `sup`, `upe`, `per`, `...`, `ing>`.
  - *OOV Resolution*: If `"supercomputing"` was not seen during training, FastText can still generate its embedding vector by summing the vectors of its constituent character n-grams, whereas Word2Vec and GloVe return a generic `<unk>` token.

---

> [!TIP]
> **Production Insight: Embedding Layer Memory Footprint**
> Static embedding layers require significant VRAM. A vocabulary of size $|V| = 250,000$ with dimension $d = 300$ requires storing $75,000,000$ float32 parameters ($\approx 300\text{MB}$). While small compared to LLMs, this can exceed memory budgets on edge devices. Quantizing the embedding matrix to int8 reduces this size to $75\text{MB}$ with minimal semantic decay.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Maps discrete vocabulary words to dense, low-dimensional vector representations that capture semantic similarity.
- **Why was it introduced?**
  Introduced to solve the high dimensionality and orthogonal constraints of sparse representations (One-Hot).
- **What are its limitations?**
  Static embeddings assign a single vector to each word, failing to handle polysemy (e.g. `"bank"` in `"river bank"` vs. `"money bank"`).
- **Computational Complexity (Time & Memory)**
  - **Inference Lookup Time**: $O(1)$ constant time lookup.
  - **Training Time**: Skip-gram with negative sampling scales as $O(K \cdot N \cdot C)$ where $C$ is corpus size.
- **Component Variable Denotation Legend**
  - $|V|$: Vocabulary size.
  - $d$: Vector dimension size (typically $100\text{--}300$).
  - $K$: Number of negative samples.
  - $C$: Corpus token count.
- **Production Use Cases**
  - High-speed semantic similarity searches.
  - Initializing embedding layers in recurrent or classification neural networks.
- **Follow-up questions interviewers ask**
  - *Why does CBOW train faster than Skip-gram?* (CBOW averages context embeddings into a single vector to predict one target, while Skip-gram runs multiple predictions per target word).
  - *Why is the noise distribution raised to the 3/4 power in Negative Sampling?* (It increases the relative sampling probability of rare words, ensuring the model updates rare word vectors frequently).
