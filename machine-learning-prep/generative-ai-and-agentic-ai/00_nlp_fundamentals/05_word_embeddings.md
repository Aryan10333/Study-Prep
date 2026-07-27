# Module 05: Word Embeddings & Semantic Spaces

Word embeddings project discrete vocabulary tokens into low-dimensional, dense continuous vector spaces. This module details embedding projection theory, derives the Word2Vec, GloVe, and FastText algorithms, and explains Out-of-Vocabulary (OOV) handling.

---

## 1. Embedding Space Projection Theory

Word embeddings translate semantic similarity into spatial proximity (e.g. cosine distance). The learning pipeline follows four main steps:

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

---

## 2. Word2Vec: CBOW vs. Skip-gram

Word2Vec uses local context window predictions to train embedding vectors:

### Continuous Bag-of-Words (CBOW)
CBOW predicts a target word $w_t$ given context words within a window size $k$:

$$\mathbf{h} = \frac{1}{2k} \sum_{-k \le j \le k, j \neq 0} \mathbf{v}_{w_{t+j}}$$

$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}' \mathbf{h})$$

Where $\mathbf{v}$ are input embeddings, $\mathbf{W}'$ is the output projection matrix, and $\mathbf{h}$ is the average context vector.

### Skip-gram
Skip-gram predicts context words $w_{t-k}, \dots, w_{t+k}$ given target word $w_t$:

$$P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{v}'_{w_{t+j}} \cdot \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_t})}$$

---

## 3. Negative Sampling Optimization (SGNS)

Calculating the denominator of the Softmax function over a large vocabulary $|V|$ is computationally expensive. Negative Sampling converts this multi-class classification into binary logistic regression:

$$\mathcal{L}_{\text{SGNS}} = -\log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^K \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})]$$

Where:
- $w_O$ is the true output context word.
- $w_I$ is the input target word.
- $w_i$ are $K$ negative sample words drawn from a noise distribution $P_n(w)$.
- $P_n(w)$ is the unigram count raised to the $3/4$ power to boost the probability of sampling rare words:
  $$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}$$

---

## 4. GloVe (Global Vectors) and FastText

Alternative algorithms address Word2Vec's limitations:

### GloVe: Global Co-occurrence Factorization
While Word2Vec scans local context windows, GloVe factorizes global co-occurrence matrices:

$$\mathcal{J} = \sum_{i,j=1}^{|V|} f(X_{i,j}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{i,j})^2$$

Where:
- $X_{i,j}$ is the co-occurrence count of words $i$ and $j$.
- $f(X_{i,j}) = \min\left(1, \left(\frac{X_{i,j}}{x_{\max}}\right)^\alpha\right)$ is a weighting function to prevent frequent co-occurrences from dominating.

### FastText: Subword N-grams for OOV Handling
Word2Vec and GloVe cannot construct vectors for words unseen during training (Out-of-Vocabulary, OOV). FastText solves this by representing words as a bag of character n-grams.
- *Example*: For word `"where"` with $n=3$, character n-grams are: `<wh`, `whe`, `her`, `ere`, `re>` (including boundary markers `<` and `>`).
- The word vector $\mathbf{v}_w$ is the sum of its character n-gram embeddings:
  $$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$
- *OOV Resolution*: When an unseen token is encountered during inference, FastText generates its vector by summing the embeddings of its constituent character n-grams.

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
