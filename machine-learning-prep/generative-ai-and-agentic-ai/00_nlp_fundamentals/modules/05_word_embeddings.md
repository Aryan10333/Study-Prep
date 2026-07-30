# Module 05: Word Embeddings & Semantic Spaces

Continuous word embeddings map discrete words into low-dimensional dense vectors, capturing semantic and syntactic relationships. This module covers embedding lookup mathematics, CBOW and Skip-gram architectures, Word2Vec optimization techniques (Hierarchical Softmax, Negative Sampling), and code demonstrations.

---

## 1. Dense Embeddings & Lookup Mathematics

In contrast to high-dimensional, sparse representations (like One-Hot or Bag of Words), dense word embeddings represent words in a continuous vector space $\mathbb{R}^d$ where $d \ll |V|$ (typically $50\text{--}300$ dimensions).

### Embedding Lookup as Matrix Multiplication
In deep learning frameworks, retrieving a word embedding vector $\mathbf{h} \in \mathbb{R}^d$ using a token index $i$ is implemented as a fast array lookup operation (e.g. `nn.Embedding(vocab_size, embedding_dim)`). Mathematically, this lookup is equivalent to multiplying the transposed one-hot vector $\mathbf{x} \in \mathbb{R}^{|V|}$ by the embedding weight matrix $\mathbf{W} \in \mathbb{R}^{|V| \times d}$:

$$\mathbf{h} = \mathbf{x}^T \mathbf{W}$$

This mathematical equivalence allows us to treat embedding layers as standard linear projection layers during backpropagation, bypassing manual indexing notation.

---

## 2. Word2Vec Architectures: CBOW vs. Skip-gram

Word2Vec (Mikolov et al., 2013) uses a local sliding context window of size $C$ to learn dense representations from local word correlations.

### Conceptual Intuition:
- **Continuous Bag of Words (CBOW):** Think of CBOW as a **"fill-in-the-blank" game**. The model looks at a window of surrounding context words and tries to guess the missing word in the center. For example, given the context `"The [?] sat on the mat"`, CBOW takes the averaged representation of `"The"`, `"sat"`, `"on"`, `"the"`, `"mat"` and predicts `"cat"`.
- **Skip-gram:** Think of Skip-gram as the **reverse game**. The model is given a single target word in the center and must predict the surrounding context words. For example, given `"cat"`, the model tries to predict `"The"`, `"sat"`, `"on"`, `"the"`, `"mat"`.

### CBOW vs. Skip-gram Architecture Diagram

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; justify-content: space-between; font-family: sans-serif;">
  
  <!-- CBOW Architecture -->
  <div style="flex: 1; min-width: 280px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center;">
    <div style="font-weight: bold; font-size: 14px; color: #1e3a8a; margin-bottom: 12px; text-transform: uppercase;">CBOW Model</div>
    
    <!-- Input layer -->
    <div style="display: flex; gap: 8px; margin-bottom: 12px;">
      <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-family: monospace; font-size: 11px;">w<sub>t-2</sub></div>
      <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-family: monospace; font-size: 11px;">w<sub>t-1</sub></div>
      <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-family: monospace; font-size: 11px;">w<sub>t+1</sub></div>
      <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-family: monospace; font-size: 11px;">w<sub>t+2</sub></div>
    </div>
    
    <div style="font-size: 14px; color: #64748b; margin-bottom: 8px;">&darr;</div>
    
    <!-- Projection layer -->
    <div style="background-color: #3b82f6; color: white; border-radius: 6px; padding: 8px 16px; font-size: 12px; font-weight: bold; text-align: center; width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
      SUM & AVERAGE Layer<br>
      <span style="font-size: 10px; font-weight: normal; font-family: monospace;">h = (1/2C) &Sigma; v<sub>i</sub></span>
    </div>
    
    <div style="font-size: 14px; color: #64748b; margin-top: 8px; margin-bottom: 8px;">&darr;</div>
    
    <!-- Output layer -->
    <div style="background-color: #ef4444; color: white; border-radius: 6px; padding: 8px 16px; font-size: 12px; font-weight: bold; text-align: center; width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 8px;">
      Predict Target Word<br>
      <span style="font-size: 10px; font-weight: normal; font-family: monospace;">w<sub>t</sub></span>
    </div>
    <div style="color: #64748b; font-size: 11px; text-align: center; line-height: 1.3;">Averages surrounding context vectors to predict the hidden middle word.</div>
  </div>

  <!-- Skip-gram Architecture -->
  <div style="flex: 1; min-width: 280px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center;">
    <div style="font-weight: bold; font-size: 14px; color: #7c3aed; margin-bottom: 12px; text-transform: uppercase;">Skip-gram Model</div>
    
    <!-- Input layer -->
    <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 12px; font-family: monospace; font-size: 12px; margin-bottom: 12px;">w<sub>t</sub> (Target)</div>
    
    <div style="font-size: 14px; color: #64748b; margin-bottom: 8px;">&darr;</div>
    
    <!-- Projection layer -->
    <div style="background-color: #8b5cf6; color: white; border-radius: 6px; padding: 8px 16px; font-size: 12px; font-weight: bold; text-align: center; width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
      PROJECTION Layer<br>
      <span style="font-size: 10px; font-weight: normal; font-family: monospace;">h = v<sub>w<sub>t</sub></sub></span>
    </div>
    
    <div style="font-size: 14px; color: #64748b; margin-top: 8px; margin-bottom: 8px;">&darr;</div>
    
    <!-- Output layer -->
    <div style="display: flex; gap: 8px; margin-bottom: 12px; width: 100%; justify-content: center;">
      <div style="background-color: #ec4899; color: white; border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 10px; font-weight: bold;">w<sub>t-2</sub></div>
      <div style="background-color: #ec4899; color: white; border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 10px; font-weight: bold;">w<sub>t-1</sub></div>
      <div style="background-color: #ec4899; color: white; border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 10px; font-weight: bold;">w<sub>t+1</sub></div>
      <div style="background-color: #ec4899; color: white; border-radius: 4px; padding: 4px 6px; font-family: monospace; font-size: 10px; font-weight: bold;">w<sub>t+2</sub></div>
    </div>
    <div style="color: #64748b; font-size: 11px; text-align: center; line-height: 1.3;">Uses a single center word vector to predict the surrounding vocabulary distributions.</div>
  </div>
  
</div>

| Feature | Continuous Bag of Words (CBOW) | Skip-gram |
| :--- | :--- | :--- |
| **Objective** | Predicts the target word $w_t$ given context words $w_{t-C}, \dots, w_{t+C}$. | Predicts context words $w_{t-C}, \dots, w_{t+C}$ given target word $w_t$. |
| **Input Layer** | Averages the embedding vectors of context words: $\mathbf{h} = \frac{1}{2C} \sum_{-C \le j \le C, j \neq 0} \mathbf{v}_{w_{t+j}}$. | The single embedding vector of the target word: $\mathbf{h} = \mathbf{v}_{w_t}$. |
| **Output Layer** | Single Softmax calculation over $|V|$ classes to predict $w_t$. | $2C$ separate Softmax calculations over $|V|$ classes to predict context. |
| **Data Efficiency** | Faster to train; captures syntactic patterns well. | Slower to train; represents rare words much better. |

---

## 3. The Softmax Bottleneck & Negative Sampling

Calculating a full Softmax over a massive vocabulary $|V| = 1,000,000$ words requires normalizing the target score by summing exponentiated dot products across all one million vocabulary entries:

$$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}_{w_O}^{\prime T} \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_{w}^{\prime T} \mathbf{v}_{w_I})}$$

This scales as $O(|V|)$ per training step, creating a massive computational bottleneck for GPUs during backpropagation.

### The Negative Sampling Solution (Binary Game)
Instead of picking 1 correct word out of a million candidates (multi-class classification), Skip-Gram with Negative Sampling (SGNS) reframes the objective as a **binary classification game**:
1. **The Positive Pair:** The model is given a real adjacent pair from the corpus (e.g. `("cat", "sat")`) and is trained to output a high probability (close to 1).
2. **The Negative Pairs:** The model is given $K$ random noise pairs (e.g. `("cat", "democracy")`, `("cat", "refrigerator")`) and is trained to output a low probability (close to 0).

By training on $K$ negative samples instead of $|V|$ candidates, the computational complexity drops from $O(|V|)$ to $O(K)$, where $K$ is typically small ($5\text{--}20$).

#### SGNS Loss Function:
$$\mathcal{L}_{\text{SGNS}} = -\log \sigma(\mathbf{v}_{w_O}^{\prime T} \mathbf{v}_{w_I}) - \sum_{i=1}^k \log \sigma(-\mathbf{v}_{w_i}^{\prime T} \mathbf{v}_{w_I})$$

Where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function.
- $\mathbf{v}_{w_I}$ is the input representation vector of the target word.
- $\mathbf{v}_{w_O}^{\prime}$ is the output context vector of the positive word.
- $\mathbf{v}_{w_i}^{\prime}$ are the output context vectors of the $k$ negative samples.

---

### Step-by-Step Hand-Calculation of SGNS Loss:
Let's calculate the SGNS loss for $k=1$ negative sample using a 3-dimensional embedding space:
- Target input vector ($w_I$ = `"cat"`): $\mathbf{v}_{\text{cat}} = [0.1, \ 0.8, \ -0.2]^T$
- Positive context vector ($w_O$ = `"sat"`): $\mathbf{v}^{\prime}_{\text{sat}} = [0.2, \ 0.9, \ 0.1]^T$
- Negative context vector ($w_1$ = `"refrigerator"`): $\mathbf{v}^{\prime}_{\text{refrig}} = [-0.9, \ 0.1, \ 0.8]^T$

#### 1. Calculate Score for Positive Pair:
- Dot product:
  $$\mathbf{v}^{\prime T}_{\text{sat}} \mathbf{v}_{\text{cat}} = (0.2 \times 0.1) + (0.9 \times 0.8) + (0.1 \times -0.2) = 0.02 + 0.72 - 0.02 = 0.7200$$
- Sigmoid activation:
  $$\sigma(0.7200) = \frac{1}{1 + e^{-0.7200}} \approx 0.6726$$
- Positive loss component:
  $$-\log \sigma(0.7200) = -\log(0.6726) \approx 0.3966$$

#### 2. Calculate Score for Negative Pair:
- Dot product:
  $$\mathbf{v}^{\prime T}_{\text{refrig}} \mathbf{v}_{\text{cat}} = (-0.9 \times 0.1) + (0.1 \times 0.8) + (0.8 \times -0.2) = -0.09 + 0.08 - 0.16 = -0.1700$$
- Negated Sigmoid activation:
  $$\sigma(-(-0.1700)) = \sigma(0.1700) = \frac{1}{1 + e^{-0.1700}} \approx 0.5424$$
- Negative loss component:
  $$-\log \sigma(0.1700) = -\log(0.5424) \approx 0.6117$$

#### 3. Total Loss:
$$\mathcal{L}_{\text{SGNS}} = 0.3966 + 0.6117 = 1.0083$$

##### Findings & Interpretation:
The computed loss is $1.0083$. For the model to improve, backpropagation updates the parameters to increase the dot product of the positive pair (aligning `"cat"` and `"sat"` vectors closer together in direction) and decrease the dot product of the negative pair (pushing `"cat"` and `"refrigerator"` vectors orthogonal or in opposite directions), minimizing total loss.

---

### Why the $3/4$ Exponent Noise Distribution?
The negative samples are drawn from a unigram probability distribution scaled by a $3/4$ exponent:

$$P_n(w) = \frac{U(w)^{3/4}}{\sum_{j=1}^{|V|} U(j)^{3/4}}$$

Where $U(w)$ is the raw unigram frequency of word $w$.
- *Intuition:* The $3/4$ exponent suppresses the probability of high-frequency stopwords (like `"the"` or `"of"`) and boosts the relative selection probability of rare words. For example, if word A has a unigram probability of $0.9$ and word B has $0.01$:
  - Ratio under unigram distribution: $\frac{0.9}{0.01} = 90$
  - Ratio under $3/4$ distribution: $\frac{0.9^{3/4}}{0.01^{3/4}} \approx \frac{0.924}{0.0316} \approx 29.2$
  This makes negative sampling significantly more active on rare words during training.

---

### PyTorch Code Integration

The following PyTorch snippet validates embedding lookup equivalence and shows Skip-Gram forward pass projections:

```python
import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

vocab_size = 5
embed_dim = 3

# Define embedding weight matrix W
embedding = nn.Embedding(vocab_size, embed_dim)
W = embedding.weight.data.clone()

# Token index to lookup
idx = 2
idx_tensor = torch.tensor([idx])

# 1. Standard index lookup
vector_lookup = embedding(idx_tensor)

# 2. Equivalent matrix multiplication (x^T * W)
x = torch.zeros(vocab_size)
x[idx] = 1.0  # One-hot vector
vector_matmul = torch.matmul(x, W)

print("Embedding Weight Matrix W:")
print(W.numpy())
print(f"\nLookup vector (Index {idx}):", vector_lookup.detach().numpy().flatten())
print(f"Matmul vector (x^T * W):  ", vector_matmul.numpy())

# Verify exact equivalence
assert torch.allclose(vector_lookup[0], vector_matmul), "Lookup and Matmul methods do not match!"
```

---

> [!TIP]
> **Production Insight: Fine-Tuning Embeddings**
> When using pre-trained embeddings (e.g. GloVe, FastText) for downstream tasks (like classification):
> - **Freeze Weights (`requires_grad=False`)** if the training corpus is small. This prevents the model from overfitting and altering semantic relationships.
> - **Fine-Tune Weights (`requires_grad=True`)** if you have a massive domain-specific corpus (like medical or financial texts) to adapt the vectors to domain-specific terminology.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Compresses discrete words into continuous low-dimensional vectors, capturing semantic associations (cosine proximity).
- **Why was it introduced?**
  Introduced to solve the vocabulary dimensionality blowup and term orthogonality limits of one-hot sparse models.
- **What are its limitations?**
  Word2Vec assigns a single vector to each word, failing to handle homophones or polysemy (e.g. `"bank"` in financial river bank vs. river bank).
- **Computational Complexity (Time & Memory)**
  - **Flat Softmax Time**: $O(|V|)$ per target prediction.
  - **Negative Sampling Time**: $O(K \cdot d)$ where $K$ is sample count and $d$ is embedding dimension.
- **Component Variable Denotation Legend**
  - $|V|$: Vocabulary dimension.
  - $d$: Embedding space size.
  - $C$: Context window radius.
  - $k$: Negative sampling noise count.
- **Production Use Cases**
  - Building semantic search retrieval matching systems.
  - Initializing token representations in classification networks.
- **Follow-up questions interviewers ask**
  - *Why do we need separate input (W) and output (W') matrices in Word2Vec?* (Using a single shared matrix causes words to match with themselves too strongly, inflating similarity coordinates and degrading semantic clustering).
  - *What is the advantage of FastText over Word2Vec?* (FastText splits words into character n-grams, allowing it to generate embeddings for unseen or out-of-vocabulary words by summing subword vector components).
