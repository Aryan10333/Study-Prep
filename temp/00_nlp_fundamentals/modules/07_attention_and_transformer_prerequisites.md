# Module 07: Attention & Transformer Prerequisites

Attention mechanisms resolved the architectural limits of recurrent neural networks. This module details the sequence bottleneck, contrasts Bahdanau and Luong attention, defines Query-Key-Value projections, and explains the attention scaling factor.

---

## 1. The Seq2Seq Encoder Bottleneck & Context Decay

In classical sequence-to-sequence (Seq2Seq) models, the encoder compresses the entire source sentence into a single, fixed-size context vector:

$$\mathbf{c} = \mathbf{h}_L$$

This introduces the **encoder bottleneck**: the context vector must store all information from the source sentence, regardless of sequence length.
- **Context Decay:** As sequence length $L$ increases, information from the early tokens (e.g. at the beginning of a long paragraph) must travel through a long chain of recurrent transitions. Since the hidden state has a fixed dimensionality, new information at step $t$ overwrites the historical context.
- **Gradient Decay:** The gradient backpropagating from the decoder must flow back through the entire recurrent chain to update early encoder weights, leading to vanishing updates. This causes the model to "forget" details from the beginning of the input, severely degrading translation quality on sentences longer than 30 tokens.

---

## 2. Attention Intuition & Weights: Bahdanau vs. Luong

Attention solves this bottleneck by letting the decoder view all intermediate encoder hidden states $\mathbf{h}_i$ at each decoding step. The decoder dynamically weighs the significance of each encoder hidden state:

$$\mathbf{c}_t = \sum_{i=1}^L \alpha_{t,i} \mathbf{h}_i$$

The weight $\alpha_{t,i}$ measures the alignment between decoder target position $t$ and encoder source position $i$:

$$\alpha_{t,i} = \frac{\exp(\text{score}(\mathbf{s}_{\text{target}}, \mathbf{h}_i))}{\sum_{j=1}^L \exp(\text{score}(\mathbf{s}_{\text{target}}, \mathbf{h}_j))}$$

There are two primary formulations for the score function:

### 1. Bahdanau (Additive) Attention
Bahdanau attention calculates scores using a single-layer feedforward network with a non-linear activation. It evaluates the alignment between the **previous** decoder state $\mathbf{s}_{t-1}$ and the encoder states:

$$\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$

Where:
- $\mathbf{W}_a \in \mathbb{R}^{d_a \times d_h}$ and $\mathbf{U}_a \in \mathbb{R}^{d_a \times d_h}$ are projection weight matrices.
- $\mathbf{v}_a \in \mathbb{R}^{d_a}$ is the output score projection vector.
- $d_a$ is the attention hidden size.

### 2. Luong (Multiplicative) Attention
Luong attention simplifies score calculation by using a multiplicative bilinear projection. It evaluates the alignment using the **current** decoder state $\mathbf{s}_t$:

$$\text{score}(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i$$

Where $\mathbf{W}_a \in \mathbb{R}^{d_h \times d_h}$ is a shared projection matrix.

#### Production Trade-offs (Interview Insight):
- **Additive (Bahdanau):** Historically performs slightly better on very long context sequences due to the non-linear $\tanh$ layer, but is computationally slow. Because it requires summing two projected states before applying tanh, it cannot be easily reduced to a single batch matrix multiplication.
- **Multiplicative (Luong):** Computes significantly faster. The dot product $\mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i$ across all target steps $T$ and source steps $L$ can be calculated simultaneously as a single highly optimized batch matrix multiplication: $\mathbf{S}^T \mathbf{W}_a \mathbf{H}$. This leverages GPU tensor cores, making it the mathematical foundation for modern Transformer self-attention.

---

## 3. Query, Key, and Value Intuition

Self-attention generalizes this alignment mechanism by mapping query vectors against key-value pairs:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Query ($Q$)**: The current token search vector (representing what information the model is looking for).
- **Key ($K$)**: The indexing labels of all available tokens (representing what characteristics each token offers).
- **Value ($V$)**: The actual content vectors of the tokens (representing the information that is retrieved).

### The E-commerce Search Analogy
When you search for `"running shoes"` on Amazon:
- Your search text is the **Query**.
- The database matches this query against product tags and titles (**Keys**).
- The search engine returns the actual product descriptions and images (**Values**) associated with the highest-scoring matches.

---

## 4. Step-by-Step Hand-Calculation of Scaled Dot-Product Attention:
Let's calculate the scaled dot-product attention for a single query vector $\mathbf{q}$ and a key-value database containing two entries ($L=2$) in a 2-dimensional space ($d_k=2$):
- Query vector: $\mathbf{q} = [1.0, \ 2.0]^T$
- Key vectors: $\mathbf{k}_1 = [1.0, \ 0.0]^T, \quad \mathbf{k}_2 = [0.0, \ 2.0]^T$
- Value vectors: $\mathbf{v}_1 = [10.0, \ 20.0]^T, \quad \mathbf{v}_2 = [30.0, \ 40.0]^T$

#### 1. Calculate Dot Products:
- Score 1 (Query and Key 1):
  $$\mathbf{q} \cdot \mathbf{k}_1 = (1.0 \times 1.0) + (2.0 \times 0.0) = 1.0000$$
- Score 2 (Query and Key 2):
  $$\mathbf{q} \cdot \mathbf{k}_2 = (1.0 \times 0.0) + (2.0 \times 2.0) = 4.0000$$

#### 2. Scale by $\sqrt{d_k}$:
Since $d_k=2$, our scaling factor is $\sqrt{2} \approx 1.4142$:
- Scaled score 1:
  $$\text{score}_1 = \frac{1.0000}{1.4142} \approx 0.7071$$
- Scaled score 2:
  $$\text{score}_2 = \frac{4.0000}{1.4142} \approx 2.8284$$

#### 3. Softmax Activation:
- Exponentiate scaled scores:
  $$e^{0.7071} \approx 2.0281$$
  $$e^{2.8284} \approx 16.9188$$
- Sum of exponents:
  $$\text{Sum} = 2.0281 + 16.9188 = 18.9469$$
- Compute Softmax attention weights:
  $$\alpha_1 = \frac{2.0281}{18.9469} \approx 0.1070$$
  $$\alpha_2 = \frac{16.9188}{18.9469} \approx 0.8930$$

#### 4. Weighted Sum of Values (Context Vector $\mathbf{c}$):
$$\mathbf{c} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2$$
$$\mathbf{c} = 0.1070 \begin{bmatrix} 10.0 \\ 20.0 \end{bmatrix} + 0.8930 \begin{bmatrix} 30.0 \\ 40.0 \end{bmatrix} = \begin{bmatrix} 1.0700 \\ 2.1400 \end{bmatrix} + \begin{bmatrix} 26.7900 \\ 35.7200 \end{bmatrix} = \begin{bmatrix} 27.8600 \\ 37.8600 \end{bmatrix}$$

##### Findings & Interpretation:
Using rounded intermediate calculations yields exactly $[27.8600, 37.8600]^T$. In full unrounded floating-point precision, the exact weights are $\alpha_1 \approx 0.10704$ and $\alpha_2 \approx 0.89296$, yielding context vector $[27.8592, 37.8592]^T$. Because the query $\mathbf{q}$ aligned much more closely with key $\mathbf{k}_2$ (resulting in a higher dot product), the Softmax gate assigned $89.30\%$ of the retrieval attention weight to value vector $\mathbf{v}_2$, causing the output to cluster closely to $\mathbf{v}_2$.

---

## 5. Why Attention Scales: The $1/\sqrt{d_k}$ Factor

Scaled dot-product attention scales query-key dot products by $\frac{1}{\sqrt{d_k}}$, where $d_k$ is the key dimension size.

- **Why is this necessary?**
  As the key dimension $d_k$ grows large, the dot product values grow in magnitude. This leads to large absolute inputs to the Softmax function.
- **Softmax Saturation**:
  When Softmax receives very large inputs, its output distribution concentrates (assigning a probability near $1$ to the largest item and near $0$ to the rest). In these flat regions, the gradient of the Softmax function is extremely small, causing **vanishing gradients** during training.
- **The Fix**:
  Dividing the dot product by the standard deviation $\sqrt{d_k}$ scales the input variance to $1$. This keeps the inputs in a range where Softmax remains sensitive to weight updates.

---

## 6. Transition to LLMs: Why Transformers Replaced RNNs

The shift from recurrent models to Transformers was driven by two key limitations of RNNs:
1. **No Parallel Training**: RNN hidden state updates require sequential processing ($t-1 \rightarrow t$). Transformers process all tokens in parallel by replacing recurrent steps with self-attention matrix operations.
2. **No Long-Range Path Decay**: In RNNs, information must travel through sequential steps, causing path decay. In self-attention, the path length between any two tokens is $O(1)$, resolving the vanishing gradient problem over long sequences.

---

> [!TIP]
> **Production Insight: Quadratic Scaling Limits**
> Self-attention requires computing alignment scores for all query-key pairs, scaling as $O(L^2)$ quadratic space and time complexity. For a sequence length $L = 2048$, this requires storing $4,194,304$ attention entries per head. Processing very long contexts (e.g. $100,000$ tokens) crashes GPU memory, requiring specialized sparse attention (e.g., FlashAttention) to run in production.

---

### PyTorch Code Integration

The following PyTorch snippet implements scaled dot-product attention using the exact hand-calculated tensors:

```python
import torch
import torch.nn.functional as F

# Query, Keys, and Values matching hand calculation
q = torch.tensor([[1.0, 2.0]]) # shape (1, d_k)
k = torch.tensor([[1.0, 0.0],  # k1
                  [0.0, 2.0]]) # k2, shape (L, d_k)
v = torch.tensor([[10.0, 20.0], # v1
                  [30.0, 40.0]]) # v2, shape (L, d_v)

d_k = q.size(-1)

# 1. Compute dot products (q * K^T)
scores = torch.matmul(q, k.t()) # shape (1, L)

# 2. Scale by sqrt(d_k)
scaled_scores = scores / (d_k ** 0.5)

# 3. Softmax weights
weights = F.softmax(scaled_scores, dim=-1)

# 4. Weighted sum of values
context = torch.matmul(weights, v)

print("Scaled Scores:     ", scaled_scores.numpy())
print("Attention Weights: ", weights.numpy())
print("Context Vector:    ", context.numpy())

# Verify exact equivalence to hand calculations
assert torch.allclose(context, torch.tensor([[27.8592, 37.8592]]), atol=1e-4)
```

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Allows models to process long-range token relationships without information bottlenecks.
- **Why was it introduced?**
  Introduced to solve the encoder bottleneck in Seq2Seq models.
- **What are its limitations?**
  - **Quadratic Compute Cost**: Self-attention requires computing all query-key pairs, scaling as $O(L^2)$ time and memory.
- **Computational Complexity (Time & Memory)**
  - **Self-Attention Time**: $O(L^2 \cdot d)$ where $L$ is sequence length.
  - **VRAM Memory Footprint**: $O(L^2)$ matrix storage.
- **Component Variable Denotation Legend**
  - $L$: Token sequence length.
  - $d_k$: Key vector dimension size.
  - $d$: Hidden state vector dimension size.
- **Production Use Cases**
  - Text translation engines.
  - Parallelized token sequence learning.
- **Follow-up questions interviewers ask**
  - *Why is self-attention memory cost quadratic with sequence length?* (Because it computes an $L \times L$ attention matrix storing attention weights for every query-key pair).
  - *How does positional encoding help attention capture order?* (Attention is permutation-invariant; it treats tokens as a bag of words. Positional encodings add positional vectors to word vectors, letting the model distinguish token positions).
