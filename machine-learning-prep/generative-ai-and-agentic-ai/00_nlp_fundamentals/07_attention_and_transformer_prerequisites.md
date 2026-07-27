# Module 07: Attention & Transformer Prerequisites

Attention mechanisms resolved the architectural limits of recurrent neural networks. This module details the sequence bottleneck, contrasts Bahdanau and Luong attention, defines Query-Key-Value projections, and provides the mathematical proof of the attention scaling factor.

---

## 1. The Seq2Seq Encoder Bottleneck

In classical sequence-to-sequence (Seq2Seq) models, the encoder compresses the entire source sentence into a single, fixed-size context vector:

$$\mathbf{c} = \mathbf{h}_L$$

This introduces the **encoder bottleneck**: the context vector must store all information from the source sentence, regardless of sequence length. If the sequence length $L$ is long (e.g. $>30$ tokens), the fixed hidden state cannot store the entire context, causing translation quality to decay.

---

## 2. Attention Intuition: Bahdanau vs. Luong

Attention solves this bottleneck by letting the decoder view all intermediate encoder hidden states $\mathbf{h}_i$ at each decoding step. The decoder dynamically weighs the significance of each encoder hidden state:

$$\mathbf{c}_t = \sum_{i=1}^L \alpha_{t,i} \mathbf{h}_i$$

Where $\alpha_{t,i}$ are attention weights representing the alignment between decoder state at step $t$ and encoder state at step $i$.

### Bahdanau vs. Luong Attention
- **Bahdanau (Additive) Attention**:
  - The alignment score is calculated using a single-layer feedforward network:
    $$\text{Score}(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$
  - Uses the previous decoder hidden state $\mathbf{s}_{t-1}$ to calculate alignment scores.
- **Luong (Multiplicative) Attention**:
  - Uses simpler multiplicative dot products, which can be computed via efficient matrix multiplications:
    $$\text{Score}(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i$$
  - Uses the current decoder hidden state $\mathbf{s}_t$ to compute alignment.

---

## 3. Query, Key, and Value Intuition

Attention models map query vectors against key-value pairs:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Query ($Q$)**: The current token search vector (representing what information the model is looking for).
- **Key ($K$)**: The indexing labels of all available tokens (representing what characteristics each token offers).
- **Value ($V$)**: The actual content vectors of the tokens (representing the information that is retrieved).

*Search Analogy*: When searching for a video on YouTube, your search text is the **Query**. YouTube matches this text against video metadata tags (**Keys**) and retrieves the actual video stream (**Value**) associated with the highest-scoring match.

---

## 4. Mathematical Proof of the Attention Scaling Factor

Scaled dot-product attention scales query-key dot products by $\frac{1}{\sqrt{d_k}}$, where $d_k$ is key dimension size.

### Mathematical Proof
Assume the query vector $\mathbf{q} \in \mathbb{R}^{d_k}$ and key vector $\mathbf{k} \in \mathbb{R}^{d_k}$ are independent random vectors whose components are independent random variables with mean $0$ and variance $1$:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$

$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

The dot product is:
$$u = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

Let's find the mean of the dot product:
$$\mathbb{E}[u] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

Since $q_i$ and $k_i$ are independent, the variance of each product term $q_i k_i$ is:
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2$$

Using independence:
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - (\mathbb{E}[q_i]\mathbb{E}[k_i])^2 = (1)(1) - (0)^2 = 1$$

Because the components are independent, the variance of the sum is the sum of the variances:
$$\text{Var}(u) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

Thus, the dot product $u = \mathbf{q} \cdot \mathbf{k}$ has mean $0$ and variance $d_k$.

### Softmax Saturation and Vanishing Gradients
If the key dimension $d_k$ is large, the variance of the dot products is high. This leads to large absolute values of $u$, pushing the Softmax function into regions with extremely small gradients:

$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$$

The derivative of Softmax is:
$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)$$

If one input $z_i$ is much larger than the others, $\sigma_i \approx 1$ and $\sigma_j \approx 0$ (for $j \neq i$). In both cases, the derivatives are near $0$, causing vanishing gradients.
Dividing by the standard deviation $\sqrt{d_k}$ scales the dot products to have a variance of $1$:

$$\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(\mathbf{q} \cdot \mathbf{k}) = \frac{d_k}{d_k} = 1$$

This keeps the inputs in a range where Softmax remains sensitive to weight updates, preventing vanishing gradients during training.

---

## 5. Transition to LLMs: Why Transformers Replaced RNNs

The shift from recurrent models to Transformers was driven by two key limitations of RNNs:

1. **No Parallel Training**: RNN hidden state updates require sequential processing ($t-1 \rightarrow t$). Transformers process all tokens in parallel by replacing recurrent steps with self-attention matrix operations.
2. **No Long-Range Path Decay**: In RNNs, information must travel through sequential steps, causing path decay. In self-attention, the path length between any two tokens is $O(1)$, resolving the vanishing gradient problem over long sequences.

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
