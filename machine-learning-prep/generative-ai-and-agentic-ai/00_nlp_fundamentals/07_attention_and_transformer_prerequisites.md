# Module 07: Attention & Transformer Prerequisites

Attention mechanisms resolved the architectural limits of recurrent neural networks. This module details the sequence bottleneck, contrasts Bahdanau and Luong attention, defines Query-Key-Value projections, and explains the attention scaling factor.

---

## 1. The Seq2Seq Encoder Bottleneck

In classical sequence-to-sequence (Seq2Seq) models, the encoder compresses the entire source sentence into a single, fixed-size context vector:
$$\mathbf{c} = \mathbf{h}_L$$

This introduces the **encoder bottleneck**: the context vector must store all information from the source sentence, regardless of sequence length. If the sequence length $L$ is long (e.g. $>30$ tokens), the fixed hidden state cannot store the entire context, causing translation quality to decay.

---

## 2. Attention Intuition: Bahdanau vs. Luong

Attention solves this bottleneck by letting the decoder view all intermediate encoder hidden states $\mathbf{h}_i$ at each decoding step. The decoder dynamically weighs the significance of each encoder hidden state:
$$\mathbf{c}_t = \sum_{i=1}^L \alpha_{t,i} \mathbf{h}_i$$

- **Bahdanau (Additive) Attention**: Uses a single-layer feedforward network to calculate alignment scores based on the *previous* decoder state $\mathbf{s}_{t-1}$ and encoder states.
- **Luong (Multiplicative) Attention**: Uses simpler multiplicative dot products to compute alignment based on the *current* decoder state $\mathbf{s}_t$, which can be computed via efficient matrix multiplications.

---

## 3. Query, Key, and Value Intuition

![Attention Matrix Heatmap](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/attention_matrix_heatmap.png)

#### Plot Explanation & Intuition: Attention Alignment Matrix Heatmap
This heatmap visualizes the self-attention alignment weights computed between query words and key words:
- **Semantic Mapping**: The query word `"feline"` aligns strongly with the key word `"cat"` ($0.90$ alignment score), while `"sat"` aligns with `"sat"` ($0.95$).
- **Alignment Matrix**: The values are soft probability distributions across the keys, summing to $1.0$ along each query row.
- **Production Takeaway**: The heatmap demonstrates the concept of dynamic alignment. Unlike recurrent hidden states that compress all history into a single vector, self-attention maps query-key relationships directly, allowing the model to focus on related terms across long sequence gaps.

Attention models map query vectors against key-value pairs:

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

## 4. Why Attention Scales: The $1/\sqrt{d_k}$ Factor

Scaled dot-product attention scales query-key dot products by $\frac{1}{\sqrt{d_k}}$, where $d_k$ is key dimension size.

- **Why is this necessary?**
  As the key dimension $d_k$ grows large, the dot product values grow in magnitude. This leads to large absolute inputs to the Softmax function.
- **Softmax Saturation**:
  When Softmax receives very large inputs, its output distribution concentrates (assigning a probability near $1$ to the largest item and near $0$ to the rest). In these flat regions, the gradient of the Softmax function is extremely small, causing **vanishing gradients** during training.
- **The Fix**:
  Dividing the dot product by the standard deviation $\sqrt{d_k}$ scales the input variance to $1$. This keeps the inputs in a range where Softmax remains sensitive to weight updates.

---

## 5. Transition to LLMs: Why Transformers Replaced RNNs

The shift from recurrent models to Transformers was driven by two key limitations of RNNs:
1. **No Parallel Training**: RNN hidden state updates require sequential processing ($t-1 \rightarrow t$). Transformers process all tokens in parallel by replacing recurrent steps with self-attention matrix operations.
2. **No Long-Range Path Decay**: In RNNs, information must travel through sequential steps, causing path decay. In self-attention, the path length between any two tokens is $O(1)$, resolving the vanishing gradient problem over long sequences.

---

> [!TIP]
> **Production Insight: Quadratic Scaling Limits**
> Self-attention requires computing alignment scores for all query-key pairs, scaling as $O(L^2)$ quadratic space and time complexity. For a sequence length $L = 2048$, this requires storing $4,194,304$ attention entries per head. Processing very long contexts (e.g. $100,000$ tokens) crashes GPU memory, requiring specialized sparse attention (e.g., FlashAttention) to run in production.

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
