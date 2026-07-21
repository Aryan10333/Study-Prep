# Module 07: Attention Mechanisms: Bahdanau, Luong & Bridge to Transformers

This study guide details the Seq2Seq Information Bottleneck, introduction of Attention, Bahdanau Additive Attention vs. Luong Multiplicative Attention, score functions (Dot, General, Concat), alignment weights calculation, context vector construction, step-by-step Luong Dot Attention hand-calculations, PyTorch implementation, Seaborn heatmap alignment visualization, and the direct mathematical transition to Self-Attention and Transformers.

> **Notebook Companion**: [07_attention_mechanisms_bahdanau_luong.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/07_attention_mechanisms_bahdanau_luong.ipynb)

---

## 1. Solving the Seq2Seq Information Bottleneck

In standard Encoder-Decoder architectures (Sutskever et al., 2014), the encoder compresses an entire input sentence $X_{1:T}$ into a single fixed-size final hidden state $h_T$ (the context vector).

```text
Standard Seq2Seq (Bottleneck):
  Input (T tokens) ──► [Encoder] ──► Fixed Vector h_T ──► [Decoder] ──► Output (S tokens)

Attention-Based Seq2Seq (Dynamic Access):
  Input (T tokens) ──► [Encoder (h_1, h_2, ..., h_T)] ──┐
                                                        ├──► [Attention Mechanism] ──► Context Vector c_i ──► [Decoder]
  Decoder State s_{i-1} ────────────────────────────────┘
```

When input sentences grow beyond 30 tokens, compressing all information into a single vector $h_T$ causes severe loss of fine-grained details.

**Attention** (Bahdanau et al., 2014) eliminates this bottleneck by retaining **ALL** encoder hidden states $H = [h_1, h_2, \dots, h_T]$ and allowing the decoder to dynamically "attend" to relevant encoder states at each generation step $i$.

---

## 2. Bahdanau Additive vs. Luong Multiplicative Attention

```text
Dimension             Bahdanau Attention (2014)                    Luong Attention (2015)
----------------------------------------------------------------------------------------------------------------------------------
Score Type            Additive (Feed-Forward Neural Network)       Multiplicative (Matrix Dot Product)
Score Equation        e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)  e_{ij} = s_i^T h_j  (Dot)  or  s_i^T W_a h_j (General)
Decoder State Used    Previous Decoder Hidden State s_{i-1}        Current Decoder Hidden State s_i
Computational Speed   Slower (requires matrix additions + tanh)    Faster (optimized matrix multiplication)
Legacy Impact         First Attention paper in Deep Learning       Direct predecessor to Transformer Scaled Dot-Product
```

### 1. Bahdanau Additive Attention Score Function:
$$e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

Where:
- $s_{i-1}$ is the decoder state at previous step $i-1$.
- $h_j$ is the $j$-th encoder hidden state.
- $W_a, U_a, v_a$ are trainable attention projection weights.

### 2. Luong Multiplicative Attention Score Functions:
Luong et al. (2015) simplified attention scores using fast multiplicative matrix operations:

- **Dot Score**: Assumes encoder and decoder hidden states have matching dimension $d$:

  $$\text{Score}_{\text{Dot}}(s_i, h_j) = s_i^\top h_j$$

- **General Score**: Uses projection matrix $W_a$ to handle dimension mismatches:

  $$\text{Score}_{\text{General}}(s_i, h_j) = s_i^\top W_a h_j$$

- **Concat Score**: Concatenates states into a linear layer:

  $$\text{Score}_{\text{Concat}}(s_i, h_j) = v_a^\top \tanh(W_a [s_i; h_j])$$

---

## 3. Attention Alignment Weights & Context Vector Construction

Once raw alignment scores $e_{ij}$ are calculated between decoder state $s_i$ and all encoder hidden states $h_1, \dots, h_T$:

### Step 1: Compute Softmax Alignment Probabilities ($\alpha_{ij}$):
The scores are normalized into a probability distribution over input tokens using Softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}$$

Where $\alpha_{ij} \in [0, 1]$ represents the relative attention weight allocated to input token $j$ when decoding output token $i$.

### Step 2: Compute Dynamic Context Vector ($c_i$):
The context vector $c_i$ is computed as the weighted linear combination of all encoder hidden states:

$$c_i = \sum_{j=1}^T \alpha_{ij} h_j$$

### Step 3: Compute Combined Attentional Vector ($\tilde{s}_i$):
The context vector $c_i$ and decoder state $s_i$ are concatenated to predict the next token probability:

$$\tilde{s}_i = \tanh(W_c [c_i; s_i])$$

$$P(y_i | y_{<i}, X) = \text{softmax}(W_s \tilde{s}_i)$$

---

## 4. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we have an Encoder sentence with $T = 3$ tokens producing 2-dimensional hidden states:
- $h_1 = [1.0, 0.0]^\top$
- $h_2 = [0.0, 2.0]^\top$
- $h_3 = [1.0, 1.0]^\top$

And a Decoder state at step $i$:
- $s_i = [2.0, 1.0]^\top$

We use **Luong Dot Attention**: $e_j = s_i^\top h_j$.

### 1. Compute Raw Attention Scores ($e_j$):
- $e_1 = s_i^\top h_1 = (2.0 \times 1.0) + (1.0 \times 0.0) = 2.0 + 0 = \mathbf{2.0}$
- $e_2 = s_i^\top h_2 = (2.0 \times 0.0) + (1.0 \times 2.0) = 0 + 2.0 = \mathbf{2.0}$
- $e_3 = s_i^\top h_3 = (2.0 \times 1.0) + (1.0 \times 1.0) = 2.0 + 1.0 = \mathbf{3.0}$

### 2. Compute Softmax Alignment Weights ($\alpha_j$):
Exponentials:
- $\exp(e_1) = \exp(2.0) \approx 7.3891$
- $\exp(e_2) = \exp(2.0) \approx 7.3891$
- $\exp(e_3) = \exp(3.0) \approx 20.0855$

Sum of Exponentials $Z = 7.3891 + 7.3891 + 20.0855 = \mathbf{34.8637}$

Alignment Weights $\alpha_j = \exp(e_j) / Z$:
- $\alpha_1 = 7.3891 / 34.8637 \approx \mathbf{0.2119}$ ($21.2\%$ attention on Token 1)
- $\alpha_2 = 7.3891 / 34.8637 \approx \mathbf{0.2119}$ ($21.2\%$ attention on Token 2)
- $\alpha_3 = 20.0855 / 34.8637 \approx \mathbf{0.5762}$ ($57.6\%$ attention on Token 3)

### 3. Compute Dynamic Context Vector ($c_i$):
$$c_i = \alpha_1 h_1 + \alpha_2 h_2 + \alpha_3 h_3$$

$$c_i = 0.2119 \times \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + 0.2119 \times \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} + 0.5762 \times \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$

$$c_i = \begin{bmatrix} 0.2119 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.4238 \end{bmatrix} + \begin{bmatrix} 0.5762 \\ 0.5762 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.7881 \\ 1.0000 \end{bmatrix}}$$

$$\mathbf{\alpha = [0.2119, 0.2119, 0.5762]^\top, \quad c_i = [0.7881, 1.0000]^\top}$$

---

## 5. Bridge to Transformers: Scaled Dot-Product Attention

The introduction of Luong Dot Attention revealed a profound mathematical insight: **recurrent hidden states are unneeded for context retrieval if we compute scaled dot products directly across token projections**.

Vaswani et al. (2017) extended Luong Dot Attention into **Scaled Dot-Product Self-Attention** by projecting input representations into Query ($Q$), Key ($K$), and Value ($V$) matrices:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Where $\frac{1}{\sqrt{d_k}}$ scaling prevents large dot products from driving Softmax gradients into extremely small regions (vanishing gradients).

---

## 6. Production PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongDotAttention(nn.Module):
    """Production PyTorch Luong Multiplicative Dot-Product Attention Module."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, decoder_state: torch.Tensor, encoder_states: torch.Tensor):
        # decoder_state shape: [batch_size, 1, hidden_dim]
        # encoder_states shape: [batch_size, seq_len, hidden_dim]
        
        # 1. Compute Dot Attention Scores: e_j = s_i^T * h_j
        # transpose(1, 2) shape: [batch_size, hidden_dim, seq_len]
        attn_scores = torch.bmm(decoder_state, encoder_states.transpose(1, 2)) # [batch_size, 1, seq_len]
        
        # 2. Compute Softmax Alignment Weights
        attn_weights = F.softmax(attn_scores, dim=-1) # [batch_size, 1, seq_len]
        
        # 3. Compute Weighted Context Vector c = sum(alpha_j * h_j)
        context_vector = torch.bmm(attn_weights, encoder_states) # [batch_size, 1, hidden_dim]
        
        return context_vector, attn_weights

# Demonstration Execution
batch_size, seq_len, hidden_dim = 2, 4, 8

encoder_h = torch.randn(batch_size, seq_len, hidden_dim)
decoder_s = torch.randn(batch_size, 1, hidden_dim)

attn_layer = LuongDotAttention()
context, weights = attn_layer(decoder_s, encoder_h)

print("=== PyTorch Luong Dot Attention Execution ===")
print("Context Vector Shape:   ", context.shape)  # [2, 1, 8]
print("Alignment Weights Shape:", weights.shape)  # [2, 1, 4]
print("Sample Alignment Weights (Batch #1):", weights[0, 0, :].detach().numpy().round(4))
```

> [!NOTE]
> **Attention Matrix Alert:**
> - `torch.bmm` executes batch matrix multiplication efficiently on GPUs.
> - Plotting `weights` matrix as a Seaborn heatmap provides full interpretability into which source words the model attended to when generating target words.

---

## 7. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Quadratic Alignment Memory Bottleneck ($O(T_{\text{enc}} \times T_{\text{dec}})$)**: Storing the full attention matrix $\alpha \in \mathbb{R}^{T_{\text{enc}} \times T_{\text{dec}}}$ for long sequences ($T=10,000$) causes massive GPU VRAM consumption.
   - *Remediation*: Deploy FlashAttention or sparse windowed attention patterns.
2. **Unscaled Large Dot Products**: In high embedding dimensions ($d_k = 512$), unscaled dot products $s_i^\top h_j$ grow large in magnitude, pushing Softmax outputs into near-binary values ($0$ or $1$) with zero gradients.
   - *Remediation*: Always scale dot product scores by $\frac{1}{\sqrt{d_k}}$.

---

## 8. Master Interview Flashcards & Questions

#### Q1: What problem does the Attention mechanism solve in standard Seq2Seq architectures?
- **Answer:** Standard Seq2Seq models compress the entire input sequence into a single fixed-size context vector $h_T$ at the end of the Encoder. For long sentences ($T > 30$), this creates an Information Bottleneck, causing severe accuracy degradation. Attention retains all encoder hidden states $[h_1, \dots, h_T]$ and allows the decoder to dynamically compute alignment weights and extract a tailored context vector $c_i$ at each decoding step.

#### Q2: Compare Bahdanau Additive Attention vs. Luong Multiplicative Attention.
- **Answer:** Bahdanau Attention computes alignment scores using a feed-forward neural network layer $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ using previous decoder state $s_{i-1}$. Luong Attention uses multiplicative matrix operations $e_{ij} = s_i^\top W_a h_j$ using current decoder state $s_i$. Luong attention is computationally faster and GPU matrix-multiplication friendly.

#### Q3: How does Luong Dot Attention mathematically transition to Transformer Scaled Dot-Product Attention?
- **Answer:** Luong Dot Attention computes $e = s_i^\top h_j$ between decoder state $s_i$ and encoder state $h_j$. Transformer Scaled Dot-Product Attention replaces recurrent hidden states with linear projections of input tokens into Query ($Q$), Key ($K$), and Value ($V$) matrices, scaling scores by $\frac{1}{\sqrt{d_k}}$: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.
