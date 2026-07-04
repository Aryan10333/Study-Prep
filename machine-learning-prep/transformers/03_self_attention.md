# Transformers: Self-Attention Mechanics

This guide details the mathematical equations, projections, complexity profiles, and step-by-step matrix calculations for the Self-Attention layer, accompanied by a clean PyTorch implementation.

---

## 1. What is Self-Attention?

Self-Attention allows a model to build context-aware token representations by comparing every token in a sequence with every other token in the same sequence.
- **The Core Goal:** For example, in the sentence *"The bank of the river,"* the word *"bank"* is contextualized by *"river"*. In *"The bank deposits money,"* the word *"bank"* is contextualized by *"deposits"* and *"money"*. Self-attention dynamically projects these semantic linkages into activation values.

---

## 2. Mathematical Projections & Vectorization

Let $X \in \mathbb{R}^{n \times d_{\text{model}}}$ be the input token embedding matrix, where $n$ is the sequence length (number of tokens) and $d_{\text{model}}$ is the embedding dimension.

We compute three projection matrices using learnable weight parameters $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$:
- **Queries:** $Q = X W_Q \in \mathbb{R}^{n \times d_k}$
- **Keys:** $K = X W_K \in \mathbb{R}^{n \times d_k}$
- **Values:** $V = X W_V \in \mathbb{R}^{n \times d_v}$

Using these projected matrices, the self-attention calculation is vectorized:
$$\text{Attention Matrix } A = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{n \times n}$$
$$\text{Context Matrix } Y = A V \in \mathbb{R}^{n \times d_v}$$

---

## 3. Computational Complexity Analysis: $O(n^2)$

During self-attention, we compute dot products between every query and key, producing an $n \times n$ attention weight matrix.

- **Computational Complexity:**
  1. Projection phase ($X W_{Q,K,V}$): $O(n \cdot d_{\text{model}}^2)$ operations.
  2. Attention scores ($Q K^T$): $O(n^2 \cdot d_k)$ operations.
  3. Weighted sum of values ($A V$): $O(n^2 \cdot d_v)$ operations.
  - **Overall Complexity:** $O(n^2 \cdot d + n \cdot d^2)$. When sequence length $n$ is large (e.g. $n > 4096$), the quadratic term $O(n^2 \cdot d)$ dominates execution.
- **VRAM Memory Footprint:**
  Storing the $n \times n$ attention matrix requires $O(n^2)$ memory. For a sequence length of 16,384 tokens, a single attention head requires storing $16384^2 = 2.68 \times 10^8$ float values ($\approx 1\text{GB}$ of VRAM) per layer, leading to GPU memory constraints.

---

## 4. Step-by-Step Hand Calculations: Matrix Projections (Andrew Ng Style)

Let's compute the self-attention pass for a tiny sequence of $n=3$ tokens, with dimensions $d_{\text{model}} = 2$ and $d_k = d_v = 2$.
- **Input Matrix ($X$):**
  $$X = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{3 \times 2} \quad \left(\text{Tokens: } x_1, x_2, x_3\right)$$
- **Projection Weights:**
  $$W_Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$

---

### Step 1: Compute Q, K, V Matrices
$$Q = X W_Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}$$
$$K = X W_K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix}$$
$$V = X W_V = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \\ 2.0 & 2.0 \end{bmatrix}$$

---

### Step 2: Compute Raw Dot Product Scores ($Q K^T$)
$$Q K^T = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.0 & 1.0 & 1.0 \\ 1.0 & 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 & 1.0 \\ 1.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 2.0 \end{bmatrix}$$

---

### Step 3: Scale Scores ($\sqrt{d_k} = \sqrt{2} \approx 1.4142$)
$$S = \frac{Q K^T}{1.4142} \approx \begin{bmatrix} 0.0 & 0.7071 & 0.7071 \\ 0.7071 & 0.0 & 0.7071 \\ 0.7071 & 0.7071 & 1.4142 \end{bmatrix}$$

---

### Step 4: Apply Softmax Per Row
For Row 1 ($[0.0, \ 0.7071, \ 0.7071]$):
- $\text{exps} = [e^{0.0}, e^{0.7071}, e^{0.7071}] \approx [1.0, 2.0281, 2.0281]$ | $\text{sum} = 5.0562$
- $\text{row}_1 = [1.0/5.0562, 2.0281/5.0562, 2.0281/5.0562] \approx [\mathbf{0.1978}, \mathbf{0.4011}, \mathbf{0.4011}]$

For Row 2 ($[0.7071, \ 0.0, \ 0.7071]$):
- $\text{exps} = [e^{0.7071}, e^{0.0}, e^{0.7071}] \approx [2.0281, 1.0, 2.0281]$ | $\text{sum} = 5.0562$
- $\text{row}_2 \approx [\mathbf{0.4011}, \mathbf{0.1978}, \mathbf{0.4011}]$

For Row 3 ($[0.7071, \ 0.7071, \ 1.4142]$):
- $\text{exps} = [e^{0.7071}, e^{0.7071}, e^{1.4142}] \approx [2.0281, 2.0281, 4.1132]$ | $\text{sum} = 8.1694$
- $\text{row}_3 \approx [2.0281/8.1694, 2.0281/8.1694, 4.1132/8.1694] \approx [\mathbf{0.2483}, \mathbf{0.2483}, \mathbf{0.5034}]$

$$A = \begin{bmatrix} 0.1978 & 0.4011 & 0.4011 \\ 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5034 \end{bmatrix}$$

---

### Step 5: Compute Final Context Matrix ($Y = A V$)
$$Y = A V = \begin{bmatrix} 0.1978 & 0.4011 & 0.4011 \\ 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5034 \end{bmatrix} \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \\ 2.0 & 2.0 \end{bmatrix}$$

- **Row 1 Output:**
  $$y_{1,1} = (0.1978 \times 2) + (0.4011 \times 0) + (0.4011 \times 2) = 0.3956 + 0.8022 = \mathbf{1.1978}$$
  $$y_{1,2} = (0.1978 \times 0) + (0.4011 \times 2) + (0.4011 \times 2) = 0.8022 + 0.8022 = \mathbf{1.6044}$$
- **Row 2 Output:**
  $$y_{2,1} = (0.4011 \times 2) + (0.1978 \times 0) + (0.4011 \times 2) = 0.8022 + 0.8022 = \mathbf{1.6044}$$
  $$y_{2,2} = (0.4011 \times 0) + (0.1978 \times 2) + (0.4011 \times 2) = 0.3956 + 0.8022 = \mathbf{1.1978}$$
- **Row 3 Output:**
  $$y_{3,1} = (0.2483 \times 2) + (0.2483 \times 0) + (0.5034 \times 2) = 0.4966 + 1.0068 = \mathbf{1.5034}$$
  $$y_{3,2} = (0.2483 \times 0) + (0.2483 \times 2) + (0.5034 \times 2) = 0.4966 + 1.0068 = \mathbf{1.5034}$$

$$Y = \begin{bmatrix} 1.1978 & 1.6044 \\ 1.6044 & 1.1978 \\ 1.5034 & 1.5034 \end{bmatrix}$$

---

## 5. Production Selection Rules

- **Standard Self-Attention vs. Linear Attention Alternatives:**
  - *Standard Self-Attention:* Scales quadratically $O(n^2)$. Best for sequence lengths $< 8192$ where full token-to-token contextual accuracy is required.
  - *Linear Attention (e.g. State Space Models, Mamba):* Scales linearly $O(n)$ by rewriting the dot product order. Best for ultra-long context environments (e.g., $100\text{k}+$ tokens) where quadratic VRAM growth is unacceptable.

---

## 6. PyTorch Custom Self-Attention Layer

This is a clean, production-grade PyTorch implementation of a standard Self-Attention layer:

```python
import torch
import torch.nn as nn
import math

class CustomSelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super(CustomSelfAttention, self).__init__()
        self.d_k = d_k
        
        # Learnable projection matrices
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # 1. Project inputs to Q, K, V
        q = self.w_q(x)  # (batch_size, seq_len, d_k)
        k = self.w_k(x)  # (batch_size, seq_len, d_k)
        v = self.w_v(x)  # (batch_size, seq_len, d_k)
        
        # 2. Compute dot product scores
        # transpose(-2, -1) swaps seq_len and d_k dims for Key transpose
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        
        # 3. Scale scores
        scaled_scores = scores / math.sqrt(self.d_k)
        
        # 4. Apply Mask (if active)
        if mask is not None:
            # Mask is True where we want to keep, False where we hide (-inf)
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
            
        # 5. Softmax weights
        attention_weights = torch.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 6. Aggregate values
        output = torch.matmul(attention_weights, v)  # (batch_size, seq_len, d_k)
        
        return output, attention_weights

# Verification
x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
attn = CustomSelfAttention(d_model=2, d_k=2)

# Set pre-defined weights matching our hand-calculation
with torch.no_grad():
    attn.w_q.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    attn.w_k.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    attn.w_v.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))

out, weights = attn(x)
print("PyTorch Weights Output:\n", weights)
print("PyTorch Context Output:\n", out)
```
