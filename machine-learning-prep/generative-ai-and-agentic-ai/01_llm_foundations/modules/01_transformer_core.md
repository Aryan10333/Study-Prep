# Module 01: The Transformer Architecture (Encoder & Decoder Core)

## 1. Introduction & Intuition

### The Core Bottleneck
Legacy sequence modeling architectures, such as RNNs, LSTMs, and GRUs, process sequences sequentially. To compute the hidden state $h_t$ at time step $t$, the recurrent cell must ingest the hidden state $h_{t-1}$ from the previous step. This sequential dependency creates a severe computational bottleneck: it is mathematically impossible to parallelize training across the sequence dimension. As a result, training on modern web-scale text corpora takes an unacceptable amount of time. Furthermore, recurrent hidden states suffer from vanishing or exploding gradients over long sequence lengths $L$, limiting their effective context window. The bottleneck is finding an architecture that can model sequence relationships in parallel while preserving long-range dependencies.

### High-Level Intuition
The Transformer architecture resolves the parallelization bottleneck by discarding recurrence entirely and replacing it with **Self-Attention**. Instead of passing hidden states step-by-step through time, the model processes all tokens in the sequence simultaneously. 
To model the relationships between tokens, the Transformer uses an attention mechanism where every token "looks" at every other token in the sequence. Each token calculates its similarity to all other tokens, creating a weighted sum of their representations. 

To organize this processing, a standard Transformer block consists of:
1. **Multi-Head Self-Attention (MHA)**: Allows tokens to gather context from other parts of the sequence.
2. **Feed-Forward Network (FFN)**: Applies non-linear transformations to each token's representation independently.
3. **Layer Normalization (LN) & Residual Connections**: Provide gradient highways to allow stable training of deep networks.

Depending on normalizations placement, we have:
*   **Post-LN**: Normalized after the residual addition (used in the original Vaswani et al. model). Requires careful learning rate warm-up to prevent gradient instability.
*   **Pre-LN**: Normalized on the shortcut branch *before* entering the attention or FFN layer. This is standard in modern LLMs (e.g., Llama, GPT) because it enables much more stable gradient flow.

---

### Layer Normalization Placement Comparison
Below is an inline SVG illustrating the architectural difference between Post-LN and Pre-LN Transformer blocks:

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 800 350" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <!-- POST-LN BLOCK -->
  <g transform="translate(40, 20)">
    <rect x="0" y="0" width="320" height="310" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="160" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">Post-LN Transformer Block</text>
    
    <!-- Input -->
    <text x="160" y="295" text-anchor="middle" font-size="12" fill="#64748b">Input Vector x</text>
    <path d="M 160 280 L 160 240" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    
    <!-- Multi-Head Attention -->
    <rect x="60" y="190" width="200" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="160" y="220" text-anchor="middle" font-size="12" font-weight="semibold" fill="#1e3a8a">Multi-Head Attention (MHA)</text>
    
    <!-- Add & Norm (MHA) -->
    <path d="M 160 190 L 160 140" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    <rect x="60" y="90" width="200" height="50" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" />
    <text x="160" y="115" text-anchor="middle" font-size="12" font-weight="semibold" fill="#7f1d1d">Add &amp; LayerNorm</text>
    <text x="160" y="130" text-anchor="middle" font-size="10" fill="#991b1b">LN(x + MHA(x))</text>
    
    <!-- Residual Shortcut -->
    <path d="M 160 260 L 30 260 L 30 115 L 60 115" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4" fill="none" marker-end="url(#arrow)" />
    
    <!-- Output -->
    <path d="M 160 90 L 160 55" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    <text x="160" y="50" text-anchor="middle" font-size="12" fill="#64748b">Output to FFN</text>
  </g>

  <!-- PRE-LN BLOCK -->
  <g transform="translate(440, 20)">
    <rect x="0" y="0" width="320" height="310" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="160" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">Pre-LN Transformer Block</text>
    
    <!-- Input -->
    <text x="160" y="295" text-anchor="middle" font-size="12" fill="#64748b">Input Vector x</text>
    <path d="M 160 280 L 160 250" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    
    <!-- LayerNorm first -->
    <rect x="60" y="200" width="200" height="50" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" />
    <text x="160" y="225" text-anchor="middle" font-size="12" font-weight="semibold" fill="#7f1d1d">Layer Normalization (LN)</text>
    <text x="160" y="240" text-anchor="middle" font-size="10" fill="#991b1b">LN(x)</text>
    
    <!-- Multi-Head Attention -->
    <path d="M 160 200 L 160 160" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    <rect x="60" y="110" width="200" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="160" y="140" text-anchor="middle" font-size="12" font-weight="semibold" fill="#1e3a8a">Multi-Head Attention (MHA)</text>
    
    <!-- Add node -->
    <path d="M 160 110 L 160 70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    <circle cx="160" cy="70" r="12" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" />
    <text x="160" y="74" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">+</text>
    
    <!-- Residual Shortcut -->
    <path d="M 160 265 L 30 265 L 30 70 L 148 70" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4" fill="none" marker-end="url(#arrow)" />
    
    <!-- Output -->
    <path d="M 160 58 L 160 45" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)" />
    <text x="160" y="40" text-anchor="middle" font-size="12" fill="#64748b">Output = x + MHA(LN(x))</text>
  </g>
  
  <!-- Marker -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### The Scaled Dot-Product Attention
The core of the Transformer is the Scaled Dot-Product Attention mechanism. It takes three input matrices: Queries ($Q$), Keys ($K$), and Values ($V$).
*   **Queries ($Q$)**: The current representations searching for context.
*   **Keys ($K$)**: The representations acting as index cards to match against queries.
*   **Values ($V$)**: The actual content payload vector representation of the tokens.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

*   **Purpose & High-level Intuition:** The dot product $Q K^T$ calculates raw similarity scores between every query and key token. We scale by $1 / \sqrt{d_k}$ to prevent the dot products from growing excessively large for high dimensions $d_k$. Large values push the softmax function into regions of extremely small gradients (gradient saturation), causing vanishing gradients. The softmax converts raw similarity scores into a probability distribution, which is used to construct a weighted sum of the values $V$.

---

### Hand Calculations: Scaled Dot-Product Attention
Let's perform a step-by-step arithmetic walk-through.

#### 1. Setup Input Matrices
Let sequence length $L = 2$, and query/key dimension $d_k = 2$.
Let Query ($Q$), Key ($K$), and Value ($V$) matrices be:
$$Q = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix}, \quad K = \begin{pmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \end{pmatrix}, \quad V = \begin{pmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{pmatrix}$$

#### Step 1: Compute Dot-Product Similarity ($S = Q K^T$)
$$\begin{aligned}
S &= \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix} \begin{pmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \end{pmatrix} \\
&= \begin{pmatrix} 1.0 \times 1.0 + 0.0 \times 0.0 & 1.0 \times 1.0 + 0.0 \times 1.0 \\ 0.0 \times 1.0 + 1.0 \times 0.0 & 0.0 \times 1.0 + 1.0 \times 1.0 \end{pmatrix} \\
&= \begin{pmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \end{pmatrix}
\end{aligned}$$

#### Step 2: Apply Scaling Factor ($1 / \sqrt{d_k}$)
Since $d_k = 2$, the scaling factor is $1 / \sqrt{2} \approx 0.7071$.
$$\begin{aligned}
S_{\text{scaled}} &= \begin{pmatrix} 1.0 \times 0.7071 & 1.0 \times 0.7071 \\ 0.0 \times 0.7071 & 1.0 \times 0.7071 \end{pmatrix} \\
&= \begin{pmatrix} 0.7071 & 0.7071 \\ 0.0 & 0.7071 \end{pmatrix}
\end{aligned}$$

#### Step 3: Compute Row-Wise Softmax ($A = \text{softmax}(S_{\text{scaled}})$)
For row 0: $x_1 = 0.7071$, $x_2 = 0.7071$.
$$e^{0.7071} \approx 2.0281$$
$$\text{Sum} = 2.0281 + 2.0281 = 4.0562$$
$$A_{0,0} = \frac{2.0281}{4.0562} = 0.5, \quad A_{0,1} = 0.5$$

For row 1: $x_1 = 0.0$, $x_2 = 0.7071$.
$$e^0 = 1.0, \quad e^{0.7071} \approx 2.0281$$
$$\text{Sum} = 1.0 + 2.0281 = 3.0281$$
$$A_{1,0} = \frac{1.0}{3.0281} \approx 0.3302, \quad A_{1,1} = \frac{2.0281}{3.0281} \approx 0.6698$$

So the attention weight matrix is:
$$A = \begin{pmatrix} 0.5 & 0.5 \\ 0.3302 & 0.6698 \end{pmatrix}$$

#### Step 4: Multiply by Values Matrix ($O = A V$)
$$\begin{aligned}
O &= \begin{pmatrix} 0.5 & 0.5 \\ 0.3302 & 0.6698 \end{pmatrix} \begin{pmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{pmatrix} \\
&= \begin{pmatrix} 
(0.5 \times 1.0) + (0.5 \times 3.0) & (0.5 \times 2.0) + (0.5 \times 4.0) \\ 
(0.3302 \times 1.0) + (0.6698 \times 3.0) & (0.3302 \times 2.0) + (0.6698 \times 4.0) 
\end{pmatrix} \\
&= \begin{pmatrix} 
0.5 + 1.5 & 1.0 + 2.0 \\ 
0.3302 + 2.0094 & 0.6604 + 2.6792 
\end{pmatrix} \\
&\approx \begin{pmatrix} 2.0 & 3.0 \\ 2.3396 & 3.3396 \end{pmatrix}
\end{aligned}$$

---

### Tensor & Shape Tracking
*   **Input Token Embeddings**: `[B, L, d]` (where $B$ is batch size, $L$ is sequence length, and $d$ is model dimension).
*   **Queries ($Q$) / Keys ($K$) / Values ($V$)**: `[B, L, d]`
*   **Multi-Head split (per head)**: `[B, h, L, d_k]` (where $h$ is head count, and $d_k = d/h$ is head dimension).
*   **Dot-Product Similarity matrix ($Q K^T$)**: `[B, h, L, L]`
*   **Attention Probabilities ($A$)**: `[B, h, L, L]`
*   **Output Vector ($A V$)**: `[B, L, d]`

---

## 3. Implementation & Reference Code

Below is a complete, self-contained PyTorch implementation of a Pre-LN Multi-Head Self-Attention block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreLNMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: [B, L, d]
        B, L, d = x.shape
        
        # 1. Apply LayerNorm (Pre-LN style)
        norm_x = self.ln(x) # [B, L, d]
        
        # 2. Project inputs to Q, K, V
        q = self.q_proj(norm_x) # [B, L, d]
        k = self.k_proj(norm_x) # [B, L, d]
        v = self.v_proj(norm_x) # [B, L, d]
        
        # 3. Reshape for Multi-Head Attention: [B, L, h, d_k] -> [B, h, L, d_k]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Compute Scaled Dot-Product Similarity: [B, h, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 5. Apply causal mask or padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 6. Apply Softmax to get attention weights: [B, h, L, L]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. Weighted sum of values: [B, h, L, d_k]
        context = torch.matmul(attn_weights, v)
        
        # 8. Concatenate heads and project: [B, L, d]
        context = context.transpose(1, 2).contiguous().view(B, L, d)
        out = self.out_proj(context)
        
        # 9. Residual connection
        return x + out # [B, L, d]

# Simple verification block
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d, h = 2, 8, 16, 4
    x = torch.randn(B, L, d)
    
    # Causal Mask (lower triangular)
    causal_mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
    
    mha = PreLNMultiHeadAttention(embed_dim=d, num_heads=h)
    out = mha(x, mask=causal_mask)
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Shape mismatch!"
    print("Block executed successfully and verified shapes!")
```

---

## Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** The computational bottleneck of sequential recurrent transitions ($O(L)$ sequential dependencies) preventing full GPU parallelization during training.
*   **Why Introduced over Legacy Approaches:** It replaced LSTMs/RNNs because Self-Attention processes the entire sequence in parallel, utilizing massive GPU parallelism and enabling training on multi-billion token datasets.
*   **Key Failure Modes & Limitations:** Vanilla self-attention has a memory and computation footprint scaling quadratically $O(L^2)$ with sequence length, making long context inputs extremely expensive.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Projection matrices scale as $O(L \cdot d^2)$. Attention dot-product scales as $O(L^2 \cdot d)$. FFN scales as $O(L \cdot d^2)$.
*   **Space/Memory Footprint:** Intermediate activation storage scales as $O(L^2 \cdot h)$ per layer due to storing the attention weight matrix.
*   **Primary Bottleneck Type:** Memory-bandwidth-bound during self-attention softmax and lookup; Compute-bound during projection multiplications ($Q, K, V$, FFN linear layers).
*   **Variable Legend:** $B$ = Batch Size, $L$ = Sequence Length, $d$ = Hidden Model Dimension, $h$ = Head Count, $d_k$ = Head Dimension ($d/h$).

### 3. Production & Scalability
*   **Deployment Considerations:** Attention weight tables ($L^2$) run out of memory quickly. Optimizations like FlashAttention (Module 07) must be used to bypass storing the full attention weight matrix in global memory.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why is the scaling factor $\sqrt{d_k}$ so critical in the attention denominator?
        *   *A:* Without scaling, for large $d_k$ (e.g. 128), the variance of the dot product $Q K^T$ grows to $d_k$. This produces large values in the inputs to the softmax, pushing the outputs to extremely small or large elements where gradients are close to 0. Dividing by $\sqrt{d_k}$ maintains a variance of 1, preserving gradient flow.
    2.  *Q:* Why is Pre-LN preferred over Post-LN in deep LLMs?
        *   *A:* In Post-LN, the gradient through the residual block scales with the depth of the network, which forces the use of a delicate warm-up learning rate schedule to avoid gradient explosion. In Pre-LN, the gradients can pass directly through the identity residual branch without scaling, ensuring stability during initial training phases and allowing deep models to converge reliably.
