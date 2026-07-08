# Interview Flashcards: 40 Transformer Questions & Answers

This guide contains high-impact, conceptual, mathematical, coding, and systems-level answers to the top 40 Transformer and LLM systems interview questions.

---

## Part 1: Theory & Core Concepts (10 Questions)

### Q1: Why is attention permutation invariant, and how do positional encodings resolve this?
- **Answer:** Attention computes scores using vector dot products ($q_i \cdot k_j$), which are independent of token position indices. Shuffling the inputs results in the exact same output values (just shuffled). Positional Encodings (PE) inject absolute or relative position coordinates (either via addition or rotation) into the input representations, allowing attention calculations to incorporate order.

### Q2: What are the key architectural differences between BERT, GPT, and T5?
- **Answer:**
  - **BERT (Encoder-only):** Uses bidirectional attention masks. Fully contextualizes representations by looking at both left and right contexts.
  - **GPT (Decoder-only):** Uses causal attention masks. Restricts tokens from looking at future steps, facilitating autoregressive text generation.
  - **T5 (Encoder-Decoder):** Uses bidirectional masks in the encoder and causal masks in the decoder, passing encoder representations to decoder layers via cross-attention.

### Q3: Why does standard Self-Attention scale quadratically ($O(n^2)$) in time and memory?
- **Answer:** Attention computes dot products between all $n$ queries and all $n$ keys, producing an $n \times n$ attention matrix. This requires $n^2$ multiplication operations and stores $n^2$ activation variables, causing computational and VRAM bounds as sequence length $n$ grows.

### Q4: Why does Pre-LayerNorm lead to more stable training than Post-LayerNorm in deep Transformers?
- **Answer:**
  - **Post-LN:** Normalizes after the residual addition ($x_{t+1} = \text{Norm}(x_t + F(x_t))$). Gradients flowing through early layers decay exponentially, causing instability and requiring learning rate warmups.
  - **Pre-LN:** Normalizes before the sub-layers ($x_{t+1} = x_t + F(\text{Norm}(x_t))$). Gradients flow directly through the residual highway without decay, ensuring stable training updates even without warmup.

### Q5: What is RMSNorm, and how does it differ from standard LayerNorm?
- **Answer:** LayerNorm centers and rescales activations using mean ($\mu$) and variance ($\sigma^2$). RMSNorm assumes centering is redundant and scales activations by their Root Mean Square:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$
  RMSNorm avoids computing the mean, saving approximately $10\%$ of computational overhead per layer.

### Q6: Explain the vanishing gradient problem in vanilla RNNs during BPTT.
- **Answer:** Backpropagating through $T$ steps requires multiplying the transpose of the hidden weights matrix $W_{hh}^T$ repeatedly: $\frac{\partial h_T}{\partial h_0} \propto (W_{hh}^T)^T$. If the spectral radius of $W_{hh}$ is less than 1, the gradient decays exponentially to $0.0$, preventing the model from learning long-term dependencies.

### Q7: What is SwiGLU, and why has it replaced ReLU in modern feed-forward layers?
- **Answer:** SwiGLU is a Gated Linear Unit where the gating path uses a Swish activation:
  $$\text{SwiGLU}(x) = (xW \cdot \text{sigmoid}(\beta xW)) \odot xV$$
  SwiGLU has a smoother gradient profile and has been shown to improve convergence speeds and overall validation metrics across large-scale pre-training runs.

### Q8: Explain how a causal mask is mathematically applied to attention logits.
- **Answer:** An upper-triangular matrix filled with $-\infty$ (diagonal=1) is added to the raw attention scores before softmax:
  $$\text{Scores}_{\text{masked}} = \frac{QK^T}{\sqrt{d_k}} + M_{\text{causal}}, \quad \text{where } M_{\text{causal}} = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$$
  Softmax maps $-\infty$ to exactly $0.0$ probability, preventing tokens from attending to future positions.

### Q9: What is Grouped-Query Attention (GQA), and why was it introduced?
- **Answer:** GQA groups query heads into groups, where each group shares a single Key-Value head. It was introduced to reduce the size of the KV Cache in memory, improving serving throughput and batch sizes while maintaining MHA's performance.

### Q10: How does a Mixture of Experts (MoE) network work, and what is active parameters count?
- **Answer:** MoE replaces the static FFN layer with $E$ independent "expert" FFN networks. A routing network selects the top-$k$ experts (typically $k=2$) for each token. The active parameters count is the size of the parameters actually activated for a single token ($k$ experts + routing parameters), while the total parameter count includes all experts in VRAM.

---

## Part 2: Mathematical & Dimensions (10 Questions)

### Q11: Show that the variance of the dot product $Q K^T$ is equal to $d_k$.
- **Answer:** Let $q, k \in \mathbb{R}^{d_k}$ be independent random vectors with $E[q_i] = E[k_i] = 0$ and $\text{Var}(q_i) = \text{Var}(k_i) = 1$. The dot product is $S = \sum_{i=1}^{d_k} q_i k_i$.
  - $E[q_i k_i] = E[q_i] E[k_i] = 0$.
  - $\text{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = E[q_i^2] E[k_i^2] = 1 \times 1 = 1$.
  - Since dimensions are independent:
    $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

### Q12: If a model has $d_{\text{model}} = 4096$, $h=32$ heads, what is the shape of the Q, K, and V tensors for batch size $B=2$ and sequence $T=512$?
- **Answer:** 
  - Head dimension $d_k = d_{\text{model}} / h = 4096 / 32 = 128$.
  - Before head transpose: Query shape is **$(2, \ 512, \ 32, \ 128)$**.
  - After head transpose: Query, Key, and Value shapes are **$(2, \ 32, \ 512, \ 128)$**.

### Q13: Given a 70 Billion parameter model utilizing Grouped-Query Attention (8 KV heads, 64 Query heads, head dim 128, 80 layers) serving in FP16 format, calculate the exact VRAM required to hold the KV Cache for a single request with a 4,000 token context window.
- **Answer:**
  - **Identify Model Configuration parameters:**
    - Layers ($L$) = $80$
    - KV Heads ($H_{\text{KV}}$) = $8$ (Note: the $64$ query heads are grouped to share these $8$ KV heads. For KV cache sizing, we only care about $H_{\text{KV}}$, not the query head count.)
    - Head Dimension ($d_{\text{head}}$) = $128$
    - Sequence Length ($T$) = $4000$
    - Batch Size ($B$) = $1$ (single request)
    - Precision scale ($P$) = $2$ bytes (FP16 precision)
  - **KV Cache VRAM Sizing Formula:**
    $$\text{Memory}_{\text{Bytes}} = 2 \text{ (Key \& Value)} \times P \text{ (Bytes/float)} \times L \times H_{\text{KV}} \times d_{\text{head}} \times T \times B$$
  - **Step-by-Step Calculation:**
    $$\text{Memory} = 2 \times 2 \times 80 \times 8 \times 128 \times 4000 \times 1 \text{ bytes}$$
    $$\text{Memory} = 4 \times 80 \times 8 \times 128 \times 4000 \text{ bytes}$$
    $$\text{Memory} = 320 \times 8 \times 128 \times 4000 = 2560 \times 128 \times 4000 = 327,680 \times 4000 \text{ bytes}$$
    $$\text{Memory} = 1,310,720,000 \text{ bytes}$$
  - **Unit Conversions:**
    - **Decimal system (GB, base-10):**
      $$\text{Memory}_{\text{GB}} = \frac{1,310,720,000}{1,000,000,000} = \mathbf{1.31072 \text{ GB}}$$
    - **Binary system (GiB, base-2):**
      $$\text{Memory}_{\text{GiB}} = \frac{1,310,720,000}{1,073,741,824} \approx \mathbf{1.2207 \text{ GiB}}$$

### Q14: Estimate the parameter count of a single Multi-Head Attention layer with $d_{\text{model}}=4096, h=32$.
- **Answer:**
  - $W_Q, W_K, W_V$ projection weights: $3 \times (4096 \times 4096) = 50,331,648$.
  - $W_O$ output projection weight: $4096 \times 4096 = 16,777,216$.
  - Total Parameters: $50,331,648 + 16,777,216 = \mathbf{67,108,864}$ parameters ($\approx 67.1\text{M}$).

### Q15: Show step-by-step how a 2D query vector $[1.0, 1.0]$ is rotated by RoPE at step index $m=2$ with base angle $\theta = \pi/2$.
- **Answer:**
  - Rotation angle $\phi = m\theta = 2 \times (\pi/2) = \pi$ ($180^\circ$).
  - Rotation Matrix $R = \begin{bmatrix} \cos(\pi) & -\sin(\pi) \\ \sin(\pi) & \cos(\pi) \end{bmatrix} = \begin{bmatrix} -1.0 & 0.0 \\ 0.0 & -1.0 \end{bmatrix}$.
  - Rotated Query:
    $$q_{\text{rot}} = R q = \begin{bmatrix} -1.0 & 0.0 \\ 0.0 & -1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-1.0} \\ \mathbf{-1.0} \end{bmatrix}$$

### Q16: Show step-by-step how RMSNorm normalizes vector $[1.0, 3.0, 4.0]$ (assume $\gamma = [1, 1, 1], \epsilon=0$).
- **Answer:**
  - $D = 3$.
  - $\sum_i x_i^2 = 1.0^2 + 3.0^2 + 4.0^2 = 1 + 9 + 16 = 26.0$.
  - $\text{RMS}(x) = \sqrt{26.0 / 3} = \sqrt{8.6667} \approx 2.9439$.
  - Normalized:
    $$y = \frac{x}{\text{RMS}(x)} = \frac{\begin{bmatrix} 1.0 & 3.0 & 4.0 \end{bmatrix}}{2.9439} \approx \begin{bmatrix} \mathbf{0.3397} & \mathbf{1.0190} & \mathbf{1.3587} \end{bmatrix}$$

### Q17: Contrast the parameter count of standard MHA vs. GQA (with $g=8$) vs. MQA.
- **Answer:**
  - **MHA:** Keys and Values are projected into $h$ heads. Parameter count $= 4 \times d_{\text{model}}^2$.
  - **GQA:** Keys and Values are projected into $g$ heads. Parameter count $= 2(1 + g/h) d_{\text{model}}^2$.
  - **MQA:** Keys and Values are projected into $1$ head. Parameter count $= 2(1 + 1/h) d_{\text{model}}^2$.

### Q18: Derive the time complexity of the self-attention layer projection phase vs. the scores calculation phase.
- **Answer:**
  - **Projection Phase:** Multiplying input $X \in \mathbb{R}^{n \times d}$ by weights $W \in \mathbb{R}^{d \times d}$ takes **$O(n \cdot d^2)$**.
  - **Scores Phase:** Multiplying $Q \in \mathbb{R}^{n \times d_k}$ by $K^T \in \mathbb{R}^{d_k \times n}$ takes **$O(n^2 \cdot d_k)$**.

### Q19: Why does dividing by $\sqrt{d_k}$ prevent the softmax gradient from vanishing?
- **Answer:** For large $d_k$, the dot product variance is $d_k$, meaning scores can be large in magnitude. Large scores saturate the softmax outputs to $0$ or $1$, where the derivative $\sigma'(z) \approx 0.0$. Dividing by $\sqrt{d_k}$ scales the variance back to $1.0$, keeping softmax in its active, high-gradient range.

### Q20: Show the binary representation layouts of FP16 vs. BF16, and explain why BF16 prevents underflow without loss scaling.
- **Answer:**
  - **FP16:** 1 sign bit, 5 exponent bits, 10 mantissa bits.
  - **BF16:** 1 sign bit, 8 exponent bits, 7 mantissa bits.
  - **Why BF16 prevents underflow:** BF16 shares the same 8 exponent bits as FP32, allowing it to represent the same dynamic range ($[2^{-126}, 2^{127}]$) as FP32. It does not suffer from FP16's limited range, eliminating the need for loss scaling.

---

## Part 3: Coding & PyTorch Layer Implementations (10 Questions)

### Q21: Write a PyTorch module implementing Scaled Dot-Product Attention from scratch.
```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights
```

### Q22: Write a PyTorch module implementing Grouped-Query Attention (GQA).
```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_k = d_model // n_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return torch.matmul(torch.softmax(scores, dim=-1), v)
```

### Q23: Write a PyTorch module implementing RMSNorm.
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.gamma
```

### Q24: Write a PyTorch module implementing a SwiGLU activation layer.
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return F.silu(self.w(x)) * self.v(x)
```

### Q25: Write a PyTorch script to create and apply a causal/padding mask.
```python
# Causal mask for sequence length T
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# Apply mask by replacing True values with -inf
scores = scores.masked_fill(mask, float('-inf'))
```

### Q26: Write a PyTorch function implementing the RoPE rotation logic.
```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

### Q27: How do you implement gradient norm clipping in PyTorch to prevent exploding gradients?
- **Answer:** Call `nn.utils.clip_grad_norm_` after backprop and before optimizer step:
  ```python
  loss.backward()
  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ```

### Q28: Write a PyTorch module representing a complete Encoder layer (Pre-LN, Self-Attention, FFN, Residual).
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### Q29: What is the difference between nn.Embedding and a standard linear layer in PyTorch?
- **Answer:** `nn.Embedding` acts as a lookup table (index indexing), extracting row vectors directly using token IDs. A linear layer performs full matrix multiplication on one-hot encoded vectors, which is computationally less efficient.

### Q30: Write a PyTorch script demonstrating how to save and load only the attention weights of a pre-trained model.
```python
# Save only attention weights
torch.save({k: v for k, v in model.state_dict().items() if 'attn' in k}, 'attn_weights.pth')
# Load back into model
weights = torch.load('attn_weights.pth')
model.load_state_dict(weights, strict=False)
```

---

## Part 4: System Design & Scenarios (10 Questions)

### Q31: How does FlashAttention reduce HBM access, and what are tile-based memory loads?
- **Answer:** Standard attention reads/writes the intermediate $N \times N$ attention matrix to GPU HBM multiple times. FlashAttention partitions Q, K, V matrices into tiles, loads them into fast GPU SRAM, computes the attention locally, and writes only the final results to HBM, bypassing HBM access bottlenecks.

### Q32: Your model is throwing CUDA OutOfMemory errors during LLM fine-tuning. What memory optimizations would you implement?
- **Answer:**
  1. Reduce the training `batch_size`.
  2. Enable **FP16/BF16 Mixed Precision**.
  3. Implement **Activation Checkpointing**.
  4. Use **Gradient Accumulation** steps.
  5. Implement parameter-efficient fine-tuning (**LoRA**).

### Q33: What is the KV Cache prefill phase vs. decode phase, and why is the decode phase memory-bandwidth bound?
- **Answer:** Prefill processes prompt tokens in parallel, which is compute-bound. Decode generates tokens one-by-one, requiring the GPU to read all weights and the KV cache from HBM for a single token computation, which is bottlenecked by HBM memory bandwidth.

### Q34: Explain the difference between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).
- **Answer:**
  - **PTQ:** Quantizes a pre-trained FP32 model directly to INT8 using calibration datasets to compute scale factors, which can degrade validation accuracy.
  - **QAT:** Simulates INT8 quantization rounding errors during training, allowing weights to adjust and preserving model accuracy.

### Q35: How does vLLM’s PagedAttention optimize serving memory and prevent physical fragmentation?
- **Answer:** PagedAttention divides the KV Cache into small, fixed-size blocks and stores them in non-contiguous physical memory pages (managed by a page lookup table). This prevents physical memory fragmentation, saving up to $96\%$ of wasted cache space.

### Q36: Explain Tensor Parallelism vs. Pipeline Parallelism in distributed LLM training.
- **Answer:**
  - **Tensor Parallelism:** Splits individual weight matrices across multiple GPUs (e.g. column/row parallel linear layers), synchronizing activations within each layer.
  - **Pipeline Parallelism:** Partitions model layers sequentially across GPUs (e.g., GPU 0 holds layers 1-8, GPU 1 holds 9-16), passing intermediate activations between devices.

### Q37: How would you choose between using an Encoder-only model (BERT) vs. a Decoder-only model (GPT) for a search engine extraction task?
- **Answer:** Choose an **Encoder-only model**. Bidirectional attention allows the model to fully contextualize representations by looking at both left and right contexts, which is more accurate and computationally cheaper for extraction/classification than decoder generation.

### Q38: Your training loss curve flatlines at step 1500. How would you diagnose and fix this stalled convergence?
- **Answer:**
  1. Check if the learning rate is too low.
  2. Implement **Learning Rate Schedulers** (like Cosine Annealing) to decay the rate dynamically.
  3. Increase the batch size.
  4. Verify that gradients are flowing properly (e.g., checking if hook outputs are non-zero).

### Q39: Your gradients evaluated in FP16 evaluate to NaN during LLM pre-training. How would you investigate and resolve this?
- **Answer:** Gradients are likely overflowing.
  1. Implement **Gradient Norm Clipping**.
  2. Verify that the **Dynamic Loss Scaler** is active and working.
  3. Switch to **BF16 precision** if hardware supports it to bypass FP16's narrow exponent range.

### Q40: Explain the relative communication overhead of DP, DDP (All-Reduce steps count), and PP.
- **Answer:**
  - **DP:** High overhead. The master GPU acts as a central hub, creating a PCIe bottleneck.
  - **DDP:** Bypasses central bottlenecks by using Ring All-Reduce, which synchronizes gradients in $2(N-1)$ communication steps independently of the master node.
  - **PP:** Low communication volume but introduces GPU idle time (bubble overhead), which is mitigated by using micro-batching.

### Q41: Whiteboard Component Review: Draw a standard decoder block utilizing Pre-LN and SwiGLU. Explain where residual connections pass through and what happens mathematically if we remove the residual connections completely.
- **Answer:**
  - **Schematic Drawing of a Pre-LN SwiGLU Decoder Layer:**
    ```text
             Input (x_l)
                 │
         ┌───────┴────────┐
         │               ▼
         │           RMSNorm
         │               │
         │         Self-Attention
         │               │
         ▼               ▼
         └───► [ + ] ◄───┘ (Residual Connection 1)
                 │
             (x_mid)
                 │
         ┌───────┴────────┐
         │               ▼
         │           RMSNorm
         │               │
         │          SwiGLU FFN
         │               │
         ▼               ▼
         └───► [ + ] ◄───┘ (Residual Connection 2)
                 │
            Output (x_{l+1})
    ```
  - **Residual Path Routing:**
    - The input to the block $x_l$ is split. One path goes through the normalization (`RMSNorm`) and the sub-layer (`Self-Attention`), and the other path bypasses them (the residual highway). They are added back: $x_{\text{mid}} = x_l + \text{Attention}(\text{RMSNorm}(x_l))$.
    - The output of the first addition $x_{\text{mid}}$ is similarly split: one path goes through normalization and the FFN sub-layer (`SwiGLU`), while the other path bypasses them. They are added: $x_{l+1} = x_{\text{mid}} + \text{SwiGLU}(\text{RMSNorm}(x_{\text{mid}}))$.
  - **Mathematical Analysis of Removing Residual Connections Completely:**
    If we remove residual connections, the layer equation becomes:
    $$x_{l+1} = \text{SwiGLU}\left(\text{RMSNorm}\left(\text{Attention}\left(\text{RMSNorm}(x_l)\right)\right)\right)$$
    Mathematically, this causes two critical issues that prevent training:
    1. **Vanishing Gradients:** 
       With residual connections, the gradient chain rule from final layer $L$ to layer $l$ is:
       $$\frac{\partial x_L}{\partial x_l} = \frac{\partial}{\partial x_l} \left[ x_l + \sum_{k=l}^{L-1} F(x_k) \right] = I + \sum_{k=l}^{L-1} \frac{\partial F(x_k)}{\partial x_l}$$
       The additive identity term $I$ ensures gradients flow back to early layers without exponential decay.
       Without residual connections, the gradient is a product of Jacobians:
       $$\frac{\partial x_L}{\partial x_l} = \prod_{k=l+1}^L J_k, \quad \text{where } J_k = \frac{\partial F(x_{k-1})}{\partial x_{k-1}}$$
       If the spectral radius of the sub-layer Jacobians $J_k$ is less than 1 (which it is for typical initializations and activations), the gradient decays exponentially with depth $L-l$, vanishing completely before reaching the initial layers ($l \to 0$), making deep architectures untrainable.
    2. **Representation Collapse / Loss of Identity:**
       As representation vectors pass through multiple successive non-linear operations without residual shortcuts, the dimensionality of the representation space collapses, and the network loses the ability to preserve identity mappings. The representations of different tokens contract to a single point, rendering representation learning impossible.

### Q42: Hardware-Level Deep Dive: Explain why the decoding step of an LLM is memory-bound rather than compute-bound, and explain how FlashAttention addresses GPU register utilization to optimize this.
- **Answer:**
  - **Prefill vs. Decode Bottlenecks (Arithmetic Intensity):**
    - **Arithmetic Intensity** is defined as the ratio of floating-point operations (FLOPs) to memory bytes transferred from high-bandwidth global memory (HBM) to local caches/registers:
      $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$
    - **Prefill Phase (Compute-Bound):** Processes the prompt tokens in parallel. For a prompt of length $N$, computing the projections and self-attention scores involves dense matrix-matrix multiplications ($Q \cdot K^T$ of shapes $(N \times d) \times (d \times N)$). The FLOPs count scales as $O(N^2 d)$, while the parameters and inputs loaded scale as $O(N d + d^2)$. The arithmetic intensity is very high, saturating GPU Tensor Cores.
    - **Decode Phase (Memory-Bound):** Generates one token at a time. The query is a single token vector ($q \in \mathbb{R}^{1 \times d}$). The key and value caches are loaded from HBM to compute attention. The FLOPs count scales as $O(d)$ per token, while we must load the massive model weights of size $W$ and the historical KV Cache of size $2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times T$ bytes from HBM to compute a single token's representation. The arithmetic intensity is extremely low ($< 1$ FLOP per byte), so GPU execution units sit idle waiting for memory transfer to complete.
  - **How FlashAttention Addresses GPU Register and SRAM Utilization:**
    Standard attention computes the intermediate attention matrix $S = \frac{QK^T}{\sqrt{d_k}}$ (size $N \times N$), writes it to HBM, reads it from HBM to perform softmax, writes the softmax matrix $P$ to HBM, and reads $P$ again to compute $O = P V$.
    FlashAttention optimizes this by utilizing local **SRAM** and **Registers** directly:
    1. **Tiling:** FlashAttention loads small, fixed-size blocks (tiles) of $Q$, $K$, and $V$ from HBM into local SRAM cache ($128 \times 128$ blocks, fitting within the GPU's SRAM memory limit).
    2. **Online Softmax in Registers:** To compute softmax on SRAM tiles incrementally, it maintains local scaling statistics (running maximum $m$ and running sum of exponents $d$) inside the fast **register file** of each streaming multiprocessor (SM). As new $K, V$ tiles are loaded, it updates these registers and scales the accumulated output tile in SRAM, avoiding writing the intermediate $N \times N$ matrix to HBM.
    3. **Backward Pass Recomputation:** By saving only the final output $O$ and scaling statistics ($m, d$) to HBM, it saves $O(N^2)$ memory bandwidth. During the backward pass, it loads the $Q, K, V$ tiles from HBM into SRAM again and recomputes the intermediate gradients locally on-the-fly using the register scaling statistics, substituting slow HBM read transactions with fast, register-assisted compute FLOPs.
