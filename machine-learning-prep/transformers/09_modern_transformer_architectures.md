# Transformers: Modern Architectures (SwiGLU, MoE & context extensions)

This guide details the evolution of modern Large Language Model (LLM) architectures (such as Llama-3, Mistral, and Mixtral), explaining the mathematical changes behind SwiGLU and Mixture of Experts (MoE), walking through a SwiGLU hand-calculation, and detailing their production systems tradeoffs.

---

## 1. How Modern LLMs Differ from the Original Transformer

Modern LLMs are built on the original Transformer architecture but introduce key optimizations to stabilize training and improve inference speed:

```text
Architectural Component   Original Transformer (Attention Is All You Need)   Modern LLMs (Llama-3, Mistral)
----------------------------------------------------------------------------------------------------------------------
Normalization Layering    Post-LN (Norm after residuals)                     Pre-LN (Norm before sub-layers)
Normalization Engine      LayerNorm (Requires mean calculation)              RMSNorm (Root Mean Square scale only)
Activation Function       ReLU (FFN)                                         SwiGLU (Gated Linear Units)
Positional Encoding       Absolute Sinusoidal Embeddings                    Rotary Positional Embeddings (RoPE)
Attention Mechanism       Multi-Head Attention (MHA)                         Grouped-Query Attention (GQA)
```

---

## 2. SwiGLU Activation Function

SwiGLU replaces standard ReLU in the Feed-Forward Network (FFN) layers. It is a **Gated Linear Unit (GLU)** where the gating path is activated by a Swish function:
$$\text{SwiGLU}(x) = \text{Swish}_\beta(x W) \odot x V$$
$$\text{Swish}_\beta(z) = z \cdot \sigma(\beta z) = \frac{z}{1 + e^{-\beta z}}$$

Where:
- $W$ and $V$ are weight projection matrices.
- $\odot$ represents element-wise multiplication.
- $\beta$ is typically set to $1.0$.
- **Production Utility:** SwiGLU has a smoother gradient profile and has been shown to improve convergence speeds and overall validation metrics across large-scale pre-training runs.

---

## 3. Mixture of Experts (MoE)

Instead of passing every token through the same static FFN block, MoE models contain a collection of $E$ independent "expert" FFN networks ($E_1, \dots, E_E$) and a **Gating/Router Network** $G(x)$.

The gating network outputs routing weights for each expert, selecting only the top-$k$ experts (typically $k=2$) to process the token:
$$y = \sum_{i \in \text{top-k}} G(x)_i E_i(x)$$

- **Conditional Computation:** Only the selected $k$ experts are activated for a given token. 
- **Production Utility:** Allows models to scale their parameter capacity (e.g. up to 100B+ parameters) while only activating a subset of them (e.g. 12B active parameters) per token, keeping the compute cost ($O(\text{active params})$) low.

---

## 4. Step-by-Step Hand Calculations: SwiGLU Activation (Andrew Ng Style)

Let's evaluate SwiGLU on a single scalar input $x = 1.0$ with $\beta=1.0$ and parameters:
- $W = [2.0]$ | $V = [0.5]$ | Biases = $0$

1. **Compute Projections:**
   $$x W = 1.0 \times 2.0 = 2.0$$
   $$x V = 1.0 \times 0.5 = 0.5$$
2. **Compute Swish Gate Output ($\text{Swish}(2.0)$):**
   $$\text{Swish}(2.0) = \frac{2.0}{1 + e^{-2.0}} \approx \frac{2.0}{1 + 0.1353} = \frac{2.0}{1.1353} \approx \mathbf{1.7616}$$
3. **Compute Final SwiGLU Output:**
   $$\text{SwiGLU}(x) = \text{Swish}(2.0) \times (x V) = 1.7616 \times 0.5 = \mathbf{0.8808}$$

**Result:** The SwiGLU activation output is **$0.8808$**.

---

## 5. Production Scenario & Example

### Scenario: Deploying Mixtral-8x7B (MoE) under high throughput constraints
You deploy a Mixtral-8x7B model (which contains 47B total parameters and activates 13B parameters per token) on an edge server node.
- **The Failure Mode:** During peak loads, GPU utilization is low, and routing calculations introduce significant latency overhead. Profiling shows that since different tokens are routed to different experts, the workload cannot be batched efficiently (known as **expert capacity skew**). Some experts sit idle while others are overloaded.
- **The Solution:** 
  1. You enable **capacity factor routing limits** that drop tokens if an expert receives more than its fair share of inputs, routing them to the next best expert.
  2. You configure **tensor parallelism** to split experts across GPUs, reducing HBM transfer overhead and keeping serving latency within SLA bounds.

---

## 6. PyTorch SwiGLU Implementation

This code implements a SwiGLU FFN layer in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # We need three projection weight matrices: W, V, and Output
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # x shape: (B, T, d_model)
        
        # 1. Compute parallel projections
        proj_w = self.w(x)  # (B, T, d_ff)
        proj_v = self.v(x)  # (B, T, d_ff)
        
        # 2. Apply Swish activation to the gate path (W)
        # silu is PyTorch's native Swish function: x * sigmoid(x)
        gate = F.silu(proj_w)
        
        # 3. Combine gating and value paths (V)
        # Element-wise multiplication
        gated_activation = gate * proj_v  # (B, T, d_ff)
        
        # 4. Project back to model dimension
        return self.out_proj(gated_activation)

# Verify shapes
x = torch.randn(2, 5, 128)  # Batch=2, Seq=5, d_model=128
ffn = SwiGLUFFN(d_model=128, d_ff=256)
out = ffn(x)
print("SwiGLU FFN Output Tensor Shape:", out.shape)  # Expected: (2, 5, 128)
```
