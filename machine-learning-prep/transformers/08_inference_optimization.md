# Transformers: Inference Optimizations & KV Caching

This guide details the systems-level bottlenecks of autoregressive decoding, the prefill vs. decode phases, the KV Cache memory equation, FlashAttention hardware tiling, a step-by-step KV cache size hand-calculation, and throughput serving optimizations.

---

## 1. Autoregressive Decoding Phases

Autoregressive inference generates text token-by-token. This occurs in two distinct execution phases:

```text
Decoders Phase      Hardware Bottleneck       Compute Pattern                     Execution Efficiency
----------------------------------------------------------------------------------------------------------------------
Prefill Phase       Compute-Bound             Parallel (Processes prompt tokens)  High GPU Tensor Core utilization
Decode Phase        Memory-Bandwidth Bound   Sequential (Generates one token)    Low GPU execution efficiency
```

### A. The Prefill Phase (Parallel)
Processes the input prompt tokens in parallel to generate the initial hidden states and Key-Value representations. This is a compute-bound operation, leveraging GPU Tensor Cores efficiently.

### B. The Decode Phase (Sequential)
Generates subsequent tokens one-by-one. Each step reads the single newly generated token, projects it to QKV, retrieves the stored KV states of all prior tokens from GPU memory, runs self-attention, and updates the memory cache.
- **The Memory Bandwidth Bottleneck:** In the decode phase, the GPU execution cores sit idle most of the time because loading model parameters and historical KV cache tensors from High-Bandwidth Memory (HBM) to local registers takes much longer than the actual mathematical computations.

---

## 2. The KV Cache

During decoding, computing attention requires comparing the current token with all prior tokens in the sequence. To avoid re-calculating the key-value projections of all historical tokens at every step, we save them in memory as the **KV Cache**.

- **Why it is helpful:** It reduces the computation complexity of each decoding step from $O(T^2)$ to $O(T)$, significantly decreasing latency.
- **The Trade-off:** The KV cache grows linearly with batch size and context length, consuming gigabytes of VRAM.

### The KV Cache VRAM Formula
$$\text{Memory}_{\text{KV\_Cache}} = 2 \times P \times L \times H_{\text{KV}} \times d_{\text{head}} \times T \times B \text{ bytes}$$

Where:
- **$2$** represents storing both Key and Value matrices.
- **$P$** is the precision weight scale (e.g., $2$ bytes for FP16/BF16, $1$ byte for INT8).
- **$L$** is the number of transformer layers.
- **$H_{\text{KV}}$** is the number of Key-Value attention heads.
- **$d_{\text{head}}$** is the dimensionality of each head.
- **$T$** is the target sequence context length.
- **$B$** is the serving batch size.

---

## 3. FlashAttention: High-Level Overview

Standard self-attention reads and writes the intermediate $N \times N$ attention weight matrix back and forth between GPU High-Bandwidth Memory (HBM) and fast local cache (SRAM) multiple times, creating a memory bottleneck.

- **The FlashAttention Solution:**
  - Uses **Tiling** to break the Query, Key, and Value matrices into small blocks.
  - Loads these blocks into SRAM, performs the attention calculations locally, and writes only the final context results back to HBM.
  - Completely avoids writing the intermediate $N \times N$ attention matrix to HBM.
  - **Impact:** Decreases memory access times, yielding a **$2\text{x} - 4\text{x}$ execution speedup** without losing mathematical accuracy.

---

## 4. Step-by-Step Hand Calculations: KV Cache Size (Andrew Ng Style)

Let's calculate the VRAM requirements of the KV Cache for a standard **Llama-2-7B** model under production serving conditions:
- **Configurations:**
  - Number of Layers ($L$) = $32$
  - Number of KV heads ($H_{\text{KV}}$) = $32$
  - Head dimension ($d_{\text{head}}$) = $128$
  - Context Length ($T$) = $2048$ tokens
  - Batch Size ($B$) = $4$
  - Precision = FP16 ($P = 2$ bytes)

---

### Step 1: Compute Parameters per Token per Layer
Each token requires storing both Key and Value vectors:
$$\text{Values} = 2 \times H_{\text{KV}} \times d_{\text{head}} = 2 \times 32 \times 128 = 8,192 \text{ float values}$$

---

### Step 2: Scale across all Layers
$$\text{Values}_{\text{Total}} = L \times \text{Values} = 32 \times 8,192 = 262,144 \text{ float values per token}$$

---

### Step 3: Scale across Batch and Context Length
$$\text{Values}_{\text{Batch}} = 262,144 \times T \times B = 262,144 \times 2048 \times 4 = 2,147,483,648 \text{ values}$$

---

### Step 4: Convert to Bytes using Precision ($P=2$)
$$\text{Memory}_{\text{Bytes}} = \text{Values}_{\text{Batch}} \times P = 2,147,483,648 \times 2 = 4,294,967,296 \text{ bytes}$$
$$\text{Memory}_{\text{Gigabytes}} = \frac{4,294,967,296}{1,073,741,824} = \mathbf{4.00 \text{ GB}}$$

**Result:** Storing key-value states for a small batch size of 4 consumes exactly **$4.00\text{ GB}$ of VRAM**.

---

## 5. Production Scenario & Example

### Scenario: Scaling chatbot API latency under concurrent loads
You deploy a customer-facing LLM API. During peak hours, as concurrent batch loads grow, output generation latency spikes from $30$ tokens/sec to $2$ tokens/sec, violating SLA latency agreements.
- **The Failure Mode:** The system memory is saturated. The GPU must constantly page segments of the KV cache in and out of GPU memory (HBM) to process concurrent requests, causing severe memory bandwidth bottlenecks.
- **The Solution:** 
  1. You enable **vLLM PagedAttention**, which allocates KV cache memory dynamically in non-contiguous physical memory pages (like virtual memory in operating systems), preventing fragmentation and saving up to $96\%$ of wasted cache space.
  2. You implement **INT8 Quantization** on the KV cache, halving the memory footprint to $2\text{GB}$ and doubling generation throughput.

---

## 6. PyTorch Mock KV Cache Update Loop

This code demonstrates how Key-Value tensors are cached and retrieved during decode iterations:

```python
import torch
import torch.nn as nn

class KVDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        # x shape: (B, 1, d_model) - single decode token input
        batch_size = x.shape[0]
        
        # 1. Project current token representation
        q = self.q_proj(x).view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        k_new = self.k_proj(x).view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        v_new = self.v_proj(x).view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Update KV Cache
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            # Concatenate along the token sequence dimension
            k = torch.cat([k_cached, k_new], dim=-2)
            v = torch.cat([v_cached, v_new], dim=-2)
        else:
            k = k_new
            v = v_new
            
        current_cache = (k, v)
        
        # 3. Compute Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, current_cache

# Mock Decode Loop
batch, d_model, n_heads = 1, 64, 4
decoder = KVDecoderBlock(d_model, n_heads)

x_token1 = torch.randn(batch, 1, d_model)
x_token2 = torch.randn(batch, 1, d_model)

# Decoding step 1 (generates cache)
out1, cache = decoder(x_token1)
print("Step 1 Cache Shapes (Key, Value):", cache[0].shape, cache[1].shape)

# Decoding step 2 (uses cache)
out2, cache = decoder(x_token2, kv_cache=cache)
print("Step 2 Cache Shapes (Key, Value):", cache[0].shape, cache[1].shape)
```
