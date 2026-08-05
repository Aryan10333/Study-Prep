# Module 06: KV Cache Mechanics & Memory Bottlenecks

## 1. Introduction & Intuition

### The Core Bottleneck
In an autoregressive decoder-only LLM, generation proceeds token-by-token. To generate token $t$, the model must compute self-attention scores between token $t$ and all prior tokens $1, 2, \dots, t-1$. If we implement this naively, we must recalculate the Query, Key, and Value vectors for all past tokens at every generation step. This results in $O(L^2)$ redundant computations, causing generation latency to explode. 

To bypass this redundant calculation, we use a **Key-Value Cache (KV Cache)**. We compute and save the Key and Value vectors of past tokens in VRAM. At step $t$, we only project the single newly generated token to get $q_t, k_t, v_t$, and append $k_t, v_t$ to our cache. 

While this eliminates redundant computations, the KV cache creates a new bottleneck: **GPU memory capacity and memory bandwidth**. The VRAM required to store the KV cache scales linearly with sequence length and batch size. Under large batch sizes and context windows, the KV cache can consume tens of gigabytes of VRAM, choking the GPU and causing Out-Of-Memory (OOM) errors.

### High-Level Intuition
Think of generating text like writing a book one word at a time. To write the next word, you must reread everything you have written so far. 
*   **Without KV Cache**: You read the entire book from page 1 to the current page to write a single word. Then, you write the word, go back to page 1, and read the entire book again to write the next word. This is slow and redundant.
*   **With KV Cache**: As you write each page, you summarize and index the key details on sticky notes. To write the next word, you only read the single word you just wrote and refer to your sticky notes. You don't go back to reread the whole book. However, as the book gets longer, your desk (VRAM) gets covered in sticky notes. If the desk runs out of space, you hit an Out-Of-Memory error.

---

### Autoregressive Decoding with KV Cache
Below is an inline SVG illustrating the execution flow of the KV Cache update during the decoding phase:

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 750 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <!-- Prefill state (grayed background) -->
  <rect x="20" y="40" width="280" height="180" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-dasharray="3" />
  <text x="160" y="25" text-anchor="middle" font-size="12" font-weight="bold" fill="#475569">Step 1: Prefill Phase (Prompt)</text>
  <text x="160" y="65" text-anchor="middle" font-size="11" fill="#334155">Processes entire prompt in parallel</text>
  <text x="160" y="85" text-anchor="middle" font-size="11" fill="#334155">Computes &amp; stores all K &amp; V states</text>
  
  <rect x="60" y="120" width="200" height="50" rx="4" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5" />
  <text x="160" y="150" text-anchor="middle" font-size="11" font-weight="semibold" fill="#334155">KV Cache Initialized</text>
  
  <!-- Arrow to decoding -->
  <path d="M 310 130 L 370 130" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow-kv)" />
  
  <!-- Decoding step t -->
  <rect x="390" y="40" width="340" height="180" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
  <text x="560" y="25" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">Step 2: Decoding Phase (Token t)</text>
  
  <!-- Newly generated token -->
  <rect x="410" y="60" width="130" height="40" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
  <text x="475" y="85" text-anchor="middle" font-size="10" font-weight="semibold" fill="#1e3a8a">New token input x_t</text>
  
  <!-- Projects to q_t, k_t, v_t -->
  <path d="M 475 100 L 475 130" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#arrow-kv)" />
  <text x="490" y="120" font-size="9" fill="#1e3a8a">Project</text>
  
  <text x="475" y="150" text-anchor="middle" font-size="10" fill="#334155">q_t, k_t, v_t</text>
  
  <!-- Append to cache -->
  <rect x="580" y="120" width="130" height="80" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" />
  <text x="645" y="140" text-anchor="middle" font-size="10" font-weight="bold" fill="#7f1d1d">KV Cache Storage</text>
  <rect x="590" y="155" width="110" height="18" rx="2" fill="#fee2e2" />
  <text x="645" y="168" text-anchor="middle" font-size="9" fill="#991b1b">K_prev , V_prev</text>
  <rect x="590" y="177" width="110" height="18" rx="2" fill="#ef4444" />
  <text x="645" y="190" text-anchor="middle" font-size="9" fill="#ffffff" font-weight="bold">+ [k_t, v_t]</text>
  
  <!-- Draw line showing routing to merge -->
  <path d="M 515 150 L 580 150" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3" marker-end="url(#arrow-kv)" />
  
  <!-- Marker -->
  <defs>
    <marker id="arrow-kv" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### Mathematical Formulations

#### 1. KV Cache Memory Footprint Sizing
$$\text{VRAM}_{\text{KVCache}} = 2 \times 2 \times B \times L \times n_{\text{layers}} \times n_{\text{heads\_kv}} \times d_{\text{head}} \times \text{BytesPerParam}$$

*   **Purpose & High-level Intuition:** This formula calculates the exact memory required to store Key and Value tensors for all active inference runs. 
    *   The first factor $2$ accounts for storing **two separate matrices**: Keys and Values.
    *   The second factor $2$ represents the float16 or bfloat16 precision format (which uses **2 bytes per parameter**). If float32 is used, this factor is $4$.
    *   The remaining factors map the total parameter count: Batch Size ($B$) $\times$ Sequence Length ($L$) $\times$ Layer count ($n_{\text{layers}}$) $\times$ KV Head Count ($n_{\text{heads\_kv}}$) $\times$ Head Dimension ($d_{\text{head}}$).

---

### Hand Calculations: Llama-3-8B KV Cache Footprint
Let's estimate the KV Cache memory footprint of Llama-3-8B under a standard production load.

#### Model Parameters
*   Layers ($n_{\text{layers}}$) $= 32$
*   KV Heads ($n_{\text{heads\_kv}}$) $= 8$ (Grouped-Query Attention)
*   Head Dimension ($d_{\text{head}}$) $= 128$
*   Inference Precision: bfloat16 ($2$ bytes per parameter)

#### Inference Load
*   Batch Size ($B$) $= 4$
*   Sequence Length ($L$) $= 8192$ (maximum context)

#### Step 1: Calculate Total Parameter Count
$$\begin{aligned}
\text{Total Params} &= 2 \times B \times L \times n_{\text{layers}} \times n_{\text{heads\_kv}} \times d_{\text{head}} \\
&= 2 \times 4 \times 8192 \times 32 \times 8 \times 128 \\
&= 8 \times 8192 \times 32 \times 1024 \\
&= 65,536 \times 32,768 \\
&= 2,147,483,648 \text{ parameters}
\end{aligned}$$

#### Step 2: Multiply by Byte Precision (2 Bytes for bf16)
$$\begin{aligned}
\text{Total Bytes} &= 2,147,483,648 \times 2 \text{ bytes} \\
&= 4,294,967,296 \text{ bytes}
\end{aligned}$$

#### Step 3: Convert to Gigabytes
$$\begin{aligned}
\text{VRAM}_{\text{KVCache}} &= \frac{4,294,967,296}{1024^3} \\
&= 4.0\text{ GB}
\end{aligned}$$

*   **Conclusion:** Just hosting the KV Cache for 4 users at 8k context length consumes **4.0 GB** of VRAM, entirely separate from the 16.0 GB of VRAM required to store the static model weights!

---

### Tensor & Shape Tracking
For a single generation step $t$, where input is `x_new` of shape `[B, 1, d]`:
*   **Query vector ($q_t$)**: `[B, H, 1, d_k]`
*   **New Key/Value vectors ($k_t, v_t$)**: `[B, G, 1, d_k]`
*   **Stored Cache Tensors ($K_{\text{cache}}, V_{\text{cache}}$)**: `[B, G, t-1, d_k]`
*   **Appended Cache Tensors ($K_{\text{cache\_new}}, V_{\text{cache\_new}}$)**: `[B, G, t, d_k]`
*   **Attention weight output ($Q K^T$)**: `[B, H, 1, t]` (representing query attending to all $t$ past positions).

---

## 3. Implementation & Reference Code

Below is a self-contained PyTorch simulation of an autoregressive generation step using a KV cache helper class.

```python
import torch
import torch.nn as nn

class KVCacheManager:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Caches are stored as lists of tuples: (k_cache, v_cache)
        self.k_caches = [None] * num_layers
        self.v_caches = [None] * num_layers
        
    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # k_new, v_new shapes: [B, h, 1, d_k]
        if self.k_caches[layer_idx] is None:
            # First token (prefill)
            self.k_caches[layer_idx] = k_new
            self.v_caches[layer_idx] = v_new
        else:
            # Concatenate along sequence dimension (dim=2)
            self.k_caches[layer_idx] = torch.cat([self.k_caches[layer_idx], k_new], dim=2)
            self.v_caches[layer_idx] = torch.cat([self.v_caches[layer_idx], v_new], dim=2)
            
        return self.k_caches[layer_idx], self.v_caches[layer_idx]
        
    def reset(self):
        self.k_caches = [None] * self.num_layers
        self.v_caches = [None] * self.num_layers

# Verification block simulating 3 decode steps
if __name__ == "__main__":
    B, h, d_k = 2, 8, 64
    n_layers = 12
    
    cache_manager = KVCacheManager(num_layers=n_layers, num_heads=h, head_dim=d_k)
    
    # Simulate step 1: Prefill sequence length L=5
    print("--- STEP 1: PREFILL PHASE ---")
    k_prefill = torch.randn(B, h, 5, d_k)
    v_prefill = torch.randn(B, h, 5, d_k)
    k_cached, v_cached = cache_manager.update(layer_idx=0, k_new=k_prefill, v_new=v_prefill)
    print("Cached Keys shape after prefill:", k_cached.shape) # Expected: [2, 8, 5, 64]
    
    # Simulate step 2: Generate first token (new sequence length L=1)
    print("\n--- STEP 2: DECODING STEP 1 ---")
    k_new_1 = torch.randn(B, h, 1, d_k)
    v_new_1 = torch.randn(B, h, 1, d_k)
    k_cached, v_cached = cache_manager.update(layer_idx=0, k_new=k_new_1, v_new=v_new_1)
    print("Cached Keys shape after decode 1:", k_cached.shape) # Expected: [2, 8, 6, 64]
    
    assert k_cached.shape == (B, h, 6, d_k), "Cache concatenation shape mismatch!"
    print("KV Cache Manager successfully concatenated cache states!")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Eliminating the $O(L^2)$ redundant computations of calculating past token representations during token-by-token decoding.
*   **Why Introduced over Legacy Approaches:** Autoregressive models must attend to historical contexts. KV Caching trade computation flops for memory bytes, enabling fast generations at the cost of VRAM footprint.
*   **Key Failure Modes & Limitations:** Dynamic allocations on memory heaps create massive page fragmentations. This forces maximum batch size reductions to avoid OOMs, even when average VRAM usage is low.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Projection flops scale as $O(B \cdot 1 \cdot d^2)$ per step. Attention dot product scales as $O(B \cdot H \cdot 1 \cdot t \cdot d_k)$.
*   **Space/Memory Footprint:** Scaled VRAM allocation of $O(2 \cdot B \cdot L \cdot n_{\text{layers}} \cdot n_{\text{heads\_kv}} \cdot d_{\text{head}} \cdot \text{BytesPerParam})$.
*   **Primary Bottleneck Type:** Memory-bandwidth-bound (HBM to SRAM transfers).
*   **Variable Legend:** $B$ = Batch Size, $L$ = Sequence Length, $t$ = Current token step index, $n_{\text{layers}}$ = Layer count.

### 3. Production & Scalability
*   **Deployment Considerations:** Under long context windows, KV Caches can exceed the weights parameters size. Dynamic memory management tools like PagedAttention (Module 07) must be deployed to avoid static pre-allocation waste.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Contrast the computing profile of the prefill phase vs. the decoding phase.
        *   *A:* The prefill phase processes the entire prompt sequence $L$ in parallel, making it highly compute-bound (dominated by matrix-matrix multiplications). The decoding phase processes only one token at a time, making it memory-bandwidth bound (dominated by loading model weights and the KV Cache from HBM into SRAM caches).
    2.  *Q:* Calculate the KV Cache VRAM saving when migrating from MHA to GQA on Llama-3-70B (80 layers, 64 Query heads, 8 KV heads, head dim 128, batch 8, sequence length 4096, fp16).
        *   *A:* 
            *   Under MHA (64 KV heads):
                $$\text{VRAM}_{\text{MHA}} = 4 \times 8 \times 4096 \times 80 \times 64 \times 128 \times 2 = 171.79\text{ GB}$$
            *   Under GQA (8 KV heads):
                $$\text{VRAM}_{\text{GQA}} = 4 \times 8 \times 4096 \times 80 \times 8 \times 128 \times 2 = 21.47\text{ GB}$$
            *   GQA saves $150.32\text{ GB}$ of VRAM, making long-context 70B inference possible on a single multi-GPU node.
