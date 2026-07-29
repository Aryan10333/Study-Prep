# 03. KV Cache & Attention Mechanics: Memory Bottlenecks and Virtualization

Autoregressive transformer inference requires storing the Key ($K$) and Value ($V$) state projections of all past context tokens. This mechanism—the **KV Cache**—saves the system from recomputing self-attention over historical tokens at every single generation step, converting a quadratic time complexity $O(N^2)$ attention pass into an efficient $O(1)$ query projection. However, caching historical states shifts the primary system bottleneck from processing computation directly to managing VRAM memory footprints.

---

## 1. The KV Cache Bottleneck

In a standard Multi-Head Attention (MHA) block, when generating token $t$, the query vector $q_t$ must be multiplied by keys from all previous tokens:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}}\right) V$$

Without caching, to generate token $t$, we would have to project the key and value vectors for all past $t-1$ tokens from scratch. This would require $O(t^2)$ projection operations over the sequence.
By caching past key and value vectors in memory, at step $t$ we only calculate $q_t, k_t, v_t$ for the single new token, load the historical keys $K_{1:t-1}$ and values $V_{1:t-1}$ from VRAM, perform attention, and write the new $k_t, v_t$ to the cache.
While this reduces FLOPs, it creates a massive VRAM footprint that scales linearly with sequence length ($l$) and batch size ($b$).

---

## 2. Exact KV Cache Memory Formula

The exact VRAM footprint (in Bytes) required to store the KV Cache across an entire LLM inference pass is given by:

$$\text{VRAM}_{\text{KV}} = 2 \times b \times l \times s \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}$$

Where:
- $b$: Batch size.
- $l$: Sequence context length (in tokens).
- $s$: Number of layers in the model.
- $n_{\text{heads}}$: Number of key-value attention heads per layer (varies based on attention architecture).
- $d_{\text{head}}$: Dimensionality of each attention head.
- $\text{bytes\_per\_element}$: Data format size ($2 \text{ bytes}$ for FP16/BF16, $1 \text{ byte}$ for FP8/INT8).
- The leading factor of $2$ accounts for storing *both* Key ($K$) and Value ($V$) tensors.

### Step-by-Step Hand Calculations for Llama-3 Models

Let's calculate the required VRAM for Llama-3 8B and Llama-3 70B models at batch size $b=16$ using FP16 precision ($\text{bytes\_per\_element} = 2$) across varying context windows.

#### Specifications Table
| Model | Layers ($s$) | KV Heads ($n_{\text{heads}}$) | Head Dim ($d_{\text{head}}$) |
|---|---|---|---|
| **Llama-3 8B** | $32$ | $8$ (Grouped-Query Attention) | $128$ |
| **Llama-3 70B** | $80$ | $8$ (Grouped-Query Attention) | $128$ |

---

#### 1. Llama-3 8B (at $b=16$)

##### Case A: Context Length $l = 32\text{k}$ ($32,768$ tokens)
- Formula substitution:
  $$\text{VRAM}_{\text{KV}} = 2 \times 16 \times 32768 \times 32 \times 8 \times 128 \times 2 \text{ bytes}$$
- Calculate step-by-step:
  - $2 \times 16 \times 32768 = 1,048,576$ (tokens across batch)
  - $1,048,576 \times 32 = 33,554,432$ (layer-tokens)
  - $33,554,432 \times 8 = 268,435,456$ (active heads)
  - $268,435,456 \times 128 = 34,359,738,368$ (elements)
  - $34,359,738,368 \times 2 \text{ bytes} = 68,719,476,736 \text{ bytes}$
- Convert to GiB:
  $$\text{GiB} = \frac{68,719,476,736}{1024^3} = 64.00 \text{ GiB}$$

##### Case B: Context Length $l = 128\text{k}$ ($131,072$ tokens)
- Calculate:
  $$\text{VRAM}_{\text{KV}} = 2 \times 16 \times 131072 \times 32 \times 8 \times 128 \times 2 = 274,877,906,944 \text{ bytes}$$
- Convert to GiB:
  $$\text{GiB} = \frac{274,877,906,944}{1024^3} = 256.00 \text{ GiB}$$

---

#### 2. Llama-3 70B (at $b=16$)

##### Case A: Context Length $l = 32\text{k}$ ($32,768$ tokens)
- Formula substitution:
  $$\text{VRAM}_{\text{KV}} = 2 \times 16 \times 32768 \times 80 \times 8 \times 128 \times 2 \text{ bytes}$$
- Calculate step-by-step:
  - $2 \times 16 \times 32768 = 1,048,576$ (tokens across batch)
  - $1,048,576 \times 80 = 83,886,080$ (layer-tokens)
  - $83,886,080 \times 8 = 671,088,640$ (active heads)
  - $671,088,640 \times 128 = 85,899,345,920$ (elements)
  - $85,899,345,920 \times 2 \text{ bytes} = 171,798,691,840 \text{ bytes}$
- Convert to GiB:
  $$\text{GiB} = \frac{171,798,691,840}{1024^3} = 160.00 \text{ GiB}$$

##### Case B: Context Length $l = 128\text{k}$ ($131,072$ tokens)
- Calculate:
  $$\text{VRAM}_{\text{KV}} = 2 \times 16 \times 131072 \times 80 \times 8 \times 128 \times 2 = 687,194,767,360 \text{ bytes}$$
- Convert to GiB:
  $$\text{GiB} = \frac{687,194,767,360}{1024^3} = 640.00 \text{ GiB}$$

---

## 3. Attention Variants for Memory Reduction

To mitigate the VRAM demand of the KV Cache, modern LLM architectures adjust the ratio of query heads to KV heads:

```html
<div style="display: flex; flex-direction: column; gap: 16px; font-family: 'Segoe UI', sans-serif; margin: 20px 0;">
  <!-- MHA -->
  <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 14px;">1. Multi-Head Attention (MHA)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">Each Query head has a dedicated Key and Value head.</div>
    <div style="font-family: monospace; font-size: 11px; color: #e11d48; font-weight: bold;">KV Heads : Query Heads = 1 : 1</div>
  </div>

  <!-- MQA -->
  <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 14px;">2. Multi-Query Attention (MQA)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">All Query heads share a single Key and Value head. Reduces KV Cache VRAM footprint significantly but degrades capability.</div>
    <div style="font-family: monospace; font-size: 11px; color: #e11d48; font-weight: bold;">KV Heads : Query Heads = 1 : H</div>
  </div>

  <!-- GQA -->
  <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 14px;">3. Grouped-Query Attention (GQA)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">Query heads are grouped; each group shares a single Key and Value head. Combines the speed of MQA with the capacity of MHA.</div>
    <div style="font-family: monospace; font-size: 11px; color: #e11d48; font-weight: bold;">KV Heads : Query Heads = 1 : G (Typically 1 : 8 or 1 : 4)</div>
  </div>
</div>
```

### Attention Variant Trade-offs

| Variant | KV : Query Heads | Pros | Cons | Production Choice |
|---|---|---|---|---|
| **Multi-Head Attention (MHA)** | $1 : 1$ | • Maximum representational capacity (each query head has private KV context). | • Maximum VRAM footprint; severely limits batch sizes and sequence length. | Older baseline models (e.g., LLaMA-1, GPT-3). |
| **Multi-Query Attention (MQA)** | $1 : H$ | • Minimal KV Cache VRAM footprint (up to 8x-32x footprint reduction). | • Degrades model capacity; performance drops on long, complex documents. | Used in specialized low-resource models (e.g., Falcon). |
| **Grouped-Query Attention (GQA)** | $1 : G$ (e.g., $1 : 8$) | • Combines MHA's high quality with MQA's low memory consumption. | • Marginally more complex query-to-group mapping logic in code. | Standard for modern SOTA models (LLaMA-3, Mistral, Command-R). |

- **VRAM Saving**: If a model uses a Grouped-Query Attention ratio of $8:1$ (e.g. Llama-3), its KV Cache VRAM footprint is reduced by **8x** compared to standard MHA.

---

## 4. PagedAttention & Virtual Memory Management

### The Problem: Memory Fragmentation
Traditional serving systems allocated KV Cache memory for each request as a contiguous chunk of VRAM sized to the *maximum* possible sequence length (e.g. $4096$ tokens). 
- **Internal Fragmentation**: If a request terminated early after generating 10 tokens, the remaining pre-allocated memory blocks remained reserved and unused.
- **External Fragmentation**: Memory allocations of varying sizes created gaps of un-allocatable space, leading to CUDA Out-Of-Memory errors despite having free memory bytes.
This wasted up to **$60\% - 80\%$** of available GPU memory.

### The Solution: PagedAttention (vLLM)
PagedAttention borrows the concept of virtual memory and paging from operating systems:
1. **Physical Blocks**: The GPU's VRAM is divided into non-contiguous physical blocks of a fixed size (e.g. $16$ tokens).
2. **Logical Blocks**: The serving engine maps a request's logical sequence of tokens to these physical blocks.
3. **Block Table**: A dynamic lookup table tracks which logical blocks map to which physical memory blocks.
4. **On-Demand Allocation**: As new tokens are generated, the engine allocates new blocks dynamically. Blocks do not need to be contiguous in VRAM, eliminating fragmentation and allowing VRAM utilization to approach **$96\%$**.

---

## 5. Prefix Caching vs. SGLang RadixAttention

For multi-turn conversations, agent execution steps, and RAG applications, successive queries share identical system prompts or reference documents. Caching these shared prefixes prevents recomputing their KV states.

### vLLM Block-level Hash Caching
vLLM implements Automatic Prefix Caching (APC) by computing a cryptographic hash of the token IDs inside a block. If a new request's starting blocks match the hash of cached blocks, the engine reuses the physical blocks directly.
- **Limitation**: The prefix matching is rigid and block-grained. It cannot handle dynamic, arbitrary prefix paths or branching chats efficiently.

### SGLang RadixAttention
SGLang models the KV Cache as a dynamic **Radix Tree** data structure:
- Prompt prefixes (e.g. System Prompt, Document Context, Chat Turn 1) are stored as nodes in a tree.
- When a new request arrives, SGLang traverses the tree to find the longest matching prefix path.
- The matching KV cache is reused, and the new execution branches off as a child node.
- If the GPU memory fills up, SGLang uses a Least Recently Used (LRU) eviction policy to evict child leaves while keeping the common root prefixes cached.
This accelerates TTFT in agent loops by **$2\times - 5\times$** compared to standard block hashing.

---

## 6. FlashAttention Hardware Acceleration

FlashAttention (1, 2, and 3) optimizes attention compute speed, but it is important to understand its hardware limits:
- **How it works**: Standard attention writes intermediate $Q K^T$ matrices to slow High Bandwidth Memory (HBM). FlashAttention uses *tiling* and *online softmax* to load blocks of queries, keys, and values into fast GPU SRAM, perform computations, and write only the final outputs back to HBM.
- **VRAM Impact**: FlashAttention reduces the **intermediate activation memory** during the forward pass. However, it does *not* reduce the VRAM footprint of the stored KV Cache in VRAM. The stored key-value states must still occupy the exact VRAM size derived by the memory formula.

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
