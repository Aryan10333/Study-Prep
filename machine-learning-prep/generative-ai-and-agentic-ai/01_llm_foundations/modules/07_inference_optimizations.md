# Module 07: GPU Memory Bounds & Inference Optimizations (FlashAttention & PagedAttention)

## 1. Introduction & Intuition

### The Core Bottleneck
GPUs possess massive computational power (thousands of TFLOPS), but accessing memory is slow. There are two primary types of memory on a GPU:
1.  **High Bandwidth Memory (HBM)**: Large global storage (e.g. 80GB on H100) where model weights and KV caches are stored. Accessing HBM is slow.
2.  **SRAM**: Fast, local cache memory (a few megabytes) located directly on-chip near the GPU processors. Accessing SRAM is fast.

The core bottleneck in self-attention is the intermediate attention weight matrix $A = \text{softmax}(QK^T/\sqrt{d_k})$. This matrix has size $L \times L$. For a sequence length $L = 64\text{k}$, this matrix occupies 8 GB of memory for a single head! In standard PyTorch, this matrix is computed, written to HBM, read back from HBM to apply softmax, and read again to multiply by $V$. This roundtrip read/write cycle of $O(L^2)$ intermediate values to slow HBM is a major memory-bandwidth bottleneck, slowing execution down to a crawl.

Furthermore, when serving models, the KV cache of multiple users must be stored. Standard deep learning frameworks allocate KV cache memory statically as contiguous arrays. Because sequence lengths are dynamic, frameworks pre-allocate memory for the maximum possible length. This results in **memory fragmentation and waste (up to 60-80%)**, limiting the number of requests a GPU can serve concurrently.

### High-Level Intuition
*   **FlashAttention**: Bypasses the HBM bottleneck by using **tiling**. Instead of loading the entire Query, Key, and Value matrices to HBM to compute the $L \times L$ grid, it loads small blocks (tiles) of Q, K, and V into fast on-chip **SRAM**. It computes attention locally on these tiles, updates the output, and writes the output back to HBM. By using **online softmax** tracking, it normalizes the values incrementally. The intermediate $L \times L$ attention matrix is never written to HBM, reducing memory accesses from quadratic $O(L^2)$ to linear $O(L)$.
*   **PagedAttention**: Solves the memory allocation bottleneck by copying the concept of **virtual memory paging** from operating systems. Instead of allocating a contiguous block of VRAM for each request's KV Cache, PagedAttention splits the KV Cache into small, fixed-size **pages** (representing e.g. 16 tokens). These pages are mapped to non-contiguous locations in VRAM via a lookup table. When a new token is generated, it is written to the current physical page. If the page is full, a new physical block is allocated from a shared pool. This eliminates fragmentation, enabling up to 4x higher throughput.

---

### FlashAttention Tiling & PagedAttention Virtual Mapping
Below is an inline SVG illustrating the architectural layout of FlashAttention and PagedAttention:

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 800 300" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <!-- FLASHATTENTION BLOCK -->
  <g transform="translate(20, 20)">
    <rect x="0" y="0" width="360" height="260" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="180" y="25" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">FlashAttention (SRAM Tiling)</text>
    
    <!-- HBM -->
    <rect x="20" y="50" width="320" height="50" rx="4" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5" />
    <text x="180" y="70" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">GPU Global Memory (HBM)</text>
    <text x="180" y="85" text-anchor="middle" font-size="9" fill="#475569">Stores large Q, K, V inputs &amp; Output O (No L x L stored!)</text>
    
    <!-- Read/Write Paths -->
    <path d="M 100 100 L 100 160" stroke="#ef4444" stroke-width="1.5" marker-end="url(#arrow-opt)" />
    <text x="70" y="135" font-size="9" fill="#ef4444">Tile Read</text>
    
    <path d="M 280 160 L 280 100" stroke="#10b981" stroke-width="1.5" marker-end="url(#arrow-opt)" />
    <text x="290" y="135" font-size="9" fill="#10b981">Final O Write</text>
    
    <!-- SRAM -->
    <rect x="20" y="160" width="320" height="80" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="180" y="180" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">Fast On-Chip SRAM (Tiling Cache)</text>
    <text x="180" y="200" text-anchor="middle" font-size="9" fill="#1d4ed8">Computes block-wise dot products &amp; tracks running scaling</text>
    <text x="180" y="215" text-anchor="middle" font-size="9" fill="#1d4ed8">e<sup>x - m</sup> statistics updated incrementally</text>
  </g>

  <!-- PAGEDATTENTION BLOCK -->
  <g transform="translate(420, 20)">
    <rect x="0" y="0" width="360" height="260" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="180" y="25" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">PagedAttention (Virtual Mapping)</text>
    
    <!-- Logical -->
    <rect x="20" y="55" width="100" height="150" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.5" />
    <text x="70" y="75" text-anchor="middle" font-size="10" font-weight="bold" fill="#78350f">Logical Cache</text>
    <rect x="30" y="90" width="80" height="25" rx="2" fill="#fef3c7" stroke="#d97706" />
    <text x="70" y="106" text-anchor="middle" font-size="9" fill="#78350f">Block 0 (T0-15)</text>
    <rect x="30" y="125" width="80" height="25" rx="2" fill="#fef3c7" stroke="#d97706" />
    <text x="70" y="141" text-anchor="middle" font-size="9" fill="#78350f">Block 1 (T16-31)</text>
    
    <!-- Mapping -->
    <path d="M 130 100 Q 180 80 230 110" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3" marker-end="url(#arrow-opt)" />
    <path d="M 130 135 Q 180 170 230 145" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3" marker-end="url(#arrow-opt)" />
    <text x="180" y="115" text-anchor="middle" font-size="8" fill="#64748b">Page Table Mapping</text>
    
    <!-- Physical -->
    <rect x="240" y="55" width="100" height="150" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5" />
    <text x="290" y="75" text-anchor="middle" font-size="10" font-weight="bold" fill="#065f46">Physical VRAM</text>
    <rect x="250" y="105" width="80" height="25" rx="2" fill="#d1fae5" stroke="#059669" />
    <text x="290" y="121" text-anchor="middle" font-size="9" fill="#065f46">VRAM Block 14</text>
    <rect x="250" y="140" width="80" height="25" rx="2" fill="#d1fae5" stroke="#059669" />
    <text x="290" y="156" text-anchor="middle" font-size="9" fill="#065f46">VRAM Block 89</text>
    <text x="290" y="190" text-anchor="middle" font-size="8" fill="#047857">Scatter Allocated</text>
  </g>
  
  <!-- Marker -->
  <defs>
    <marker id="arrow-opt" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### Mathematical Formulations

#### 1. Roofline Model Operational Intensity
$$\text{Operational Intensity} = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$$

*   **Purpose & High-level Intuition:** Measures whether an execution phase is bound by memory latency or processor speeds.
    *   **Compute-Bound**: Operational intensity is high. The GPU spends its time performing arithmetic. High GPU utilization is achieved.
    *   **Memory-Bandwidth-Bound**: Operational intensity is low. The arithmetic units are idle, waiting for data to load from HBM. GPU utilization drops.

#### 2. Online Softmax Normalization
To prevent numerical overflow when computing softmax on hardware, we subtract the maximum value:
$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad \text{where } m = \max_j x_j$$

In FlashAttention, this is computed incrementally for tiles. If we have local block maximum $m^{(1)}$ and sum $s^{(1)}$, and receive new block with maximum $m^{(2)}$ and sum $s^{(2)}$, we update the running maximum and sum:
$$m^{\text{new}} = \max(m^{(1)}, m^{(2)})$$
$$s^{\text{new}} = s^{(1)} e^{m^{(1)} - m^{\text{new}}} + s^{(2)} e^{m^{(2)} - m^{\text{new}}}$$

---

### Hand Calculations: Decoding Operational Intensity
Let's calculate the operational intensity of generating a single token in the decoding phase.

#### Parameters
*   Model Parameters ($N$) $= 7,000,000,000$ (7B parameters).
*   Precision: float16 ($2$ bytes per parameter).
    *   **FLOPs required**: $2 \times N$ calculations (matrix-vector multiplication of input token with all weights).
        $$\begin{aligned}
        \text{FLOPs} &= 2 \times N \\
        &= 2 \times 7 \times 10^9 \\
        &= 14 \times 10^9 \text{ FLOPs (14 GFLOPs)}
        \end{aligned}$$
    *   **Memory bytes read**: Load all parameters from HBM to registers.
        $$\begin{aligned}
        \text{Bytes Read} &= N \times 2 \text{ bytes} \\
        &= 7 \times 10^9 \times 2 \text{ bytes} \\
        &= 14 \times 10^9 \text{ Bytes (14 GB)}
        \end{aligned}$$

#### Calculate Operational Intensity
$$\begin{aligned}
\text{Operational Intensity} &= \frac{\text{FLOPs}}{\text{Bytes Read}} \\
&= \frac{14 \times 10^9 \text{ FLOPs}}{14 \times 10^9 \text{ Bytes}} \\
&= 1.0\text{ FLOP/Byte}
\end{aligned}$$

*   **Conclusion:** The operational intensity is exactly **1.0 FLOP/Byte**. 
    An NVIDIA A100 GPU has a roofline threshold of around **150 FLOPs/Byte**. Since our intensity ($1.0$) is way below the threshold, the decoding phase is extremely **memory-bandwidth bound**. The GPU processors spend 99% of their time idle, waiting for weights to load from global memory.

---

### Tensor & Shape Tracking
*   **Logical KV Cache Block**: `[Block_Size, G, d_k]` (where `Block_Size` is typically 16 tokens).
*   **FlashAttention Block (Tile)**:
    *   Query Tile ($Q_i$): `[B_r, d_k]` (where $B_r$ is row block size, e.g. 64).
    *   Key Tile ($K_j$): `[B_c, d_k]` (where $B_c$ is column block size, e.g. 64).
    *   Value Tile ($V_j$): `[B_c, d_k]`.

---

## 3. Implementation & Reference Code

Below is a Python simulation of the incremental online softmax math used during FlashAttention tiling.

```python
import numpy as np

def compute_standard_softmax(x):
    # Standard softmax subtraction to avoid numerical overflow
    m = np.max(x)
    exp_x = np.exp(x - m)
    return exp_x / np.sum(exp_x), m, np.sum(exp_x)

def simulate_online_softmax():
    # Simulate an attention row split into two blocks
    np.random.seed(42)
    x = np.random.randn(8)
    
    block1 = x[:4]
    block2 = x[4:]
    
    # 1. Process Block 1
    m1 = np.max(block1)
    s1 = np.sum(np.exp(block1 - m1))
    o1 = np.exp(block1 - m1) / s1
    
    # 2. Process Block 2
    m2 = np.max(block2)
    s2 = np.sum(np.exp(block2 - m2))
    o2 = np.exp(block2 - m2) / s2
    
    # 3. Apply FlashAttention Online Softmax Update Formula
    m_new = max(m1, m2)
    s_new = s1 * np.exp(m1 - m_new) + s2 * np.exp(m2 - m_new)
    
    # Normalize output representations to the new scale
    o1_scaled = o1 * (s1 * np.exp(m1 - m_new) / s_new)
    o2_scaled = o2 * (s2 * np.exp(m2 - m_new) / s_new)
    
    final_online = np.concatenate([o1_scaled, o2_scaled])
    
    # Compare with standard softmax
    final_standard, _, _ = compute_standard_softmax(x)
    
    print("Standard Softmax Output:", final_standard)
    print("Online Softmax Output  :", final_online)
    assert np.allclose(final_standard, final_online), "Outputs do not match!"
    print("Online softmax arithmetic successfully verified!")

if __name__ == "__main__":
    simulate_online_softmax()
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** The memory-bandwidth wall of writing/reading $L \times L$ attention matrices, and static KV cache allocation fragmentation.
*   **Why Introduced over Legacy Approaches:** PyTorch's native attention creates quadratic HBM roundtrips. FlashAttention bypasses this by doing block-wise SRAM reductions. PagedAttention replaces contiguous arrays with dynamic page mappings to maximize serving batch sizes.
*   **Key Failure Modes & Limitations:** PagedAttention adds lookup overhead via the page table, slightly increasing routing latency.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Both standard attention and FlashAttention scale as $O(L^2 \cdot d)$ computation FLOPs.
*   **Space/Memory Footprint:** Standard attention requires $O(L^2 \cdot h)$ HBM storage for intermediate activations. FlashAttention requires $O(L \cdot d)$ linear storage.
*   **Primary Bottleneck Type:** FlashAttention is Compute-bound (or memory-bandwidth bound depending on context length); PagedAttention overhead is Memory-bandwidth bound.
*   **Variable Legend:** $L$ = Sequence Length, $d$ = Hidden Model Dimension, $h$ = Head Count.

### 3. Production & Scalability
*   **Deployment Considerations:** FlashAttention is implemented as custom CUDA kernels. It requires compilation and direct hardware support (e.g. FP16/BF16 tensor cores on Ampere/Hopper).
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does FlashAttention perform *more* FLOPs than standard attention during backpropagation, but runs significantly faster?
        *   *A:* In the backward pass, FlashAttention does not load the $L \times L$ attention matrix from HBM (since it was never stored). Instead, it recomputes the attention matrix tiles on-the-fly in SRAM. This requires performing extra FLOP calculations. However, because HBM memory reads/writes are 10-100x slower than SRAM compute FLOPs, recomputing is much faster than loading from memory.
    2.  *Q:* Explain how PagedAttention enables "Copy-on-Write" for parallel decoding.
        *   *A:* When generating multiple completions for a prompt (e.g. beam search or system branching), the initial prompt's KV cache is shared. Instead of copying the prompt's KV cache for each path, PagedAttention points all virtual tables to the same physical pages (with read-only locks). When one path generates a new token, only its target page is copied and modified (Copy-on-Write). This saves massive amounts of VRAM.
