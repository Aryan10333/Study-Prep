# Module 06: Context Windows & KV Cache Dynamics

This study guide explains the engineering mechanics of context window management and Key-Value (KV) caching during LLM inference, detailing prefill vs. decode phases, and VRAM memory footprint estimations.

---

## 1. LLM Inference Phases: Prefill vs. Decode

LLM text generation is an autoregressive process split into two distinct execution phases:

```text
User Prompt ──► [Prefill Phase] ──► First Token ──► [Decode Phase] ──► Next Tokens...
                (Compute-Bound)                      (Memory-Bound)
```

### 1. Prefill Phase (First Token Generation)
- **Concept**: The model ingests the entire user prompt and processes all tokens in parallel.
- **Compute Profile**: Highly **compute-bound** (dominated by parallel Matrix Multiplications utilizing GPU tensor cores).
- **Behavior**: This step is highly parallelized, but because it processes the full prompt length at once, it consumes significant time, explaining *why the first token is slow*.

### 2. Decode Phase (Subsequent Token Generation)
- **Concept**: The model generates tokens one-by-one. Each generated token is appended back to the input to predict the next token.
- **Compute Profile**: Highly **memory-bandwidth bound** (dominated by transferring model parameter matrices from GPU VRAM to the processor registers to compute a single token step).
- **Behavior**: Execution speed is capped by memory bandwidth speeds, not floating-point operations.

---

## 2. The KV Cache: Mathematical Necessity

During decoding, to predict token $t$, the attention layer needs Query vector $\mathbf{q}_t$, and Key/Value representations for all historical tokens: $\mathbf{K}_{\le t}, \mathbf{V}_{\le t}$.
- **Without KV Cache**: At every step $t$, the model must re-run the Transformer forward pass over *all* preceding tokens $1 \dots t-1$ to compute their $\mathbf{K}$ and $\mathbf{V}$ vectors. This requires $O(L^2)$ redundant computations.
- **With KV Cache**: The model stores computed $\mathbf{K}$ and $\mathbf{V}$ vectors of historical tokens in VRAM. At step $t$, it projects *only* the new token to $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$, appends $\mathbf{k}_t, \mathbf{v}_t$ to the cache, and computes attention, reducing computation from $O(L^2)$ to $O(L)$ per decoding step.

---

## 3. KV Cache VRAM Footprint Calculations

While the KV Cache reduces computation, it consumes significant VRAM.

### Mathematical Formulation (FP16 Precision)
$$\text{KV Cache VRAM Size} = 2 \times 2 \times b \times n_{\text{layers}} \times n_{\text{heads\_kv}} \times d_{\text{head}} \times s_{\text{len}} \text{ bytes}$$
- The first factor **$2$** represents the Key and Value matrices.
- The second factor **$2$** represents precision bytes per floating-point value (FP16 or BF16 = 2 bytes).
- $b$: Batch size.
- $n_{\text{layers}}$: Transformer layer count.
- $n_{\text{heads\_kv}}$: Key-Value attention heads (varies based on MHA/GQA/MQA).
- $d_{\text{head}}$: Vector size per attention head.
- $s_{\text{len}}$: Active sequence length in tokens.

#### Step-by-Step Hand Calculation (Llama-3-8B Configuration)
- **Scenario**: Let model configuration be Llama-3-8B:
  - $n_{\text{layers}} = 32$
  - $n_{\text{heads\_kv}} = 8$ (using Grouped-Query Attention with $8$ KV heads and $32$ Query heads)
  - $d_{\text{head}} = 128$
- **Inference Parameters**: Batch size $b = 4$, context length $s_{\text{len}} = 4096$ tokens in FP16.
- **Calculation**:
  1. Compute floating-point elements stored per token per layer:
     $$\text{Elements}_{\text{token, layer}} = 2 \times n_{\text{heads\_kv}} \times d_{\text{head}} = 2 \times 8 \times 128 = 2048 \text{ elements}$$
  2. Compute total elements stored per token across all layers:
     $$\text{Elements}_{\text{token, total}} = 2048 \times n_{\text{layers}} = 2048 \times 32 = 65,536 \text{ elements}$$
  3. Calculate VRAM byte size per token per batch item in FP16 (2 bytes):
     $$\text{Bytes}_{\text{token, batch\_item}} = 65,536 \times 2 = 131,072 \text{ bytes } (\approx 128 \text{ KB})$$
  4. Multiply by Batch Size $b = 4$:
     $$\text{Bytes}_{\text{token, batch}} = 131,072 \times 4 = 524,288 \text{ bytes } (512 \text{ KB})$$
  5. Multiply by context length $s_{\text{len}} = 4096$:
     $$\text{Total VRAM} = 524,288 \times 4096 = 2,147,483,648 \text{ bytes}$$
     $$\text{VRAM in GB} = \frac{2,147,483,648}{1024^3} = 2.00 \text{ GB}$$
- **Result**: The KV Cache consumes exactly $2.00\text{ GB}$ of VRAM at runtime.

---

## 4. Context Window Limitations & Anomalies

### Sliding Window Attention
- **Concept**: Restricts attention bounds to a local window of size $W$ (e.g. $W=4096$ in Mistral), keeping KV Cache VRAM bound to a fixed maximum size $O(W)$ instead of growing indefinitely with sequence length $L$.

### Lost-in-the-Middle Phenomenon
- **Concept**: LLMs retrieve information successfully at the beginning and end of long context windows, but their retrieval accuracy degrades on facts placed in the middle.
- **Mitigation**: Place critical prompt instructions and context documents near the beginning or end of input sequences.

---

## 5. Interview Questions & Production Trade-offs

### What problem does this solve?
Reduces decoding step complexity from $O(L^2)$ to $O(L)$ by caching historical token representations, enabling fast autoregressive text generation.

### Why was it introduced?
Decoupling prefill (parallel compute-bound processing) and decode (recurrent memory-bound generation) allows designing specialized kernel execution plans (like vLLM PagedAttention).

### What are its limitations?
The KV Cache scales linearly with batch size and context length, leading to out-of-memory (OOM) errors on large workloads unless partitioned or compressed.

### Computational Complexity (Time & Memory)
- **Decoding step (with KV Cache)**: $O(L \cdot d)$ operations.
- **KV Cache VRAM Footprint**: $O(b \cdot L \cdot d)$ bytes.

### Component Variable Denotation Legend
- $b$: Batch size.
- $n_{\text{layers}}$: Transformer layers.
- $n_{\text{heads\_kv}}$: Key-Value heads.
- $d_{\text{head}}$: Vector size per attention head.
- $s_{\text{len}}$: Active sequence length in tokens.

### Production Use Cases:
- Production inference engines (vLLM, TensorRT-LLM) using PagedAttention to fragment KV Cache memories like virtual OS page blocks, avoiding VRAM fragmentation.
- RAG applications arranging document placements to prevent Lost-in-the-Middle retrieval decay.

### Follow-up Questions Interviewers Ask:
1. *Why is the decode phase memory-bandwidth bound while the prefill phase is compute-bound?*
   - **Answer**: In prefill, the model processes $L$ prompt tokens concurrently, allowing GPUs to load weights once and apply them to $L$ columns of tokens, keeping Tensor cores saturated (compute-bound). In decode, only $1$ token is processed. The GPU must load the entire parameter weights (e.g., $16\text{ GB}$ for an $8\text{B}$ model in FP16) from VRAM to compute activations for a single token, capping latency by memory bandwidth transfer limits ($O(1)$ arithmetic intensity).
2. *How does Grouped-Query Attention (GQA) reduce KV Cache memory footprints?*
   - **Answer**: Standard Multi-Head Attention (MHA) defines equal query and key-value heads: $h_Q = h_{KV} = 32$. GQA groups queries into clusters (e.g. $G=4$), sharing a single key-value head among $4$ query heads ($h_{KV} = h_Q / G = 8$). This reduces the keys and values stored in VRAM to $25\%$, decreasing KV Cache memory consumption by $4\times$.
3. *What is the "First Token Latency" (Time to First Token - TTFT) and why is it a primary metric in production?*
   - **Answer**: TTFT measures the latency of the Prefill Phase (reading prompt and generating the first token). Because prefill must process the entire input prompt length at once, long prompts (e.g. $20\text{k}+$ tokens) introduce significant parallel matrix multiply latency. Splitting TTFT from subsequent token generation speed (Inter-Token Latency) helps engineers optimize system bottlenecks separately (e.g., flash decoding).
