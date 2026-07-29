# 01. Inference Fundamentals: Prefill, Decode, and Roofline Mechanics

Autoregressive Large Language Model (LLM) serving presents a unique hardware and systems engineering challenge. Unlike traditional deep learning models where training and inference share similar compute-bound profiles, LLM inference changes bottlenecks dynamically between its execution phases. Understanding these bottlenecks is critical for optimizing throughput, satisfying latency Service Level Agreements (SLAs), and designing high-scale serving infrastructure.

---

## 1. Training vs. Inference Bottlenecks

An LLM's life cycle is split into two compute paradigms: training (which is compute-bound) and autoregressive inference (which is memory-bandwidth-bound during token generation).

### Training Phase (Compute-Bound)
During pre-training or fine-tuning, the model processes large batches of sequences simultaneously. The training step utilizes the *causal mask* to calculate self-attention across all tokens in a sequence concurrently:
- We compute projections and attention scores for all sequence positions at once.
- Weights are loaded into SRAM once and reused to compute activations for thousands of tokens in parallel.
- The GPU Tensor Cores are kept highly saturated, making training efficiency scale with the hardware's peak FLOP/s capacity.

### Inference Phase (Autoregressive Decode is Memory-Bandwidth-Bound)
Autoregressive inference generates text token-by-token. To predict token $t$, the model must process all previous tokens $1, \dots, t-1$.
- Generating each new token requires loading the entire model's weights (gigabytes of parameters) from High Bandwidth Memory (HBM) to GPU SRAM.
- The GPU performs only a single token's matrix-vector multiplication (GEMV) before discarding the weights or loading the next layer.
- Because memory read speeds (HBM bandwidth) are orders of magnitude slower than arithmetic processing speeds, the GPU Tensor Cores sit idle waiting for weights, making token decoding bound by memory bandwidth.

---

## 2. The Roofline Model & Arithmetic Intensity

The **Roofline Model** maps a system's execution performance limits against its **Arithmetic Intensity** ($I$), which represents the ratio of arithmetic work (FLOPs) performed per byte of data transferred:

$$I = \frac{\text{Work (FLOPs)}}{\text{Memory Traffic (Bytes)}}$$

Hardware performance is bounded by two physical constraints:
1. **Peak Compute Performance** ($P_{\text{peak}}$): The maximum floating-point operations the processor can run per second (FLOP/s).
2. **Peak Memory Bandwidth** ($B_{\text{peak}}$): The maximum rate at which the processor can read or write data to memory (Bytes/sec).

The attainable performance ($P$) is modeled as:

$$P = \min\left(P_{\text{peak}}, I \cdot B_{\text{peak}}\right)$$

The transition point where the bottleneck shifts from memory-bound to compute-bound is the **Hardware Ridge Point** ($I_{\text{ridge}}$):

$$I_{\text{ridge}} = \frac{P_{\text{peak}}}{B_{\text{peak}}}$$

> [!TIP]
> **The Memory Bus Analogy**: 
> Think of the GPU as a massive assembly line (the Tensor Cores) and VRAM (HBM) as the parts warehouse. 
> - **In Prefill (Compute-Bound)**: A single massive pallet of materials (the model weights) is loaded onto the assembly line, and the workers spend a long time processing thousands of tasks (tokens) in parallel. The workers are kept fully saturated—this is compute-bound.
> - **In Decode (Memory-Bandwidth-Bound)**: The assembly line must load the entire blueprint catalogue (the whole model weights) from the warehouse just to stamp a single custom label (one new token), after which the catalog is discarded, and the next layer's catalog is loaded. The shipping trucks (memory bus) are bottlenecked, while the workers sit idle waiting for the next catalog—this is memory-bandwidth-bound.

### Step-by-Step Hand Calculation on a 1B Parameter Model

Let's compute the arithmetic intensity for an LLM layer with $P = 1 \times 10^9$ (1 Billion) parameters executing on an NVIDIA A100 GPU (SXM4, 80GB VRAM) in FP16 precision:
- **NVIDIA A100 SXM4 peak FP16/BF16 compute** ($P_{\text{peak}}$): $312 \text{ TFLOP/s} = 312 \times 10^{12} \text{ FLOP/s}$.
- **NVIDIA A100 SXM4 peak memory bandwidth** ($B_{\text{peak}}$): $2.0 \text{ TB/s} = 2.0 \times 10^{12} \text{ Bytes/s}$.
- **Hardware Ridge Point** ($I_{\text{ridge}}$):
  $$I_{\text{ridge}} = \frac{312 \times 10^{12}}{2.0 \times 10^{12}} = 156 \text{ FLOP/s per Byte}$$
  - If a kernel's arithmetic intensity $I < 156$, it is **memory-bandwidth-bound**.
  - If $I > 156$, it is **compute-bound**.

#### Scenario A: Prefill Phase (Batch Size $b=1$, Prompt Length $N=1024$ tokens)
During prefill, the model processes all 1024 prompt tokens in parallel.
1. **FLOPs Estimation**: A forward pass requires approximately $2 \times P$ operations per token:
   $$\text{FLOPs} = 2 \times (1 \times 10^9) \times 1024 = 2.048 \times 10^{12} \text{ FLOPs} = 2.048 \text{ TFLOPs}$$
2. **Memory Traffic Estimation**: We load the model weights in FP16 ($2 \text{ bytes per parameter}$):
   $$\text{Bytes} = P \times 2 = 1 \times 10^9 \times 2 = 2.0 \times 10^9 \text{ Bytes} = 2 \text{ GB}$$
3. **Arithmetic Intensity ($I_{\text{prefill}}$)**:
   $$I_{\text{prefill}} = \frac{2.048 \times 10^{12} \text{ FLOPs}}{2.0 \times 10^9 \text{ Bytes}} = 1024 \text{ FLOP/s per Byte}$$
4. **Attainable Performance**: Since $1024 > 156$ (well past the ridge point), the prefill phase is **compute-bound**.

#### Scenario B: Decode Phase (Batch Size $b=1$, generating $N=1$ new token)
During decoding, we process exactly 1 token at a time.
1. **FLOPs Estimation**:
   $$\text{FLOPs} = 2 \times (1 \times 10^9) \times 1 = 2 \times 10^9 \text{ FLOPs} = 2 \text{ GFLOPs}$$
2. **Memory Traffic Estimation**: We must load the entire 2 GB model weights to calculate the output for this single token:
   $$\text{Bytes} = P \times 2 = 2.0 \times 10^9 \text{ Bytes} = 2 \text{ GB}$$
3. **Arithmetic Intensity ($I_{\text{decode}}$)**:
   $$I_{\text{decode}} = \frac{2 \times 10^9 \text{ FLOPs}}{2.0 \times 10^9 \text{ Bytes}} = 1.0 \text{ FLOP/s per Byte}$$
4. **Attainable Performance**: Since $1.0 \ll 156$, the decode phase is **memory-bandwidth-bound**. The attainable performance is capped at:
   $$P_{\text{decode}} = 1.0 \times (2.0 \times 10^{12} \text{ Bytes/s}) = 2.0 \text{ TFLOP/s}$$
   *Only $0.64\%$ ($2 / 312$) of the A100 GPU's raw compute capability is utilized!*

---

## 3. Two-Phase Inference Lifecycle

LLM serving engines split generation into two sequential processing phases:

```html
<div style="display: flex; flex-direction: column; gap: 20px; font-family: 'Segoe UI', sans-serif; margin: 20px 0;">
  <!-- Prefill Phase Card -->
  <div style="border: 1px solid #3b82f6; border-radius: 8px; background-color: #f8fafc; padding: 16px; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.05);">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
      <span style="background-color: #3b82f6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase;">Phase 1</span>
      <h4 style="margin: 0; font-size: 16px; color: #1e3a8a;">Prefill Phase (Prompt Processing)</h4>
    </div>
    <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.5;">
      Processes all prompt tokens simultaneously in a single forward pass. Computes and saves the initial key-value (KV) states into the KV Cache. Highly parallelized, maximizing tensor core utilization. Bounded by <strong>Compute (FLOP/s)</strong>.
    </p>
  </div>

  <!-- Arrow -->
  <div style="text-align: center; color: #64748b; font-size: 20px; font-weight: bold;">↓</div>

  <!-- Decode Phase Card -->
  <div style="border: 1px solid #10b981; border-radius: 8px; background-color: #f8fafc; padding: 16px; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.05);">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
      <span style="background-color: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase;">Phase 2</span>
      <h4 style="margin: 0; font-size: 16px; color: #065f46;">Decode Phase (Token Generation)</h4>
    </div>
    <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.5;">
      Generates tokens one-by-one autoregressively. Each step takes the single newly generated token, reads the prior KV Cache tokens from VRAM, performs a matrix-vector product, and appends the new token's key-value states to the cache. Bounded by <strong>Memory Bandwidth (GB/s)</strong>.
    </p>
  </div>
</div>
```

---

## 4. Prefill-Decode (PD) Disaggregation

### The Problem
When prefill and decode phases run on the same GPU instance, they compete for execution slots. 
- A compute-heavy prefill request arriving in the queue can block ongoing decode requests. This is called **Head-of-Line Blocking**.
- Because prefill and decode require different optimal batch sizes (prefill benefits from small batches to minimize latency; decode benefits from large batches to saturate memory bandwidth), running them on the same GPU compromises execution efficiency.

### The Solution: PD Disaggregation
**PD Disaggregation** decouples the physical hardware nodes assigned to these phases:

```html
<div style="display: flex; justify-content: space-between; align-items: stretch; gap: 20px; font-family: 'Segoe UI', sans-serif; margin: 24px 0;">
  <!-- Prefill Node Pool -->
  <div style="flex: 1; border: 1.5px solid #2563eb; border-radius: 8px; background-color: #f1f5f9; padding: 16px; text-align: center;">
    <div style="font-weight: 700; color: #1e3a8a; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">Prefill Nodes (GPUs)</div>
    <div style="font-size: 12px; color: #475569; margin-bottom: 8px;">Compute-optimized batching</div>
    <div style="background-color: #dbeafe; color: #1e40af; font-size: 12px; padding: 6px; border-radius: 4px; font-weight: 600; margin-top: 4px; border: 1px dashed #93c5fd;">High Tensor Core Saturation</div>
  </div>

  <!-- Network Interconnect (Arrow Bridge) -->
  <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 120px; text-align: center;">
    <span style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">KV Cache Transfer</span>
    <div style="width: 100%; height: 6px; background: linear-gradient(90deg, #2563eb, #10b981); border-radius: 3px;"></div>
    <span style="font-size: 10px; color: #94a3b8; margin-top: 4px; font-family: monospace;">RDMA / NVLink</span>
  </div>

  <!-- Decode Node Pool -->
  <div style="flex: 1; border: 1.5px solid #10b981; border-radius: 8px; background-color: #f1f5f9; padding: 16px; text-align: center;">
    <div style="font-weight: 700; color: #065f46; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">Decode Nodes (GPUs)</div>
    <div style="font-size: 12px; color: #475569; margin-bottom: 8px;">Memory-bandwidth optimized batching</div>
    <div style="background-color: #d1fae5; color: #065f46; font-size: 12px; padding: 6px; border-radius: 4px; font-weight: 600; margin-top: 4px; border: 1px dashed #6ee7b7;">Max Memory Bandwidth Saturation</div>
  </div>
</div>
```

1. **Prefill Nodes**: Process the prompt sequences and compute the initial KV states.
2. **KV Cache Transfer**: Send the generated KV states over a high-speed network (InfiniBand, RDMA, or NVLink) to the Decode nodes.
3. **Decode Nodes**: Append the states to memory and execute the low-intensity decode steps without blocking from new prompt arrivals.

### Serving Topology Trade-offs (Unified vs. PD Disaggregated)

| Topology | Pros | Cons | Production Use Case |
|---|---|---|---|
| **Unified serving (Single GPU / Node)** | • Simple orchestration and zero network overhead.<br>• No need for expensive high-speed RDMA interconnects.<br>• Easy deployment on standard cloud instances. | • Heavy Head-of-Line blocking (a long prefill prompt pauses decodes).<br>• Poor GPU resource utilization due to mixed compute/memory bounds. | Low-to-medium volume deployments, personal assistants, or offline jobs. |
| **PD Disaggregation (Split Pool serving)** | • Bypasses Head-of-Line blocking entirely.<br>• Allows compute-optimized nodes (H100) for prefill, and memory-bandwidth nodes (A100) for decode.<br>• Distinct queue batching policies. | • Heavy KV Cache transfer overhead across network.<br>• High infrastructure complexity and cost (requires RDMA/InfiniBand).<br>• Routing layer overhead. | High-scale LLM API providers (OpenAI, Anyscale) serving millions of concurrent requests. |

---

## 5. Core Latency & Throughput Metrics

Production LLM hosting targets three metrics to satisfy client SLAs:

| Metric | Definition | Critical For | Hard Limits |
|---|---|---|---|
| **TTFT**<br>(Time to First Token) | Time elapsed from sending a request to receiving the first generated token. | Interactive Chat, Real-Time Interfaces. | Bounded by prefill execution speed and request queue delays. |
| **TPOT / ITL**<br>(Time Per Output Token) | The average time interval between generating successive tokens. | Reading comfort, downstream streaming consumers. | Bounded by GPU HBM memory bandwidth. |
| **TPS**<br>(Tokens Per Second) | Total output tokens generated per second across all users ($TPS = \text{Batch Size} \times \frac{1}{\text{TPOT}}$). | Offline batch processing pipelines (summarization, indexing). | Bounded by peak scheduling efficiency and VRAM capacity. |

---

### Interview Questions & Production Trade-offs

#### What problem does this solve?
It decouples the end-to-end LLM inference workflow into a high-parallelism prefill phase and a sequential, token-by-token decoding loop. This separation highlights the distinction between compute-bound matrix processing and memory-bandwidth-bound matrix-vector operations, laying the groundwork for latency optimizations.

#### Why was it introduced?
As models scaled from millions to billions of parameters, naive sequential rendering became too slow for user interfaces. Formulating prefill and decode stages enabled systems architects to isolate bottlenecks (FLOPs limits vs. HBM bus throughput) and design specialized scheduling, memory cache managers, and hardware kernels.

#### What are its limitations?
- **Low Compute Occupancy during Decode**: Because decode is memory-bound, GPU Tensor Cores sit idle up to 85%-90% of the time, leading to low execution occupancy.
- **Head-of-Line Blocking**: Long prompt prefill processing can stall active decoding streams in a shared execution queue.

#### Computational Complexity (Time & Memory)
- **Prefill (Prompt Processing)**:
  - *Time Complexity*: $O(L_{\text{prompt}}^2 \cdot d + L_{\text{prompt}} \cdot d^2)$ operations per attention layer.
  - *Memory Complexity*: $O(b \cdot L_{\text{prompt}}^2 + b \cdot L_{\text{prompt}} \cdot d)$ space for intermediate attention score activations.
- **Decode (Token Generation)**:
  - *Time Complexity*: $O(b \cdot L_{\text{output}} \cdot d^2)$ operations per attention layer (for query projections and output feed-forwards).
  - *Memory Complexity*: $O(b \cdot d)$ per generation step.

#### Component Variable Denotation Legend
- $b$: Batch size (number of concurrent client queries).
- $L$: Sequence token length (where $L = L_{\text{prompt}} + L_{\text{output}}$).
- $d$: Hidden model dimension ($d = n_{\text{heads}} \times d_{\text{head}}$).
- $s$: Number of attention layers.
- $P$: Model parameter count (in billions).
- $B_{\text{peak}}$: Peak GPU memory bus bandwidth (in Bytes/sec).
- $P_{\text{peak}}$: Peak GPU floating-point compute capacity (in FLOPs/sec).

#### Production Use Cases
- **Low-Latency Streaming API Gateways**: Optimizing TTFT and TPOT to meet strict customer Service Level Agreements (SLAs).
- **Batch Processing Systems**: Maximizing overall model throughput (tokens/sec) for offline document categorization or database summaries.

#### Follow-up questions interviewers ask
1. *Why does the prefill phase transition from compute-bound to memory-bound when sequence lengths exceed a certain threshold?*
   - **Answer**: Attention matrix size scales quadratically ($O(L^2)$). As $L$ becomes extremely large, loading massive query-key-value attention score metrics from HBM to SRAM dominates execution time, shifting the bottleneck to memory bandwidth.
2. *Describe the quantitative trade-off between collocated serving vs. Prefill-Decode (PD) Disaggregated Serving.*
   - **Answer**: Collocated serving has zero inter-node communication latency but suffers from p99 tail latency spikes due to prefill tasks blocking decodes. PD disaggregation stabilizes decode latencies (lowering p99 TPOT by up to 50%) but introduces KV cache transport latency across networks (RDMA/InfiniBand) between prefill and decode nodes.
