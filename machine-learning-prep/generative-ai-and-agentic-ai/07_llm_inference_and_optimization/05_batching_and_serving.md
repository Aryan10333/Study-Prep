# 05. Batching, Scheduling & Serving: Iteration Scheduling and Parallelism

LLM hosting engines must manage multiple concurrent client requests. Because generating tokens autoregressively is a memory-bound process, serving requests sequentially results in low GPU utility. To maximize throughput and satisfy strict Service Level Agreements (SLAs), engines use advanced batching schedulers and multi-GPU tensor scaling.

---

## 1. Batching Evolution

Batching multiple requests together allows the GPU to share loaded model weights across multiple input tokens, increasing arithmetic intensity and Tensor Core utilization.

### Static Batching
In static batching, requests are batched together and padded with dummy tokens to match the length of the longest request in the batch:
- **Wasted FLOPs**: The GPU executes computation on padding tokens, consuming memory and cycles.
- **Wasted VRAM**: Padding tokens consume block allocations in the KV Cache.
- **Easiest Finish Bottleneck**: The batch cannot release resources until the longest request completes, keeping shorter requests locked.

### Dynamic Batching
Dynamic batching groups incoming requests into batches when they enter a queue buffer within a fixed time window. While it improves alignment, it still suffers from padding issues and head-of-line blocking under varying sequence lengths.

### Continuous Batching (Iteration-level Scheduling)
Continuous batching operates at the iteration level rather than the sequence level:
1. The batch is stepped token-by-token.
2. At the end of a single decoding iteration, the scheduler inspects the queue.
3. Completed requests (those that hit `[EOS]` or max tokens) are immediately evicted from the active batch.
4. New requests (waiting in the queue) are inserted into the vacant slots.
This eliminates padding overheads, boosting throughput by **$2\times - 4\times$** compared to static batching.

---

## 2. Request Scheduling Algorithms

Schedulers manage the allocation of physical blocks for the KV Cache:
- **First-Come-First-Served (FCFS)**: Default queue. Requests are processed in order of arrival. Can lead to preemption if KV Cache memory fills up.
- **Chunked Prefill**: Long prompt prefills require high compute time, which stalls ongoing decode requests and causes Inter-Token Latency (ITL) spikes. Chunked prefill splits a long prefill prompt into chunks (e.g. 512 tokens). It schedules one chunk of prefill alongside active decode requests in a single batch iteration, stabilizing ITL.

---

## 3. Multi-GPU Parallelism Strategies

When a model is too large to fit in the VRAM of a single GPU, or when we need to distribute the memory footprint, we apply parallelism:

### Tensor Parallelism (TP - Megatron-LM)
Splits individual weight matrices across multiple GPUs within a single node. This requires fast GPU-to-GPU communication (e.g. NVLink) because synchronization is required at every layer.

```html
<div style="display: flex; flex-direction: column; gap: 20px; font-family: 'Segoe UI', sans-serif; margin: 24px 0;">
  <!-- Column Parallel -->
  <div style="border: 1px solid #3b82f6; border-radius: 8px; background-color: #f8fafc; padding: 16px;">
    <div style="font-weight: 700; color: #1e3a8a; font-size: 14px; margin-bottom: 8px; text-transform: uppercase;">1. Column-Parallel Linear Layer</div>
    <div style="font-size: 13px; color: #334155; margin-bottom: 8px;">
      Weights $W$ are split vertically: $W = [W_1 \mid W_2]$. Input $X$ is replicated on all GPUs.
    </div>
    <div style="background-color: #f1f5f9; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 12px; color: #0f172a; border-left: 3px solid #3b82f6;">
      GPU 1: Y1 = X * W1 <br>
      GPU 2: Y2 = X * W2 <br>
      Output: Y = Concatenate(Y1, Y2) [Requires All-Gather synchronization]
    </div>
  </div>

  <!-- Row Parallel -->
  <div style="border: 1px solid #10b981; border-radius: 8px; background-color: #f8fafc; padding: 16px;">
    <div style="font-weight: 700; color: #065f46; font-size: 14px; margin-bottom: 8px; text-transform: uppercase;">2. Row-Parallel Linear Layer</div>
    <div style="font-size: 13px; color: #334155; margin-bottom: 8px;">
      Weights $W$ are split horizontally: $W = [W_1^T, W_2^T]^T$. Input $X$ is split column-wise: $X = [X_1 \mid X_2]$.
    </div>
    <div style="background-color: #f1f5f9; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 12px; color: #0f172a; border-left: 3px solid #10b981;">
      GPU 1: Y1 = X1 * W1 <br>
      GPU 2: Y2 = X2 * W2 <br>
      Output: Y = Y1 + Y2 [Requires All-Reduce Sum synchronization]
    </div>
  </div>
</div>
```

In a standard Transformer block, we chain these layers:
- Attention input projections (Q, K, V) are Column-Parallel.
- Attention output projection (O) is Row-Parallel.
- MLP Gate/Up projections are Column-Parallel.
- MLP Down projection is Row-Parallel.
This arrangement allows us to execute the entire attention block with only one `All-Reduce` operation at the end of attention and one `All-Reduce` at the end of MLP, minimizing communication overhead.

### Pipeline Parallelism (PP)
Splits model layers across nodes (e.g. GPU 0 holds layers 1-10, GPU 1 holds 11-20).
- **Communication**: Communication is sequential, only passing activations at the boundaries of the pipeline stages.
- **Pipeline Bubble**: Idle wait times occur while nodes wait for activations from previous stages. This is mitigated by dividing batches into micro-batches (e.g., using 1F1B scheduling).

### Context Parallelism (CP)
Splits the sequence length dimension of a long context input across GPUs. Each GPU processes a chunk of the tokens and computes local attention queries, keys, and values, exchanging attention metrics via ring-attention rings. Useful for sequences exceeding 100k tokens.

### Parallelism Strategy Comparison Matrix

| Strategy | Partition Dimension | Pros | Cons | Production Choice |
|---|---|---|---|---|
| **Tensor Parallelism (TP)** | Layer Weights (Columns/Rows) | • Lowest latency; highly efficient weight reuse.<br>• Replicates execution states cleanly. | • Requires ultra-fast interconnects (NVLink); typically limited to 8 GPUs within a single node. | Scaling 70B class models within a single HGX node. |
| **Pipeline Parallelism (PP)** | Model Layers (Depth-wise) | • Scales across multiple server boxes over standard networking.<br>• Low bandwidth demands. | • Introduces pipeline bubbles (nodes waiting for boundary activations).<br>• High scheduling complexity (1F1B loops). | Scaling 405B class models across multiple interconnected nodes. |
| **Context Parallelism (CP)** | Sequence Length (Tokens) | • Enables processing of ultra-long contexts (128k+) that would overflow a single GPU's KV VRAM. | • Requires complex ring-attention communications to calculate global softmax scores. | Specialized long-context generation pipelines, stacked on top of TP/PP. |

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
