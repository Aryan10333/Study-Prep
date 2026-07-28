# 09. Performance Monitoring, Profiling & Debugging: Production Incidents

Running LLM serving clusters at scale requires continuously monitoring infrastructure telemetry and system metrics. Because LLM workloads dynamically transition between compute-bound and memory-bound phases, diagnosing production incidents—such as latency spikes, memory crashes, and poor hardware utilization—demands a structured debugging protocol.

---

## 1. Operational Telemetry Metrics

To maintain health and reliability, serving platforms monitor two categories of metrics:

### Latency & Quality Metrics
- **TTFT (p50, p95, p99)**: High p99 TTFT values indicate queue saturation or prefill scheduling bottlenecks.
- **TPOT / ITL (p50, p95, p99)**: The inter-token generation speed. Steady TPOT values confirm stable memory reads.
- **End-to-End Latency**: The total time from user submission to final response delivery.
- **Token Generation Ratio**: The ratio of generated tokens to prompt tokens, which helps project workload evolution.

### Hardware & Memory Metrics
- **GPU HBM Memory Utilization**: The fraction of High Bandwidth Memory occupied.
- **KV Cache Block Allocation / Hit Rate**: The percentage of physical blocks allocated to cached sequences. A high hit rate indicates efficient prefix sharing.
- **Model FLOPs Utilization (MFU)**: The ratio of attained execution FLOP/s to the GPU's peak theoretical FLOP/s.
- **CUDA Memory Fragmentation Ratio**: The ratio of requested memory allocations to the largest contiguous free memory block.

---

## 2. Debugging Real-World Production Incidents

Below is a structured diagnostic guide for the three most common production LLM serving issues:

### Incident A: CUDA Out of Memory (OOM) Errors under Load

#### Root Causes
1. **KV Cache Over-allocation**: The scheduler allocates too many physical blocks relative to the maximum sequence length, leading to memory exhaustion under high concurrency.
2. **Activation Spikes**: Compute-heavy prefill operations for long prompts create massive intermediate activation matrices in SRAM/HBM.
3. **Memory Leaks**: Non-contiguous allocations fragment CUDA memory, preventing the engine from allocating space for new request blocks.

#### Diagnostic Checklist
1. Inspect the stack trace: Did the crash occur during the prefill phase or the decode phase?
2. Check `max_model_len` and `gpu_memory_utilization` parameters in your configuration.
3. Query `nvidia-smi` or PyTorch's `memory_allocated()` to trace active allocations.
4. Set the vLLM parameter `max_num_seqs` to restrict the concurrent batch size.

---

### Incident B: TTFT Latency Spikes (e.g. p99 TTFT > 10 Seconds)

#### Root Causes
1. **Queue Saturation**: A surge in request volume saturates the prefill queue, leaving incoming requests waiting for GPU scheduling slots.
2. **Un-cached System Prompts**: Multi-turn chats or RAG prompts with large context prefixes bypass the prefix cache, requiring the engine to recompute the entire prompt prefill.
3. **Head-of-Line Blocking**: Long prefill requests block active decoding steps.

#### Diagnostic Checklist
1. Inspect the scheduler queue status: Measure the wait time in the queue vs. the active compute time.
2. Verify the prefix cache hit rate. If the hit rate is low, inspect why prompts are changing slightly (e.g., dynamic timestamps or user IDs in system messages).
3. Enable **Chunked Prefill** to split long prompts and interleave prefill compute with active decode iterations.

---

### Incident C: Low Throughput & GPU Under-utilization (Low MFU)

#### Root Causes
1. **Small Batch Sizes**: Under low traffic volume, the engine cannot form large batches, forcing the GPU to execute memory-bandwidth-bound single-token decodes.
2. **CPU-GPU Transfer Overheads**: Repeatedly transferring data between host memory (RAM) and GPU VRAM stalls Tensor Core processing.
3. **Un-batched Tool Calls**: Synchronously waiting for external API queries halts the generation loop.

#### Diagnostic Checklist
1. Check the active batch size. If traffic is low, consider deploying smaller, quantized models to reduce serving costs.
2. Use **CUDA Graphs** to record and execute sequences of GPU kernels, bypassing CPU-launch overheads.
3. Ensure tool execution and routing are asynchronous.

---

## 3. Profiling Tools

To diagnose deep performance bottlenecks, developers use profiling frameworks:
- **PyTorch Profiler**: Captures CPU/GPU execution trace files, identifying slow operators and memory allocation spikes.
- **NVIDIA Nsight Systems (`nsys`)**: A system-wide profiling tool that captures CUDA kernel execution timelines, NVLink communication overheads, and hardware stalls.
- **Prometheus vLLM endpoints**: vLLM exposes `/metrics` endpoints tracking current queue sizes, KV cache block hit rates, and iteration latencies, which can be visualized in Grafana dashboards.

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
