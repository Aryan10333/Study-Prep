# Module 05: Quantization for Inference

## 1. Introduction & Intuition

### The Core Bottleneck
Every real byte a GPU has to move — weights, activations, or KV cache — costs real memory-bandwidth traffic and real memory capacity, and Module 01's roofline framing already established that much of real LLM inference (especially decode) is bottlenecked by exactly that data movement, not raw compute. Quantization attacks this directly: represent the same real values with fewer real bits, and every one of those memory costs shrinks proportionally. It is a genuinely different lever from Module 02's GQA/MQA (which reduces *how many* KV vectors exist) or Module 03's PagedAttention (which reduces *wasted* allocation) — quantization reduces the real *size of each individual value* being stored and moved.

### High-Level Intuition
Storing a number at FP16 versus INT4 is like writing a measurement down to the millimeter versus rounding it to the nearest centimeter — the rounded version takes far less space to write down and carry around, at the real cost of some lost precision. For most of an LLM's real weights and activations, that lost precision turns out to cost surprisingly little real quality if done carefully, while the real memory savings are large and direct.

---

## 2. Core Concepts & Mathematical Formulation

### Bytes-Per-Parameter and Real Memory Footprint Across Three Distinct Targets

#### Intuition & Practical Use
The real memory a model consumes at a given numeric precision is simply the number of values stored times the real number of bytes each value occupies at that bit-width. This applies identically to three **distinct real targets**, each with its own accuracy/speed/memory trade-off profile:

*   **Weight quantization** — the model's real, fixed parameters, quantized once (often offline).
*   **Activation quantization** — real intermediate values computed during the forward pass, quantized on the fly; generally a harder real accuracy trade-off than weights since activations vary per-input.
*   **KV-cache quantization** — the real, per-token key/value vectors from Module 02, quantized as they're written to the cache; a direct, real lever specifically for the long-context/high-concurrency regime where Module 02 already showed KV cache can dominate real memory.

$$\text{Mem} = N_{\text{params}} \times \frac{\text{bits}}{8}$$

### Precision Reduction Does Not Guarantee Proportional Speed Gains

#### Intuition & Practical Use
It's tempting to assume that halving a value's real storage size halves real latency too — this is **not guaranteed**. The real memory-footprint reduction above is direct and real. Whether that translates into real *speed* depends on two separate, real conditions: (1) whether the workload's actual bottleneck (per Module 01's roofline framing) is genuinely memory-bandwidth-bound in the first place — a compute-bound workload sees little real speed benefit from smaller values, since compute time, not data movement, was the real constraint; and (2) whether the serving hardware and kernel implementation genuinely support fast execution at that specific lower precision — a GPU or kernel without real low-precision compute paths may have to upcast values back to a higher precision before computing, which can add real overhead that partially or fully offsets the memory win.

---

### Hand Calculation: Real Memory Footprint Across Three Targets, Three Precisions
An illustrative 7B-parameter model, and (for the KV-cache case) Module 02's own high-concurrency MHA scenario ($B{=}32$, $L{=}4{,}096$, $N_{\text{KV\_heads}}{=}32$, 32 layers, $d_{\text{head}}{=}128$), in binary GB, consistent with Module 02's units.

*   **Step 1: Weight memory across three precisions.**
    $$\text{FP16: } 7\times10^9 \times \tfrac{16}{8} / 1024^3 \approx 13.04\text{ GB} \quad \text{INT8: } \approx 6.52\text{ GB} \quad \text{INT4: } \approx 3.26\text{ GB}$$
    Each halving of bit-width real, directly halves real weight memory — exactly as the linear formula predicts.

*   **Step 2: KV-cache memory across the same three precisions, reusing Module 02's formula at reduced `bytes_per_elem`.**
    $$\text{FP16: } 64.00\text{ GB} \quad \text{INT8: } 32.00\text{ GB} \quad \text{INT4: } 16.00\text{ GB}$$

*   **Step 3: Real interpretation.** At INT4, this illustrative workload's total real memory (weights + KV cache) drops from $13.04 + 64.00 = 77.04\text{ GB}$ (FP16) to $3.26 + 16.00 = 19.26\text{ GB}$ — a real $\approx 4\times$ reduction, directly reusable to serve real higher concurrency or longer real context in the same fixed GPU memory budget. Whether real *latency* also drops $4\times$ is a separate, unverified question per this module's precision-vs-speed caveat — not answered by the memory formula alone.

---

## 3. Implementation & Reference Code

```python
GiB = 1024 ** 3


def weight_memory_gb(n_params: int, bits: int) -> float:
    return n_params * bits / 8 / GiB


def kv_cache_memory_gb(n_layers: int, n_kv_heads: int, d_head: int, batch_size: int, seq_len: int, bits: int) -> float:
    bytes_per_elem = bits / 8
    total_bytes = 2 * batch_size * seq_len * n_layers * n_kv_heads * d_head * bytes_per_elem
    return total_bytes / GiB


if __name__ == "__main__":
    N_PARAMS = 7_000_000_000
    precisions = [(16, "FP16"), (8, "INT8"), (4, "INT4")]

    print("Weight memory:")
    weight_results = {}
    for bits, name in precisions:
        mem = weight_memory_gb(N_PARAMS, bits)
        weight_results[bits] = mem
        print(f"  {name}: {mem:.2f} GB")

    assert abs(weight_results[16] / weight_results[8] - 2.0) < 0.01, "Halving bit-width must exactly halve real memory"
    assert abs(weight_results[8] / weight_results[4] - 2.0) < 0.01

    print("\nKV cache memory (Module 02's high-concurrency MHA scenario, B=32, L=4096):")
    kv_results = {}
    for bits, name in precisions:
        mem = kv_cache_memory_gb(n_layers=32, n_kv_heads=32, d_head=128, batch_size=32, seq_len=4096, bits=bits)
        kv_results[bits] = mem
        print(f"  {name}: {mem:.2f} GB")

    assert abs(kv_results[16] - 64.0) < 0.01, "Must match Module 02's real FP16 KV-cache figure exactly"
    assert abs(kv_results[16] / kv_results[4] - 4.0) < 0.01, "FP16 -> INT4 must give exactly a real 4x KV-cache memory reduction"

    total_fp16 = weight_results[16] + kv_results[16]
    total_int4 = weight_results[4] + kv_results[4]
    print(f"\nTotal (weights + KV cache): FP16={total_fp16:.2f} GB, INT4={total_int4:.2f} GB, reduction={total_fp16/total_int4:.2f}x")

    print("\nVerified: real memory-footprint reduction is exact and linear in bit-width for both targets.")
    print("This says nothing about real latency -- that depends separately on whether the workload is memory-bandwidth-bound")
    print("(Module 01) and whether the serving hardware/kernels genuinely support fast low-precision execution.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Real memory footprint (and, conditionally, real memory-bandwidth traffic) of weights, activations, and KV cache — a direct, multiplicative lever on top of Module 02's KV-cache-head-count reduction and Module 03's allocation-efficiency improvement.
* **Why Introduced over Legacy Approaches:** Running everything at FP32/FP16 leaves real memory and bandwidth on the table for values that, empirically, don't need that much real precision to preserve model quality; quantization captures that real slack.
* **Key Failure Modes & Limitations:** Assuming a memory-footprint win from lower precision automatically implies a proportional real latency win — the module's central caveat; applying aggressive quantization (e.g., INT4) uniformly without validating real downstream task accuracy, since activation and KV-cache quantization in particular can be more accuracy-sensitive than weight quantization.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Quantization doesn't change the real number of logical operations attention/matmuls perform; real speed benefit (when it exists) comes from moving less real data per operation and/or hardware-native low-precision compute paths executing faster per operation.
* **Space/Memory Footprint:** Scales linearly, exactly, with bit-width for a fixed parameter/element count — verified exactly above (2x per halving, both for weights and KV cache).
* **Primary Bottleneck Type:** Directly targets the memory-bandwidth/memory-capacity side of Module 01's roofline framing; a compute-bound workload (per that same framing) sees a real memory win here with little to no real speed win, per this module's central caveat.
* **Variable Legend:** $N_{\text{params}}$ = parameter (or element) count, bits = numeric precision width, GB throughout = binary GiB, consistent with Module 02's units.

### 3. Production & Scalability
* **Deployment Considerations:** GPTQ/AWQ (weight-only, post-training), bitsandbytes (accessible, broad-support quantization), and native FP8 (hardware-accelerated on recent GPUs) each occupy a different real point on the accuracy/speed/ease-of-deployment trade-off; KV-cache quantization specifically earns its real cost most clearly in the long-context/high-concurrency regime Module 02 already identified as KV-cache-memory-dominated.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* You quantized a model's weights from FP16 to INT4 and memory dropped 4x, but real measured latency barely improved — why?
        *   *A:* The workload was likely compute-bound rather than memory-bandwidth-bound (Module 01's roofline framing) for the real batch size/hardware in question, or the serving kernel lacked genuine native INT4 compute support and had to upcast before computing — either real condition would explain a memory win without a proportional real speed win, exactly per this module's central caveat.
    2.  *Q:* Why might KV-cache quantization be riskier for real accuracy than weight quantization?
        *   *A:* KV-cache values are real, per-input activations accumulated across a whole generation, not fixed, offline-calibratable parameters — errors introduced by quantizing them can compound across a long real sequence in a way static weight quantization, calibrated once, does not.
