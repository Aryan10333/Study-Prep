# 🎯 LLM Inference & Optimization: High-Frequency Question Bank

> **Target Role:** AI Engineer / Applied AI Engineer / LLM Engineer (3+ Years Experience)  
> **Scope:** Two-phase execution, KV Cache VRAM math, PagedAttention, quantization mechanics, continuous batching, speculative tree decoding, vLLM/SGLang engines, PD disaggregation, and production incident debugging.

---

## 1. Inference Fundamentals (8 Questions)

### 1. End-to-End Inference Pipeline
**Explain the complete LLM inference pipeline step-by-step: Prompt $\rightarrow$ Tokenization $\rightarrow$ Embedding $\rightarrow$ Prefill GEMM $\rightarrow$ Autoregressive Decode GEMV $\rightarrow$ Sampling $\rightarrow$ Detokenization.**
- **Short Answer**: The inference pipeline consists of processing the text input into discrete tokens, mapping them to embeddings, executing a parallel prefill phase (compute-bound matrix-matrix multiplication) to ingest the prompt and allocate the KV Cache, running sequential decode loops (memory-bandwidth-bound matrix-vector multiplication) to generate successive tokens, applying sampling logits transformations to select the next token, and detokenizing the resulting IDs back to human-readable characters.
- **Key Interview Points**: Prefill GEMM vs. Decode GEMV, KV Cache allocation, token-to-embedding mapping, and sampling constraints.
- **Technical Intuition**: Text is mapped to token IDs via BPE/SentencePiece. Embeddings convert these IDs to vector spaces $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$. Prefill calculates self-attention for all tokens in parallel using general matrix multiplication (GEMM) since the entire sequence is available. Decode computes queries, keys, and values for the new token, querying the cached KV tensors to compute single-token attention via matrix-vector products (GEMV).
- **Production Perspective**: Ensure detokenizers support streaming Server-Sent Events (SSE) so users receive tokens incrementally instead of waiting for the full output block.
- **Follow-up**: *What causes detokenization lag?* (Mainly buffering issues with multi-byte UTF-8 characters split across token boundaries).

### 2. Prefill Phase vs. Decode Phase
**What are the fundamental computational and execution differences between the Prefill (Prompt Processing) phase and the Decode (Token Generation) phase?**
- **Short Answer**: The Prefill phase processes all prompt tokens in parallel, saturating GPU Tensor Cores via compute-bound General Matrix Multiplications (GEMM). The Decode phase generates one token at a time sequentially, loading the entire model's weights and past KV caches for each step, which makes it memory-bandwidth-bound General Matrix-Vector Multiplication (GEMV).
- **Key Interview Points**: Compute-bound vs. memory-bandwidth-bound, GEMM vs. GEMV kernels, parallel input vs. sequential output.
- **Technical Intuition**: During prefill, the query, key, and value shapes are $[B, L_{\text{prompt}}, d_{\text{model}}]$. Since $L_{\text{prompt}} \gg 1$, matrix multiplications reuse model weights, yielding high arithmetic intensity. During decode, the query shape is $[B, 1, d_{\text{model}}]$, requiring the GPU to load the entire layer weights from HBM to SRAM to perform a single vector-matrix multiply.
- **Production Perspective**: Optimizing serving requires adjusting batching policies. Heavy prefill batches should be chunked to avoid halting active decode cycles.
- **Follow-up**: *Why does prefill have higher GPU power draw?* (Because Tensor Cores are active continuously, whereas decode leaves them mostly idle while waiting for memory transfers).

### 3. Arithmetic Intensity & Roofline Bottlenecks
**Define **Arithmetic Intensity** ($\text{FLOPs/Byte}$). Use the Roofline model to prove why the prefill phase is compute-bound ($\text{GEMM}$) while the decode phase is memory-bandwidth-bound ($\text{GEMV}$).**
- **Short Answer**: Arithmetic Intensity ($I$) is the ratio of FLOPs executed to memory bytes read/written. The Roofline Model states performance is bounded by $\min(P_{\text{peak}}, I \cdot B_{\text{peak}})$. For prefill, $I \approx L_{\text{prompt}}$, exceeding the hardware ridge point and making it compute-bound. For decode, $I \approx 1$ FLOP/Byte, which is well below the ridge point, making it memory-bandwidth-bound.
- **Key Interview Points**: Arithmetic Intensity formula, Roofline Model regions, Hardware Ridge Point calculation, GEMM vs. GEMV bounds.
- **Technical Intuition**: Given a model of size $P$ parameters:
  - Prefill ($N$ tokens): $\text{FLOPs} = 2PN$, $\text{Bytes} = 2P$ (weights) + cache writes. Thus, $I \approx N$. For $N=1024$ and FP16 weights, $I = 1024$ FLOPs/Byte.
  - Decode ($1$ token): $\text{FLOPs} = 2P$, $\text{Bytes} = 2P$ (weights) + cache reads. Thus, $I \approx 1$.
  Since an NVIDIA A100 has a ridge point of $\approx 156$ FLOPs/Byte, prefill ($I=1024 > 156$) is compute-bound, while decode ($I=1 < 156$) is memory-bandwidth-bound.
- **Production Perspective**: Increasing memory bandwidth (e.g. upgrading H100 to H200 with faster HBM3e) directly improves decode throughput, while compute upgrades improve prefill speeds.
- **Follow-up**: *How does quantization affect arithmetic intensity?* (Quantizing to INT8/INT4 halves or quarters the denominator, increasing $I$ and moving decode execution closer to the compute-bound boundary).

### 4. Latency & Throughput Metrics
**Define TTFT (Time To First Token), TPOT (Time Per Output Token / Inter-Token Latency), TPS (Tokens Per Second), and End-to-End Latency. When is each metric critical for user SLAs?**
- **Short Answer**:
  - **TTFT**: Delay until the first token is received. Critical for real-time user-facing chat.
  - **TPOT (ITL)**: Average time between successive tokens. Critical for reading comfort.
  - **TPS**: Total token generation throughput. Critical for background batch jobs.
  - **End-to-End Latency**: Total time for request completion. Critical for non-streaming programmatic API integration.
- **Key Interview Points**: Latency component definitions, client experience mappings, and throughput metrics.
- **Technical Intuition**: $\text{TTFT} = \text{Queue Delay} + \text{Prefill Latency}$. $\text{TPOT} = \text{Decode Iteration Latency}$. $\text{End-to-End Latency} = \text{TTFT} + (L_{\text{output}} - 1) \times \text{TPOT}$.
- **Production Perspective**: Trade-off batch sizes: larger batches increase throughput (TPS) but degrade latency (TTFT and TPOT).
- **Follow-up**: *Why do chat interfaces care more about TTFT than End-to-End latency?* (Because a low TTFT creates a responsive feel, hiding the total generation time).

### 5. Prefill-Decode (PD) Disaggregation
**What is Prefill-Decode Disaggregation, and why does splitting compute-bound prefill nodes from memory-bandwidth-bound decode nodes across NVLink/RDMA networks prevent head-of-line blocking?**
- **Short Answer**: PD Disaggregation separates physical GPUs into dedicated prefill workers and decode workers. This prevents compute-heavy prompt processing steps from blocking active, iteration-level decode streams in the serving queue, resolving head-of-line blocking and stabilizing inter-token latency (TPOT).
- **Key Interview Points**: Head-of-line blocking, GPU workload isolation, KV Cache transfer cost, NVLink/RDMA fabrics.
- **Technical Intuition**: In collocated execution, a new prefill request takes over GPU compute, stalling decode iterations of active sessions for several hundred milliseconds. By splitting them, prefill nodes calculate keys/values and transfer them via high-speed RDMA networks directly to the decode nodes' memory pool, ensuring decode nodes execute uninterrupted.
- **Production Perspective**: PD disaggregation is highly beneficial for long-context workloads (e.g. RAG, document analysis) where prefill tasks are extremely heavy.
- **Follow-up**: *What is the main bottleneck in PD disaggregation?* (The network latency of transferring massive KV cache tensors between nodes).

### 6. Sequential Autoregressive Dependence
**Why is autoregressive token generation inherently sequential, and why does this prevent standard intra-sequence parallelization during the decode phase?**
- **Short Answer**: Autoregressive decoding relies on the causal dependency where each output token is conditioned on all preceding tokens: $P(x_t \mid x_{1:t-1})$. Because token $x_t$ cannot be computed until the token ID of $x_{t-1}$ is selected and appended to the input sequence, we cannot parallelize the execution of sequential tokens within a single request.
- **Key Interview Points**: Causal attention, sequence dependency, sequential decoding loop.
- **Technical Intuition**: Self-attention requires computing the query vector of the current step against the keys of all previous steps. Because the query vector $q_t$ depends on the token embedding of $x_{t-1}$, the matrix operations for step $t$ cannot begin until step $t-1$ completes.
- **Production Perspective**: To maximize hardware utilization despite sequence sequentiality, we use batching to process different sequences in parallel.
- **Follow-up**: *Can we generate multiple tokens at once losslessly?* (Only via Speculative Decoding, which validates multiple draft tokens in parallel).

### 7. Inference vs. Training Memory Footprints
**Why is LLM inference generally more memory-bandwidth-bound and VRAM-sensitive per generated token than LLM pre-training or fine-tuning?**
- **Short Answer**: During training, activation tensors are saved for backpropagation, and weights are reused across massive batches of tokens processed in parallel. During inference, we do not store gradients, but we must store the KV Cache for every concurrent request, which grows linearly with batch size and context length, making the VRAM footprint highly sensitive to dynamic traffic workloads.
- **Key Interview Points**: Activation storage, gradient caching, KV Cache footprint, weight reuse differences.
- **Technical Intuition**: Training uses $B \times L$ tokens per step, sharing weight overheads. Inference uses a batch size $B$ but processes only $1$ token per step, requiring the model to load all weights from HBM to SRAM for that single token's computation.
- **Production Perspective**: In training, we scale compute clusters to maximize FLOPs. In inference, we scale clusters primarily to accommodate the aggregate VRAM footprint of active KV Caches.
- **Follow-up**: *What is the activation memory footprint during prefill?* (It scales with the square of sequence length, which can trigger CUDA OOMs on extremely long prompts).

### 8. Latency Breakdown Factors
**What hardware and algorithmic factors contribute most to TTFT spikes vs. TPOT slowdowns in production LLM clusters?**
- **Short Answer**: TTFT spikes are caused by queue wait times, long prompt lengths (prefill computation), and uncached system prompts. TPOT slowdowns are caused by HBM bandwidth limits, high active batch sizes (requiring large weight/KV cache memory reads), and inter-GPU communication latency in Tensor Parallel setups.
- **Key Interview Points**: Queue delays, HBM bandwidth constraints, tensor parallel comms (All-Reduce overheads), and prefix cache misses.
- **Technical Intuition**: $\text{TTFT} \propto \frac{\text{Prompt Length} \times \text{Layers}}{\text{Compute FLOP/s}} + \text{Queue Delay}$. $\text{TPOT} \propto \frac{\text{Model Weights} + \text{KV Cache Size}}{\text{HBM Bandwidth}} + \text{Communication Latency}$.
- **Production Perspective**: To stabilize TTFT, deploy prefix caching and chunked prefill. To stabilize TPOT, use GQA, quantization (FP8/INT4), and optimize TP communications.
- **Follow-up**: *How does TP width affect TPOT?* (Increasing TP width reduces compute time per GPU but increases All-Reduce communication overhead. Beyond a certain width, communication latency outweighs compute gains, slowing down TPOT).

---

## 2. Decoding & Sampling (8 Questions)

### 9. Deterministic vs. Stochastic Sampling
**Compare Greedy Decoding, Beam Search, Top-K, Top-p (Nucleus), and **Min-p** sampling across output quality, diversity, and computational overhead.**
- **Short Answer**: Greedy decoding and Beam Search are deterministic; they produce high-factuality, low-diversity outputs. Beam Search has high memory overhead due to maintaining $B$ active KV caches. Top-K, Top-p, and Min-p are stochastic; they introduce creativity. Top-K uses a static cutoff count, Top-p dynamically adjusts the cutoff based on cumulative probability, and Min-p dynamically filters based on probability relative to the top token.
- **Key Interview Points**: Deterministic vs. stochastic, beam search memory footprint, static vs. dynamic truncation thresholds.
- **Technical Intuition**:
  - Greedy: $x_t = \arg\max P(x)$.
  - Beam Search: Tracks $B$ paths, maximizing $\sum \log P(x)$.
  - Top-K: Sorts vocabulary, samples from top $K$ candidates.
  - Top-p: Samples from smallest set $V$ where $\sum_{i \in V} P_i \ge p$.
  - Min-p: Samples from tokens where $P_i \ge P_{\text{max}} \times p_{\text{min\_threshold}}$.
- **Production Perspective**: Min-p is preferred for production chatbots because it maintains quality under high temperatures while preserving creative vocabulary options.
- **Follow-up**: *Why is Beam Search avoided in chat servers?* (Because it requires duplicating the KV cache $B$ times for each concurrent request, exhausting VRAM).

### 10. Temperature Scaling Mechanics
**Mathematically explain how Temperature ($T$) modifies the raw logit vector before the Softmax transformation ($P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$). What happens as $T \to 0$ and $T \to \infty$?**
- **Short Answer**: Temperature acts as a scaling divisor for the raw logits $z_i$. When $T \to 0$, the logit differences are amplified, making the distribution highly peaky and converging to deterministic greedy selection. When $T \to \infty$, the logit values converge to 0, making the exponentiated terms approach 1, which results in a uniform random distribution.
- **Key Interview Points**: Logit scaling, Softmax exponentiation, entropy changes.
- **Technical Intuition**: Given logits $z = [2.0, 1.0, 0.0]$:
  - At $T=1.0$: $P \approx [0.665, 0.245, 0.090]$.
  - At $T=0.5$ (low temp): scaled $z/T = [4.0, 2.0, 0.0] \implies P \approx [0.867, 0.117, 0.016]$ (higher confidence).
  - At $T=2.0$ (high temp): scaled $z/T = [1.0, 0.5, 0.0] \implies P \approx [0.506, 0.307, 0.186]$ (flatter).
- **Production Perspective**: Set $T=0.0$ (or greedy) for code generation, JSON parsing, and mathematical calculations. Use $T=0.7 - 0.9$ for creative text generation.
- **Follow-up**: *Does temperature change the ranking of tokens?* (No, because the division by $T > 0$ is a monotonic transformation. It only alters their relative probabilities).

### 11. Top-K vs. Top-p vs. Min-p Truncation
**Contrast static Top-K, cumulative Top-p, and dynamic confidence-relative **Min-p** truncation ($p_{\text{threshold}} = p_{\text{max}} \times \text{scale}$). Why is Min-p superior at preserving output coherence under high temperatures?**
- **Short Answer**: Top-K keeps a fixed number of tokens, ignoring the shape of the distribution. Top-p keeps a dynamic number of tokens based on cumulative probability, which can include low-probability tail tokens if the top token is weak. Min-p filters out tokens whose probability falls below a threshold relative to the top token's probability. This prevents low-confidence tail tokens from being sampled when the top token is highly confident, maintaining coherence under high temperatures.
- **Key Interview Points**: Fixed cutoffs vs. dynamic distributions, tail token suppression, dynamic confidence scaling.
- **Technical Intuition**: In a highly confident setting where the top token has $p_{\text{max}} = 0.90$:
  - Min-p (with scale $0.05$) sets the threshold to $0.045$. Any token with probability $< 4.5\%$ is pruned.
  - Top-p (with threshold $0.95$) might still include tokens down to $0.1\%$ to fill the remaining $5\%$ cumulative probability.
  This makes Min-p more effective at pruning low-quality tail tokens.
- **Production Perspective**: Deploying Min-p allows using higher temperatures (e.g. $T=1.2$) for creative variety without suffering from grammar collapse or gibberish.
- **Follow-up**: *Can Top-K and Top-p be combined?* (Yes, traditionally models apply Top-K first, then Top-p, then Temperature. Min-p replaces Top-p).

### 12. Application-Specific Sampling Parameters
**How would you configure temperature, top-p/min-p, and penalties for: Factual QA / Code Generation, Conversational Chatbots, Creative Writing & Brainstorming, and Structured JSON Data Extraction?**
- **Short Answer**:
  - **Factual QA / Code**: $T=0.0$ (Greedy), no truncation, no penalties.
  - **Structured JSON**: $T=0.0$ with grammar-guided constraints (Outlines).
  - **Conversational Chat**: $T=0.7$, Top-p = $0.90$ (or Min-p = $0.05$), Repetition Penalty = $1.05$.
  - **Creative Writing**: $T=1.1$, Min-p = $0.10$, Presence Penalty = $0.10$.
- **Key Interview Points**: Logit tuning, deterministic configurations, repetition penalty tuning.
- **Technical Intuition**: High factuality requires minimizing entropy (low temperature). Creative tasks require expanding the search space (high temperature, lower truncation thresholds). Repetition penalties prevent looping behaviors in open-ended generations.
- **Production Perspective**: Exposing these parameters via APIs allows clients to customize inference behavior per task domain.
- **Follow-up**: *Why do repetition penalties occasionally break JSON generation?* (Because they penalize repetitive structural characters like brackets and quotes, causing schema validation errors).

### 13. Logit Penalties
**Explain the mathematical mechanics of Frequency Penalty, Presence Penalty, and Repetition Penalty during logit post-processing.**
- **Short Answer**: Frequency and Presence penalties reduce logits linearly based on token occurrences. Frequency penalty scales with the count of appearances, while Presence penalty applies a constant deduction if the token appears at least once. Repetition penalty divides the logit by a scaling factor if the logit is positive, or multiplies it if negative, reducing the probability of repeating tokens.
- **Key Interview Points**: Linear vs. multiplicative penalties, frequency-dependent scaling, token presence tracking.
- **Technical Intuition**:
  - Frequency / Presence Penalty:
    $$z_i = z_i - (\text{count}_i \times \alpha_{\text{frequency}} + \delta(\text{count}_i > 0) \times \alpha_{\text{presence}})$$
  - Repetition Penalty ($\theta \ge 1.0$):
    $$z_i = \begin{cases} z_i / \theta & \text{if } z_i \ge 0 \\ z_i \cdot \theta & \text{if } z_i < 0 \end{cases}$$
- **Production Perspective**: Keep penalties subtle (e.g. $1.02 - 1.08$). High values can distort vocabulary distributions, causing grammar collapse.
- **Follow-up**: *Do these penalties apply to prompt tokens?* (Typically they only apply to generated output tokens, but some engines support penalizing prompt tokens as well).

### 14. Stop Sequences & Token Termination
**How are stop sequences monitored during streaming generation, and how do engines handle multi-token stop sequences across token boundaries?**
- **Short Answer**: Serving engines maintain a sliding buffer of recent output tokens. At each step, they check if the buffer suffix matches any configured stop sequence. If a stop sequence spans across multiple tokens, the engine buffers the potential match and delays sending the streaming response to the client until it confirmed a complete match or a mismatch.
- **Key Interview Points**: Sliding buffer validation, partial match buffering, streaming token delays.
- **Technical Intuition**: If the stop sequence is `"\nUser:"` and the model generates `"\n"`, then `"User"`, the engine buffers these tokens. If the next token is `":"`, the generation terminates and the buffered tokens are discarded. If the next token is `"agent"`, the engine flushes the buffered tokens (`"\nUser agent"`) to the stream.
- **Production Perspective**: Instruct client applications to handle partial stop sequence strings to avoid flickering in the user interface.
- **Follow-up**: *What happens if the model generates a stop sequence token but the engine misses it?* (The model continues generating text, causing prompt leaks or multi-turn sequence failures).

### 15. Constrained & Structured Decoding Overhead
**How do context-free grammars (CFGs) and regex state-masking (e.g., using Outlines or XGrammar) enforce strict JSON schemas during decoding without causing massive per-token latency penalties?**
- **Short Answer**: Constrained decoding compiles the target JSON schema or regex into a Finite State Machine (FSM). At each step, the FSM determines the valid next tokens, and the engine masks out invalid logits. To avoid per-token latency penalties, frameworks like XGrammar pre-compute token transition maps and cache them, keeping structured generation near native speeds.
- **Key Interview Points**: FSM token state transitions, logit masking ($P_i = -\infty$), transition map caching.
- **Technical Intuition**: The vocabulary is indexed by trie structures. When the FSM state requires a digit, the engine sets the logits of all non-digit tokens to $-\infty$. Doing this on the GPU via cached index lists keeps validation overhead under a few microseconds.
- **Production Perspective**: Structured generation guarantees that outputs are parseable, eliminating client-side retries and parsing crashes.
- **Follow-up**: *Can we enforce CFGs on closed APIs?* (Only via prompt engineering or schema parameters if supported. Logit masking requires direct access to the model's logits before sampling).

### 16. Beam Search Memory & Compute Footprint
**Why is Beam Search rarely used for online conversational LLM serving despite its utility in machine translation?**
- **Short Answer**: Beam Search maintains $B$ candidate sequences (beams) at each step. This requires keeping $B$ distinct KV Caches for a single request, multiplying memory usage by $B$. This memory footprint limits serving capacity and causes CUDA OOMs under high concurrency.
- **Key Interview Points**: KV Cache duplication, beam expansion complexity, memory overhead.
- **Technical Intuition**: For beam width $B=4$, the engine must store 4 key-value states for every token generated. During beam expansion, if paths split, the engine must copy and reorganize the KV cache blocks in VRAM, which creates high memory-copy overhead and slows down TPOT.
- **Production Perspective**: Use Beam Search only for offline batch translation or document generation where quality is critical and latency is not a bottleneck.
- **Follow-up**: *Does Beam Search improve conversational quality?* (No, it often leads to repetitive, generic responses in open-ended conversations compared to stochastic sampling).

---

## 8. Production Deployment, Monitoring & Debugging (8 Questions)

### 48. High-Availability Serving Architecture
**Design a production multi-region LLM serving infrastructure featuring dynamic model routing, semantic caching (GPTCache), blue-green model swaps, and zero-downtime failover.**
- **Short Answer**: A high-availability LLM infrastructure uses a global CDN load balancer routing to regional gateways. Regional gateways check incoming queries against a semantic cache (e.g., Redis + GPTCache) to instantly serve cached responses for redundant queries. For cache misses, requests are routed to local vLLM/SGLang engine nodes. If regional queues saturate or error rates spike, gateways automatically failover queries to standby regions or fallback third-party APIs. Model updates are managed via blue-green swaps where traffic is routed to the new cluster only after successful health checks.
- **Key Interview Points**: Global vs. regional gateways, semantic caching matching, multi-region failover latency, blue-green resource allocation.
- **Technical Intuition**: Semantic caches compute prompt embeddings and query a vector database: if cosine similarity $\ge 0.95$, the cached completion is returned. Model routers track regional GPU active cache block usage and queue backlogs. Zero-downtime failover redirects traffic to fallback zones when regional latencies cross SLA boundaries, protecting the user experience.
- **Production Perspective**: Keep connection timeouts short (e.g., <500ms) on regional endpoints to ensure instant failover triggers under load spikes.
- **Follow-up**: *How do you synchronize prefix caches across nodes?* (You don't sync them dynamically; instead, route requests with identical prompt prefixes to the same GPU worker nodes to maximize local cache hits).

### 49. Provider Abstraction & Fallback Policies
**How do you build a resilient provider abstraction layer that dynamically routes queries between self-hosted models (vLLM/SGLang) and cloud APIs (OpenAI/Anthropic/Gemini) based on latency, cost, and rate limits?**
- **Short Answer**: Design a central gateway router middleware that wraps self-hosted and cloud LLM APIs behind a single unified interface. The router uses a priority-based routing policy: it routes queries to self-hosted vLLM clusters first to minimize token costs. If the self-hosted cluster returns 429 (rate limit), 503 (service unavailable), or if active latency monitoring detects TPOT crossing SLA limits, the router fails over to commercial cloud APIs dynamically.
- **Key Interview Points**: Resiliency middleware, unified schema parsing, rate-limit headers, cost-latency thresholding.
- **Technical Intuition**: The gateway tracks downstream worker health. It parses rate-limit headers (`x-ratelimit-remaining`) to preemptively avoid sending traffic to exhausted nodes. Fallback transitions are implemented using circuit breakers that pause routes to failing self-hosted clusters for a cooldown window.
- **Production Perspective**: Map self-hosted and third-party outputs to a unified Pydantic schema (e.g., using Instructor) to prevent parser failures during fallback events.
- **Follow-up**: *How do you minimize fallback latency?* (Parallelize requests by starting a standby fallback call if the primary self-hosted call does not return a token within a tight timeout window).

### 50. Blue-Green & Canary Deployments
**How do you execute canary deployments for fine-tuned LLM service updates, and what automated regression checks (perplexity drift, output latency, guardrail pass rate) trigger rollback?**
- **Short Answer**: Deploy the new model (Green) alongside the production model (Blue) on a small subset of serving nodes (e.g., 5% traffic). Use shadow traffic to duplicate production prompts to the Green instances in the background. Automated monitors calculate Green's metrics: perplexity drift against calibration sets, TTFT/TPOT latency compliance, and guardrail pass rates. If any metric violates safety bounds, the gateway automatically rolls back traffic to the Blue instances.
- **Key Interview Points**: Shadow testing setups, metric evaluation pipelines, automatic rollback triggers.
- **Technical Intuition**: The gateway duplicates incoming payloads, executing inference on both models. Green's responses are analyzed but discarded, allowing the platform to verify performance under real-world traffic profiles without risking regression exposure to active users.
- **Production Perspective**: Canary testing is critical for LLMs because offline benchmark evaluations (like MMLU) often fail to capture subtle language drift or formatting regressions.
- **Follow-up**: *How do you measure perplexity dynamically?* (Perplexity requires calculating cross-entropy loss over logit values, which is only supported on self-hosted engines; for closed APIs, use LLM-as-a-judge consistency scoring).

### 51. Production Health & Telemetry Metrics
**What specific operational metrics (p50/p95/p99 TTFT, TPOT/ITL, GPU HBM utilization, KV cache block hit rate, CUDA memory fragmentation) do you monitor continuously?**
- **Short Answer**: We monitor p50/p95/p99 TTFT to evaluate prompt processing speeds and queue delays, TPOT/ITL to track output generation speeds, GPU HBM memory utilization to monitor overall capacity, KV Cache block hit rates to verify prefix caching reuse, and CUDA memory fragmentation ratios to identify potential memory exhaustion.
- **Key Interview Points**: Telemetry monitoring categories, latency distributions, memory usage indicators.
- **Technical Intuition**:
  - High TTFT indicates prefill queue congestion or cache misses.
  - Slow TPOT indicates HBM bandwidth bottlenecks or network overheads in Tensor Parallel clusters.
  - Dropping KV Cache hit rates indicates prompt template drift or cache thrashing.
  - CUDA fragmentation indicates non-contiguous memory allocations.
- **Production Perspective**: Configure Prometheus to scrape metrics directly from the serving engine (e.g., `/metrics` on vLLM) and compile them into Grafana dashboards.
- **Follow-up**: *What alert thresholds would you set for KV cache utilization?* (Trigger alerts when active block utilization exceeds 90% for sustained periods, indicating cache saturation and imminent preemption).

### 52. Debugging High TTFT Spikes
**Your monitoring system flags p99 TTFT spikes reaching 10 seconds while TPOT remains normal. Walk through your systematic diagnostic process (prefill queue backlogs, un-cached prompt prefixes, un-chunked prefills).**
- **Short Answer**: Identify the bottleneck by isolating the queue delay from the compute time:
  1. Check the queue size: if the queue is large, the spike is caused by prefill task backlogs.
  2. Check prefix cache hit rates: a sudden drop indicates that prompts are not matching cached prefixes, forcing the model to re-run full prefill compute passes.
  3. If the queue is small but prefill time is high, the cause is un-chunked prefills of massive prompts. Enable chunked prefill to interleave prompt processing and stabilize TTFT.
- **Key Interview Points**: Queue delay vs. compute delay, cache hit rate metrics, head-of-line blocking resolution.
- **Technical Intuition**: Because prefill is compute-bound, processing a 32k prompt can block the GPU for several seconds. If multiple long prompts arrive simultaneously, they block all decode slots, driving up TTFT.
- **Production Perspective**: Prevent prefill bottlenecks by enforcing prompt length limits at the gateway and standardizing system prompt formats to maximize Radix tree matches.
- **Follow-up**: *How do you identify prompt prefix mismatch causes?* (Inspect prompt templates: check if dynamic values like timestamps, user IDs, or random seeds are being injected into the system prefix).

### 53. Debugging GPU Out-Of-Memory (OOM) Errors
**A production LLM worker crashes with a CUDA OOM error under load. How do you investigate whether the root cause was activation spikes, KV cache over-allocation, or CUDA memory fragmentation?**
- **Short Answer**: Analyze the crash trace and allocator state:
  1. If the crash occurs during prefill, the cause is an activation memory spike from an excessively long prompt.
  2. If it occurs during decode under high concurrency, the cause is KV cache over-allocation.
  3. If the allocator logs show free VRAM bytes but no contiguous block of the requested size can be allocated, the cause is CUDA memory fragmentation.
- **Key Interview Points**: OOM timing checks, configuration bounds, allocation fragmentation patterns.
- **Technical Intuition**:
  - Activation memory scales with sequence length and batch size ($O(L^2)$).
  - KV Cache VRAM is bound by the engine's `gpu_memory_utilization` configuration.
  - Fragmentation occurs when allocating and deallocating memory of varying sizes without contiguous mapping.
- **Production Perspective**: Set `gpu_memory_utilization` to 0.90 (leaving a 10% buffer for activations) and configure PyTorch's allocator split limits to reduce fragmentation.
- **Follow-up**: *How does PagedAttention prevent fragmentation?* (By virtualizing the KV cache memory space, allowing non-contiguous physical block allocations to act as a single logical sequence).

### 54. Diagnosing Poor GPU Utilization (Low MFU)
**A GPU node shows 95% power consumption but Model FLOPs Utilization (MFU) is below 15%. What are the primary causes (small batch size, memory bandwidth saturation, CPU-GPU data transfer overhead)?**
- **Short Answer**: The primary cause is memory bandwidth saturation during the decode phase. Under low batch sizes, the GPU Tensor Cores sit idle while the system spends cycles loading model weights from HBM to SRAM for single tokens. Other causes include CPU-GPU synchronization stalls and un-batched tool execution delays.
- **Key Interview Points**: Memory-bandwidth bound idle states, low batch size compute limits, CPU-GPU data transfer bottlenecks.
- **Technical Intuition**: The GPU draws peak power because the memory buses are fully saturated reading parameters from HBM. However, because $I \approx 1$ FLOP/Byte during decode, the Tensor Cores execute very few operations, keeping MFU extremely low.
- **Production Perspective**: To raise MFU, increase batch sizes using continuous batching or implement speculative decoding to execute more FLOPs per memory read.
- **Follow-up**: *What is a typical MFU during prefill vs. decode?* (Prefill MFU can reach 40-50% due to compute parallelism, while decode MFU is often below 5% for small batches).

### 55. Cost Optimization Framework
**Walk through your exact strategy to reduce an enterprise LLM platform's monthly GPU cloud bill by 50% without dropping response quality or violating latency SLAs.**
- **Short Answer**: Implement a four-stage optimization pipeline:
  1. **Quantization**: Quantize model weights to FP8 or 4-bit AWQ, reducing model VRAM size and memory bandwidth pressure by 2x to 4x.
  2. **Prefix Caching**: Enable RadixAttention prefix caching to bypass prefill compute on repeated system prompts and RAG contexts.
  3. **Continuous Batching**: Use iteration-level scheduling to maximize throughput per GPU node.
  4. **Auto-scaling**: Set up horizontal auto-scaling to scale down GPU nodes during low-traffic hours.
- **Key Interview Points**: Quantization cost benefits, cache efficiency gains, batch serving density, dynamic scaling.
- **Technical Intuition**: Moving from FP16 to INT4 AWQ allows running large models on cheaper, memory-bound GPUs (e.g. A10G instead of A100) while maintaining accuracy, directly cutting cloud spend.
- **Production Perspective**: Pair prefix caching with continuous batching to maximize the capacity of existing nodes before scaling cluster sizes.
- **Follow-up**: *How does semantic caching fit in this cost framework?* (Reclaims 10% to 30% of query costs by returning cached text from a vector database for matching inputs, bypassing LLM compute entirely).

---

## 9. System Design & Scenario Questions (5 Questions)

### 56. Design ChatGPT-Scale Inference Service
**Design an inference platform capable of serving millions of concurrent active users with streaming SSE responses, prompt prefix caching, multi-tenant rate limits, and sub-second TTFT SLAs.**
- **Short Answer**: Design a multi-tier serving architecture:
  1. **Gateway Tier**: Handles rate-limiting (token bucket), routes users to regional endpoints, and executes semantic caching.
  2. **Orchestrator Tier**: Inspects prompt headers and routes requests to model workers holding matching RadixAttention prefixes.
  3. **Worker Tier**: Run vLLM or SGLang clusters configured with PagedAttention, continuous batching, and chunked prefills to maintain sub-second TTFT SLAs under heavy traffic.
- **Key Interview Points**: API gateway rate limiting, prefix-routing load balancing, continuous batching, and chunked prefill coordination.
- **Technical Intuition**: Directing prompts with identical system prefixes to the same worker node maximizes Radix tree cache matches. This reduces the prefill phase compute to a memory lookup, keeping p99 TTFT below target thresholds.
- **Production Perspective**: Expose telemetry metrics (Prometheus) on engines to monitor KV Cache usage and trigger horizontal auto-scaling before memory saturation.
- **Follow-up**: *How do you handle regional outages?* (The global load balancer automatically redirects traffic to healthy regions, while gateways apply fallback routing to alternative APIs).

### 57. Design Multi-Model Router Platform
**Design an enterprise AI gateway supporting 20+ fine-tuned models with varying latency SLAs (sub-100ms vs. batch) and cost constraints.**
- **Short Answer**: Build a gateway with a dynamic router and a shared model worker cluster:
  - **Dynamic Router**: Routes incoming prompts to regional nodes based on SLAs and costs.
  - **Shared Cluster**: Uses multi-LoRA serving (e.g., vLLM multi-LoRA) to serve 20+ fine-tuned models on a single GPU cluster. The base model is loaded in VRAM once, and custom LoRA adapters are dynamically loaded into memory on-demand.
- **Key Interview Points**: Multi-LoRA serving architectures, dynamic gateway routing parameters, cost-latency trade-offs.
- **Technical Intuition**: Standard serving requires loading one base model per GPU, which is memory-prohibitive for 20+ models. Multi-LoRA serving processes adapters as small parameter modifications, allowing the GPU to run multiple fine-tuned models concurrently.
- **Production Perspective**: Set routing thresholds: send low-latency queries to fine-tuned edge models, and high-complexity queries to base models with LoRA adapters.
- **Follow-up**: *What is the latency overhead of LoRA adapter swapping?* (Loading adapters from host RAM to GPU memory can introduce minor latency, which is mitigated by caching active adapters in VRAM).

### 58. GPU Memory Bottleneck Priority Checklist
**You have a single 24GB GPU and need to serve an 8B model at 32k context length. Walk through your prioritized sequence of optimizations (GQA, INT4/FP8 quantization, PagedAttention, sliding window) to fit the workload.**
- **Short Answer**: Follow this sequence:
  1. **Weight Quantization**: Quantize weights to 4-bit AWQ. This reduces weight footprint from 16GB to 4GB, freeing up 12GB.
  2. **PagedAttention**: Reclaim fragmented memory blocks, maximizing active cache space.
  3. **Grouped-Query Attention (GQA)**: Ensure GQA is enabled (standard on Llama-3 8B), reducing KV Cache VRAM size by 8x.
  4. **Sliding Window / Context Trimming**: Apply context window limits if sequence length exceeds VRAM capacity under load.
- **Key Interview Points**: Quantization savings, PagedAttention memory mapping, GQA reduction ratios, context limits.
- **Technical Intuition**:
  - Weights: FP16 = 16GB, INT4 = 4GB (saving 12GB).
  - KV Cache at 32k context ($b=1$, Llama-3 8B GQA):
    $$\text{VRAM}_{\text{KV}} = 2 \times 1 \times 32768 \times 32 \times 8 \times 128 \times 2 \approx 4 \text{ GB}$$
  - The model (4GB) and KV cache (4GB) easily fit within the 24GB VRAM, leaving a 16GB buffer for activation spikes and batch concurrency.
- **Production Perspective**: Prioritize optimizations that maintain output quality (quantization, PagedAttention) before applying destructive window limits.
- **Follow-up**: *Can we run this setup without quantization?* (No, because FP16 weights (16GB) and 32k context cache (4GB) would consume 20GB, leaving insufficient memory for activations and batch concurrency).

### 59. Optimizing a 70B Latency SLA Violation
**A 70B model fails its p95 TPOT SLA (requires 50 tokens/sec, achieves 20 tokens/sec). How do you optimize the system (tensor parallelism, AWQ/FP8 quantization, speculative decoding, continuous batching) before downgrading to an 8B model?**
- **Short Answer**: Apply these optimizations systematically:
  1. **Quantization**: Quantize the model to FP8 or INT4 (AWQ). This reduces weight data transfer sizes by 2x-4x, accelerating memory reads.
  2. **Tensor Parallelism (TP)**: Scale the TP width (e.g. from TP=2 to TP=4 or TP=8) across NVLink nodes to distribute compute and memory bandwidth loads.
  3. **Speculative Decoding**: Implement a fast draft model (e.g. Llama-3 8B) or Prompt-Lookup to validate multiple candidate tokens in parallel, bypassing decode bandwidth limits.
  4. **Continuous Batching**: Maximize batch utilization to raise average throughput.
- **Key Interview Points**: Quantization speedups, TP width scaling, speculative decoding acceleration.
- **Technical Intuition**:
  - Quantizing to INT4 AWQ cuts model read traffic from 140GB/token to 35GB/token, directly increasing TPOT throughput.
  - Scaling TP width to 4 nodes reduces weights loaded per node, raising memory-read speeds.
  - Speculative decoding generates multiple tokens per target step, bypassing the sequential generation limit.
- **Production Perspective**: Evaluate the cost-latency tradeoff of scaling TP nodes before deciding to downgrade to a smaller model.
- **Follow-up**: *Does TP scaling always decrease TPOT?* (No, if the network interconnect is slow, the All-Reduce communication overhead can exceed compute savings, slowing down TPOT).

### 60. Fair Inference Benchmarking Methodology
**How do you design a fair, reproducible benchmarking methodology to evaluate two competing inference engines (e.g., vLLM vs. SGLang) on real-world multi-turn conversational traces?**
- **Short Answer**: Design a benchmarking framework that replays real-world traffic profiles:
  - **Dataset**: Use a representative dataset of multi-turn conversational traces (e.g. ShareGPT).
  - **Traffic Profile**: Replay requests using a Poisson process to simulate realistic arrival intervals.
  - **Metrics**: Measure TTFT, TPOT, and overall TPS.
  - **Controls**: Ensure both engines run on identical hardware, using identical weights, quantization parameters, and temperature configurations.
- **Key Interview Points**: Poisson request distribution, representative test traces, latency/throughput metrics, control variables.
- **Technical Intuition**: Replaying requests with realistic prompt-to-response length distributions is critical because prefill-to-decode ratios determine scheduler efficiency. Static mock sequences (e.g. 512 in, 512 out) fail to test prefix cache evictions or continuous batching scheduler performance under dynamic loads.
- **Production Perspective**: Run benchmarks under varying levels of concurrency to identify the saturation point where engines begin preemption or fail to meet SLAs.
- **Follow-up**: *Why is the Poisson process preferred for arrival times?* (Because it models independent, random request arrivals, which matches the behavior of real-world user traffic).
