# Module 09: Scaling Laws & Mixture of Experts (MoE)

This study guide explains the engineering mathematics of LLM scaling properties and sparse architectures, detailing Kaplan vs. Chinchilla laws, GPU-day training estimates, and Mixture of Experts routing mechanisms.

---

## 1. Power-Law Scaling: Kaplan vs. Chinchilla

Scaling laws state that cross-entropy loss $L$ decreases as a power-law relative to parameter count $N$, dataset size $D$, and compute budget $C$:

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

### 1. Kaplan et al. Scaling Law (OpenAI)
- **Finding**: Parameter count is the dominant factor in model performance. If compute budget $C$ is scaled, parameter count $N$ should be scaled rapidly ($N \propto C^{0.73}$), while dataset size $D$ is scaled slowly ($D \propto C^{0.27}$).
- *Consequence*: Led to models trained on relatively small token counts (e.g. GPT-3 with 175B parameters trained on only 300B tokens).

### 2. Chinchilla Scaling Law (Hoffmann et al., DeepMind)
- **Finding**: Kaplan's team underestimated the value of training data size. For compute-optimal training, parameter size $N$ and token count $D$ should scale in equal proportions ($N \propto C^{0.5}, D \propto C^{0.5}$).
- **Compute-Optimal Ratio**:
  $$D \approx 20 \cdot N$$
  For every 1B parameters, a model should be trained on 20B tokens.
- **Total Compute Cost Formula**:
  $$C \approx 6ND \text{ FLOPS}$$
  (The coefficient $6$ represents the operations required for forward (2 ops) and backward (4 ops) passes per parameter per token).

---

## 2. Step-by-Step Hand Calculation (Training Budget & GPU-Days)

- **Scenario**: We want to train a model with $N = 7\text{B}$ parameters ($7 \times 10^9$) compute-optimally.
- **GPU Specifications**: H100 GPU with peak FP16 tensor performance of $9.89 \times 10^{14}$ FLOPS ($989$ TFLOPS). Real-world Model Flop Utilization (MFU) is $40\%$, making effective throughput:
  $$\text{Effective FLOPS} = 9.89 \times 10^{14} \times 0.40 \approx 3.956 \times 10^{14} \text{ FLOPS}$$
- **Calculation**:
  1. Compute the optimal dataset size $D$ according to Chinchilla scaling:
     $$D = 20 \times N = 20 \times (7 \times 10^9) = 1.4 \times 10^{11} \text{ tokens (140B tokens)}$$
  2. Compute total training compute budget $C$:
     $$C \approx 6ND = 6 \times (7 \times 10^9) \times (1.4 \times 10^{11}) = 5.88 \times 10^{21} \text{ FLOPS}$$
  3. Calculate training time in seconds on a single GPU:
     $$\text{Time}_{\text{sec}} = \frac{C}{\text{Effective FLOPS}} = \frac{5.88 \times 10^{21}}{3.956 \times 10^{14}} \approx 1.4863 \times 10^7 \text{ seconds}$$
  4. Convert to days:
     $$\text{Time}_{\text{days}} = \frac{1.4863 \times 10^7}{86,400} \approx 172.0 \text{ GPU-days}$$
- **Result**: Training a 7B model compute-optimally requires $\approx 172.0$ H100 GPU-days.

---

## 3. Mixture of Experts (MoE) Architecture

To scale model parameters without multiplying compute costs, Mixture of Experts (MoE) replaces dense FFN layers with multiple parallel experts, routing each token to a subset of them.

```text
Input Token ──► [Router / Gating] ──► Select Top-2 Experts ──► [Expert 1] \
                                                           ──► [Expert 4] ──► Concat Output
```

- **Router (Gating Network)**: Computes a probability distribution over $E$ experts:
  $$\mathbf{g} = \text{Softmax}(\mathbf{x} \mathbf{W}_g)$$
- **Top-2 Routing**: Sets gating coefficients to $0$ for all but the top 2 experts, saving computation.
- **MoE Challenges**:
  1. **Expert Capacity Limits**: The maximum number of tokens an expert can process in a batch. If exceeded, extra tokens are dropped, degrading quality.
  2. **Expert Load Imbalance**: The router can over-allocate to a few favorite experts. To prevent this, models train with an auxiliary **balancing loss**:
     $$\mathcal{L}_{\text{balance}} = E \sum_{i=1}^{E} f_i \cdot P_i$$
     where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the mean routing probability allocated to expert $i$.

---

## 4. Comparison of Architectures

| Metric | Dense Model (BERT/GPT) | Sparse MoE Model (Mixtral) |
|---|---|---|
| **Total Parameters** | $N$ (all active) | $N_{\text{total}}$ (highly scaled) |
| **Active Parameters** | $N$ (all active) | $N_{\text{active}} \ll N_{\text{total}}$ |
| **Latency Profile** | Constant per token | Subject to routing and inter-GPU communication |
| **VRAM Requirement** | Fits $N$ parameters | Fits $N_{\text{total}}$ parameters (high VRAM ceiling) |

---

## 5. Interview Questions & Production Trade-offs

### What problem does this solve?
Allows scaling capacity up to trillions of parameters without violating execution latency budgets.

### Why was it introduced?
Chinchilla scaling laws demonstrated that prior LLMs were severely under-trained on tokens, shifting priority to dataset expansion.

### What are its limitations?
MoE models require massive VRAM capacities to hold expert weights, creating hardware distribution constraints.

### Computational Complexity (Time & Memory)
- **Dense Layer training cost**: $6 \cdot N \cdot D$ FLOPS.
- **MoE routing decision overhead**: $O(E \cdot d)$ operations.

### Component Variable Denotation Legend
- $N$: Model parameter count.
- $D$: Training dataset token count.
- $E$: Number of parallel MoE experts.
- $C$: Total FLOPS compute budget.

### Production Use Cases:
- Large-scale pre-training pipelines allocating GPU resources based on Chinchilla compute estimates.
- Efficient text generation endpoints running Mixtral models to reduce token serving costs.

### Follow-up Questions Interviewers Ask:
1. *Why does Llama-3-8B violate Chinchilla scaling laws by training on 15T tokens ($D \approx 1875 N$ instead of $20 N$)?*
   - **Answer**: Chinchilla scaling optimizes for the lowest *training* compute budget. However, in production, a model is queried billions of times. Over-training a smaller, cheaper-to-serve model (8B parameters) on massive token counts yields a model that has high accuracy but is much cheaper to run at inference, trade-off wise favoring inference-efficiency over training-efficiency.
2. *Why do MoE models require expert capacity limits during distributed training?*
   - **Answer**: On distributed hardware, experts are placed on different GPUs. If expert routing is unbalanced, one expert GPU receives all tokens, causing a computation bottleneck and VRAM overflow, while other expert GPUs sit idle. Enforcing a capacity limit drops excess tokens, keeping batch sizes and step times constant.
3. *Explain how MoE expert caching mitigates weight loading bottlenecks during decoding.*
   - **Answer**: Because MoE routing changes token-by-token, consecutive tokens in a sequence target different experts. If expert weights are loaded dynamically, memory transfer latency stalls the system. Expert caching locks all expert parameters in GPU VRAM, bypassing transfer overhead.
