# Module 02: Transformer Architecture: Encoder, Decoder & Sub-layers

This study guide covers the structural design of the Transformer architecture, its primary operational variants (Encoder, Decoder, and Encoder-Decoder), and the evolution of its sub-layers (RMSNorm, FFN Gated Units, and Residuals).

---

## 1. Architectural Variations: Encoder, Decoder, and Encoder-Decoder

```text
       [Encoder-Decoder (T5)]
       /                   \
[Encoder-only (BERT)]    [Decoder-only (GPT/Llama)]
```

- **Encoder-only (BERT)**: Uses bidirectional self-attention. Every token can attend to every other token in the sequence. Ideal for classification, extractive QA, and sequence labeling.
- **Decoder-only (GPT, Llama, Mistral)**: Uses causal masking self-attention. Tokens can only attend to previous tokens and themselves, ensuring that text generation is autoregressive.
- **Encoder-Decoder (T5, BART)**: Features a bidirectional encoder linked via cross-attention to a causal decoder. Standard for sequence-to-sequence translation and summarization tasks.

---

## 2. Normalization Architectures: LayerNorm vs. RMSNorm

Layer normalization stabilizes training dynamics by scaling activation values across feature dimensions.

### LayerNorm (Post-LN vs. Pre-LN)
- **Post-LN (Original Transformer)**: Normalization is applied *after* residual addition:
  $$\mathbf{x}_{l} = \text{LN}(\mathbf{x}_{l-1} + \text{SubLayer}(\mathbf{x}_{l-1}))$$
  - *Limitation*: Gradients near the output layer are large, while gradients in early layers decay. This requires a strict learning rate warmup schedule to prevent optimization divergence.
- **Pre-LN (Modern Standard)**: Normalization is applied to sub-layer inputs *before* execution, adding a clean highway backpropagation path:
  $$\mathbf{x}_{l} = \mathbf{x}_{l-1} + \text{SubLayer}(\text{LN}(\mathbf{x}_{l-1}))$$
  - *Advantage*: Enables stable training from epoch 1, resolving early gradient vanishing issues.

### RMSNorm (Root Mean Square Normalization)
RMSNorm simplifies LayerNorm by omitting mean subtraction, showing that variance scaling alone provides sufficient regularizing properties:
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \quad \text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
- **Why it is used**: Omiting mean calculation simplifies computation, improving training hardware throughput.

#### Step-by-Step Hand Calculation (RMSNorm)
- **Scenario**: Input vector $\mathbf{x} = [2.0, -1.0, 3.0]$. Hidden dimension $d = 3$. Let scale weights $\boldsymbol{\gamma} = [1.0, 1.0, 1.0]$ and bias stabilizer $\epsilon = 1\text{e-}5$.
- **Calculation**:
  1. Calculate Mean Square:
     $$\text{MS} = \frac{1}{3} \left( 2.0^2 + (-1.0)^2 + 3.0^2 \right) = \frac{1}{3} (4.0 + 1.0 + 9.0) = \frac{14}{3} \approx 4.6667$$
  2. Compute Root Mean Square (RMS):
     $$\text{RMS}(\mathbf{x}) = \sqrt{4.6667 + 1\text{e-}5} \approx \sqrt{4.66671} \approx 2.1603$$
  3. Normalize $\mathbf{x}$:
     $$\text{Normalized } \mathbf{x} = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} = \left[ \frac{2.0}{2.1603}, \frac{-1.0}{2.1603}, \frac{3.0}{2.1603} \right] \approx [0.9258, -0.4629, 1.3887]$$
  4. Scale with $\boldsymbol{\gamma}$:
     $$\text{Output} = [0.9258, -0.4629, 1.3887] \odot [1.0, 1.0, 1.0] = [0.9258, -0.4629, 1.3887]$$

---

## 3. Feed Forward Networks (FFN) & Activation Evolution

The FFN layer processes the output of the Self-Attention block token-by-token.

### 1. Classical FFN (ReLU)
$$\text{FFN}_{\text{ReLU}}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
- *Limitation*: ReLU outputs exactly zero for negative inputs (the "dead ReLU" problem), stopping gradient updates during backpropagation.

### 2. GELU (Gaussian Error Linear Unit)
GELU weights inputs by their cumulative probability under a normal distribution, creating a smooth, non-zero gradient baseline for negative values:
$$\text{GELU}(\mathbf{x}) = \mathbf{x} \Phi(\mathbf{x}) \approx 0.5 \mathbf{x} \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} (\mathbf{x} + 0.044715 \mathbf{x}^3) \right) \right)$$

### 3. SwiGLU (Swish-Gated Linear Unit)
SwiGLU replaces the standard feed-forward layer with a gated representation, where one linear projection acts as a dynamic gate over a second projection:
$$\text{SwiGLU}(\mathbf{x}) = \left( \text{Swish}(\mathbf{x} \mathbf{W}_1) \odot \mathbf{x} \mathbf{W}_2 \right) \mathbf{W}_3$$
$$\text{Swish}(a) = a \cdot \sigma(\beta a) = \frac{a}{1 + e^{-\beta a}} \quad (\text{with } \beta = 1)$$
- **Why Llama and Mistral use SwiGLU**: The multiplicative gate structure allows the layer to modulate inputs dynamically, yielding improved parameter efficiency and faster learning convergence relative to ReLU/GELU.

#### Step-by-Step Hand Calculation (SwiGLU Gating)
- **Scenario**: Input $\mathbf{x} = [0.5, 0.5]$.
- **Given Weights** (projecting to intermediate state size of 1):
  - $\mathbf{W}_1^T = [2.0, -2.0]$
  - $\mathbf{W}_2^T = [-1.0, 3.0]$
  - $\mathbf{W}_3 = [1.5]$ (output projection scale)
- **Calculation**:
  1. Compute linear projection inputs for gate and value:
     $$a_1 = \mathbf{x} \mathbf{W}_1^T = (0.5 \times 2.0) + (0.5 \times -2.0) = 1.0 - 1.0 = 0.0$$
     $$a_2 = \mathbf{x} \mathbf{W}_2^T = (0.5 \times -1.0) + (0.5 \times 3.0) = -0.5 + 1.5 = 1.0$$
  2. Compute Swish activation on the gate representation $a_1$:
     $$\text{Swish}(a_1) = \text{Swish}(0.0) = \frac{0.0}{1 + e^{-0.0}} = 0.0$$
  3. Perform Hadamard product gating step:
     $$\text{Gated Value} = \text{Swish}(a_1) \odot a_2 = 0.0 \times 1.0 = 0.0$$
  4. Project to output layer:
     $$\text{Output} = 0.0 \times 1.5 = 0.0$$

---

## 4. Execution Modes: Training vs. Inference

| Attribute | Training Mode | Inference Mode |
|---|---|---|
| **Data Processing** | Parallelized (processes all tokens at once). | Sequential (autoregressive, one token at a time). |
| **Masking Style** | Causal mask matrix block-applied. | Single token query projection. |
| **KV Cache** | Not used (all keys/values computed in parallel). | Essential (caches keys/values to avoid redundant recomputation). |
| **Compute Profile** | Compute-bound (high matrix multiply utilization). | Memory-bandwidth bound (dominated by loading weights). |

---

## 5. Interview Questions & Production Trade-offs

### What problem does this solve?
Sequential recurrent models cannot parallelize updates. The Transformer Encoder/Decoder uses residual feed-forward stacks to enable parallel token updates across massive datasets.

### Why was it introduced?
Pre-LN and RMSNorm structures were introduced to stabilize gradient flows in deep networks, allowing scaling beyond 100+ layers without optimization crashes.

### What are its limitations?
Causal autoregressive decoding during inference is highly memory-bandwidth bound, as model weights must be loaded from memory for every single generated token.

### Computational Complexity (Time & Memory)
- **Transformer Layer forward pass**: $O(L \cdot d^2)$ operations (excluding self-attention).
- **FFN Activation Memory VRAM**: $O(L \cdot d_{\text{ffn}})$ activation tensors stored during training.

### Component Variable Denotation Legend
- $L$: Sequence length.
- $d$: Hidden model dimension.
- $d_{\text{ffn}}$: Intermediate feed-forward layer expansion dimension (typically $\frac{8}{3}d$ in SwiGLU).

### Production Use Cases:
- Large-scale pre-training pipelines utilizing Pre-LN or RMSNorm to prevent gradient explosions.
- Low-latency inference serving systems using SwiGLU layers to achieve high token-generation accuracy with smaller parameter footprints.

### Follow-up Questions Interviewers Ask:
1. *Why does Pre-LN scale better than Post-LN in deep Transformers?*
   - **Answer**: In Post-LN, the residual update is added before normalization, scaling activation magnitudes as depth increases. Early layer updates are scaled down, leading to gradient vanishing. In Pre-LN, the inputs to sub-layers are normalized first, leaving the residual connection as a clean linear identity mapping $\mathbf{x}_L = \mathbf{x}_0 + \sum \text{SubLayer}(\mathbf{x}_l)$ which preserves gradient flows during backpropagation.
2. *Why does SwiGLU use three weight matrices ($\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$) instead of the standard two found in GELU/ReLU FFNs?*
   - **Answer**: SwiGLU is a gated architecture. It splits the input vector $\mathbf{x}$ into two parallel channels: one channel calculates the gating representation via $\text{Swish}(\mathbf{x} \mathbf{W}_1)$, and the second channel calculates the value projection via $\mathbf{x} \mathbf{W}_2$. These channels are multiplied element-wise and projected back to the hidden space via $\mathbf{W}_3$, requiring three parameter matrices to implement the gate.
3. *Why does Pre-LN require an extra normalization step before output projections?*
   - **Answer**: Since sub-layers add directly to the identity residual highway without intermediate normalization, the final output layer activation magnitude is the unnormalized summation of all residual additions. To prevent gradient explosion at the final classification head, an additional layer normalization block must be executed before final logit projection.
