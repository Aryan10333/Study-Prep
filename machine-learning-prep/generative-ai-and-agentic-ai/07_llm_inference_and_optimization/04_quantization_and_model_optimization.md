# 04. Model Quantization & Compression: Precision and Conversion Mechanics

LLM serving requires significant VRAM memory. To reduce costs and allow large models to run on standard hardware, we utilize **Model Quantization** to reduce the bit-width of weights and activations (e.g. from 16-bit floating point down to 8-bit or 4-bit integers), dramatically lowering VRAM usage and boosting memory-bound decoding speeds.

---

## 1. Floating Point Formats & Data Precision

Modern GPUs process numbers in several precision formats:
- **FP32 (Single Precision)**: 32 bits. 1 sign bit, 8 exponent bits (range), 23 fraction/mantissa bits (precision). Standard for training base calculations, but too memory-intensive for online serving.
- **FP16 (Half Precision)**: 16 bits. 1 sign, 5 exponent, 10 fraction. Standard model weights format.
- **BF16 (Brain Floating Point)**: 16 bits. 1 sign, 8 exponent, 7 fraction. Matches FP32's dynamic range but with reduced numerical precision. Prevents underflow/overflow during fine-tuning.
- **FP8 (8-Bit Float)**: Used in NVIDIA Hopper Tensor Cores:
  - **E4M3**: 1 sign, 4 exponent, 3 mantissa. Maximizes precision; preferred for activation layers during forward inference steps.
  - **E5M2**: 1 sign, 5 exponent, 2 mantissa. Matches FP16's dynamic range; preferred for weight storage and gradients.

---

## 2. Quantization Mechanics: Symmetric vs. Asymmetric

Quantization maps a continuous range of real-world values $r \in [\min(X), \max(X)]$ to a discrete integer grid $q \in [q_{\text{min}}, q_{\text{max}}]$ (such as $[-128, 127]$ for signed INT8).

### Symmetric Quantization
Maps the real range symmetrically around zero ($z = 0$).
- **Scale Factor ($s$)**:
  $$s = \frac{\max(|X|)}{q_{\text{max}}}$$
- **Quantization**:
  $$q = \text{clip}\left(\text{round}\left(\frac{X}{s}\right), q_{\text{min}}, q_{\text{max}}\right)$$
- **De-quantization**:
  $$r \approx s \cdot q$$

### Asymmetric Quantization
Maps the real minimum and maximum values exactly to the integer minimum and maximum boundaries, offset by a **Zero-Point** ($z$).
- **Scale Factor ($s$)**:
  $$s = \frac{\max(X) - \min(X)}{q_{\text{max}} - q_{\text{min}}}$$
- **Zero-Point ($z$)**:
  $$z = \text{round}\left(\frac{-\min(X)}{s}\right) + q_{\text{min}}$$
- **Quantization**:
  $$q = \text{clip}\left(\text{round}\left(\frac{X}{s}\right) + z, q_{\text{min}}, q_{\text{max}}\right)$$
- **De-quantization**:
  $$r \approx s \cdot (q - z)$$

---

### Step-by-Step Hand Calculation: Asymmetric INT8 Quantization

Let our real-valued activation vector be:

$$X = [-1.5, 0.5, 2.0]$$

We want to quantize $X$ to signed INT8, where $q_{\text{min}} = -128$ and $q_{\text{max}} = 127$.

#### Step 1: Identify bounds
- $\min(X) = -1.5$
- $\max(X) = 2.0$

#### Step 2: Compute Scale ($s$)
$$s = \frac{\max(X) - \min(X)}{q_{\text{max}} - q_{\text{min}}} = \frac{2.0 - (-1.5)}{127 - (-128)} = \frac{3.5}{255} \approx 0.0137255$$

#### Step 3: Compute Zero-Point ($z$)
$$z = \text{round}\left(\frac{-(-1.5)}{0.0137255}\right) + (-128) = \text{round}(109.28) - 128 = 109 - 128 = -19$$

#### Step 4: Quantize values $q = \text{round}(X / s) + z$

1. **For $X_1 = -1.5$**:
   $$q_1 = \text{round}\left(\frac{-1.5}{0.0137255}\right) - 19 = \text{round}(-109.28) - 19 = -109 - 19 = -128$$
2. **For $X_2 = 0.5$**:
   $$q_2 = \text{round}\left(\frac{0.5}{0.0137255}\right) - 19 = \text{round}(36.43) - 19 = 36 - 19 = 17$$
3. **For $X_3 = 2.0$**:
   $$q_3 = \text{round}\left(\frac{2.0}{0.0137255}\right) - 19 = \text{round}(145.71) - 19 = 146 - 19 = 127$$

Quantized vector: $q = [-128, 17, 127]$

#### Step 5: De-quantize values $r \approx s \cdot (q - z)$
1. $r_1 \approx 0.0137255 \times (-128 - (-19)) = 0.0137255 \times (-109) = -1.496 \approx -1.5$
2. $r_2 \approx 0.0137255 \times (17 - (-19)) = 0.0137255 \times 36 = 0.494 \approx 0.5$
3. $r_3 \approx 0.0137255 \times (127 - (-19)) = 0.0137255 \times 146 = 2.004 \approx 2.0$

---

## 3. Post-Training Quantization (PTQ) Paradigms

PTQ converts pre-trained models without retraining. We categorize PTQ methods by whether they target weights only or weight-activations.

### Weight-Only Quantization
Quantizes weights to 4-bit, keeping activations in 16-bit. Weights are de-quantized to FP16 in SRAM before execution.
- **GPTQ**: Uses a second-order optimization method. It updates remaining weights to compensate for the error introduced by quantizing a specific column weight, using the inverse Hessian matrix.
- **AWQ (Activation-aware Weight Quantization)**: Observes that not all weights are equally important. Only 1% of weights (those corresponding to features with high activation magnitude) are salient. AWQ protects these channels from quantization error by applying a scaling factor rather than running expensive optimization passes.

### Weight-Activation Quantization
Quantizes both weights and activations to INT8, allowing the model to utilize integer compute engines (INT8 Tensor Cores) directly.
- **SmoothQuant**: Emergent outlier features in activations (spikes up to 100x larger than median values) make activations difficult to quantize to INT8. SmoothQuant mathematically migrates this quantization difficulty from the activations ($X$) to the weights ($W$) by applying a scaling factor $s$:
  $$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

### PTQ Algorithm Comparison Matrix

| Algorithm       | Quantization Scope         | Pros                                                                                                                             | Cons                                                                                   | Production Choice                                                |
| -----------------| ----------------------------| ----------------------------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------| ------------------------------------------------------------------|
| **GPTQ**        | Weight-only (INT4/INT3)    | • High quality compression at extreme bit-widths.                                                                                | • High calibration computation overhead (requires inverse Hessian solvers).            | Static weight-only deployment for memory reduction.              |
| **AWQ**         | Weight-only (INT4)         | • Preserves outlier weights dynamically; yields higher accuracy than GPTQ.<br>• Does not require complex Hessian approximations. | • Relies on specialized de-quantization kernels in SRAM at execution time.             | Modern standard for INT4 serving on vLLM and SGLang.             |
| **SmoothQuant** | Weight + Activation (W8A8) | • Enables native INT8 GEMM on Tensor Cores, yielding compute speedups.                                                           | • Higher perplexity degradation on very large models with massive activation outliers. | Enterprise serving where both latency and throughput must scale. |

---

## 4. K-Quantization & GGUF (llama.cpp)

For edge servers and CPU execution, `llama.cpp` uses **Block-wise Quantization** formats (GGUF):
- Instead of scaling the entire layer, GGUF groups parameters into small blocks (e.g. 32 or 256 weights).
- Each block has a local scale factor and zero-point. This local grouping prevents outliers from degrading accuracy, allowing models like Llama-3 8B to run in 4-bit on standard laptops with minimal Perplexity loss.

---

## 5. Accuracy vs. VRAM Trade-Off Matrix

The table below summarizes the trade-offs of quantizing a Llama-3 8B model:

| Precision Format | Weights VRAM | Perplexity Degradation | Optimal Hardware |
|---|---|---|---|
| **BF16 / FP16** | $16 \text{ GB}$ | Baseline ($0.0$ change) | All GPUs |
| **FP8 (E4M3/E5M2)** | $8 \text{ GB}$ | Negligible ($< 0.02$) | H100, B200 |
| **INT8 (SmoothQuant)** | $8 \text{ GB}$ | Low ($< 0.05$) | A100, T4, L4 |
| **INT4 (AWQ)** | $4 \text{ GB}$ | Minor ($< 0.15$) | Latency-bound GPUs |
| **INT4 (GPTQ)** | $4 \text{ GB}$ | Minor ($< 0.18$) | Batch-bound GPUs |

---

### Interview Questions & Production Trade-offs

#### What problem does this solve?
It reduces the precision bit-width of model parameters and activations (from FP16 to FP8/INT8/INT4), shrinking the memory size of weights and KV Cache tensors to save VRAM and accelerate memory-bound decode latency.

#### Why was it introduced?
Deploying massive models in production requires multiple enterprise GPUs (e.g. A100s/H100s) purely to accommodate weight parameters, leading to high capital expenditures. Quantizing parameters down to 4-bit cuts server costs by 4x, allowing 70B models to run on single nodes and 8B models on consumer laptops.

#### What are its limitations?
- **Perplexity Loss**: Lower precision introduces mathematical rounding noise, which can degrade the model's accuracy on reasoning, mathematics, or code generation tasks.
- **Dequantization Latency**: On memory-bandwidth-bound GPUs, dynamically dequantizing INT4 weights back to FP16 in SRAM at runtime can add CPU/kernel latency, reducing theoretical throughput gains.

#### Computational Complexity (Time & Memory)
- **Quantized Matrix Multiplication**:
  - *Time Complexity*: $O(M \cdot N \cdot K)$ operations per layer, but executing integer matrix multiplies (e.g. W8A8 via SmoothQuant) runs faster on dedicated GPU Tensor Cores.
  - *Memory Complexity*: Weight parameter VRAM footprints scale down linearly by the compression factor: $\text{VRAM}_{\text{quant}} = P \cdot \frac{\text{bits}}{8}$ Bytes.

#### Component Variable Denotation Legend
- $P$: Model parameter count (number of weights).
- $\text{bits}$: Precision bit-width ($16$ for FP16, $8$ for FP8/INT8, $4$ for INT4).
- $s$: Scale factor (real-valued multiplier).
- $z$: Zero-point integer offset (for asymmetric mapping).
- $q$: Quantized integer representation.
- $r$: Reconstructed real float value.
- $q_{\text{min}}, q_{\text{max}}$: Minimum and maximum bounds of the integer grid.

#### Production Use Cases
- **Local Edge Hosting (llama.cpp / GGUF)**: Running LLMs on consumer devices (macBooks, laptops) using 4-bit block-wise quantization.
- **High-Throughput Serving (vLLM / AWQ)**: Scaling batch sizes on server clusters by deploying AWQ INT4 weights to save KV cache space.

#### Follow-up questions interviewers ask
1. *Why are activations much harder to quantize to INT8 than weights in LLMs?*
   - **Answer**: Weights have relatively static, uniform distributions that easily map to integer ranges. Activations, however, suffer from "emergent outliers" (certain channels spike to values 100x larger than median values). Quantizing these activations directly squashes normal features, degrading performance. SmoothQuant resolves this by mathematically migrating the quantization difficulty from activations to weights.
2. *Contrast Post-Training Quantization (PTQ) with Quantization-Aware Training (QAT).*
   - **Answer**: PTQ quantizes a pre-trained model directly without retraining, using a small calibration dataset to estimate scale and zero-point parameters. It is fast and cheap but can cause perplexity drops on lower bit-widths ($<4$ bits). QAT simulates quantization rounding errors during model training using straight-through estimators. It is compute-expensive (requires training loops) but preserves accuracy.
