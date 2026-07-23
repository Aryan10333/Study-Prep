# Parameter-Efficient Fine-Tuning (PEFT)

Parameter-Efficient Fine-Tuning (PEFT) adapts pre-trained LLMs to downstream tasks by training a small subset of parameters (usually $<1\%$) while keeping the base model weights frozen. This minimizes memory overhead, reduces storage requirements, and enables rapid model deployment.

---

## 1. Why PEFT? Memory vs. Compute Savings

During full fine-tuning with the AdamW optimizer under FP16/BF16 precision, the GPU memory overhead scales linearly with the number of model parameters $N$. The VRAM footprint for states is distributed as follows:
- **Frozen/Active Weights**: $2N$ bytes (FP16/BF16)
- **Gradients**: $2N$ bytes
- **AdamW Optimizer States**: $12N$ bytes ($4N$ bytes for FP32 weights copy, $4N$ bytes for first momentum $m_t$, $4N$ bytes for second momentum $v_t$)

Total optimizer and model state VRAM = $16N$ bytes. For a 7B parameter model, this requires $112\text{ GB}$ of VRAM just to store training states, excluding activation memory and key-value cache.

PEFT methods freeze the base model parameters $W_0$, requiring gradients and optimizer states only for a tiny fraction of active parameters. This reduces optimizer memory overhead by over 95%.

---

## 2. Adapters, Prefix Tuning, and Prompt Tuning

Modern PEFT techniques fall into three categories:

1. **Parameter Addition (Adapters)**: Insert small feed-forward blocks inside the transformer layers.
2. **Prefix Tuning**: Prepends virtual key-value prompt tokens directly to the keys ($K$) and values ($V$) inside all self-attention layers.
3. **Prompt Tuning**: Prepends virtual trainable embedding vectors to the input sequence (only at the first embedding layer).

---

## 3. LoRA (Low-Rank Adaptation)

LoRA freezes the pre-trained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ and models its update $\Delta \mathbf{W}$ using two low-rank matrices $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$.

### Mathematical Formulation

The forward pass of a modified linear layer is:
$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$$

Where:
- $\mathbf{x} \in \mathbb{R}^{k}$ is the input vector.
- $\alpha$ is a constant scaling factor. When training starts, $\alpha$ is typically set to twice the rank ($2r$) or equal to the rank ($r$). Adjusting $r$ scales the learning capacity, while $\frac{\alpha}{r}$ scales the contribution of the adapter.
- **Initialization**: Matrix $\mathbf{A}$ is initialized from a random Gaussian distribution $\mathcal{N}(0, \sigma^2)$ so that initial projections are active. Matrix $\mathbf{B}$ is initialized to all zeros. Thus:
  $$\Delta \mathbf{W} = \mathbf{B} \mathbf{A} = \mathbf{0} \cdot \mathbf{A} = \mathbf{0}$$
  At step 0, the adapter contributes nothing, ensuring the model behaves identically to the original pre-trained model.

---

### Step-by-Step Hand Calculation: LoRA Forward Pass

Let's trace a forward pass on a single-token representation through a toy LoRA layer.

#### 1. Setup Parameters
- Input dimension $k = 4$, output dimension $d = 4$, rank $r = 2$.
- Scaling factor $\alpha = 4$. The scaling ratio $\frac{\alpha}{r} = \frac{4}{2} = 2.0$.
- Input vector:
  $$\mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \end{bmatrix}$$
- Frozen base weight matrix $\mathbf{W}_0$ (set to identity for simplicity):
  $$\mathbf{W}_0 = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & 0.0 & 1.0 \end{bmatrix}$$
- Trainable adapter matrix $\mathbf{A}$ (representing projection to rank space):
  $$\mathbf{A} = \begin{bmatrix} 0.5 & 1.0 & -0.5 & 0.0 \\ -1.0 & 0.0 & 0.5 & 1.0 \end{bmatrix}$$
- Trainable adapter matrix $\mathbf{B}$ (representing projection back to output space after some updates):
  $$\mathbf{B} = \begin{bmatrix} 1.0 & 0.0 \\ 0.5 & -0.5 \\ 0.0 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}$$

#### 2. Step 1: Base Output calculation ($\mathbf{W}_0 \mathbf{x}$)
$$\mathbf{h}_{\text{base}} = \mathbf{W}_0 \mathbf{x} = \begin{bmatrix} 1.0(1.0) \\ 1.0(2.0) \\ 1.0(3.0) \\ 1.0(4.0) \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \end{bmatrix}$$

#### 3. Step 2: Projection to Low-Rank Space ($\mathbf{A} \mathbf{x}$)
$$\mathbf{h}_{\text{proj}} = \mathbf{A} \mathbf{x} = \begin{bmatrix} (0.5 \times 1.0) + (1.0 \times 2.0) + (-0.5 \times 3.0) + (0.0 \times 4.0) \\ (-1.0 \times 1.0) + (0.0 \times 2.0) + (0.5 \times 3.0) + (1.0 \times 4.0) \end{bmatrix}$$
$$\mathbf{h}_{\text{proj}} = \begin{bmatrix} 0.5 + 2.0 - 1.5 + 0.0 \\ -1.0 + 0.0 + 1.5 + 4.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 4.5 \end{bmatrix}$$

#### 4. Step 3: Projection back to Output Space ($\mathbf{B} \mathbf{h}_{\text{proj}}$)
$$\mathbf{h}_{\text{adapt\_raw}} = \mathbf{B} \mathbf{h}_{\text{proj}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.5 & -0.5 \\ 0.0 & 1.0 \\ -1.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 4.5 \end{bmatrix} = \begin{bmatrix} 1.0(1.0) + 0.0(4.5) \\ 0.5(1.0) - 0.5(4.5) \\ 0.0(1.0) + 1.0(4.5) \\ -1.0(1.0) + 0.5(4.5) \end{bmatrix} = \begin{bmatrix} 1.0 \\ -1.75 \\ 4.5 \\ 1.25 \end{bmatrix}$$

#### 5. Step 4: Apply Scaling factor and Sum Output
Multiply the raw adapter activation by the scaling ratio $\frac{\alpha}{r} = 2.0$:
$$\mathbf{h}_{\text{adapt}} = 2.0 \times \begin{bmatrix} 1.0 \\ -1.75 \\ 4.5 \\ 1.25 \end{bmatrix} = \begin{bmatrix} 2.0 \\ -3.5 \\ 9.0 \\ 2.5 \end{bmatrix}$$
Add the base output to obtain the final activation:
$$\mathbf{h} = \mathbf{h}_{\text{base}} + \mathbf{h}_{\text{adapt}} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \end{bmatrix} + \begin{bmatrix} 2.0 \\ -3.5 \\ 9.0 \\ 2.5 \end{bmatrix} = \begin{bmatrix} 3.0 \\ -1.5 \\ 12.0 \\ 6.5 \end{bmatrix}$$

---

## 4. DoRA (Weight-Decomposed LoRA)

DoRA analyzes the weight update $\Delta \mathbf{W}$ by breaking weights into magnitude and direction components. In standard full fine-tuning, the updates to weight matrices modify both direction and magnitude. In LoRA, however, direction and magnitude updates are highly coupled, limiting capacity.

### Mathematical Formulation
DoRA decomposes the weight matrix $\mathbf{W}$ as:
$$\mathbf{W} = \mathbf{m} \frac{\mathbf{V}}{\|\mathbf{V}\|_c}$$
Where:
- $\mathbf{m} \in \mathbb{R}^{d}$ is the magnitude vector.
- $\mathbf{V} \in \mathbb{R}^{d \times k}$ is the direction matrix.
- $\|\cdot\|_c$ denotes the vector norm of each column.

To train efficiently, DoRA sets the directional matrix update to run via a LoRA adapter, while the magnitude vector is trained directly:
$$\mathbf{W} = \mathbf{m} \frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{\|\mathbf{W}_0 + \mathbf{B}\mathbf{A}\|_c}$$

This allows DoRA to decouple updates, achieving accuracy comparable to full SFT with zero extra inference latency (as the weights are merged back in production).

---

## 5. QLoRA (Quantized LoRA)

QLoRA improves on LoRA by loading base weights at 4-bit precision, introducing three key optimizations to reduce VRAM requirements without degrading output accuracy.

### 1. 4-bit NormalFloat (NF4)
Standard 4-bit quantization maps float values uniformly. However, model weights are naturally distributed in a zero-centered Gaussian curve. NF4 designs a non-uniform distribution quantile mapping.
For a zero-mean unit-variance normal distribution $\mathcal{N}(0, 1)$, NF4 defines 16 quantization bins $q_i$ such that each bin has an equal probability mass. The quantization function maps weight $w_j$ to:
$$q_j = \text{argmin}_i |w_j - c_i|$$
Where $c_i$ is the centroid of quantile interval $i$.

### 2. Double Quantization (DQ)
Quantization scales (constants used to dequantize block-wise parameters) require memory. QLoRA quantizes the quantization constants themselves. 
For 32-bit floats scales grouped in blocks of 64, Double Quantization compresses these FP32 constants to 8-bit floats with blocks of 256, saving approximately $0.37\text{ GB}$ of VRAM on a 65B model.

### 3. Paged Optimizers
To prevent out-of-memory errors due to activation peaks during gradient calculation over large batches, Paged Optimizers utilize NVIDIA Unified Memory to perform page-swapping between the GPU memory (VRAM) and CPU RAM for optimizer states.

---

## 6. Quantitative Comparison of PEFT Methods

| PEFT Method | Trainable Params | VRAM Footprint | Throughput | Latency Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning**| $100\%$ | $16 \times N$ | High | Zero |
| **LoRA ($r=16$)** | $\sim 0.1\%$ | $\sim 4 \times N$ | Medium-High | Zero (if merged) |
| **DoRA ($r=16$)** | $\sim 0.15\%$ | $\sim 4.1 \times N$ | Medium | Zero (if merged) |
| **QLoRA ($r=16$)**| $\sim 0.1\%$ | $\sim 1.5 \times N$ | Low-Medium | Dequantization latency overhead during training |

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  PEFT reduces the multi-GPU memory footprint required for model adaptation, enabling training on consumer-grade hardware and significantly lowering adapter weight sizes for multi-tenant storage.
- **Why was it introduced?**
  Full parameter training scales quadratically in parameter size, making it cost-prohibitive for organizations to fine-tune unique model parameters per downstream customer or task.
- **What are its limitations?**
  QLoRA introduces a small dequantization latency overhead during training due to converting NF4 to BF16 for activation multiplication. Standard LoRA can suffer from lower task capacity if rank $r$ is set too small.
- **Computational Complexity (Time & Memory)**
  - **LoRA Forward**: $O(L \cdot d \cdot r)$ operations, where $L$ is sequence length.
  - **Memory complexity**: Reduced from $O(N)$ optimizer states to $O(M)$ where $M \ll N$ (typically $M < 0.01 N$).
- **Component Variable Denotation Legend**
  - $N$: Baseline model parameter count.
  - $r$: Rank value of the adapter.
  - $\alpha$: LoRA adapter scaling multiplier.
  - $d, k$: Input and output linear dimensions.
  - $m$: DoRA magnitude vector.
  - $V$: DoRA directional weight matrix.
- **Production Use Cases**
  - Fine-tuning a 70B parameter model on a single 80GB A100 GPU using QLoRA.
  - Quick-swapping customer-specific adapter files ($<100\text{ MB}$) on top of a single base model API.
- **Follow-up questions interviewers ask**
  - *Why do we calculate the forward pass activations in BF16/FP16 if the weights are stored in NF4 format?*
  - *If a task requires deep conceptual reasoning updates, would you prefer increasing rank $r$ or switching from LoRA to DoRA?*
