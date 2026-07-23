# Module 03: Self-Attention & Multi-Head Attention Math

This study guide delivers a mathematical deep dive into the Self-Attention and Multi-Head Attention layers, deriving their computational complexities, detailing causal masking logic, and explaining high-level system optimizations like FlashAttention.

---

## 1. Attention Components: Query, Key, and Value

Self-Attention allows tokens to weigh the relevance of other tokens in a sequence dynamically. 
- Input sequence representation: $\mathbf{X} \in \mathbb{R}^{L \times d}$
- Projections: Each token projects its embedding to three vectors using weight matrices:
  - **Query** ($\mathbf{Q}$): Represents what the token is searching for: $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$ where $\mathbf{W}_Q \in \mathbb{R}^{d \times d}$
  - **Key** ($\mathbf{K}$): Represents what information the token holds: $\mathbf{K} = \mathbf{X} \mathbf{W}_K$ where $\mathbf{W}_K \in \mathbb{R}^{d \times d}$
  - **Value** ($\mathbf{V}$): Contains the actual semantic payload: $\mathbf{V} = \mathbf{X} \mathbf{W}_V$ where $\mathbf{W}_V \in \mathbb{R}^{d \times d}$

---

## 2. Scaled Dot Product Attention

The attention score represents the alignment (dot product similarity) between Query vector $\mathbf{q}_i$ and Key vector $\mathbf{k}_j$, scaled to stabilize variance, softmax-normalized, and multiplied by Value vectors $\mathbf{v}_j$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

- **Variance Stabilization Derivation ($\sqrt{d_k}$)**:
  Let components of Query and Key vectors be independent random variables with zero mean and unit variance: $q_i, k_i \sim \mathcal{N}(0, 1)$. Their dot product is:
  $$\mathbf{q} \cdot \mathbf{k} = \sum_{m=1}^{d_k} q_m k_m$$
  - The mean is $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$.
  - The variance is:
    $$\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{m=1}^{d_k} \text{Var}(q_m k_m) = \sum_{m=1}^{d_k} \mathbb{E}[q_m^2] \mathbb{E}[k_m^2] = \sum_{m=1}^{d_k} (1)(1) = d_k$$
  - As head dimension $d_k$ increases, the variance of the raw dot product scales to $d_k$. Large dot products yield extremely high logit values, pushing the Softmax function into flat saturation regions where gradients become near-zero ($\nabla \text{Softmax} \approx 0$). Dividing by scaling factor $\sqrt{d_k}$ pulls the variance back to $1.0$, stabilizing gradients during backpropagation.

---

## 3. Step-by-Step Hand Calculation (Causal Attention)

- **Scenario**: Input sequence length $L = 3$. Dimension per head $d_k = 2$.
- **Given Matrices (Head 1)**:
  - Query matrix:
    $$\mathbf{Q} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}$$
  - Key matrix:
    $$\mathbf{K} = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix}$$
  - Value matrix:
    $$\mathbf{V} = \begin{bmatrix} 2.0 & -1.0 \\ 0.0 & 3.0 \\ 1.0 & 1.0 \end{bmatrix}$$
- **Calculation**:
  1. Compute raw dot products $\mathbf{A} = \mathbf{Q} \mathbf{K}^T$:
     $$\mathbf{A} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 0.0 \\ 2.0 & 1.0 & 1.0 \end{bmatrix}$$
  2. Scale by factor $\sqrt{d_k} = \sqrt{2} \approx 1.4142$:
     $$\mathbf{S} = \frac{\mathbf{A}}{1.4142} \approx \begin{bmatrix} 0.7071 & 0.0000 & 0.7071 \\ 0.7071 & 0.7071 & 0.0000 \\ 1.4142 & 0.7071 & 0.7071 \end{bmatrix}$$
  3. Apply Causal Masking (tokens cannot attend to future indices; replace values above diagonal with $-\infty$):
     $$\mathbf{M} = \begin{bmatrix} 0.7071 & -\infty & -\infty \\ 0.7071 & 0.7071 & -\infty \\ 1.4142 & 0.7071 & 0.7071 \end{bmatrix}$$
  4. Apply Softmax to each row:
     - **Row 0**: Softmax of $[0.7071, -\infty, -\infty]$:
       $$\text{Weights}_0 = [1.0000, 0.0000, 0.0000]$$
     - **Row 1**: Softmax of $[0.7071, 0.7071, -\infty]$:
       $$\text{Denominator} = e^{0.7071} + e^{0.7071} = 2.0281 + 2.0281 = 4.0562$$
       $$\alpha_{1,0} = \frac{2.0281}{4.0562} = 0.5000, \quad \alpha_{1,1} = \frac{2.0281}{4.0562} = 0.5000$$
       $$\text{Weights}_1 = [0.5000, 0.5000, 0.0000]$$
     - **Row 2**: Softmax of $[1.4142, 0.7071, 0.7071]$:
       $$\text{Denominator} = e^{1.4142} + e^{0.7071} + e^{0.7071} = 4.1133 + 2.0281 + 2.0281 = 8.1695$$
       $$\alpha_{2,0} = \frac{4.1133}{8.1695} = 0.5035, \quad \alpha_{2,1} = \frac{2.0281}{8.1695} = 0.2483, \quad \alpha_{2,2} = \frac{2.0281}{8.1695} = 0.2483$$
       $$\text{Weights}_2 = [0.5035, 0.2483, 0.2483]$$
     - Attention weight matrix $\mathbf{P}$:
       $$\mathbf{P} \approx \begin{bmatrix} 1.0000 & 0.0000 & 0.0000 \\ 0.5000 & 0.5000 & 0.0000 \\ 0.5035 & 0.2483 & 0.2483 \end{bmatrix}$$
  5. Compute final weighted output $\mathbf{O} = \mathbf{P} \mathbf{V}$:
     - **Row 0 Output**:
       $$\mathbf{O}_0 = 1.0 \times [2.0, -1.0] = [2.0, -1.0]$$
     - **Row 1 Output**:
       $$\mathbf{O}_1 = 0.5 \times [2.0, -1.0] + 0.5 \times [0.0, 3.0] = [1.0, -0.5] + [0.0, 1.5] = [1.0, 1.0]$$
     - **Row 2 Output**:
       $$\mathbf{O}_2 = 0.5035 \times [2.0, -1.0] + 0.2483 \times [0.0, 3.0] + 0.2483 \times [1.0, 1.0]$$
       $$\mathbf{O}_2 = [1.0070, -0.5035] + [0.0, 0.7449] + [0.2483, 0.2483] = [1.2553, 0.4897]$$
     - Final Output Matrix $\mathbf{O}$:
       $$\mathbf{O} \approx \begin{bmatrix} 2.0000 & -1.0000 \\ 1.0000 & 1.0000 \\ 1.2553 & 0.4897 \end{bmatrix}$$
     This walkthrough matches output code simulations exactly.

---

## 4. Multi-Head Attention & Head Concatenation

Instead of performing single-attention over hidden states of dimension $d$, Multi-Head Attention projects variables into $h$ independent, parallel heads of size $d_k = d / h$.
1. **Parallel Execution**: Compute attention output for each head:
   $$\text{head}_i = \text{Attention}(\mathbf{X} \mathbf{W}_{Q_i}, \mathbf{X} \mathbf{W}_{K_i}, \mathbf{X} \mathbf{W}_{V_i})$$
2. **Concatenation**: Combine outputs from all heads:
   $$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)$$
3. **Projection**: Project the concatenation back to hidden dimension $d$ using matrix $\mathbf{W}_O$:
   $$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$

---

## 5. Computational Complexity & Memory Bottlenecks

### The Quadratic Bottleneck $O(L^2)$
Computing attention scores requires multiplying $Q \in \mathbb{R}^{L \times d_k}$ and $K^T \in \mathbb{R}^{d_k \times L}$, resulting in a raw similarity matrix of size $L \times L$.
- **Time Complexity**: $O(L^2 \cdot d)$ operations.
- **Memory Complexity**: $O(L^2)$ to store the intermediate attention weights. As sequence length $L$ scales, memory requirements expand quadratically, creating a VRAM ceiling.

### FlashAttention: IO-Aware Acceleration (High-Level)
- **The Problem**: Standard PyTorch attention writes the intermediate $L \times L$ attention matrix to slow High Bandwidth Memory (HBM), causing memory transfer latency bottlenecks.
- **FlashAttention Solution**: Uses **SRAM Tiling**. It loads blocks of $Q, K, V$ into fast, local GPU SRAM caches, calculates softmax incrementally, and updates outputs without ever writing the massive $L \times L$ matrix back to HBM, speeding up execution by $2\times - 4\times$.

---

## 6. Interview Questions & Production Trade-offs

### What problem does this solve?
Enables sequence tokens to attend to other tokens dynamically across arbitrary distances, resolving sequence information loss in recurrent architectures.

### Why was it introduced?
Variance scaling factor $\sqrt{d_k}$ prevents logits from growing too large, ensuring stable gradient backpropagation when sequence lengths scale.

### What are its limitations?
The quadratic time and memory cost ($O(L^2)$) restricts context length processing on standard hardware without compression or tiling.

### Computational Complexity (Time & Memory)
- **MHA Computation**: $O(L^2 \cdot d + L \cdot d^2)$ operations.
- **Attention VRAM Space**: $O(L^2 \cdot h + L \cdot d)$ bytes.

### Component Variable Denotation Legend
- $L$: Sequence length.
- $d$: Hidden state model dimension.
- $h$: Attention heads count.
- $d_k$: Attention head vector size ($d/h$).

### Production Use Cases:
- Long-context LLM inference servers implementing FlashAttention to reduce latency and VRAM footprints.
- Retrieval models querying document matches using dot-product semantic projections.

### Follow-up Questions Interviewers Ask:
1. *Why does attention compute complexity scale as $O(L^2 \cdot d)$?*
   - **Answer**: Calculating the attention scores requires multiplying Query matrix $\mathbf{Q} \in \mathbb{R}^{L \times d_k}$ by transposed Key matrix $\mathbf{K}^T \in \mathbb{R}^{d_k \times L}$ for all $h$ heads. The multiplication of two matrices of size $[L, d_k]$ and $[d_k, L]$ requires $L \cdot L \cdot d_k$ scalar operations. Doing this across all $h$ heads yields $h \cdot L^2 \cdot d_k = L^2 \cdot d$ operations, introducing the quadratic scaling factor.
2. *How does FlashAttention calculate Softmax incrementally during block updates?*
   - **Answer**: Softmax requires a global normalization denominator $\sum e^{x_i}$. To compute this incrementally over blocks without global access, FlashAttention tracks local normalization stats (maximum block logit value $m$ and local denominator sum $d$) for each block. On loading new blocks, it scales the old local outputs and updates the denominator using the relation $e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$, ensuring exact mathematical equivalence to standard softmax while processing data in local SRAM tiles.
3. *Why does a higher vocabulary size or sequence length lead to softmax saturation?*
   - **Answer**: If vectors are unscaled, the variance of their dot products matches hidden size $d$. With large dimensions (e.g. $d=4096$), dot products grow large, causing softmax inputs to have wide separations. The derivative of Softmax with respect to inputs $x_i$ is $s_i(\delta_{ij} - s_j)$. When one value dominates, its softmax probability approaches $1.0$ and others approach $0.0$, pushing the derivatives to zero and causing vanishing gradients.
