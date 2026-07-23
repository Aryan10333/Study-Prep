# Model Merging & Adapters

In production environments, deploying distinct full-sized models for each task is computationally inefficient. Model Merging and Adapter Composition enable the consolidation of multiple capabilities into a single model, bypassing multi-GPU storage and routing limitations.

---

## 1. Multi-Task Adapters & Adapter Fusion

Rather than merging weight parameters directly, **Adapter Fusion** keeps multiple task-specific adapters frozen and inserts a trainable gating network. The gating network calculates dynamic attention scores over active adapters at runtime, routing token activations to the most relevant adapter blocks:
$$\mathbf{h}_l = \sum_{i=1}^{M} g_i(\mathbf{x}) \cdot \text{Adapter}_i(\mathbf{x})$$

Where $g_i(\mathbf{x})$ represents the dynamic softmax gating weight for adapter $i$ given token representation $\mathbf{x}$.

---

## 2. Model Merging Algorithms

Model merging combines the parameter weights of multiple models fine-tuned from the same base model, without requiring further gradient updates.

### 1. SLERP (Spherical Linear Interpolation)
Standard linear interpolation (LERP) over high-dimensional spaces causes a collapse in variance, leading to degraded model outputs. SLERP interpolates weights spherically along the geometric surface:
$$\text{SLERP}(\mathbf{W}_1, \mathbf{W}_2; t) = \frac{\sin((1-t)\theta)}{\sin\theta}\mathbf{W}_1 + \frac{\sin(t\theta)}{\sin\theta}\mathbf{W}_2$$
Where $\theta = \arccos(\frac{\mathbf{W}_1 \cdot \mathbf{W}_2}{\|\mathbf{W}_1\| \|\mathbf{W}_2\|})$ is the angle between the weight tensors, and $t$ is the interpolation scale.

### 2. TIES (Trim, Elect Sign, & Merge)
When merging multiple models, parameter changes conflict (some weights increase while others decrease). TIES resolves this in three steps:
1. **Trim**: Retain only the top $k\%$ most significant parameter changes (deltas) from the base model, setting the rest to zero.
2. **Elect Sign**: Resolve parameter direction conflicts by calculating a consensus sign ($+1$ or $-1$) for each position based on the majority of models.
3. **Consensus Merge**: Average the parameter changes that align with the consensus sign.

### 3. DARE (Drop and Rescale)
DARE randomly zeroes out parameter changes (deltas) with a drop probability $p$ and rescales the remaining weights by $\frac{1}{1-p}$ to keep the output expectation consistent, matching performance with cleaner representations.

### 4. Task Arithmetic
Task Arithmetic calculates task-specific changes (task vectors) relative to the base model, and combines them using linear scaling.

---

### Step-by-Step Hand Calculation: Task Arithmetic

Let's calculate a merged weight matrix parameter using Task Arithmetic.

#### 1. Setup Base and Aligned Parameters
Let's consider a single weight parameter index:
- Base model parameter: $W_0 = 0.50$
- Code-expert model parameter: $W_{\text{code}} = 0.80$
- Math-expert model parameter: $W_{\text{math}} = 0.30$

#### 2. Calculate Task Vectors
We define the task vector $\tau_i$ as the parameter change from the base model:
$$\tau_{\text{code}} = W_{\text{code}} - W_0 = 0.80 - 0.50 = 0.30$$
$$\tau_{\text{math}} = W_{\text{math}} - W_0 = 0.30 - 0.50 = -0.20$$

#### 3. Merge Task Vectors with Scaling Factors
Let's combine the vectors using scaling coefficients $\lambda_{\text{code}} = 0.60$ and $\lambda_{\text{math}} = 0.40$:
$$W_{\text{merged}} = W_0 + \lambda_{\text{code}} \tau_{\text{code}} + \lambda_{\text{math}} \tau_{\text{math}}$$
$$W_{\text{merged}} = 0.50 + (0.60 \times 0.30) + (0.40 \times -0.20)$$
$$W_{\text{merged}} = 0.50 + 0.18 - 0.08 = 0.60$$

The merged parameter value is $0.60$, integrating contributions from both specialized models.

---

## 3. Merging LoRA Adapters into Base weights

During SFT with LoRA, the final model parameters are represented as:
$$\mathbf{W}_{\text{final}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}$$

To deploy this model in production without introducing execution latency (due to executing separate low-rank forward passes), we perform a **LoRA Merge**:
1. Calculate the explicit tensor update:
   $$\Delta \mathbf{W} = \frac{\alpha}{r} (\mathbf{B} \cdot \mathbf{A})$$
2. Add this update directly to the base weights:
   $$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \Delta \mathbf{W}$$
3. Save the resulting consolidated tensors as standard model weights.

This eliminates the adapter routing code during inference, matching base model execution speeds.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Model merging creates multi-task models from single-domain variants without requiring expensive combined training runs.
- **Why was it introduced?**
  Training multi-task models from scratch is computationally expensive and prone to task interference. Model merging allows parallel training of experts followed by zero-cost compilation.
- **What are its limitations?**
  Unchecked merging can lead to parameter interference, where conflicting signs degrade performance across both target domains.
- **Computational Complexity (Time & Memory)**
  - **Memory Complexity**: $O(K \cdot N)$ where $K$ is the number of models to merge, and $N$ is parameter size.
  - **Time Complexity**: $O(N)$ operations for linear addition, requiring minimal compute.
- **Component Variable Denotation Legend**
  - $\mathbf{W}_0$: Base model weight tensor.
  - $\tau_i$: Task vector delta tensor for task $i$.
  - $\lambda_i$: Scaled blending multiplier.
  - $t$: Spherical interpolation coordinate.
- **Production Use Cases**
  - Blending a math-aligned model and a chat-aligned model to create a math-tutor agent.
  - Merging LoRA adapters back into base models to minimize latency for high-throughput serving endpoints.
- **Follow-up questions interviewers ask**
  - *Why does simple linear interpolation (LERP) degrade high-dimensional model representations compared to SLERP?*
  - *How does TIES resolve sign disagreement issues when merging three or more expert models?*
