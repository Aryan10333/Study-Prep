# Module 04: Positional Representations: From Sinusoidal to RoPE & ALiBi

This study guide explains the engineering mechanics of positional representation in Transformer architectures, detailing the math behind coordinate transformations (RoPE), linear attention biases (ALiBi), and their implementation tradeoffs.

---

## 1. Why Position Matters

Because the Self-Attention mechanism computes dot product similarities across all tokens in parallel, it is inherently permutation-invariant. Changing word order (e.g. `"not bad, actually good"` vs. `"good, actually not bad"`) yields the exact same attention score vectors. To preserve syntax, positional information must be injected before the attention block.

---

## 2. Positional Representation Paradigms

```text
               Positional Encodings
             /                      \
   [Absolute Position]           [Relative Position]
   ├── Sinusoidal (Static)       ├── Attention Biases (ALiBi)
   └── Learned Embeddings        └── Rotary Embeddings (RoPE)
```

### Absolute Positional Encoding (Sinusoidal)
- **Concept**: Adds fixed, deterministic sinusoidal coordinates directly to the input embedding:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
  - *Advantage*: No parameters to learn.
  - *Limitation*: Cannot extrapolate. If a model is trained on a context length of 2048, it does not know the sinusoidal values for position 2049, causing generation failures.

### Learned Positional Embeddings
- **Concept**: Initializes a parameter matrix $\mathbf{W}_{\text{pos}} \in \mathbb{R}^{L_{\text{max}} \times d}$ trained alongside token embeddings.
  - *Limitation*: Strict upper-bound context limit ($L_{\text{max}}$). Extrapolation is impossible.

### Relative Positional Encoding (ALiBi)
- **Concept**: Attention with Linear Biases (ALiBi) injects relative position directly into attention heads by subtracting a linear penalty scaled by distance:
  $$a_{i,j} = \text{Softmax}\left( \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}} - m \cdot |i - j| \right)$$
  - *Advantage*: High context extrapolation capacity at runtime.

---

## 3. Rotary Position Embeddings (RoPE)

RoPE encodes relative position by applying a rotation matrix to Query and Key vectors in 2D slices of the hidden dimension, maintaining absolute coordinates while naturally yielding relative scores inside attention dot products.

### Mathematical Formulation
For a 2D vector $\mathbf{x} = [x_1, x_2]^T$ at position $m$, the rotated vector is:
$$\mathbf{R}_{\Theta, m}^2 \mathbf{x} = \begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 \cos m\theta - x_2 \sin m\theta \\ x_1 \sin m\theta + x_2 \cos m\theta \end{bmatrix}$$
- **Relative Distance Preservation**:
  The inner product of rotated query $\mathbf{q}_m$ and key $\mathbf{k}_n$ is:
  $$\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$$
  This shows that attention scores depend only on relative token separation $(n-m)$.

#### Step-by-Step Hand Calculation (RoPE 2D Rotation)
- **Scenario**: Token at position $m = 1$. Let base rotation angle $\theta = \frac{\pi}{2}$ radians ($90^\circ$).
- **Input vector**: $\mathbf{x} = [1.0, 2.0]^T$.
- **Calculation**:
  1. Evaluate trigonometric functions for position $m\theta = 1 \times \frac{\pi}{2} = \frac{\pi}{2}$:
     $$\cos\left(\frac{\pi}{2}\right) = 0.0, \quad \sin\left(\frac{\pi}{2}\right) = 1.0$$
  2. Multiply by rotation matrix:
     $$\mathbf{R}_{\frac{\pi}{2}, 1}^2 \mathbf{x} = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} (1.0 \times 0.0) - (2.0 \times 1.0) \\ (1.0 \times 1.0) + (2.0 \times 0.0) \end{bmatrix} = \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix}$$
- **Result**: The rotated coordinate vector is $[-2.0, 1.0]^T$ (representing a strict $90^\circ$ counterclockwise rotation), matching target simulations.

---

## 4. Comparison of Positional Encodings

| Feature / Metric | Sinusoidal PE | Learned PE | ALiBi | RoPE |
|---|---|---|---|---|
| **Insertion Phase** | Input embeddings | Input embeddings | Attention head scores | Query/Key projections |
| **Parameters** | 0 (static) | $L_{\text{max}} \times d$ | 0 (static biases) | 0 (trig weights) |
| **Extrapolation** | Poor | Impossible | Excellent | Good (scales with RoPE base adjustments) |
| **Computation** | Very Low (add) | Very Low (add) | Low (subtract bias) | Moderate (trig rotates) |

---

## 5. Detailed Computational Complexity (Time & Memory)

- **Learned PE Lookup Time**: $O(L \cdot d)$ addition operations.
- **RoPE Rotation Time**: $O(L \cdot d)$ operations (trig operations scale linearly with tokens).
- **VRAM Memory Space**:
  - Learned PE: $O(L_{\text{max}} \cdot d)$ parameter storage.
  - RoPE: $O(1)$ static trig buffer cached in VRAM.

---

## 6. Interview Questions & Production Trade-offs

### What problem does this solve?
Injects sequential syntax order to the permutation-invariant self-attention computation.

### Why was it introduced?
Absolute learned encodings crash on context lengths exceeding $L_{\text{max}}$. Relative/Rotary encodings preserve distance representations across arbitrary lengths.

### What are its limitations?
ALiBi decays long-range dependencies too strictly. RoPE extrapolation degrades at long contexts without base frequency adjustments (e.g. RoPE theta scaling).

### Production Use Cases:
- Llama-3 models using Rotary Positional Embeddings to scale context windows up to $128\text{k}$ tokens.
- Summarization engines implementing ALiBi to achieve stable context length extrapolation.

### Follow-up Questions Interviewers Ask:
1. *Why does the relative dot product property $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$ hold true for RoPE?*
   - **Answer**: The rotation matrix $\mathbf{R}_m$ is an orthogonal transformation. Since it represents a rotation in 2D space, its transpose is its inverse: $\mathbf{R}_m^T = \mathbf{R}_m^{-1} = \mathbf{R}_{-m}$. Because rotations are additive in the exponent, we have $\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{-m} \mathbf{R}_n = \mathbf{R}_{n-m}$, demonstrating that the dot product is independent of absolute indexes $m$ and $n$ and relies solely on their difference.
2. *What is the "RoPE Base Frequency Scaling" trick and why is it used?*
   - **Answer**: The RoPE rotation angle is calculated as $\theta_i = 10000^{-2i/d}$. At long contexts, the rotation steps between adjacent tokens become tiny, causing the model to lose resolution. By scaling the base frequency (e.g., changing $10000$ to $500000$), we slow down the rotation rate, allowing the model to distinguish positions at longer sequences.
3. *Why does ALiBi perform better than learned absolute encodings during context length extrapolation?*
   - **Answer**: Learned absolute encodings are trained on absolute index keys. At sequence lengths $> L_{\text{max}}$, the model encounters index coordinates it has never seen, leading to random embeddings and generation breakdown. ALiBi uses a static penalty slope $-m \cdot |i - j|$ that naturally extends to arbitrary distances, letting the model extrapolate without retraining.
