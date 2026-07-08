# Transformers: Positional Encoding & Context Extensions

This guide details the mathematical equations, relative properties, and implementations of Sinusoidal, Learned, and Rotary Positional Embeddings (RoPE), walking through hand-calculations and PyTorch implementation.

---

## 1. Why is Positional Information Needed?

The self-attention calculation ($\text{softmax}(QK^T)V$) is **permutation invariant**.
- **The Issue:** If we shuffle the order of input tokens, the output vectors will be identical (only shuffled to match). The model has no concept of word order. For example, the model would treat the sentences *"The dog bit the man"* and *"The man bit the dog"* identically.
- **The Solution:** We inject positional information into the input representations, allowing the attention matrices to incorporate absolute or relative token order.

---

## 2. Mathematical Formulations

### A. Sinusoidal Positional Encoding (Original Transformer)
A fixed, non-learnable vector added directly to the input token embeddings:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$
Where:
- $pos$ is the token position index in the sequence.
- $i$ is the dimension index.
- **Relative Position Property:** For any fixed offset $k$, $PE_{pos+k}$ can be projected as a linear function of $PE_{pos}$. This allows the model to learn relative relationships.

---

### B. Learned Positional Embeddings
Treats position indices as discrete tokens, training an embedding matrix $PE \in \mathbb{R}^{N_{\text{max}} \times d_{\text{model}}}$.
- **Production Drawback:** The model cannot extrapolate to sequences longer than the maximum training context length $N_{\text{max}}$. If a model is trained on context length 2048, it will crash or fail at step 2049.

---

### C. Rotary Positional Embeddings (RoPE)
Instead of adding position vectors to input embeddings, RoPE multiplies the projected Query and Key vectors by a rotation matrix, rotating 2D slices of the vectors by an angle proportional to the token position $m$:
$$R_{\Theta, m}^2 \cdot x = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$
Where $\theta_i = 10000^{-2(i-1)/d}$.

#### The Relative Advantage:
The dot product of a Query at position $m$ and a Key at position $n$ rotated via RoPE becomes:
$$\langle R_{\Theta, m} q, \ R_{\Theta, n} k \rangle = q^T R_{\Theta, n-m} k$$
- **Production Utility:** The attention weights depend *only* on the relative distance $n-m$, allowing the model to naturally model relative positions. Standard in modern LLMs (Llama, Mistral, Gemma).

---

## 3. Step-by-Step Hand Calculations (Andrew Ng Style)

### A. Sinusoidal Position PE Calculation
Let's compute the positional encoding vector for a token at position $pos=1$ with a model dimension $d_{\text{model}} = 4$:
- **For dimension index $i=0$ (indexes $2i=0$, $2i+1=1$):**
  - $\text{Denominator} = 10000^{0/4} = 1.0$
  - $PE_{(1, 0)} = \sin(1 / 1.0) = \sin(1.0) \approx \mathbf{0.8415}$
  - $PE_{(1, 1)} = \cos(1 / 1.0) = \cos(1.0) \approx \mathbf{0.5403}$
- **For dimension index $i=1$ (indexes $2i=2$, $2i+1=3$):**
  - $\text{Denominator} = 10000^{2/4} = 100.0$
  - $PE_{(1, 2)} = \sin(1 / 100.0) = \sin(0.01) \approx \mathbf{0.0100}$
  - $PE_{(1, 3)} = \cos(1 / 100.0) = \cos(0.01) \approx \mathbf{0.9999}$

$$PE_1 = \begin{bmatrix} 0.8415 & 0.5403 & 0.0100 & 0.9999 \end{bmatrix}$$

---

### B. RoPE Rotation Calculation
Rotate a 2D query vector $q = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$ at position $m=2$ with base angle $\theta = \frac{\pi}{4}$ ($45^\circ$):
1. **Compute Rotation Angle:**
   $$\phi = m\theta = 2 \times \frac{\pi}{4} = \frac{\pi}{2} \ (90^\circ)$$
2. **Setup Rotation Matrix ($R$):**
   $$R = \begin{bmatrix} \cos(\pi/2) & -\sin(\pi/2) \\ \sin(\pi/2) & \cos(\pi/2) \end{bmatrix} = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix}$$
3. **Compute Rotated Vector:**
   $$q_{\text{rot}} = R q = \begin{bmatrix} 0.0 & -1.0 \\ 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} (0 \times 1 - 1 \times 1) \\ (1 \times 1 + 0 \times 1) \end{bmatrix} = \begin{bmatrix} \mathbf{-1.0} \\ \mathbf{1.0} \end{bmatrix}$$

**Result:** The rotated Query vector is $\begin{bmatrix} -1.0 \\ 1.0 \end{bmatrix}$.

---

## 4. Production Selection & Context Length Extension Rules

- **Why RoPE is standard for Context Length Extensions:**
  If you need to extend an LLM's context length beyond its pre-trained limit (e.g., from a pre-trained limit $L$ to an extended limit $L' = s \cdot L$, where $s$ is the scaling factor), RoPE allows simple context-extension scaling techniques with minimal or no retraining:

  ### A. Linear RoPE Scaling
  Linear scaling modifies the position index $m$ by dividing it by the scaling factor $s$:
  $$m \leftarrow \frac{m}{s}$$
  The rotated Query/Key vectors at position $m$ are computed by rotating with angles:
  $$\phi_i = \frac{m}{s} \theta_i = \frac{m}{s} \text{base}^{-2(i-1)/d}$$
  - **The Trade-off:** By squishing the position indices down, the model remains within its trained frequency bounds, which prevents catastrophic validation loss. However, it scales down all frequencies equally. For very close tokens (e.g., adjacent words), the relative distance is squished from $1$ to $1/s$. This dilutes the high-frequency relative distance details, leading to severe performance degradation on short-context sequences unless the model is fine-tuned extensively.

  ### B. NTK-Aware (Neural Tangent Kernel) Scaling
  Instead of scaling the position index $m$ uniformly, NTK-aware scaling leaves the position index $m$ unchanged and dynamically scales the rotation frequency base.
  
  The base is scaled from $\text{base}$ to $\text{base}'$:
  $$\text{base}' = \text{base} \cdot s^{\frac{d}{d-2}}$$
  Thus, the scaled frequencies $\theta_i'$ are:
  $$\theta_i' = \left( \text{base} \cdot s^{\frac{d}{d-2}} \right)^{-2(i-1)/d} = \text{base}^{-2(i-1)/d} \cdot s^{-\frac{2(i-1)}{d-2}}$$
  - **Why it works:** 
    - For high-frequency components (when $i \to 1$), the scaling multiplier $s^{-\frac{2(i-1)}{d-2}} \to 1$. Thus, the high-frequency components (representing close, local token interactions) remain almost identical to their pre-trained values, preserving high-resolution local attention structures.
    - For low-frequency components (when $i \to d/2$), the scaling multiplier $s^{-\frac{2(d/2-1)}{d-2}} = s^{-1} = 1/s$. The low frequencies (representing global sequence structures) are scaled down by exactly $1/s$, allowing the model to extrapolate smoothly to the extended sequence length $s \cdot L$.
    - **Production Utility:** NTK-aware scaling achieves outstanding context extensions (often up to $8\text{x}$ or $16\text{x}$ extension windows) without requiring fine-tuning, keeping local attention intact while extending global bounds.

---

## 5. PyTorch RoPE Implementation

This code implements the 2D rotary position projections on key/query tensors:

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # dim: hidden dimension of attention head (must be even)
        self.dim = dim
        
        # 1. Compute theta frequencies: theta_i = theta^(-2(i-1)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 2. Precompute cos and sin values for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate freqs with themselves to handle even/odd splits
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _rotate_half(self, x):
        # x shape: (B, seq_len, dim)
        # Split vector and rotate halves
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        # x shape: (batch, seq_len, dim)
        cos = self.cos_cached[:seq_len, :]  # (seq_len, dim)
        sin = self.sin_cached[:seq_len, :]  # (seq_len, dim)
        
        # Apply RoPE: x_rotated = x * cos(m_theta) + rotate_half(x) * sin(m_theta)
        return (x * cos) + (self._rotate_half(x) * sin)

# Verification
rope = RotaryPositionalEmbedding(dim=4, max_seq_len=10)
q = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)  # batch=1, seq=1, dim=4

# Rotate at step pos=2 (angle = pi/2 for index 0-1)
q_rot = rope(q, seq_len=1)
print("Rotated Query Tensor:\n", q_rot)
```
