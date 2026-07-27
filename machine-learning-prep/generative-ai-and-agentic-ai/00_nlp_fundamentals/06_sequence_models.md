# Module 06: Sequence Models & Recurrent Architectures

Sequence models process variable-length inputs by maintaining sequential state representations. This module details recurrent architectures, contains mathematical proofs of vanishing gradients, details LSTM cell gating, and covers sequence-to-sequence translation models.

---

## 1. RNNs and the Mathematical Proof of Vanishing Gradients

A standard Recurrent Neural Network (RNN) processes tokens sequentially, updating a hidden state vector $h_t$ at each step:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

### Mathematical Proof of Vanishing Gradients
Consider a loss function $L_T$ calculated at time step $T$. We calculate the gradient of the loss with respect to the recurrent weight matrix $W_{hh}$:

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$

The key term that determines gradient stability is the Jacobian chain product:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}$$

Let's compute the individual Jacobian $\frac{\partial h_k}{\partial h_{k-1}}$:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - h_k^2) W_{hh}^T$$

Where $\text{diag}(1 - h_k^2)$ is the diagonal matrix of the derivative of the activation function $\tanh$.
Substituting this back into the product:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \text{diag}(1 - h_k^2) W_{hh}^T$$

Taking the norm of this product:

$$\left\|\frac{\partial h_T}{\partial h_t}\right\| \le \prod_{k=t+1}^T \left\|\text{diag}(1 - h_k^2)\right\| \|W_{hh}^T\|$$

Since $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$, the diagonal matrix norm is less than or equal to $1$.
- **Vanishing Gradient**: If the largest eigenvalue (spectral radius) of the weight matrix satisfies $\rho(W_{hh}) < 1$, the norm of the product decays exponentially as the time gap $T - t$ increases:
  $$\lim_{T-t \to \infty} \frac{\partial h_T}{\partial h_t} = 0$$
  This prevents gradient updates from propagating back to the earliest steps, making the model blind to long-range dependencies.
- **Exploding Gradient**: If $\rho(W_{hh}) > 1$, the product can grow exponentially, causing weight divergence during training.

---

## 2. LSTM Gating Mechanics and the Constant Error Carousel

The Long Short-Term Memory (LSTM) network introduces a dedicated cell state vector $C_t$ to act as a linear conveyor belt for gradient flow.

### LSTM Gating Equations
At step $t$, the LSTM updates its state vectors using four gating mechanisms:

$$\text{Forget Gate: } f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

$$\text{Input Gate: } i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

$$\text{Candidate Cell State: } \tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

$$\text{Cell State Update: } C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$\text{Output Gate: } o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$\text{Hidden State: } h_t = o_t \odot \tanh(C_t)$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ projects gate activations to $(0, 1)$, and $\odot$ is the element-wise Hadamard product.

---

### Proof of the Constant Error Carousel (CEC)
To see how LSTM prevents vanishing gradients, we calculate the derivative of the cell state $C_t$ with respect to the previous state $C_{t-1}$:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \text{terms containing } \frac{\partial f_t}{\partial C_{t-1}}$$

If we set the forget gate to $f_t = 1$ (indicating the model should retain historical cell information), the derivative simplifies to:

$$\frac{\partial C_t}{\partial C_{t-1}} \approx 1$$

Consequently, the error gradient can propagate back through time indefinitely without exponential decay:

$$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^T f_k \approx 1$$

This linear update path is called the **Constant Error Carousel**, allowing LSTMs to retain information across long contexts.

---

## 3. GRU Variations

The Gated Recurrent Unit (GRU) simplifies the LSTM by merging the cell state and hidden state, and combining the input and forget gates:

$$\text{Update Gate: } z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

$$\text{Reset Gate: } r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

$$\text{Candidate State: } \tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$

$$\text{Hidden State: } h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

*Trade-off*: GRUs have fewer parameters than LSTMs, leading to faster training times, but can exhibit lower capacity on long sequences.

---

## 4. Bidirectional Recurrent Models

Bidirectional sequence models process input sequences in both directions, capturing future and past context:

$$\vec{h}_t = \text{LSTM}_{\text{forward}}(x_t, \vec{h}_{t-1})$$

$$\overleftarrow{h}_t = \text{LSTM}_{\text{backward}}(x_t, \overleftarrow{h}_{t+1})$$

$$\text{Combined Hidden State: } h_t = [\vec{h}_t \ ; \ \overleftarrow{h}_t]$$

This combined representation captures context from both sides of a word, forming the architectural foundation for encoders like BERT.

---

## 5. Sequence-to-Sequence (Seq2Seq) and Decoding Strategies

Seq2Seq models use an Encoder network to compress a variable-length source sequence into a single context vector, which a Decoder network then unpacks into a target sequence:

```
Encoder Context Mapping                     Decoder Decoding Sequence
     x1 ➔ x2 ➔ x3                           [Context Vector]
         │                                         │
         ▼                                         ▼
  [Hidden States] ➔ [Context Vector] ➔ ŷ1 ➔ ŷ2 ➔ ŷ3 (Decoder outputs)
```

### Decoder Strategies
- **Teacher Forcing**: During training, the decoder receives the ground-truth target tokens as input instead of its own previous predictions, stabilizing early training.
- **Greedy Search**: The model outputs the most likely token at each step:
  $$y_t = \arg\max P(y \mid y_{<t})$$
  *Limitation*: Cannot backtrack; an early error propagates through the rest of the generation.
- **Beam Search**: Keeps a running set of the $B$ most likely hypothesis sequences (beams). At each step, it expands all beams and retains the top $B$ paths with the highest cumulative log probability:
  $$\text{Score}(y_{1..t}) = \sum_{i=1}^t \log P(y_i \mid y_{<i})$$

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Processes variable-length sequence data while retaining historical information across time steps.
- **Why was it introduced?**
  Introduced because standard feedforward networks require fixed-sized inputs, making them unable to process sequences of arbitrary length.
- **What are its limitations?**
  - **Sequential Bottleneck**: Processing token $t$ requires computing states up to step $t-1$, preventing parallel execution.
  - **Memory Footprint**: BPTT requires storing intermediate hidden states for all steps, leading to high VRAM footprint.
- **Computational Complexity (Time & Memory)**
  - **Sequence processing Time**: $O(L \cdot d^2)$ where $L$ is sequence length and $d$ is hidden dimension size.
  - **Backpropagation Memory**: $O(L \cdot d)$ per layer.
- **Component Variable Denotation Legend**
  - $L$: Sequence token length.
  - $d$: Recurrent hidden state dimension.
  - $B$: Beam search branch width parameter.
- **Production Use Cases**
  - Text translation pipelines.
  - Named Entity Recognition sequence tagging.
- **Follow-up questions interviewers ask**
  - *Why do we use log probabilities instead of raw probabilities in Beam Search?* (Multiplying small probabilities leads to numerical underflow; summing log probabilities keeps calculations numerically stable).
  - *How does Gradient Clipping prevent exploding gradients in RNNs?* (If the norm of the gradient exceeds a threshold $g_{\max}$, it is scaled down: $\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{g_{\max}}{\|\mathbf{g}\|}$, preventing weight updates from causing network instability).
