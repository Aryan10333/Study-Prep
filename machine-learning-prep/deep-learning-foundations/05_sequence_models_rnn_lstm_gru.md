# Deep Learning: Foundational Sequence Models (RNN, LSTM, GRU)

This guide details the structural design and equations of Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTM) networks, walking through a single LSTM timestep forward pass step-by-step.

---

## 1. Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are designed to process sequential data (e.g., time series, natural language) by maintaining a hidden state $h_t$ that acts as a memory of past timesteps.

### Mathematical Equations
For each timestep $t$:
- **Hidden State:**
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
- **Output:**
  $$y_t = W_{hy} h_t + b_y$$

*Dimensions:*
- Input vector $x_t \in \mathbb{R}^{d \times 1}$.
- Hidden state vector $h_t \in \mathbb{R}^{h \times 1}$.
- Hidden weight matrix $W_{hh} \in \mathbb{R}^{h \times h}$.
- Input weight matrix $W_{xh} \in \mathbb{R}^{h \times d}$.

---

### Backpropagation Through Time (BPTT) & The Vanishing Gradient Math
When we train an RNN, we unroll the sequence over $T$ steps. The total loss $L$ is the sum of losses at each timestep:
$$L = \sum_{t=1}^T L_t$$

To calculate the gradient of the loss at step $t$ with respect to the hidden weight matrix $W_{hh}$, we unroll the dependency chain backward through time:
$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

The middle term $\frac{\partial h_t}{\partial h_k}$ is a product of Jacobians representing historical state transitions:
$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$$

Let's calculate the Jacobian matrix $\frac{\partial h_j}{\partial h_{j-1}}$:
$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\left( 1 - \tanh^2\left( W_{hh} h_{j-2} + W_{xh} x_{j-1} + b_h \right) \right) \cdot W_{hh}^T$$

- **Vanishing Gradient Failure:** Because $\tanh'(z) = 1 - \tanh^2(z)$ is bounded in $(0, 1]$, and is close to $0$ when inputs saturate, the product of these matrices is dominated by the power of the weights matrix:
  $$\frac{\partial h_t}{\partial h_k} \propto \left( W_{hh}^T \right)^{t-k}$$
  If the largest eigenvalue (spectral radius) of $W_{hh}$ is less than $1.0$, the gradient term decays exponentially to exactly $0.0$ as the gap $t-k$ grows, preventing the model from learning dependencies beyond a few steps.
  - *Sequential Bottleneck:* Since $h_t$ cannot be computed until $h_{t-1}$ is completed, training cannot be parallelized along the sequence dimension, leading to slow GPU utilization.

---

## 2. LSTMs and GRUs: Gated Architectures

To solve vanishing gradients, gated recurrent architectures introduce a linear highway bypass that prevents gradients from being repeatedly multiplied by weight matrices.

### A. Long Short-Term Memory (LSTM)
LSTMs maintain a **Cell State** $C_t$ alongside the hidden state $h_t$. Information flows linearly through addition, bypassing weight decay.

```text
Gate Name       Equation                                         Intuition
----------------------------------------------------------------------------------------------------------------------
Forget Gate     f_t = σ(W_f · [h_t-1, x_t] + b_f)                Decides what historical information to discard.
Input Gate      i_t = σ(W_i · [h_t-1, x_t] + b_i)                Decides what new information to write.
Candidate Cell  C~_t = tanh(W_c · [h_t-1, x_t] + b_c)            Represents new candidate state values.
Cell State      C_t = f_t ⊙ C_t-1 + i_t ⊙ C~_t                   Combines old memory and new input via linear addition.
Output Gate     o_t = σ(W_o · [h_t-1, x_t] + b_o)                Decides what part of the cell state to reveal as output.
Hidden State    h_t = o_t ⊙ tanh(C_t)                            Final output state passed to the next step.
```

#### Production Gating Intuition (NLP Example)
To understand why LSTMs regulate memory this way during text generation:
- **Forget Gate ($f_t$):** Discards historical context when grammar or subjects shift. *Example:* If the sentence transitions from a singular subject ("The book is...") to a plural subject ("The books are..."), $f_t$ outputs values close to $0.0$ to clear out the singular grammatical context.
- **Input Gate ($i_t$):** Determines if the current word contains new information that needs to be written to long-term memory. *Example:* Storing the gender or number characteristics of a newly introduced character.
- **Output Gate ($o_t$):** Filters the cell state to output only what is relevant to the immediate prediction. *Example:* Storing grammatical attributes needed to predict the very next verb while leaving long-term semantic knowledge hidden in the Cell State conveyor belt.

#### Why the Cell State Solves Vanishing Gradients
Calculating the gradient derivative of the cell state $C_t$ with respect to the prior state $C_{t-1}$ yields:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \text{terms involving derivatives of gates}$$

If the forget gate $f_t$ is active (close to $1.0$), the gradient propagates directly backward through time:
$$\frac{\partial L}{\partial C_{t-1}} \approx \frac{\partial L}{\partial C_t} \cdot 1.0$$
This addition-based gradient path eliminates the exponential multiplication decay, allowing LSTMs to preserve gradients over long sequences.

---

### B. Gated Recurrent Unit (GRU)
GRUs simplify the LSTM by merging the Cell State and Hidden State into a single hidden state $h_t$. They use only two gates:
- **Reset Gate ($r_t$):** Controls how much of the past hidden state to forget when proposing a candidate state:
  $$r_t = \sigma\left( W_r \cdot [h_{t-1}, x_t] + b_r \right)$$
- **Update Gate ($z_t$):** Decides whether to copy the old hidden state directly or write the new candidate state:
  $$z_t = \sigma\left( W_z \cdot [h_{t-1}, x_t] + b_z \right)$$
- **Candidate Hidden State ($\tilde{h}_t$):**
  $$\tilde{h}_t = \tanh\left( W \cdot [r_t \odot h_{t-1}, x_t] + b \right)$$
- **Hidden State Update:**
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

#### LSTM vs. GRU Trade-offs
- **GRU Advantage:** Has approximately $33\%$ fewer parameters than an LSTM, reducing memory consumption and speeding up convergence on small-to-medium scale sequential datasets.
- **LSTM Advantage:** The separate Cell State and Hidden State provide larger representational capacity, enabling LSTMs to outperform GRUs on long, complex sequences when training data is abundant.

---

## 3. Step-by-Step Hand Calculations: LSTM Cell Pass (Andrew Ng Style)

Let's compute a single LSTM timestep forward pass:
- **Inputs:** $x_t = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$, \quad prior hidden state $h_{t-1} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$ (Concatenated input $[h_{t-1}, x_t] = \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.5 \end{bmatrix}$)
- **Prior Cell State:** $C_{t-1} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$
- Assume the gate weights and biases evaluate to the following pre-activation linear combinations:
  - Forget Gate pre-activation: $Z_f = \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}$
  - Input Gate pre-activation: $Z_i = \begin{bmatrix} -2.0 \\ 2.0 \end{bmatrix}$
  - Candidate Cell pre-activation: $Z_c = \begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix}$
  - Output Gate pre-activation: $Z_o = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$

---

### Step 1: Compute Forget & Input Gates
Apply the sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$:
$$f_t = \sigma\left(\begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{-1}} \\ \frac{1}{1 + e^{1}} \end{bmatrix} \approx \begin{bmatrix} 0.731 \\ 0.269 \end{bmatrix}$$
$$i_t = \sigma\left(\begin{bmatrix} -2.0 \\ 2.0 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{2}} \\ \frac{1}{1 + e^{-2}} \end{bmatrix} \approx \begin{bmatrix} 0.119 \\ 0.881 \end{bmatrix}$$

---

### Step 2: Compute Candidate Cell State ($\tilde{C}_t$)
Apply the Tanh activation:
$$\tilde{C}_t = \tanh\left(\begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix}\right) \approx \begin{bmatrix} 0.462 \\ 0.664 \end{bmatrix}$$

---

### Step 3: Compute Updated Cell State ($C_t$)
Combine old memory and new candidate state using element-wise multiplication ($\odot$):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t = \begin{bmatrix} 0.731 \\ 0.269 \end{bmatrix} \odot \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.119 \\ 0.881 \end{bmatrix} \odot \begin{bmatrix} 0.462 \\ 0.664 \end{bmatrix}$$

$$C_{t,1} = (0.731 \times 0.5) + (0.119 \times 0.462) = 0.3655 + 0.0550 = \mathbf{0.4205}$$
$$C_{t,2} = (0.269 \times -0.5) + (0.881 \times 0.664) = -0.1345 + 0.5850 = \mathbf{0.4505}$$

$$C_t = \begin{bmatrix} 0.4205 \\ 0.4505 \end{bmatrix}$$

---

### Step 4: Compute Output Gate & Hidden State ($o_t$, $h_t$)
$$o_t = \sigma\left(\begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}\right) = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$
$$\tanh(C_t) = \tanh\left(\begin{bmatrix} 0.4205 \\ 0.4505 \end{bmatrix}\right) \approx \begin{bmatrix} 0.3973 \\ 0.4221 \end{bmatrix}$$
$$h_t = o_t \odot \tanh(C_t) = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} \odot \begin{bmatrix} 0.3973 \\ 0.4221 \end{bmatrix} = \begin{bmatrix} \mathbf{0.1987} \\ \mathbf{0.2111} \end{bmatrix}$$

**Result:** The updated Cell State is $\begin{bmatrix} 0.4205 \\ 0.4505 \end{bmatrix}$ and the output Hidden State is $\begin{bmatrix} 0.1987 \\ 0.2111 \end{bmatrix}$.

---

## 4. Production Scenario & Example

### Scenario: Customer Support Ticket Classification Latency
You are deploying a text routing engine that classifies incoming customer support tickets into 50 diagnostic categories. The ticket text sequences average 300 words.
- **The Failure Mode:** Your initial design utilizes a bidirectional LSTM. When user traffic spikes, the model's sequential training and inference bottlenecks cause API timeout exceptions. Profiling reveals that since step $h_t$ requires step $h_{t-1}$, the GPU is under-utilized, wasting compute cycles waiting on sequential unrolling steps.
- **The Solution:** You replace the recurrent model with a Transformer-based Encoder. Unlike LSTMs, which process sequences sequentially, self-attention layers compute dot-product alignment matrices for all 300 words in parallel:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  This removes the $O(T)$ step-dependency bottleneck, decreasing training latency by $10\text{x}$ and letting you batch inference inputs to maximize GPU execution throughput.
