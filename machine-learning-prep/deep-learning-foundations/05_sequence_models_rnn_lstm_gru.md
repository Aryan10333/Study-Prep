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

### The Sequential Bottleneck & Gradient Decay
- **Sequential Bottleneck:** To calculate the state $h_t$ at step $t$, you must first calculate $h_{t-1}$. This sequence dependency prevents parallel training on GPU hardware, making RNNs slow to train on long sequences.
- **Backpropagation Through Time (BPTT):** The network is unrolled across all $T$ timesteps, creating a deep computational graph.
- **Vanishing Gradient Failure:** The gradient of the loss $L$ with respect to the initial hidden state $h_0$ involves repeated matrix multiplications of the hidden weights transpose:
  $$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} \propto \left( W_{hh}^T \right)^T$$
  If the eigenvalues of $W_{hh}$ are less than $1.0$, the gradient decays exponentially to exactly $0.0$ as sequence length $T$ grows, preventing the model from learning long-term dependencies (e.g., connections across sentences). If eigenvalues exceed $1.0$, gradients explode.

---

## 2. LSTMs and GRUs: Gated Architectures

To resolve vanishing gradients, gated architectures introduce highway connections controlled by sigmoid gates.

### A. Long Short-Term Memory (LSTM)
LSTMs maintain a **Cell State** $C_t$ that acts as a linear conveyor belt. Information flows through addition, allowing gradients to propagate backward across hundreds of steps without decaying.

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

---

### B. Gated Recurrent Unit (GRU)
GRUs are a computationally lighter variant of LSTMs that merge the Cell State and Hidden State, using only two gates:
- **Reset Gate ($r_t$):** Controls how much of the past hidden state to forget:
  $$r_t = \sigma\left( W_r \cdot [h_{t-1}, x_t] + b_r \right)$$
- **Update Gate ($z_t$):** Combines the functions of the forget and input gates, deciding what to retain and what to write:
  $$z_t = \sigma\left( W_z \cdot [h_{t-1}, x_t] + b_z \right)$$
- **Candidate Hidden State ($\tilde{h}_t$):**
  $$\tilde{h}_t = \tanh\left( W \cdot [r_t \odot h_{t-1}, x_t] + b \right)$$
- **Hidden State Update:**
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

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
