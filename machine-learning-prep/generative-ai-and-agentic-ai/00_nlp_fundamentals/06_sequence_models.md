# Module 06: Sequence Models & Recurrent Architectures

Sequence models process variable-length inputs by maintaining sequential state representations. This module details recurrent architectures, explains the mechanics of vanishing gradients, details LSTM cell gating, and covers sequence-to-sequence translation models.

---

## 1. RNNs and the Intuition of Vanishing Gradients

![Gradient Flow Comparison](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/gradient_flow_comparison.png)

A standard Recurrent Neural Network (RNN) processes tokens sequentially, updating a hidden state vector $h_t$ at each step:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

### Why Gradients Vanish or Explode
During training, the model calculates the gradient of the loss at time step $T$ with respect to parameters at time step $t$. This requires backpropagating the error through all intermediate steps. 

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} \propto (W_{hh})^{T-t}$$

- **Vanishing Gradient**: If the weights in $W_{hh}$ are small (specifically, if its largest eigenvalue is $< 1$), multiplying this matrix repeatedly over a long time gap $T-t$ causes the gradient to shrink exponentially to $0$. The model becomes unable to learn dependencies from early tokens.
- **Exploding Gradient**: If the weights in $W_{hh}$ are large (largest eigenvalue $> 1$), multiplying this matrix repeatedly causes the gradient to grow exponentially, leading to weight divergence and training instability.

---

## 2. LSTM Gating Mechanics and the Constant Error Carousel

The Long Short-Term Memory (LSTM) network introduces a dedicated cell state vector $C_t$ to act as a linear conveyor belt for gradient flow.

### LSTM Gating Diagram
```
              [ Cell State C(t-1) ] ──────────────( X )──────────────────(+)─────────────▶ [ Cell State C(t) ]
                                                   │                      ▲
                                                   │ (Forget Gate f_t)    │ (Input Gate i_t * Candidate C~_t)
                                                   ▼                      │
  [ Input x_t ] ─────┐                       [ Forget Gate ]        [ Input Gate ]
                     ├─────▶ [ Gates ] ─────▶[ Candidate  ] ───────▶[ Output Gate ]
  [ Hidden h(t-1) ] ─┘                       [ Output Gate ]              │
                                                   │                      ▼
                                                   └────────────────────( X )────────────▶ [ Hidden h_t ]
```

### LSTM Gating Equations
At step $t$, the LSTM updates its state vectors using four gating mechanisms:
- **Forget Gate** ($f_t$): Controls how much of the previous cell state to discard (0 = discard, 1 = retain).
- **Input Gate** ($i_t$): Controls which new values to write to the cell state.
- **Candidate Cell State** ($\tilde{C}_t$): The new information candidate.
- **Cell State Update**: Compiles old and new information linearly:
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
- **Output Gate** ($o_t$): Controls what information from the cell state to output to the hidden state $h_t$.

### The Constant Error Carousel (CEC) Intuition
Because the cell state update uses **addition ($+$)** instead of multiplication ($*$), the derivative of $C_t$ with respect to $C_{t-1}$ contains the linear term $f_t$:
$$\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$$
If the forget gate is open ($f_t \approx 1$), the gradient of $C_T$ with respect to $C_t$ remains near $1$:
$$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^T f_k \approx 1$$
This allows the gradient to flow backward through time indefinitely without exponential decay, creating the **Constant Error Carousel**.

---

## 3. GRU Variations

The Gated Recurrent Unit (GRU) simplifies the LSTM by merging the cell state and hidden state, and combining the input and forget gates into a single update gate $z_t$, reducing training parameter counts.

---

## 4. Bidirectional RNNs and LSTMs

Bidirectional sequence models process input sequences in both directions, capturing future and past context:

$$\vec{h}_t = \text{LSTM}_{\text{forward}}(x_t, \vec{h}_{t-1})$$

$$\overleftarrow{h}_t = \text{LSTM}_{\text{backward}}(x_t, \overleftarrow{h}_{t+1})$$

$$\text{Combined Hidden State: } h_t = [\vec{h}_t \ ; \ \overleftarrow{h}_t]$$

This combined representation captures context from both sides of a word, forming the architectural foundation for encoders like BERT.

---

## 5. Sequence-to-Sequence (Seq2Seq) and Decoding Strategies

Seq2Seq models use an Encoder network to compress a variable-length source sequence into a single context vector, which a Decoder network then unpacks into a target sequence.
- **Teacher Forcing**: During training, the decoder receives the ground-truth target tokens as input instead of its own previous predictions, stabilizing early training.
- **Greedy Search**: The model outputs the most likely token at each step:
  $$y_t = \arg\max P(y \mid y_{<t})$$
  *Limitation*: Cannot backtrack; an early error propagates through the rest of the generation.
- **Beam Search**: Keeps a running set of the $B$ most likely hypothesis sequences (beams). At each step, it expands all beams and retains the top $B$ paths with the highest cumulative log probability.

---

> [!TIP]
> **Production Insight: Gradient Norm Clipping**
> To prevent exploding gradients in recurrent models, implement gradient clipping:
> ```python
> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
> ```
> This scales down the gradients if their total norm exceeds `max_norm`, preventing training loops from crashing with `NaN` losses.

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
