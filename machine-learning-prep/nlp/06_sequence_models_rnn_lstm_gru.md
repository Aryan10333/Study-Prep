# Module 06: Sequence Models (RNN, LSTM, GRU & Seq2Seq Architectures)

This study guide covers Recurrent Neural Networks (RNN), Backpropagation Through Time (BPTT), vanishing gradients, the 5 core LSTM update equations, a scalar LSTM forward pass walkthrough (forward pass only), GRU gating, PyTorch code, complexity analysis, and standardized interview Q&A.

> **Notebook Companion**: [06_sequence_models_rnn_lstm_gru.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/06_sequence_models_rnn_lstm_gru.ipynb)

---

## 1. Vanilla RNN Architecture & Vanishing Gradients

Vanilla Recurrent Neural Networks process sequential input tokens $x_1, x_2, \dots, x_T$ step-by-step, maintaining a hidden state $h_t \in \mathbb{R}^d$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)$$

### The Vanishing Gradient Problem in BPTT:
Calculating the loss gradient $\frac{\partial \mathcal{L}_T}{\partial h_k}$ at time step $k \ll T$ requires repeatedly multiplying weight matrix $W_{hh}^\top$:

$$\frac{\partial \mathcal{L}_T}{\partial h_k} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{j=k+1}^T W_{hh}^\top \text{diag}(1 - \tanh^2(\dots))$$

If the largest eigenvalue $\lambda_{\max}(W_{hh}) < 1.0$, the gradient norm decays exponentially to zero ($0.85^T \rightarrow 0$), preventing long-range dependency learning.

---

## 2. The 5 Core LSTM Update Equations

The Long Short-Term Memory (LSTM) architecture resolves vanishing gradients by introducing a cell memory state $C_t$ governed by 3 gating mechanisms:

```text
1. Forget Gate:     f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
2. Input Gate:      i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
3. Candidate Cell:  C~_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
4. Cell State:      C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t
5. Output Gate:     o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  ==>  h_t = o_t ⊙ tanh(C_t)
```

Where $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the sigmoid function and $\odot$ represents element-wise Hadamard product.

---

## 3. Step-by-Step Scalar LSTM Forward Pass Walkthrough

> [!NOTE]
> **Interview Scope**: Forward Pass Only (No BPTT derivation required).

Consider a single-cell LSTM forward pass step with simple scalar inputs:
- Inputs: $x_t = 1.0$, previous hidden state $h_{t-1} = 0.5$, previous cell state $C_{t-1} = 0.8$.
- Scalar Weights & Biases (set to $W = 1.0, b = 0.0$ for step demonstration):

### Step 1: Compute Gating Signals
- Summed activation input: $z = h_{t-1} + x_t = 0.5 + 1.0 = 1.5$
- **Forget Gate**:
  $$f_t = \sigma(1.5) = \frac{1}{1 + e^{-1.5}} \approx \mathbf{0.8176}$$
- **Input Gate**:
  $$i_t = \sigma(1.5) \approx \mathbf{0.8176}$$
- **Candidate Cell State**:
  $$\tilde{C}_t = \tanh(1.5) = \frac{e^{1.5} - e^{-1.5}}{e^{1.5} + e^{-1.5}} \approx \mathbf{0.9051}$$

### Step 2: Cell State Update ($C_t$)
$$C_t = (f_t \times C_{t-1}) + (i_t \times \tilde{C}_t)$$
$$C_t = (0.8176 \times 0.8) + (0.8176 \times 0.9051) = 0.6541 + 0.7400 = \mathbf{1.3941}$$

### Step 3: Output Gate & Hidden State ($h_t$)
- **Output Gate**:
  $$o_t = \sigma(1.5) \approx \mathbf{0.8176}$$
- **Hidden State Output**:
  $$h_t = o_t \times \tanh(C_t) = 0.8176 \times \tanh(1.3941)$$
  $$\tanh(1.3941) \approx 0.8840$$
  $$h_t = 0.8176 \times 0.8840 = \mathbf{0.7228}$$

---

## 4. Gated Recurrent Unit (GRU) Simplification

GRU merges cell state and hidden state into a single state $h_t$, utilizing only 2 gates:
1. **Reset Gate ($r_t$)**: Controls how much past hidden state to forget.
2. **Update Gate ($z_t$)**: Controls the balance between previous state $h_{t-1}$ and candidate state $\tilde{h}_t$.

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU has $25\%$ fewer parameters than LSTM, accelerating training on smaller datasets.

---

## 5. RNN vs. LSTM Gradient Flow Comparison

![RNN Vanishing Gradient vs LSTM Error Highway](images/06_lstm_gradient_flow.png)

> **Plot Interpretation & Production Insight**:
> - **Error Highway**: The LSTM cell state $C_t$ update is purely additive ($C_t = f_t \odot C_{t-1} + \dots$), creating a linear gradient highway that maintains gradient norm $\approx 1.0$ across 50+ time steps.

---

## 6. Production PyTorch LSTM Implementation Code

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)                     # (batch_size, seq_len, embed_dim)
        output, (hn, cn) = self.lstm(embedded)           # output: (batch_size, seq_len, hidden_dim*2)
        # Concatenate forward and backward final hidden states
        final_hidden = torch.cat((hn[-2], hn[-1]), dim=1) # (batch_size, hidden_dim*2)
        return self.fc(final_hidden)                     # (batch_size, num_classes)

model = SentimentLSTM()
dummy_input = torch.randint(0, 1000, (4, 32))  # Batch size 4, seq len 32
output = model(dummy_input)
print("PyTorch LSTM Output Tensor Shape:", output.shape)
```

---

## 7. Interview Questions & Production Trade-offs

### What problem do LSTMs solve over vanilla RNNs?
Vanilla RNNs suffer from vanishing gradients because backpropagating through time involves repeated matrix multiplication by $W_{hh}^\top$, causing gradients to decay exponentially to zero. LSTMs introduce an additive cell memory highway ($C_t$) that preserves gradient flow across long sequences ($T > 100$).

### Why was the Cell State constant error highway introduced?
In cell update equation $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, if forget gate $f_t = 1$, the gradient derivative $\frac{\partial C_t}{\partial C_{t-1}} = 1$, allowing gradients to pass backward without matrix multiplication attenuation.

### What are the primary limitations of LSTMs?
- **Sequential Bottleneck**: Hidden state $h_t$ depends sequentially on $h_{t-1}$, preventing parallel GPU execution over sequence length $T$.
- **Fixed Hidden Capacity**: Forcing long sequences into fixed vector $h_t$ creates an information bottleneck (resolved by Attention).

### Computational Complexity:
- **Training Step Time Complexity**: $O(T \cdot (4 \cdot d^2 + 4 \cdot d \cdot d_x))$ per sequence.
- **Inference Time Complexity**: $O(T \cdot d^2)$ strictly sequential operations.

### Production Use Cases:
- Time-series forecasting and sensor telemetry anomaly detection.
- Legacy speech recognition and optical character recognition (OCR) systems.

### Follow-up Interview Questions:
1. *Why does bidirectional LSTM double the hidden dimension size?* (Answer: BiLSTM concatenates the forward hidden state $\vec{h}_t$ and backward hidden state $\overleftarrow{h}_t$, doubling output channels).
2. *When would you choose GRU over LSTM?* (Answer: When training on small datasets with limited GPU memory; GRU has fewer parameters and trains faster while matching LSTM accuracy).
