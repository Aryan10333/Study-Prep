# Module 06: Sequence Models & Recurrent Architectures

Sequence models process variable-length inputs by maintaining sequential state representations. This module details recurrent architectures, explains the mechanics of vanishing gradients, details LSTM cell gating, and covers sequence-to-sequence translation models.

---

## 1. Recurrent State Mechanics & Parameter Sharing

A standard Recurrent Neural Network (RNN) processes tokens sequentially, updating a hidden state vector $h_t$ at each step.

### RNN Unfolded Computational Graph

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center; gap: 15px;">
  <div style="font-weight: bold; font-size: 14px; color: #1e3a8a; text-transform: uppercase;">RNN Step-by-Step Computational Flow</div>
  
  <div style="display: flex; align-items: center; gap: 15px; width: 100%; justify-content: center;">
    <!-- Step t-1 -->
    <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; text-align: center; width: 100px;">
      <div style="font-size: 10px; color: #64748b; font-weight: bold;">Prev Hidden</div>
      <div style="font-family: monospace; font-size: 14px; font-weight: bold; color: #0f172a; margin: 4px 0;">h_{t-1}</div>
    </div>
    
    <div style="font-size: 16px; color: #64748b;">+</div>
    
    <!-- Input t -->
    <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; text-align: center; width: 100px;">
      <div style="font-size: 10px; color: #64748b; font-weight: bold;">Input Token</div>
      <div style="font-family: monospace; font-size: 14px; font-weight: bold; color: #3b82f6; margin: 4px 0;">x_t</div>
    </div>
    
    <div style="font-size: 18px; color: #64748b;">&rarr;</div>
    
    <!-- RNN Cell (tanh) -->
    <div style="background-color: #3b82f6; color: white; border-radius: 8px; padding: 12px; text-align: center; width: 150px; box-shadow: 0 2px 4px rgba(59,130,246,0.15);">
      <div style="font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">RNN Cell</div>
      <div style="font-family: monospace; font-size: 12px; font-weight: bold; margin: 4px 0;">tanh(W_hh * h_{t-1} + W_xh * x_t + b)</div>
    </div>
    
    <div style="font-size: 18px; color: #64748b;">&rarr;</div>
    
    <!-- Step t Output -->
    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 10px; text-align: center; width: 100px;">
      <div style="font-size: 10px; color: #1e40af; font-weight: bold;">New Hidden</div>
      <div style="font-family: monospace; font-size: 14px; font-weight: bold; color: #1e3a8a; margin: 4px 0;">h_t</div>
    </div>
  </div>
  
  <div style="color: #64748b; font-size: 11px; text-align: center; line-height: 1.4;">
    <strong>Parameter Sharing:</strong> The same projection matrices <strong>W_hh</strong>, <strong>W_xh</strong>, and bias <strong>b</strong> are used at every step <em>t</em>.
  </div>
</div>

### RNN Hidden State Equation
At each step $t$, the RNN cell takes the current input vector $\mathbf{x}_t \in \mathbb{R}^d$ and the previous hidden state vector $\mathbf{h}_{t-1} \in \mathbb{R}^{d_h}$, and computes the updated hidden state $\mathbf{h}_t \in \mathbb{R}^{d_h}$ using:

$$\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + b_h)$$

#### Algorithmic Breakdown & Intuition:
- **$W_{hh} \mathbf{h}_{t-1}$ (State Transition):** Projects the previous hidden state (the model's memory of all tokens processed up to step $t-1$) into the current hidden space. This term preserves historical context.
- **$W_{xh} \mathbf{x}_t$ (Input Projection):** Projects the current input word representation (e.g. from the embedding lookup layer) into the hidden space. This term registers the new word.
- **$\tanh$ Activation:** The hyperbolic tangent activation function squeezes all activations to the range $[-1, 1]$. This is a scaling boundary that prevents the magnitude of the hidden state vectors from growing infinitely at each step.
- **Insight:** The recurrent state is a continuous blend of the past context ($h_{t-1}$) and the current observation ($x_t$), allowing the model to carry historical traces across variable sequence lengths.

---

## 2. Why RNN Gradients Vanish or Explode

To train an RNN, we backpropagate errors through the unfolded graph (Backpropagation Through Time, BPTT). The gradient flow from a distant loss at time step $T$ to the hidden state at step $t$ scales with the recurrent weight matrix $W_{hh}$:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \propto (W_{hh})^{T-t}$$

### Intuition & Failure Modes:
- **Vanishing Gradients:** If the weights in $W_{hh}$ are small (specifically, if its largest eigenvalue is $< 1$), repeatedly multiplying this matrix over a long time gap $T-t$ causes the gradient to decay exponentially to 0. Consequently, the model cannot learn long-term dependencies from early tokens in a sentence.
- **Exploding Gradients:** If the weights in $W_{hh}$ are large (largest eigenvalue $> 1$), repeatedly multiplying this matrix causes the gradient to grow exponentially, leading to weight updates that diverge (`NaN` losses) and training instability.

---

## 3. LSTM Gating Mechanics & The Constant Error Carousel (CEC)

Long Short-Term Memory (LSTM) networks resolve the vanishing gradient problem by splitting the hidden representation into two vectors:
1. **Cell State ($C_t$):** A linear conveyor belt that stores long-term memory, modified only by additive updates.
2. **Hidden State ($h_t$):** A short-term context vector derived from the cell state, used to predict outputs and gate inputs.

### LSTM Cell Architecture Diagram

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center; gap: 15px;">
  <div style="font-weight: bold; font-size: 14px; color: #7c3aed; text-transform: uppercase;">LSTM Cell Internal Routing</div>
  
  <div style="width: 100%; border: 1px dashed #cbd5e1; border-radius: 6px; padding: 15px; background-color: #ffffff; display: flex; flex-direction: column; gap: 12px;">
    <!-- Cell State Conveyor Belt -->
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 8px 12px; border-radius: 6px;">
      <div style="font-family: monospace; font-size: 12px; font-weight: bold; color: #16a34a;">C_{t-1} (Prev Cell)</div>
      <div style="font-size: 16px; color: #16a34a;">&rarr; [ &times; Forget Gate (f_t) ] &rarr; [ + Input Gate * Candidate (i_t * C~_t) ] &rarr;</div>
      <div style="font-family: monospace; font-size: 12px; font-weight: bold; color: #16a34a;">C_t (New Cell)</div>
    </div>
    
    <!-- Gates calculation -->
    <div style="display: flex; gap: 10px; justify-content: space-around;">
      <div style="background-color: #fef2f2; border: 1px solid #fca5a5; padding: 6px 10px; border-radius: 4px; font-size: 11px; text-align: center; width: 100px;">
        <div style="font-weight: bold; color: #dc2626;">Forget Gate (f_t)</div>
        <div style="font-family: monospace; font-size: 10px; margin-top: 4px;">&sigma;(W_f * [h_{t-1}, x_t])</div>
      </div>
      <div style="background-color: #eff6ff; border: 1px solid #93c5fd; padding: 6px 10px; border-radius: 4px; font-size: 11px; text-align: center; width: 100px;">
        <div style="font-weight: bold; color: #2563eb;">Input Gate (i_t)</div>
        <div style="font-family: monospace; font-size: 10px; margin-top: 4px;">&sigma;(W_i * [h_{t-1}, x_t])</div>
      </div>
      <div style="background-color: #fff7ed; border: 1px solid #ffedd5; padding: 6px 10px; border-radius: 4px; font-size: 11px; text-align: center; width: 110px;">
        <div style="font-weight: bold; color: #ea580c;">Candidate (~C_t)</div>
        <div style="font-family: monospace; font-size: 10px; margin-top: 4px;">tanh(W_c * [h_{t-1}, x_t])</div>
      </div>
      <div style="background-color: #faf5ff; border: 1px solid #e9d5ff; padding: 6px 10px; border-radius: 4px; font-size: 11px; text-align: center; width: 100px;">
        <div style="font-weight: bold; color: #9333ea;">Output Gate (o_t)</div>
        <div style="font-family: monospace; font-size: 10px; margin-top: 4px;">&sigma;(W_o * [h_{t-1}, x_t])</div>
      </div>
    </div>
    
    <!-- Hidden State output -->
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #f5f3ff; border: 1px solid #ddd6fe; padding: 8px 12px; border-radius: 6px; margin-top: 4px;">
      <div style="font-family: monospace; font-size: 12px; font-weight: bold; color: #6d28d9;">h_{t-1} (Prev Hidden) + x_t</div>
      <div style="font-size: 14px; color: #6d28d9;">&rarr; [ Calculate Gates ] &rarr; [ h_t = o_t * tanh(C_t) ] &rarr;</div>
      <div style="font-family: monospace; font-size: 12px; font-weight: bold; color: #6d28d9;">h_t (New Hidden)</div>
    </div>
  </div>
</div>

### LSTM Gating Equations
At step $t$, the LSTM updates its state vectors using four gating mechanisms:

$$f_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_f) \quad \text{(Forget Gate)}$$

$$i_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_i) \quad \text{(Input Gate)}$$

$$\tilde{\mathbf{C}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_c) \quad \text{(Candidate Cell State)}$$

$$\mathbf{C}_t = f_t \odot \mathbf{C}_{t-1} + i_t \odot \tilde{\mathbf{C}}_t \quad \text{(Cell State Update)}$$

$$o_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_o) \quad \text{(Output Gate)}$$

$$\mathbf{h}_t = o_t \odot \tanh(\mathbf{C}_t) \quad \text{(Hidden State Update)}$$

#### Algorithmic Breakdown & Intuition:
- **Forget Gate ($f_t$):** The forget gate output is a vector of values between 0 and 1. It acts as a **reset button**. It determines what portion of the previous cell memory $C_{t-1}$ to retain. If $f_t = 0$, the corresponding memory is wiped clean (useful when transitioning to a new topic or sentence). If $f_t = 1$, the history is perfectly preserved.
- **Input Gate ($i_t$) and Candidate State ($\tilde{C}_t$):**
  - The Candidate State $\tilde{C}_t$ creates a new vector of information extracted from the current input $x_t$ and the previous hidden state $h_{t-1}$ using a $\tanh$ activation.
  - The Input Gate $i_t$ acts as a **write-volume knob** (from 0 to 1) deciding *which dimensions of the candidate vector* are written into the long-term cell state.
- **Cell State Update ($\mathbf{C}_t$):** This combines the filtered past memory ($f_t \odot \mathbf{C}_{t-1}$) and the gated new memory ($i_t \odot \tilde{\mathbf{C}}_t$). Because this combining operation is **additive ($+$)** rather than multiplicative, the gradient flows back through time without exponential decay.
- **Output Gate ($o_t$) & Hidden State ($\mathbf{h}_t$):** The output gate decides *what parts of the updated cell state to reveal* as the hidden state $h_t$. The cell state $C_t$ is squeezed via a $\tanh$ (to keep its values bounded) and multiplied element-wise by the output gate $o_t$. The hidden state $h_t$ is then passed as the output for the current time step and recycled to the next step.

### The Constant Error Carousel (CEC) Intuition
Because the cell state update uses **addition ($+$)** instead of multiplication ($*$), the derivative of $C_t$ with respect to $C_{t-1}$ contains the linear forget gate term $f_t$:

$$\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$$

If the forget gate is open ($f_t \approx 1$), the gradient flows back continuously without exponential decay:

$$\frac{\partial C_T}{\partial C_t} \approx 1$$

This linear shortcut allows the gradient to flow backward through time indefinitely, creating the **Constant Error Carousel** that resolves the vanishing gradient problem.

---

## 4. Gated Recurrent Unit (GRU) Variations

The Gated Recurrent Unit (GRU; Cho et al., 2014) is a simplified variant of the LSTM. It consolidates states and gates to reduce the parameter footprint:
- **Single Hidden State ($h_t$):** GRU merges the cell state and hidden state into a single representation $\mathbf{h}_t \in \mathbb{R}^{d_h}$.
- **Consolidated Gates:** It combines the forget and input gates into a single **update gate** $z_t$, and uses a **reset gate** $r_t$ to control past context flow.

### GRU State Update Equations
At each step $t$, the GRU computes:

$$z_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_z) \quad \text{(Update Gate)}$$

$$r_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_r) \quad \text{(Reset Gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h [r_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + b_h) \quad \text{(Candidate Hidden State)}$$

$$\mathbf{h}_t = (1 - z_t) \odot \mathbf{h}_{t-1} + z_t \odot \tilde{\mathbf{h}}_t \quad \text{(Hidden State Update)}$$

#### Algorithmic Breakdown & Intuition:
- **Update Gate ($z_t$):** Decides how much of the historical hidden state $\mathbf{h}_{t-1}$ to retain versus how much of the new candidate state $\tilde{\mathbf{h}}_t$ to write. It acts as both a forget gate (scale $1-z_t$) and an input gate (scale $z_t$) simultaneously.
- **Reset Gate ($r_t$):** Determines *how much of the past memory to ignore* when constructing the candidate state. If $r_t = 0$, the model completely wipes out history, treating the current input $\mathbf{x}_t$ as the start of a new segment.
- **Candidate Hidden State ($\tilde{\mathbf{h}}_t$):** Evaluates the candidate update vector. The previous hidden state is scaled by the reset gate ($r_t \odot \mathbf{h}_{t-1}$), letting the model selectively filter out irrelevant context before applying the tanh projection.
- **Hidden State Update ($\mathbf{h}_t$):** Linearly interpolates between the previous hidden state and the candidate hidden state.

#### Production Trade-offs (Interview Insight)
By utilizing only $3$ sets of gate projections instead of the LSTM's $4$, the GRU has **$33\%$ fewer parameters** than standard LSTM cells. This makes GRUs computationally faster to train and less memory-intensive on GPUs, while providing comparable performance on small-to-medium sequence lengths.

---

## 5. Bidirectional RNNs and LSTMs

Standard recurrent models only process text from left to right, meaning the hidden state $\mathbf{h}_t$ cannot incorporate future context. Bidirectional models run two independent recurrent streams:
1. **Forward Stream:** Processes the sequence from left to right, computing $\vec{\mathbf{h}}_t$:
   $$\vec{\mathbf{h}}_t = \text{LSTM}_{\text{forward}}(\mathbf{x}_t, \vec{\mathbf{h}}_{t-1})$$
2. **Backward Stream:** Processes the sequence from right to left (end of sentence to start), computing $\overleftarrow{\mathbf{h}}_t$:
   $$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{backward}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

The outputs of both streams are concatenated at each time step $t$ to form the final contextual hidden state:
$$\mathbf{h}_t = [\vec{\mathbf{h}}_t \ ; \ \overleftarrow{\mathbf{h}}_t]$$

#### Production Limits (Interview Insight)
- **Strengths:** By incorporating both past and future words, bidirectional models generate highly rich feature vectors for words. This is the architectural foundation of encoders like BERT and Named Entity Recognition (NER) taggers.
- **Critical Generation Bottleneck:** **Bidirectional recurrent models cannot be used for autoregressive text generation** (predicting the next word in a sequence). Predicting word $y_t$ during generation requires accessing $\overleftarrow{\mathbf{h}}_t$, which depends on future words $y_{t+1}, y_{t+2}, \dots$ that have not been generated yet.

---

## 6. Sequence-to-Sequence (Seq2Seq) and Decoding Strategies

Sequence-to-Sequence (Seq2Seq) models (Sutskever et al., 2014) convert an input sequence of arbitrary length into an output sequence of arbitrary length (e.g. English to French translation). The system is split into two components:
1. **The Encoder:** Processes the input tokens and compresses the entire sequence into a single **context vector** $\mathbf{v}$ (often the final hidden state of the encoder).
2. **The Decoder:** Unpacks the context vector $\mathbf{v}$ auto-regressively, generating one target token at a time.

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center; gap: 10px;">
  <div style="font-weight: bold; font-size: 14px; color: #1e3a8a; text-transform: uppercase;">Seq2Seq Encoder-Decoder Context Vector Bottleneck</div>
  
  <svg width="600" height="190" viewBox="0 0 600 190" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%;">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 7 5 L 0 8.5 z" fill="#64748b"/>
      </marker>
    </defs>
    
    <!-- ENCODER GROUP -->
    <rect x="70" y="10" width="80" height="155" rx="6" fill="none" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>
    <text x="110" y="180" font-family="sans-serif" font-size="10" font-weight="bold" fill="#64748b" text-anchor="middle">ENCODER</text>
    
    <!-- Inputs x -->
    <text x="30" y="45" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">x_1</text>
    <text x="30" y="95" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">x_2</text>
    <text x="30" y="145" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">x_3</text>
    
    <!-- Encoder LSTMs -->
    <rect x="80" y="25" width="60" height="30" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="110" y="44" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">LSTM</text>
    
    <rect x="80" y="75" width="60" height="30" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="110" y="94" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">LSTM</text>
    
    <rect x="80" y="125" width="60" height="30" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="110" y="144" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">LSTM</text>
    
    <!-- Lines Input -> Encoder -->
    <line x1="55" y1="40" x2="73" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
    <line x1="55" y1="90" x2="73" y2="90" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
    <line x1="55" y1="140" x2="73" y2="140" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
    
    <!-- CONTEXT VECTOR BOTTLENECK -->
    <rect x="230" y="58" width="140" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
    <text x="300" y="82" font-family="sans-serif" font-size="11" font-weight="bold" fill="#7c2d12" text-anchor="middle">Context Vector (v)</text>
    <text x="300" y="102" font-family="sans-serif" font-size="9" font-weight="bold" fill="#ea580c" text-anchor="middle">BOTTLENECK</text>
    
    <!-- Converging lines Encoder -> Context -->
    <path d="M 140 40 L 222 78" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    <path d="M 140 90 L 222 90" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    <path d="M 140 140 L 222 102" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    
    <!-- DECODER GROUP -->
    <rect x="450" y="10" width="80" height="155" rx="6" fill="none" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>
    <text x="490" y="180" font-family="sans-serif" font-size="10" font-weight="bold" fill="#64748b" text-anchor="middle">DECODER</text>
    
    <!-- Decoder LSTMs -->
    <rect x="460" y="25" width="60" height="30" rx="4" fill="#faf5ff" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="490" y="44" font-family="sans-serif" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">LSTM</text>
    
    <rect x="460" y="75" width="60" height="30" rx="4" fill="#faf5ff" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="490" y="94" font-family="sans-serif" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">LSTM</text>
    
    <rect x="460" y="125" width="60" height="30" rx="4" fill="#faf5ff" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="490" y="144" font-family="sans-serif" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">LSTM</text>
    
    <!-- Diverging lines Context -> Decoder -->
    <path d="M 370 78 L 452 40" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    <path d="M 370 90 L 452 90" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    <path d="M 370 102 L 452 140" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
    
    <!-- Outputs y -->
    <text x="550" y="45" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">y_1</text>
    <text x="550" y="95" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">y_2</text>
    <text x="550" y="145" font-family="monospace" font-size="14" font-weight="bold" fill="#0f172a">y_3</text>
    
    <!-- Lines Decoder -> Output -->
    <line x1="520" y1="40" x2="538" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
    <line x1="520" y1="90" x2="538" y2="90" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
    <line x1="520" y1="140" x2="538" y2="140" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>
  </svg>
</div>

### The Encoder-Decoder Bottleneck (Failure Mode)
For long input sentences, compressing all information into a single fixed-size context vector $\mathbf{v}$ creates an **information bottleneck**. The encoder inevitably loses early details (e.g., the subject of a long sentence), causing the decoder to generate incorrect translations. This failure mode motivated the development of **Attention Mechanisms** (Module 07).

---

### Decoder Training vs. Inference: Teacher Forcing
- **Decoder Generation Loop:** At step $t$, the decoder predicts the next token $y_t$ based on the context vector $\mathbf{v}$ and the sequence of previously generated tokens:
  $$P(y_t \mid y_{<t}, \mathbf{v})$$
- **Teacher Forcing (Training):** During training, instead of feeding the decoder's own predicted token $\hat{y}_{t-1}$ as input to the next step, we feed the **ground-truth target token** $y^*_{t-1}$ directly.
  - *Why it's used:* Accelerates training convergence. Without teacher forcing, an early error by the model would cascade, making all subsequent predictions in the sequence wrong, and preventing the model from learning from correct examples.
  - *Exposure Bias (Production Caveat):* Creates a mismatch between training and inference. At inference time, the model has no ground truth and must feed its own predictions back into itself. If it makes an early mistake, the error propagates, leading to repetitive or nonsensical generation loops.

---

### Decoding Strategies at Inference

Once the model predicts probability distributions over the vocabulary at step $t$, we must select the token sequence.

#### 1. Greedy Search
Selects the single token with the highest probability at each step:
$$y_t = \arg\max P(y \mid y_{<t}, \mathbf{v})$$
- *Trade-off:* Computes fast ($O(1)$ search width), but is sub-optimal. If the model makes a mistake at step 1, it cannot backtrack, locking it into a poor generation path.

#### 2. Beam Search
Maintains the $B$ most likely hypothesis sequences (beams) at each step to explore a larger search space without exponential scaling.

##### Step-by-Step Example with Beam Width $B=2$:
Let our vocabulary be `{"the", "cat", "sat"}`.
- **Step 1:** Decode the first token. The model produces probabilities:
  - `"the"` ($0.6$), `"cat"` ($0.3$), `"sat"` ($0.1$)
  - Keep top $B=2$ beams: `[("the", log(0.6))]` and `[("cat", log(0.3))]`.
- **Step 2:** Expand both beams to find all possible next tokens:
  - Beam 1 (`"the"`):
    - `"the the"`: $\text{score} = \log(0.6) + \log(0.1) = -0.51 + (-2.30) = -2.81$
    - `"the cat"`: $\text{score} = \log(0.6) + \log(0.7) = -0.51 + (-0.36) = -0.87$
    - `"the sat"`: $\text{score} = \log(0.6) + \log(0.2) = -0.51 + (-1.61) = -2.12$
  - Beam 2 (`"cat"`):
    - `"cat the"`: $\text{score} = \log(0.3) + \log(0.2) = -1.20 + (-1.61) = -2.81$
    - `"cat cat"`: $\text{score} = \log(0.3) + \log(0.1) = -1.20 + (-2.30) = -3.50$
    - `"cat sat"`: $\text{score} = \log(0.3) + \log(0.7) = -1.20 + (-0.36) = -1.56$
- **Pruning Step:** Sort all 6 candidates and retain only the top $B=2$:
  - Beams kept: `["the cat" (-0.87)]` and `["cat sat" (-1.56)]`.

This ensures the model keeps highly correlated sequences, preventing early decoding mistakes.

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

