# NLP Fundamentals Interview Revision Cheatsheet

This cheatsheet provides a high-density, 1-page reference of core formulas, computational complexities, variable legends, and troubleshooting configurations for technical interviews.

---

## 1. Component Variable Denotation Legend

| Symbol | Definition | Symbol | Definition |
| :--- | :--- | :--- | :--- |
| $N$ | Total number of documents in the corpus | $d$ | Model hidden / embedding dimension size |
| $L$ | Token sequence length (context window) | $K$ | Number of negative samples in SGNS |
| $|V|$ | Vocabulary size (number of unique tokens) | $C(x)$ | Count/frequency of token $x$ |
| $c$ | Candidate token sequence length (metrics) | $r$ | Reference token sequence length (metrics) |
| $k_1$ | BM25 term frequency saturation scaling parameter | $b$ | BM25 document length penalty parameter |

---

## 2. Computational Complexity Matrix

| Algorithm | Training Time | Inference Time | Space (Memory) |
| :--- | :--- | :--- | :--- |
| **Bag-of-Words** | $O(N \cdot L)$ | $O(L)$ | $O(N \cdot |V|)$ sparse matrix |
| **TF-IDF Vectorization** | $O(N \cdot L + |V|)$ | $O(L)$ | $O(N \cdot |V|)$ sparse matrix |
| **N-gram Language Model** | $O(N \cdot L)$ | $O(1)$ lookup | $O(|V|^N)$ transition table |
| **Recurrent Neural Net (RNN)** | $O(L \cdot d^2)$ | $O(L \cdot d^2)$ (Sequential) | $O(d^2)$ parameter weights |
| **Long Short-Term Memory (LSTM)** | $O(L \cdot d^2)$ | $O(L \cdot d^2)$ (Sequential) | $O(4 \cdot d^2)$ parameter weights |
| **Gated Recurrent Unit (GRU)** | $O(L \cdot d^2)$ | $O(L \cdot d^2)$ (Sequential) | $O(3 \cdot d^2)$ parameter weights |
| **Scaled Dot-Product Attention** | $O(L^2 \cdot d)$ | $O(L^2 \cdot d)$ (Parallel) | $O(L^2)$ intermediate attention matrix |
| **BLEU-N Evaluation** | - | $O(c \cdot r)$ | $O(c)$ tokens |
| **BERTScore Evaluation** | - | $O(c \cdot r \cdot d) + \text{Forward Pass}$ | $O(c \cdot r)$ similarity table |

---

## 3. Master Formula Reference

### Vector Semantics & Retrieval:
- **Smooth IDF**: $\text{IDF}(t, D) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$
- **$L_2$ Normalization**: $\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$
- **Okapi BM25 Score**: $\text{Score}(d, q) = \sum_{t \in q} \ln\left(\frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5} + 1\right) \times \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$

### Statistical & Smoothing Models:
- **Laplace (Add-One) Bigram**: $P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$
- **Sequence Perplexity**: $\text{PPL}(W) = P(w_1, \dots, w_m)^{-\frac{1}{m}} = e^{\mathcal{L}_{\text{CE}}}$

### Continuous Word Embeddings:
- **SGNS Loss**: $\mathcal{L}_{\text{SGNS}} = -\ln \sigma(\mathbf{v}_w \cdot \mathbf{v}'_{w_c}) - \sum_{i=1}^K \ln \sigma(-\mathbf{v}_w \cdot \mathbf{v}'_{w_i})$
- **Noise Sampling Distribution**: $P_n(w) \propto U(w)^{0.75}$

### Sequence Models & Attention:
- **LSTM CEC derivative**: $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx \mathbf{f}_t \approx \mathbf{1}$ (when forget gate is open, bypassing recurrent weight vanishing decay)
- **Scaled Dot-Product Attention**: $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$
- **BLEU Brevity Penalty**: $\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$

### Production Drift Monitoring:
- **Population Stability Index**: $\text{PSI} = \sum_{i=1}^B (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$ (stable: $<0.1$, moderate: $0.1\text{--}0.25$, high: $>0.25$)

---

## 4. Top 5 Critical Troubleshooting Guidelines

1. **Vanishing Gradients in Recurrent State Paths**:
   - *Problem*: Backpropagated gradients vanish due to multiplicative chain products ($W_{hh}^{T-t}$).
   - *Fix*: Use LSTMs/GRUs where cell state CEC updates are additive, or switch to Transformer self-attention.
2. **Softmax Saturation in Self-Attention**:
   - *Problem*: For large key sizes $d_k$, query-key dot products blow up, pushing Softmax inputs into flat regions where derivative is near 0.
   - *Fix*: Divide attention scores by $\sqrt{d_k}$ to scale variance to $1.0$, keeping gradients sensitive.
3. **Exposure Bias in Seq2Seq Models**:
   - *Problem*: Decoders are trained using Teacher Forcing (feeding true tokens), but generate auto-regressively at inference (feeding model's own outputs), compounding errors.
   - *Fix*: Apply Scheduled Sampling (randomly feeding model predictions during training) or use sequence-level reinforcement learning objectives.
4. **Data (Covariate) Shift vs. Concept Drift**:
   - *Problem*: Model accuracy decays in production because vocabulary characteristics shift ($P(X_{\text{prod}}) \neq P(X_{\text{train}})$) or meanings shift ($P(Y \mid X_{\text{prod}}) \neq P(Y \mid X_{\text{train}})$).
   - *Fix*: Monitor Population Stability Index (PSI) or Wasserstein Distance on live logs; trigger retrain loops when PSI $> 0.25$.
5. **Vocabulary Serving Crash**:
   - *Problem*: Text normalization and vocabulary indexing tables mismatch between the preprocessing serving API and model container, leading to corrupted embeddings.
   - *Fix*: Package tokenizer configurations and token dictionaries directly as read-only assets in the model registry alongside model weights.
