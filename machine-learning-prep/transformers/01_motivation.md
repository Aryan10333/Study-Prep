# Transformers: Motivation & Sequence Modeling Evolution

This guide details the evolution of sequence models, tracing NLP from Rule-based systems to Statistical NLP, Static Embeddings (Word2Vec, GloVe), Recurrent Neural Networks (RNNs, LSTMs), Seq2Seq models, and the attention-driven Transformer paradigm.

---

## 1. Complete Evolution of NLP & Sequence Models

Natural Language Processing (NLP) evolved across five distinct paradigms to solve sequence representation, context modeling, and computational scalability.

```text
Era & Paradigm            Key Mechanism                        Primary Limitation
----------------------------------------------------------------------------------------------------------------------
1. Rule-Based NLP         Regular Expressions, Hand-coded CFG Rigid syntax rules, zero generalization to novel text
2. Statistical NLP        N-grams, TF-IDF, Naive Bayes         Curse of dimensionality, zero semantic similarity
3. Static Embeddings      Word2Vec (CBOW/Skip-Gram), GloVe     Polysemy blind (Bank = Financial vs Bank = Riverbank)
4. Recurrent (RNN/LSTM)   Sequential State Updates             Sequential O(N) bottleneck, vanishing gradients
5. Transformer / LLMs     Self-Attention Parallel Processing   O(N^2) quadratic memory (solvable via FlashAttention)
```

### A. Rule-Based & Statistical NLP
- **Rule-Based Systems**: Depended on manually written Context-Free Grammars (CFGs) and regex patterns. Fails on ungrammatical, real-world text.
- **Statistical NLP (N-grams & TF-IDF)**:
  - **N-gram Language Models**: Estimate word sequence probability via Markov assumption:
    $$P(w_1, w_2, \dots, w_N) \approx \prod_{i=1}^N P(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**:
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left( \frac{|D|}{|\{d \in D : t \in d\}|} \right)$$
  - *Limitation:* Treats words as orthogonal one-hot vectors ($v_{\text{cat}} \perp v_{\text{feline}}$), completely missing semantic similarity.

### B. Static Word Embeddings: Word2Vec, GloVe & FastText
Introduced low-dimensional continuous vector spaces $\mathbb{R}^d$ ($d \approx 300$) where semantically similar words sit close together.
- **Word2Vec (Mikolov et al., 2013)**:
  - **Continuous Bag-of-Words (CBOW)**: Predicts center word $w_t$ given context words $w_{t-c}, \dots, w_{t+c}$.
  - **Skip-Gram**: Predicts surrounding context words given center word $w_t$:
    $$\mathcal{L}_{\text{SkipGram}} = \sum_{t=1}^T \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} \mid w_t)$$
  - **Softmax Optimization**: Trained using **Negative Sampling** to avoid calculating the full $|V|$ denominator.
- **GloVe (Global Vectors for Word Representation)**:
  - Factorizes the global word-word co-occurrence matrix $X_{i,j}$:
    $$\mathcal{L}_{\text{GloVe}} = \sum_{i,j=1}^{|V|} f(X_{i,j}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{i,j} \right)^2$$
- **FastText**: Represents words as bags of character $n$-grams (e.g., `<wh`, `whe`, `her`, `ere`, `re>`), allowing representation of Out-Of-Vocabulary (OOV) words.
- *Fatal Limitation:* **Polysemy**. Static embeddings assign a single static vector per word, unable to distinguish *"river bank"* from *"financial bank"*.

---

## 2. Limitations of Recurrent Architectures (RNNs, LSTMs, Seq2Seq)

### A. The Sequential Compute Bottleneck
Recurrent architectures must update their hidden states step-by-step:
$$h_t = \tanh\left( W_{hh} h_{t-1} + W_{xh} x_t + b_h \right)$$
- **Why it is a systems bottleneck:** Because hidden state $h_t$ requires the output of hidden state $h_{t-1}$, training cannot be parallelized along the sequence dimension. Modern GPU hardware contains thousands of parallel processing cores. During recurrent updates, worker cores sit idle waiting for sequential timesteps to resolve, resulting in low hardware utilization and slow training throughput.

### B. Linear Information Bottleneck (LSTMs & Seq2Seq)
LSTMs improve on RNNs by introducing a cell state $C_t$ to convey long-term memory:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
- **The Information Bottleneck:** In Seq2Seq encoder-decoder setups, the encoder must compress an entire input sentence into a single final vector $h_T \in \mathbb{R}^d$. As sequence length $T$ increases, this fixed vector saturates, causing the model to forget early details.

---

## 3. Step-by-Step Hand Calculations: Gradient Decay (Andrew Ng Style)

Let's calculate the gradient transition across a recurrent sequence of length $T=3$ vs. $T=10$ to demonstrate how vanishing gradients occur.
- Assume a simplified, single-neuron scalar RNN hidden update: $h_t = \tanh(w \cdot h_{t-1} + x_t)$
- Let the recurrent weight be $w = 0.5$.
- Assume inputs are saturated such that the derivative of the activation function is approximately $\tanh'(z) \approx 1.0$.

### 1. Gradient Propagation for $T=3$ steps
To update the initial parameter weight based on the loss at step 3, we compute the derivative of hidden state $h_3$ with respect to the initial hidden state $h_0$ using the chain rule:
$$\frac{\partial h_3}{\partial h_0} = \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial h_0}$$
Since $\frac{\partial h_j}{\partial h_{j-1}} = w \cdot \tanh'(z_j) \approx w$:
$$\frac{\partial h_3}{\partial h_0} \approx w \cdot w \cdot w = w^3 = (0.5)^3 = \mathbf{0.125}$$
- *Result:* The gradient retains **$12.5\%$** of its original signal.

### 2. Gradient Propagation for $T=10$ steps
$$\frac{\partial h_{10}}{\partial h_0} \approx w^{10} = (0.5)^{10} = \frac{1}{1024} \approx \mathbf{0.0009765}$$
- *Result:* The gradient retains only **$0.097\%$** of its signal, rendering early sequence steps unlearnable.

---

## 4. Why Attention & Transformers Replaced Recurrent Models

```text
Feature                  RNN / LSTM                           Transformer
----------------------------------------------------------------------------------------------------------------------
Sequence Operations      O(N) Sequential                      O(1) Direct Self-Attention
Maximum Path Length      O(N) Steps                           O(1) Step
Training Parallelism     No (Blocked by h_{t-1})              Yes (Full Matrix Multiplication)
Long-Range Memory        Decays exponentially with distance   Preserved uniformly across entire sequence
```

By allowing every token to directly query every other token in parallel via Self-Attention matrices ($Q K^T / \sqrt{d_k}$), Transformers solved both the computational training bottleneck and long-range memory degradation.
