# Part 4: NLP Interview Cheatsheet (8 Essential Must-Know Foundations)

A high-density revision reference for AI, Applied AI, GenAI, and ML Engineer interviews covering the 8 essential formulas, core diagrams, time & memory complexity, interview Q&A, and a 1-page comparison table.

---

## 1. The 8 Must-Know Essential Formulas

| # | Topic | Must-Know Mathematical Formula |
|---|---|---|
| **1** | **TF-IDF Vectorization** | $\text{TF}(t, d) = \frac{\text{count}(t, d)}{\vert d \vert}, \quad \text{IDF}(t) = \log\left(\frac{1+\vert D \vert}{1+\text{DF}(t)}\right)+1, \quad \text{TF-IDF} = \text{TF} \times \text{IDF}$ |
| **2** | **Cosine Similarity** | $\text{CosineSim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ |
| **3** | **Naive Bayes (+ Laplace)** | $P(c \mid d) \propto P(c) \prod P(w_i \mid c), \quad P(w_i \mid c) = \frac{N_{c,i} + 1}{N_c + \vert V \vert}$ |
| **4** | **Word2Vec (Skip-Gram)** | $P(w_O \mid w_I) = \sigma({v'_{w_O}}^\top v_{w_I}) = \frac{1}{1 + \exp(- {v'_{w_O}}^\top v_{w_I})}$ |
| **5** | **LSTM Cell Updates** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \quad h_t = o_t \odot \tanh(C_t)$ |
| **6** | **Scaled Dot Attention** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ |
| **7** | **BLEU-2 Score** | $\text{BLEU-2} = \text{BP} \cdot \exp(0.5 \log p_1 + 0.5 \log p_2), \quad \text{BP} = \exp(1 - r/c) \text{ if } c \le r \text{ else } 1.0$ |
| **8** | **Perplexity (PPL)** | $\text{PPL} = \exp(\mathcal{L}_{\text{CE}}) = \left( \prod P(w_i \mid w_{<i}) \right)^{-1/N}$ |

---

## 2. Time & Memory Complexity Matrix

| Algorithm / Architecture | Training Time Complexity | Inference Time Complexity | Memory Footprint Complexity |
|---|---|---|---|
| **TF-IDF Indexing** | $O(\vert D \vert \times L)$ | $O(k \times \vert V \vert)$ | $O(\vert D \vert \times \vert V \vert)$ Sparse |
| **Multinomial Naive Bayes** | $O(\vert D \vert \times L)$ | $O(C \times L)$ | $O(C \times \vert V \vert)$ Dense |
| **Word2Vec Skip-Gram** | $O(C \cdot m \cdot K \cdot d)$ | $O(1)$ Lookup | $O(\vert V \vert \times d)$ Embedding Table |
| **LSTM Model Layer** | $O(T \cdot d^2)$ per sequence | $O(T \cdot d^2)$ Sequential | $O(4 \cdot d^2)$ Model Weights |
| **Scaled Dot-Product Attention** | $O(T^2 \cdot d)$ per batch | $O(T^2 \cdot d)$ | $O(T^2)$ RAM Attention Weights |
| **INT8 Quantized Serving** | $O(N \cdot d)$ SIMD | $O(N \cdot d)$ SIMD | $O(\frac{1}{4} M_{\text{FP32}})$ Bytes |

---

## 3. High-Frequency Interview Q&A Summary

1. **Why do we scale $QK^\top$ by $\sqrt{d_k}$ in Attention?**
   - *Answer*: Large $d_k$ causes dot products $q \cdot k$ to have variance $d_k$. Large dot products push Softmax into saturated regions with near-zero gradients ($\text{softmax}'(z) \rightarrow 0$), causing vanishing gradients during backpropagation. Dividing by $\sqrt{d_k}$ rescales variance back to $1.0$.
2. **Why does Laplace Smoothing prevent zero probability in Naive Bayes?**
   - *Answer*: Unseen vocabulary words in a training class yield raw likelihood $P(w_{\text{new}} \mid c) = 0$, causing the entire posterior product to collapse to zero. Adding $+1$ ensures every word retains a positive non-zero likelihood $P(w_i \mid c) = \frac{N_{c,i}+1}{N_c + |V|}$.
3. **How does the LSTM cell state $C_t$ resolve vanishing gradients?**
   - *Answer*: Cell update $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ is additive. When forget gate $f_t = 1$, the gradient derivative $\frac{\partial C_t}{\partial C_{t-1}} = 1$, preserving gradient norm $\approx 1.0$ across long sequences.
4. **Why is lower Perplexity better?**
   - *Answer*: $\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$. Perplexity represents the average branching factor of uncertainty when predicting the next token. A lower PPL means fewer candidate choices the model is uncertain between.
5. **Why is BLEU precision-oriented while ROUGE is recall-oriented?**
   - *Answer*: BLEU checks what fraction of generated candidate n-grams are valid (preventing hallucinated ungrounded text in translation). ROUGE checks what fraction of reference target tokens were captured in the summary (preventing key detail omission).
