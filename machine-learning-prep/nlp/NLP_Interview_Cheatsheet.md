# Part 4: NLP Interview Cheatsheet (8 Essential Must-Know Foundations)

A high-density revision reference for AI, Applied AI, GenAI, and ML Engineer interviews covering the 8 essential formulas, core diagrams, time & memory complexity, interview Q&A, and a 1-page comparison table.

---

## 1. The 8 Must-Know Essential Formulas

```text
1. TF-IDF Formula:
   TF(t, d) = count(t, d) / |d|
   IDF(t) = log((1 + |D|) / (1 + DF(t))) + 1
   TF-IDF(t, d) = TF(t, d) * IDF(t)

2. Cosine Similarity:
   CosineSim(u, v) = (u · v) / (||u||2 * ||v||2)

3. Naive Bayes (+ Laplace Add-1 Smoothing):
   P(c | d) ∝ P(c) * ∏ P(w_i | c)
   P(w_i | c) = (N_{c,i} + 1) / (N_c + |V|)

4. Word2Vec Skip-Gram Sigmoid Output:
   P(w_O | w_I) = σ(v'_{w_O}ᵀ v_{w_I}) = 1 / (1 + exp(- v'_{w_O}ᵀ v_{w_I}))

5. LSTM Cell Update Equations (Forward Pass Only):
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       (Forget Gate)
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       (Input Gate)
   C~_t = tanh(W_c · [h_{t-1}, x_t] + b_c)   (Candidate Cell)
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t          (Cell State Update)
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)       (Output Gate)
   h_t = o_t ⊙ tanh(C_t)                      (Hidden State)

6. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax( (Q Kᵀ) / sqrt(d_k) ) V

7. BLEU Score (BLEU-2 Precision-Oriented):
   BLEU-2 = BP * exp(0.5 log p_1 + 0.5 log p_2)
   BP = exp(1 - r/c) if c <= r else 1.0

8. Perplexity (Model Uncertainty Branch Factor):
   PPL = exp(L_{CrossEntropy})
```

---

## 2. Time & Memory Complexity Matrix

```text
Algorithm / Architecture     Training Time Complexity           Inference Time Complexity  Memory Footprint Complexity
-----------------------------------------------------------------------------------------------------------------------------
TF-IDF Indexing              $O(|D| \times L)$                  $O(k \times |V|)$          $O(|D| \times |V|)$ Sparse
Multinomial Naive Bayes      $O(|D| \times L)$                  $O(C \times L)$            $O(C \times |V|)$ Dense
Word2Vec Skip-Gram           $O(C \cdot m \cdot K \cdot d)$     $O(1)$ Lookup              $O(|V| \times d)$ Embedding Table
LSTM Model Layer             $O(T \cdot d^2)$ per sequence      $O(T \cdot d^2)$           $O(4 \cdot d^2)$ Weights
Scaled Dot-Product Attention $O(T^2 \cdot d)$ per sequence      $O(T^2 \cdot d)$           $O(T^2)$ RAM Attention Weights
INT8 Quantized Model         $O(N \cdot d)$ SIMD                $O(N \cdot d)$ SIMD        $O(\frac{1}{4} M_{\text{FP32}})$ Bytes
```

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
