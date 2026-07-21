# Module 10: Classical NLP vs. Modern LLMs Master Cheatsheet & 30 Flashcards

This master study guide provides a single-page architectural comparison matrix, a time & memory complexity matrix, and 30 high-frequency interview flashcards for Data Science, AI, Applied AI, GenAI, and ML Engineer interviews.

---

## 1. Master Architectural Comparison Matrix

```text
Dimension          TF-IDF + Logistic           Word2Vec / GloVe            LSTM / GRU                  Transformer / LLMs
---------------------------------------------------------------------------------------------------------------------------------------------
Representation     Sparse $\mathbb{R}^{|V|}$   Dense Static $\mathbb{R}^d$ Contextual $\mathbb{R}^{T \times d}$ Deep Contextual $\mathbb{R}^{T \times d}$
Polysemy Handling  None (1 vector per word)    None (1 vector per word)    Dynamic per sequence step   Dynamic Multi-Head Self-Attention
Context Horizon    Bag-of-Words / Local        Local Window ($m=2-5$)     Sequential ($T \approx 100$) Global Context ($128k+$ tokens)
Parallel Training  Fully Parallel (CPUs)       Fully Parallel (CPUs)       Sequential ($O(T)$ Bottleneck) Fully Parallelized GPU Matrix Multiply
Inference Latency  Ultra-fast ($<1\text{ms}$)  Lookup ($<1\text{ms}$)      Fast ($5\text{ms} - 20\text{ms}$) Autoregressive ($50\text{ms} - 1000\text{ms}$)
Primary Bottleneck Feature Sparsity            Static Polysemy             Sequential Training         Quadratic Memory $O(T^2)$
```

---

## 2. Computational Complexity & Memory Footprint Matrix

```text
Algorithm / Model            Training Time Complexity              Inference Time Complexity    Memory Footprint Complexity
---------------------------------------------------------------------------------------------------------------------------------
TF-IDF Vectorizer            $O(|D| \times L)$                     $O(k \times |V|)$            $O(|D| \times |V|)$ Sparse
Multinomial Naive Bayes      $O(|D| \times L)$                     $O(C \times L)$              $O(C \times |V|)$ Dense
Word2Vec (Skip-Gram SGNS)    $O(C \cdot m \cdot K \cdot d)$        $O(1)$ Lookup                $O(|V| \times d)$ Embedding Table
LSTM Layer                   $O(T \cdot d^2)$ per sequence         $O(T \cdot d^2)$ Sequential  $O(4 \cdot d^2)$ Model Weights
Scaled Dot-Product Attention $O(T^2 \cdot d)$ per batch            $O(T^2 \cdot d)$             $O(T^2)$ RAM Attention Weights
INT8 Quantized Serving       $O(N \cdot d)$ SIMD                   $O(N \cdot d)$ SIMD          $O(\frac{1}{4} M_{\text{FP32}})$ Bytes
```

---

## 3. 30 High-Frequency Interview Flashcards

### 1. Preprocessing & Tokenization
1. **Q: Why does Zipf's law necessitate subword tokenization?**
   - *A*: Zipf's law states $f(k) \propto 1/k$. Treating words as atomic units causes a long tail of rare words, inflating $|V|$ and causing high Out-Of-Vocabulary (OOV) rates. Subword tokenization (BPE) breaks rare words into known sub-units, achieving 0% OOV.
2. **Q: Compare Stemming and Lemmatization.**
   - *A*: Stemming chops suffixes using heuristic rules (fast, outputs invalid words like `"comput"`). Lemmatization uses morphological dictionaries and POS tags (slower, outputs valid canonical lemmas like `"compute"`).
3. **Q: What is the Byte Pair Encoding (BPE) algorithm?**
   - *A*: A bottom-up tokenization algorithm that initializes a vocabulary with individual characters and iteratively merges the most frequent adjacent symbol pair until reaching target vocabulary size $|V|$.
4. **Q: What is the Type-Token Ratio (TTR)?**
   - *A*: $\text{TTR} = \frac{|V|}{N_{\text{total}}}$. Measures lexical diversity in a corpus.

### 2. Traditional Text Representations & TF-IDF
5. **Q: What is the TF-IDF formula?**
   - *A*: $\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$, where $\text{IDF}(t) = \log\left(\frac{1+|D|}{1+\text{DF}(t)}\right) + 1$.
6. **Q: Why is IDF introduced?**
   - *A*: IDF downweights ubiquitous stop words (`"the"`, `"is"`) while magnifying rare, domain-specific keywords (`"postgresql"`, `"billing"`).
7. **Q: Why is L2 normalization applied to TF-IDF vectors?**
   - *A*: To eliminate document length bias so long documents do not produce artificially larger vector norms.
8. **Q: Why is Cosine Similarity preferred over Euclidean Distance for TF-IDF?**
   - *A*: Cosine similarity measures vector orientation angle rather than magnitude, preventing document length from distorting similarity.

### 3. Word Embeddings (Word2Vec, GloVe, FastText)
9. **Q: Compare Word2Vec CBOW and Skip-Gram.**
   - *A*: CBOW predicts target word $w_t$ from surrounding context (faster). Skip-Gram predicts context words from target word $w_t$ (better representation for rare words).
10. **Q: Why was Negative Sampling introduced in Word2Vec?**
    - *A*: Full Softmax requires computing partition function $\sum_{w=1}^{|V|} \exp(v'_w^\top v_{w_I})$ over all $|V|$ words ($O(|V|)$ complexity). Negative Sampling converts this into $K+1$ binary logistic classifications ($O(K)$ complexity).
11. **Q: What is the primary limitation of static embeddings like Word2Vec?**
    - *A*: Static embeddings assign a single fixed vector per word, failing to resolve polysemy (e.g. `"bank"` in river bank vs. investment bank).
12. **Q: How does FastText handle Out-Of-Vocabulary (OOV) terms?**
    - *A*: FastText represents words as bags of character n-grams. Unseen OOV words are embedded by summing the vectors of their constituent character n-grams.
13. **Q: How does GloVe differ from Word2Vec?**
    - *A*: Word2Vec updates online via local context windows using SGD, while GloVe factorizes the global co-occurrence log-count matrix offline.

### 4. Classical Classifiers & Baselines
14. **Q: Write Bayes' Theorem with Naive Independence.**
    - *A*: $P(c \mid d) \propto P(c) \prod_{i=1}^T P(w_i \mid c)$. Assumes features are conditionally independent given class $c$.
15. **Q: Why is Laplace Add-1 Smoothing necessary in Naive Bayes?**
    - *A*: Without smoothing, an unseen word in class $c$ yields $P(w_{\text{new}} \mid c) = 0$, causing the entire posterior product to collapse to zero. Laplace smoothing sets $P(w_i \mid c) = \frac{N_{c,i} + 1}{N_c + |V|}$.
16. **Q: Why log-probabilities are used when implementing Naive Bayes?**
    - *A*: Summing log-probabilities $\sum \log P(w_i \mid c)$ prevents floating-point numerical underflow caused by multiplying hundreds of small probabilities.
17. **Q: Why are Precision-Recall curves preferred over ROC curves for imbalanced text?**
    - *A*: On imbalanced datasets (e.g. 95% normal, 5% anomaly), ROC curves can appear overly optimistic because False Positive Rate stays small. Precision-Recall curves isolate minority positive class performance.

### 5. Sequence Models (RNN, LSTM, GRU)
18. **Q: What causes vanishing gradients in Vanilla RNNs during BPTT?**
    - *A*: Backpropagating over $T$ steps requires multiplying weight matrix $W_{hh}^\top$ repeatedly. If $\lambda_{\max}(W_{hh}) < 1.0$, gradients decay exponentially ($0.85^T \rightarrow 0$).
19. **Q: What are the 5 core LSTM update equations?**
    - *A*: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$, $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$, $\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$, $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \implies h_t = o_t \odot \tanh(C_t)$.
20. **Q: How does the LSTM Cell State ($C_t$) eliminate vanishing gradients?**
    - *A*: The cell state update $C_t = f_t \odot C_{t-1} + \dots$ acts as an additive linear highway. If $f_t = 1$, the gradient derivative $\frac{\partial C_t}{\partial C_{t-1}} = 1$, preserving gradient flow across time steps.
21. **Q: Compare LSTM and GRU.**
    - *A*: GRU merges cell state and hidden state, using 2 gates (Reset $r_t$ and Update $z_t$). GRU has 25% fewer parameters and trains faster.
22. **Q: What is the main architectural bottleneck of RNN/LSTM models?**
    - *A*: Sequential computation dependency ($h_t$ depends on $h_{t-1}$), preventing parallel GPU training over sequence length $T$.

### 6. Attention Mechanisms
23. **Q: What is the Scaled Dot-Product Attention formula?**
    - *A*: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.
24. **Q: Why do we scale by $\sqrt{d_k}$ in Attention?**
    - *A*: For large $d_k$, dot products $q \cdot k$ have variance $d_k$. Large dot products push Softmax into saturated regions with near-zero gradients ($\text{softmax}'(z) \rightarrow 0$), causing vanishing gradients during backpropagation. Dividing by $\sqrt{d_k}$ rescales variance back to $1.0$.
25. **Q: Compare Bahdanau and Luong Attention.**
    - *A*: Bahdanau uses additive score function $v_a^\top \tanh(W_a s_{i-1} + W_b h_j)$. Luong uses multiplicative score function $s_i^\top W_a h_j$ (optimized for GPU matrix multiplication).
26. **Q: What is the computational complexity of Self-Attention?**
    - *A*: $O(T^2 \cdot d)$ time and $O(T^2)$ RAM memory complexity over sequence length $T$.

### 7. Evaluation Metrics & Serving
27. **Q: What is BLEU score and why is it precision-oriented?**
    - *A*: BLEU measures modified n-gram precision multiplied by a Brevity Penalty. It focuses on precision to prevent models from outputting repetitive high-confidence tokens.
28. **Q: What is ROUGE and how does ROUGE-1 differ from ROUGE-L?**
    - *A*: ROUGE is recall-oriented for summarization. ROUGE-1 measures unigram recall, while ROUGE-L measures Longest Common Subsequence (LCS) sentence structure.
29. **Q: What is Perplexity and how is it related to Cross-Entropy Loss?**
    - *A*: $\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$. Lower perplexity indicates a better language model with lower next-token branching uncertainty.
30. **Q: What is INT8 Quantization?**
    - *A*: Converts 32-bit floating point weights ($W_{\text{FP32}}$) to 8-bit integers ($W_{\text{INT8}}$), achieving $75\%$ RAM savings ($4\text{x}$ memory compression) and accelerating CPU inference via SIMD instructions.
