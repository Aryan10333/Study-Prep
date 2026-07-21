# Module 10: Classical NLP vs. Modern LLMs Master Cheatsheet & 30 Flashcards

This master study guide provides a comprehensive comparison matrix between Classical NLP, Sequence Models, and Modern LLMs, an enterprise production decision tree, 30 top-tier interview flashcards, and a 1-page formula cheatsheet.

---

## 1. Paradigm Architectural Comparison Matrix

```text
Dimension              Regex / Rule-Based         Classical ML (TF-IDF + SVM)    Sequence Deep Learning (LSTM)    Modern Transformer LLM (GPT-4)
--------------------------------------------------------------------------------------------------------------------------------------------------
Era & Primary Tech     1960s (Regex, Grammars)    1990s - 2010s (TF-IDF, Naive)  2014 - 2018 (LSTM, GloVe)        2019 - Present (BERT, Llama 3)
Vector Space           None (String match)        Sparse R^|V| (|V| ~ 100k)      Dense R^d (d ~ 100-300)          Dense Contextual R^d (d ~ 4096+)
Context Horizon        Rule clause bounded        Unigram / Bigram window        Sequential hidden state (T~50)   Global Context (128k+ tokens)
Word Order Handling    Manual rule order          Ignored (Bag-of-Words)         Preserved via recurrence h_t     Preserved via Positional Encoding
OOV Token Handling     Fails on missing rules     Mapped to <UNK> token          Subword character n-grams        Subword BPE / tiktoken (0% OOV)
Training Data Req.     Zero (Hand-crafted)        Low (100 - 1,000 docs)         Moderate (10,000+ docs)          Massive (Trillions of tokens)
Inference Latency SLA  Ultra-fast (<1ms)          Ultra-fast (<2ms)              Fast (10ms - 50ms)               Heavy (100ms - 1000ms+)
Hardware Requirement   Single CPU thread          Standard CPU server            CPU / Single GPU                 Heavy GPU Cluster / TensorRT
Best Production Use    Input validation & noise   High-speed log filter, spam    Sequence tagging, edge NER       Complex reasoning, RAG, QA
```

---

## 2. Enterprise Production Architecture Decision Tree

```text
                                [START: NLP Task Requirement]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         Is Latency SLA < 5ms or                             Is Task Complex Reasoning,
         Compute Budget = $0 CPU?                            Code Gen, or General QA?
                    │                                                   │
          ┌─────────┴─────────┐                               ┌─────────┴─────────┐
          ▼                   ▼                               ▼                   ▼
    [Rule-Based /       [TF-IDF + Linear           [Fine-Tuned Small       [LLM API + RAG /
     Regex Pipeline]    Logistic Regression]       Transformer: BERT]      GPT-4o-mini]
    - Validation        - Spam Filtering           - Private NER           - Multi-doc QA
    - Timestamp strip   - Ticket Category          - Fast Classification   - Reasoning Agent
```

---

## 3. 30 Master Interview Flashcards & Questions

### Category 1: NLP Fundamentals & Preprocessing
#### Flashcard 1: What makes unstructured natural language text fundamentally different from tabular data in Machine Learning?
- **Answer**: Tabular data consists of continuous/categorical features in fixed Euclidean dimensions. Text consists of discrete, variable-length, non-Euclidean symbols with high vocabulary cardinality ($|V| > 100,000$), extreme matrix sparsity, and strong context dependencies.

#### Flashcard 2: Explain Zipf's Law and its impact on vocabulary explosion.
- **Answer**: Zipf's Law states that word frequency is inversely proportional to its rank ($f \propto 1/k$). A small set of common words account for $>80\%$ of text occurrences, while a massive tail of rare words causes vocabulary explosion. Subword tokenization (BPE) handles this long tail without OOV errors.

#### Flashcard 3: When should you NOT remove stop words during preprocessing?
- **Answer**: Stop words must be preserved in Sentiment Analysis (e.g. removing *"not"* converts *"not good"* to *"good"*), Machine Translation, and Transformer/LLM pipelines where attention depends on complete grammatical syntax.

#### Flashcard 4: Compare Stemming vs. Lemmatization.
- **Answer**: Stemming uses heuristic suffix slicing rules (Porter), often generating non-real word stems (`"univers"`). Lemmatization uses morphological lookup and Part-of-Speech (POS) tags to return the true canonical dictionary lemma (`"good"` for `"better"`).

#### Flashcard 5: How does Byte Pair Encoding (BPE) eliminate Out-Of-Vocabulary (OOV) errors?
- **Answer**: BPE iteratively merges frequent character pairs into subwords. Any unseen test word is decomposed into known subword fragments or individual bytes, guaranteeing $100\%$ token coverage with 0% OOV fallbacks.

---

### Category 2: Text Representations & Embeddings
#### Flashcard 6: Why are One-Hot encoded word vectors incapable of capturing semantic similarity?
- **Answer**: One-hot vectors treat words as orthogonal unit vectors in $|V|$-dimensional space ($e_i^\top e_j = 0$ for $i \neq j$). Consequently, Cosine similarity between any two distinct one-hot vectors is identically zero.

#### Flashcard 7: Define the mathematical formula for TF-IDF and explain the role of IDF.
- **Answer**: $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$. IDF penalizes words that appear across many documents (e.g. `"the"`, `"system"`), boosting rare informative terms.

#### Flashcard 8: Explain the Distributional Hypothesis.
- **Answer**: *"You shall know a word by the company it keeps"* (J.R. Firth). Words that occur in similar context window environments share similar semantic meanings in vector space.

#### Flashcard 9: Compare CBOW vs. Skip-Gram in Word2Vec.
- **Answer**: CBOW predicts target center word $w_t$ given context words. Skip-Gram predicts context words given target word $w_t$. CBOW trains faster; Skip-Gram is slower but superior for rare words and small datasets.

#### Flashcard 10: How does Negative Sampling solve the Softmax computational bottleneck in Word2Vec?
- **Answer**: Full Softmax requires computing a sum over all $|V|$ words in denominator ($O(|V|)$ complexity). Negative Sampling converts this into $K+1$ binary logistic regression tasks (1 positive pair, $K$ negative noise words), reducing update cost to $O(K)$.

#### Flashcard 11: How does FastText handle Out-Of-Vocabulary (OOV) words differently from Word2Vec?
- **Answer**: Word2Vec assigns one static vector per word and throws errors on unseen terms. FastText represents words as a bag of character n-grams (e.g. `"<wh"`, `"whe"`), building vectors for unseen words by summing their subword n-gram vectors.

#### Flashcard 12: What is the main defect of static word embeddings (Word2Vec / GloVe)?
- **Answer**: Polysemy failure. Static embeddings assign 1 single static vector per word regardless of sentence context (e.g. `"river bank"` vs `"investment bank"` get identical vectors).

---

### Category 3: Classical Classifiers & Sequence Models
#### Flashcard 13: What is the "naive" assumption in Naive Bayes?
- **Answer**: It assumes that token occurrences $w_1, \dots, w_n$ are conditionally independent given class $c$: $P(w_1, \dots, w_n | c) = \prod P(w_i | c)$.

#### Flashcard 14: Why is Laplace Add-1 Smoothing necessary in Naive Bayes?
- **Answer**: If a test word never appeared in class $c$ during training, its frequency probability is $0$. Multiplying probabilities together would cause the entire class score to become zero. Laplace smoothing adds $\alpha=1$ to numerator and $\alpha |V|$ to denominator to ensure non-zero probabilities.

#### Flashcard 15: Compare Generative (Naive Bayes) vs Discriminative (Logistic Regression) classifiers.
- **Answer**: Naive Bayes models joint probability $P(X, Y) = P(X|Y)P(Y)$ (fast, assumes feature independence). Logistic Regression directly models conditional probability $P(Y|X) = \sigma(w^\top x + b)$ (handles correlated n-grams, produces calibrated probabilities).

#### Flashcard 16: Why do standard Recurrent Neural Networks (RNNs) suffer from vanishing gradients?
- **Answer**: Backpropagation Through Time (BPTT) requires repeatedly multiplying by recurrent matrix $W_{hh}^\top$ over $T$ steps. If eigenvalues of $W_{hh}$ are $<1$, gradients decay exponentially to zero ($O(\lambda^T)$), causing the network to forget long-range dependencies.

#### Flashcard 17: How does the LSTM Cell State $C_t$ solve vanishing gradients mathematically?
- **Answer**: The LSTM Cell State $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ uses an additive linear update highway instead of multiplicative matrix steps, allowing error gradients to propagate backwards through time unchanged.

#### Flashcard 18: Name the 3 gates in an LSTM cell and their specific functions.
- **Answer**:
  1. **Forget Gate ($f_t$)**: Decides what fraction of old cell memory $C_{t-1}$ to discard.
  2. **Input Gate ($i_t$)**: Decides which new candidate values $\tilde{C}_t$ to write into cell memory.
  3. **Output Gate ($o_t$)**: Filters cell state to generate hidden state output $h_t = o_t \odot \tanh(C_t)$.

#### Flashcard 19: Compare LSTM vs GRU architectures.
- **Answer**: LSTM has 3 gates (Forget, Input, Output) and maintains separate Cell State $C_t$ and Hidden State $h_t$. GRU has 2 gates (Reset, Update), merges Cell and Hidden states into $h_t$, has $\approx 25\%$ fewer parameters, and trains faster.

#### Flashcard 20: Explain the Information Bottleneck in standard Seq2Seq Encoder-Decoder models.
- **Answer**: Standard Seq2Seq forces the Encoder LSTM to compress the entire input sequence $X_{1:T}$ into a single fixed-size final vector $h_T$. For long sentences ($T > 30$), compressing all information into one vector causes severe information loss.

---

### Category 4: Attention Mechanisms & Evaluation Metrics
#### Flashcard 21: How does Attention solve the Seq2Seq Information Bottleneck?
- **Answer**: Attention retains ALL encoder hidden states $[h_1, \dots, h_T]$ and allows the decoder to dynamically compute alignment weights $\alpha_{ij}$ and construct a custom context vector $c_i = \sum \alpha_{ij} h_j$ at each generation step.

#### Flashcard 22: Compare Bahdanau Additive Attention vs Luong Multiplicative Attention.
- **Answer**: Bahdanau uses an additive feed-forward layer $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ with previous state $s_{i-1}$. Luong uses multiplicative matrix dot products $e_{ij} = s_i^\top h_j$ with current state $s_i$, offering faster GPU matrix multiplication.

#### Flashcard 23: How does Luong Dot Attention transition to Transformer Self-Attention?
- **Answer**: Luong Dot Attention computes $s_i^\top h_j$ between decoder and encoder states. Transformer Self-Attention eliminates recurrence by projecting input tokens into Query ($Q$), Key ($K$), and Value ($V$) matrices: $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.

#### Flashcard 24: What is modified n-gram precision in BLEU?
- **Answer**: Modified n-gram precision clips candidate n-gram counts to the maximum frequency they appear in any single reference sentence, preventing short repetitive candidates (e.g. `"the the the"`) from achieving fake $100\%$ precision scores.

#### Flashcard 25: What is Brevity Penalty in BLEU, and why is it needed?
- **Answer**: Precision metrics favor short translations. Brevity Penalty $\text{BP} = \exp(1 - r/c)$ penalizes candidate outputs shorter than reference length $r$, ensuring balanced evaluation scores.

#### Flashcard 26: Compare BLEU vs ROUGE evaluation targets.
- **Answer**: BLEU is **Precision-Oriented** (used in Machine Translation to verify generated candidate accuracy). ROUGE is **Recall-Oriented** (used in Summarization to verify reference content coverage).

#### Flashcard 27: Define Perplexity (PPL) and its relationship to Cross-Entropy Loss.
- **Answer**: Perplexity is the exponentiated cross-entropy loss: $\text{PPL} = \exp(\mathcal{L}_{\text{CrossEntropy}})$. It represents the average branch factor of uncertainty when predicting the next token.

#### Flashcard 28: What is the main limitation of BLEU and ROUGE in modern GenAI evaluation?
- **Answer**: Both rely strictly on exact surface string n-gram matching, scoring valid semantically identical paraphrases as zero. Modern evaluation uses BERTScore or LLM-as-a-Judge.

#### Flashcard 29: What is INT8 Quantization, and what are its performance benefits?
- **Answer**: INT8 Quantization maps 32-bit float weights ($W_{\text{FP32}}$) to 8-bit integers ($W_{\text{INT8}}$), reducing model RAM footprint by $75\%$ and increasing CPU inference throughput by $2\text{x}-4\text{x}$ with $<0.5\%$ loss in accuracy.

#### Flashcard 30: What is ONNX Runtime, and why is it used for serving NLP models?
- **Answer**: ONNX compiles PyTorch/TensorFlow computational graphs into hardware-optimized execution engines, eliminating Python framework overhead and providing $2\text{x}-3\text{x}$ faster inference SLAs.

---

## 4. 1-Page Formula & Equation Cheatsheet

```text
1. Zipf's Law:
   f(k; s, N) = (1 / k^s) / sum_{n=1}^N (1 / n^s)

2. Sublinear Log-Scaled TF-IDF:
   TF_log(t, d) = 1 + log(f_{t,d})  if f_{t,d} > 0 else 0
   IDF(t, D)    = log((1 + |D|) / (1 + DF(t))) + 1
   TF-IDF       = TF_log * IDF
   Norm Vector  = v / sqrt(sum(v_i^2))

3. Word2Vec Negative Sampling Loss (SGNS):
   L_SGNS = -log sigma(v'_wO^T v_wI) - sum_{k=1}^K log sigma(-v'_wk^T v_wI)

4. GloVe Co-occurrence Loss:
   J = sum_{i,j=1}^|V| f(X_ij) * (w_i^T w~_j + b_i + b~_j - log X_ij)^2

5. Laplace Add-1 Smoothing (Naive Bayes):
   P(w_i | c) = (N_{c,i} + 1) / (N_c + |V|)

6. LSTM Cell Equations:
   f_t = sigma(W_f [h_{t-1}, x_t] + b_f)       (Forget Gate)
   i_t = sigma(W_i [h_{t-1}, x_t] + b_i)       (Input Gate)
   C~_t = tanh(W_c [h_{t-1}, x_t] + b_c)       (Candidate Cell)
   C_t = f_t * C_{t-1} + i_t * C~_t            (Cell State Update)
   o_t = sigma(W_o [h_{t-1}, x_t] + b_o)       (Output Gate)
   h_t = o_t * tanh(C_t)                       (Hidden State Output)

7. Luong Multiplicative Attention:
   e_ij = s_i^T h_j  (Dot)  |  alpha_ij = exp(e_ij) / sum_k exp(e_ik)  |  c_i = sum_j alpha_ij h_j

8. Transformer Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

9. BLEU Brevity Penalty & Score:
   BP = exp(1 - r/c) if c <= r else 1
   BLEU = BP * exp( sum_{n=1}^N w_n log p_n )

10. Perplexity Equivalence:
    PPL = exp( L_CrossEntropy )
```
