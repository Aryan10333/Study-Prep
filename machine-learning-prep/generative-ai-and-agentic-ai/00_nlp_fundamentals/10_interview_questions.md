# Module 10: NLP Fundamentals High-Frequency Interview Question Bank

This module provides **40 high-frequency interview questions** covering NLP foundations, mathematical derivations, debugging procedures, and production systems design.

---

## 1. Conceptual Questions (1-10)

### Q1: Contrast workflows vs. classical NLP models vs. modern LLMs across cost, context windows, and latency.
- **Classical NLP Models**: Train specialized parameters (e.g., Bi-LSTM classification head) over static vocabularies. Very low compute cost, sub-10ms inference latency, but limited to short input constraints.
- **Modern LLMs**: Leverage massive attention blocks. High cost and latency, but support long context windows and complex contextual reasoning.
- **Workflows**: Multi-model routing layers connecting classical models (e.g. classifier filtering queries) and LLMs to optimize latency and costs.

### Q2: Why did subword tokenization replace character-level and word-level tokenization in modern LLMs?
Word-level tokenization causes a vocabulary size explosion ($|V| > 1,000,000$), increasing memory footprint and resulting in high Out-of-Vocabulary (OOV) rates. Character-level tokenizers solve OOV but increase sequence length, raising attention computational cost $O(L^2)$ quadratically. Subword tokenization (BPE/WordPiece) balances these limits by representing common roots while decomposing unknown words.

### Q3: Explain how Byte-Pair Encoding (BPE) handles unseen text sequences during inference.
During BPE training, rare characters are grouped or mapped to individual byte tokens. Unseen words are decomposed into character segments or bytes already present in the vocabulary, preventing `<unk>` mapping errors.

### Q4: Explain the differences between the merging objectives of BPE and WordPiece.
BPE merges adjacent token pairs based on raw co-occurrence frequency counts. WordPiece merges pairs that maximize corpus likelihood under a unigram language model, scoring pairs by:
$$\text{Score}(A, B) = \frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$$

### Q5: How does Unigram tokenization differ from BPE in its building strategy?
BPE is a bottom-up algorithm, starting with individual characters and iteratively merging frequent pairs. Unigram is a top-down algorithm, starting with a large vocabulary and iteratively pruning tokens that have the lowest contribution to corpus likelihood.

### Q6: What is the Distributional Hypothesis and how does it relate to dense word embeddings?
The Distributional Hypothesis states that words occurring in similar contexts share semantic meaning. Dense embeddings learn representations by optimizing models to predict a word given its context (or vice versa), mapping co-occurrence patterns to vector spaces.

### Q7: Contrast Continuous Bag of Words (CBOW) and Skip-gram architectures in Word2Vec.
CBOW predicts a single target word given its surrounding context words, averaging context vectors. Skip-gram predicts multiple context words given a target word. Skip-gram performs better on small datasets and rare words, while CBOW trains faster on large corpora.

### Q8: How does FastText solve the Out-of-Vocabulary (OOV) problem that causes Word2Vec and GloVe to fail?
FastText represents words as bags of character n-grams. When encountering an OOV token, FastText sums the vectors of its constituent character n-grams to generate an embedding, whereas Word2Vec and GloVe return a generic `<unk>` vector.

### Q9: Why do standard Recurrent Neural Networks (RNNs) struggle to model long-range dependencies?
As context length increases, gradients propagated through time are scaled by the recurrent weight matrix product $\prod W_{hh}$. If the spectral radius of $W_{hh}$ is less than 1, the gradient decays exponentially to zero, blinding the model to early tokens.

### Q10: Why did Self-Attention replace Recurrent architectures as the standard sequence modeling paradigm?
Recurrent models process tokens sequentially, creating an execution bottleneck. Self-attention models process all tokens in parallel, reducing training path lengths between distant tokens to $O(1)$.

---

## 2. Mathematical Questions (11-20)

### Q11: Derive the TF-IDF calculation for a word that appears twice in a document of 10 words, where the word occurs in 5 out of 100 total documents.
- Term Frequency: $\text{TF} = 2 / 10 = 0.2$
- Smooth IDF: $\text{IDF} = \log\left(\frac{1 + 100}{1 + 5}\right) + 1 = \log(101/6) + 1 \approx \log(16.83) + 1 \approx 2.823 + 1 = 3.823$
- Unnormalized weight: $\text{TF-IDF} = 0.2 \times 3.823 = 0.7646$

### Q12: Prove mathematically why scaled dot-product attention scales scores by $1/\sqrt{d_k}$.
Assume components of query $\mathbf{q}$ and key $\mathbf{k}$ are independent random variables with mean $0$ and variance $1$. The dot product is:
$$u = \sum_{i=1}^{d_k} q_i k_i$$
- Mean: $\mathbb{E}[u] = \sum \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$
- Variance: $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = (1)(1) = 1$
- Total Variance: $\text{Var}(u) = \sum_{i=1}^{d_k} 1 = d_k$
If $d_k$ is large, the dot product values grow, pushing Softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ scales the input variance to $1$:
$$\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

### Q13: Write the cell state update equation of an LSTM and explain why it prevents vanishing gradients.
- Cell State: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- Gradient Flow: The derivative with respect to the previous cell state is:
  $$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots$$
  If the forget gate $f_t \approx 1$, the error gradient propagates back through time linearly via addition, bypassing the multiplicative decay of standard RNNs.

### Q14: Calculate the Laplace-smoothed bigram probability $P(\text{"cat"} \mid \text{"the"})$ given: $C(\text{"the", "cat"}) = 0$, $C(\text{"the"}) = 100$, and vocabulary size $|V| = 10,000$.
$$P_{\text{Laplace}}(\text{"cat"} \mid \text{"the"}) = \frac{C(\text{"the", "cat"}) + 1}{C(\text{"the"}) + |V|} = \frac{0 + 1}{100 + 10000} = \frac{1}{10100} \approx 9.9 \times 10^{-5}$$

### Q15: Prove that Perplexity is equal to $e^{H(W)}$, where $H(W)$ is the cross-entropy loss of the sequence.
$$\text{PPL}(W) = P(w_1, \dots, w_m)^{-\frac{1}{m}}$$
Taking the natural logarithm:
$$\ln(\text{PPL}(W)) = -\frac{1}{m} \ln P(w_1, \dots, w_m) = -\frac{1}{m} \sum_{i=1}^m \ln P(w_i \mid w_{1..i-1}) = H(W)$$
Exponentiating both sides:
$$\text{PPL}(W) = e^{H(W)}$$

### Q16: Show how Katz Backoff calculates bigram probabilities when co-occurrence counts are zero.
If $C(w_{i-1}, w_i) = 0$, Katz Backoff estimates the probability by backing off to the lower-order unigram probability, scaled by a normalization factor $\alpha$:
$$P_{\text{Katz}}(w_i \mid w_{i-1}) = \alpha(w_{i-1}) P(w_i)$$

### Q17: Write the loss function for Skip-gram with Negative Sampling (SGNS) and define its components.
$$\mathcal{L} = -\log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^K \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})$$
- $w_I$: Input target word.
- $w_O$: True output context word.
- $w_i$: Negative sample words.
- $K$: Number of negative samples.

### Q18: Calculate the BLEU brevity penalty (BP) for a generated candidate of 8 words compared to a reference translation of 10 words.
Since candidate length $c = 8 \le r = 10$:
$$\text{BP} = e^{1 - r/c} = e^{1 - 10/8} = e^{-0.25} \approx 0.7788$$

### Q19: Express the GloVe weighting function $f(X_{i,j})$ mathematically and explain its purpose.
$$f(x) = \min\left(1, \left(\frac{x}{x_{\max}}\right)^\alpha\right)$$
It bounds the loss contribution of extremely frequent word co-occurrences (e.g. `"the"`, `"and"`), preventing them from dominating the optimization gradient.

### Q20: Show the matrix multiplication steps to compute Self-Attention scores for input matrix $X$.
1. Project inputs: $Q = X W_Q$, $K = X W_K$, $V = X W_V$
2. Score similarity: $S = Q K^T$
3. Normalize: $A = \text{Softmax}\left(\frac{S}{\sqrt{d_k}}\right)$
4. Output: $Z = A V$

---

## 3. Production Questions (21-30)

### Q21: What is the difference between Data Drift and Concept Drift in production NLP pipelines?
- **Data Drift**: The input text distribution changes (e.g. new vocabulary, slang), but the target relationship remains the same:
  $$P(X_{\text{prod}}) \neq P(X_{\text{train}}), \quad P(Y \mid X_{\text{prod}}) = P(Y \mid X_{\text{train}})$$
- **Concept Drift**: The mapping relationship between inputs and outputs shifts (e.g. the word `"viral"` shifts from a negative health context to a positive marketing context):
  $$P(Y \mid X_{\text{prod}}) \neq P(Y \mid X_{\text{train}})$$

### Q22: How do you identify data drift in a production NLP application?
By measuring statistical divergence between token frequency distributions in production and training data using metrics like Wasserstein Distance or Population Stability Index (PSI).

### Q23: Explain the trade-offs of post-training int8 quantization for sequence classifiers.
- **Advantages**: Reduces memory footprint by up to $75\%$, increases throughput, and lowers VRAM requirements.
- **Disadvantages**: Introduces precision loss, which can degrade classification performance on edge cases.

### Q24: How does Feature Hashing save memory in high-vocabulary production models?
It maps words to a fixed-size array index using a hash function, eliminating the need to store a large vocabulary mapping dictionary in memory.

### Q25: Why is vocabulary synchronization between training and serving tokenizers critical?
If serving tokenizers use a different dictionary mapping than the training pipeline, token index values will map to incorrect word vectors, degrading model predictions.

### Q26: Under what conditions is Cosine Similarity identical to Dot Product?
When the input vectors are pre-normalized to have unit $L_2$ length ($\|\mathbf{a}\|_2 = \|\mathbf{b}\|_2 = 1$).

### Q27: How does BM25 handle term frequency saturation compared to standard TF-IDF?
TF-IDF scales linearly with term frequency, meaning a document repeating a query term 100 times is scored 100 times higher. BM25 scales term frequency non-linearly using a saturation parameter $k_1$, bounding the maximum score contribution of any single term.

### Q28: Detail the step-by-step production debugging feedback loop for resolving model decay.
`Inference` (log predictions) $\rightarrow$ `Metrics Log` (detect accuracy drop or drift) $\rightarrow$ `Error Analysis` (inspect low-confidence predictions) $\rightarrow$ `Diagnostic Actions` (retrain model, update vocabulary) $\rightarrow$ `Re-deployment`.

### Q29: What are the latency impacts of choosing Lemmatization over Stemming?
Lemmatization requires part-of-speech (POS) tagging and dictionary lookups, increasing computational latency. Stemming uses rule-based string slicing, which runs significantly faster.

### Q30: What is the main trade-off of using greedy decoding vs. beam search decoding?
Greedy search is computationally fast ($O(L)$) but can get stuck in sub-optimal local paths. Beam search explores multiple paths ($O(L \cdot B)$), improving generation quality at the cost of higher latency and memory usage.

---

## 4. Debugging Questions (31-35)

### Q31: How do you diagnose and fix exploding gradients in recurrent sequence models?
- **Diagnosis**: Training loss spikes to `NaN`, or weight gradients show extremely large norms during backpropagation.
- **Fix**: Apply **Gradient Clipping**, scaling down gradients that exceed a threshold:
  $$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{g_{\max}}{\|\mathbf{g}\|}$$

### Q32: A sequence classifier returns low accuracy on long sequences. How do you troubleshoot this?
Verify if the model is a standard RNN (suffering from vanishing gradients). If so, replace it with an LSTM or GRU to enable stable gradient flow, or add an attention layer to bypass the sequential bottleneck.

### Q33: How do you debug tokenization mismatch bugs where the model output maps to garbage characters?
Check the preprocessing pipeline to ensure training and serving code use the same Unicode normalization (e.g. NFC composition) and subword dictionary indices.

### Q34: What causes model performance to degrade when evaluating text containing emojis and URLs?
Standard tokenizers may lack vocabulary tokens for emojis, mapping them to `<unk>`. Fix this by applying pre-tokenization regex cleanups or using a byte-fallback tokenizer.

### Q35: Your model's BLEU score is high but human reviewers report poor translation quality. Why?
BLEU only measures exact n-gram overlaps. The model may be generating translations that share n-grams with the reference but contain grammatical errors or inverted meanings (e.g. negations). Resolve this by evaluating with BERTScore or human feedback loops.

---

## 5. System Design Questions (36-40)

### Q36: Design a search query auto-completion system.
- **Ingestion**: Clean input queries, normalize Unicode characters (NFC), and tokenize using BPE.
- **Representation**: Generate n-gram language models from query logs.
- **Retrieval**: Use a trie structure populated with query probabilities. For the current query prefix, look up the top $K$ completions with the highest MLE probabilities.
- **Resilience**: Implement Katz backoff or interpolation to predict completions for unseen or rare prefixes.

### Q37: Design a high-throughput spam message classifier.
- **Preprocessing**: Apply regex cleaning to normalize URLs, emojis, and phone numbers.
- **Text Representation**: Use Feature Hashing (to avoid storing a large vocabulary dictionary) to construct sparse term frequency vectors.
- **Classifier**: Train a Naïve Bayes or logistic regression model on hash indices.
- **Production Loop**: Quantize weights to int8, monitor for data drift using Wasserstein distance, and automatically route low-confidence samples to a manual review queue.

### Q38: Design an evaluation pipeline for a document summarization service.
- **Metrics**: Compute ROUGE-L (LCS mapping) and BLEU to verify token overlap.
- **Semantic Check**: Generate sentence embeddings using SBERT and measure cosine similarity against reference summaries, and calculate BERTScore to capture contextual token matches.
- **Audit**: Log outputs to an evaluation dataset, routing a percentage of generations to human reviewers for alignment checks.

### Q39: Design a Named Entity Recognition (NER) pipeline for streaming financial documents.
- **Preprocessing**: Segment text into sentences, preserving casing (capitalization is a strong signal for financial entities).
- **Tokenization**: Apply WordPiece tokenization.
- **Representation**: Generate contextual token representations using a bidirectional LSTM.
- **Classifier**: Append a Conditional Random Field (CRF) layer to decode the most likely tag sequence.
- **Production**: Deploy the pipeline using real-time streaming inference (e.g. Kafka stream processor).

### Q40: Design a semantic search system for an e-commerce platform.
- **Ingestion**: Normalize and tokenize queries using a subword tokenizer (FastText character n-grams to handle OOV typos).
- **Representation**: Retrieve dense query vectors from the embedding layer.
- **Search Index**: Query a vector database (e.g., Faiss) using Hierarchical Navigable Small World (HNSW) indexing to find products with the highest cosine similarity.
- **Fallback**: Fall back to BM25 keyword matching if search vector confidence is low.
