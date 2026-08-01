# NLP Fundamentals: Covered Topics

This document provides a clean directory of the concepts and engineering topics covered in the **NLP Fundamentals** curriculum.

## 1. NLP Introduction
* **The Standard NLP Pipeline**:
  * Text cleaning and noise reduction (removal of boilerplate metadata, markup tags, and formatting noise).
  * Word-level tokenization principles (handling punctuation boundaries, contractions, and clitics).
  * Vocabulary construction (mapping unique tokens to index tables, fallback out-of-vocabulary handling).
* **Evolution of NLP Paradigms**:
  * Rule-based processing engines (Context-Free Grammars, regular expression rules).
  * Statistical NLP (N-gram sequences, probabilistic modeling).
  * Neural and Deep Learning sequence baselines (RNNs, LSTMs, GRUs).
  * Modern Transformer architectures.
* **Tokenization Taxonomies**:
  * Character-level vs. Word-level tokenization trade-offs.
  * Subword Tokenization algorithms (Byte-Pair Encoding, WordPiece, SentencePiece).

## 2. Text Preprocessing
* **Text Normalization**:
  * Lowercase conversion and its implications on downstream tasks like Named Entity Recognition.
  * Contraction expansion and spelling normalization.
  * Unicode normalization styles (Composition NFC vs. Decomposition NFD).
* **Morphological Standardization**:
  * Stemming heuristics (suffix removal).
  * Lemmatization via dictionary mappings.
* **Filtering Strategies**:
  * Stop word removal criteria (impact on lexical density vs. semantic loss).
* **Regex and Text Cleaning**:
  * Pattern matching rules for network identifiers, web URLs, and markup noise.

## 3. Text Representation
* **One-Hot Encoding**:
  * Binary sparse representation vectors.
  * Sparsity limitations and the orthogonal distance bottleneck.
* **Bag of Words (BoW)**:
  * Word frequency document representations.
  * Discarding word ordering and grammatical context.
* **TF-IDF (Term Frequency-Inverse Document Frequency)**:
  * Term Frequency weighting options.
  * Inverse Document Frequency metrics.
  * Document retrieval using Cosine Similarity in high-dimensional vector spaces.

## 4. Statistical Language Models
* **Language Modeling Principles**:
  * Sequence joint probabilities and the conditional probability chain rule.
  * The Markov Assumption (simplifying long-range histories to fixed-size context window n-grams).
* **Zero-Probability and Sparsity Challenges**:
  * The problem of unseen contexts.
* **Smoothing Methodologies**:
  * Laplace (add-one) and Add-k smoothing.
  * Stupid Backoff penalty scaling.
  * Absolute Discounting and Kneser-Ney continuation probability smoothing.
* **Model Evaluation**:
  * Perplexity (branching factor interpretation and connection to cross-entropy loss).

## 5. Word Embeddings
* **Distributed Representation**:
  * Contrast between dense, low-dimensional continuous vector embeddings and high-dimensional sparse vectors.
* **Word2Vec Architectures**:
  * Continuous Bag of Words (CBOW) architecture.
  * Skip-gram architecture.
* **Softmax Optimizations**:
  * Hierarchical Softmax (representing vocabulary classes as binary Huffman tree paths).
  * Negative Sampling (reframing multi-class prediction as a binary logistic classification task).
* **Global Vectors (GloVe)**:
  * Log-bilinear co-occurrence matrix factorization.
  * Combining global statistics with local window contexts.
* **FastText**:
  * Subword representation using character n-grams.
  * Handling out-of-vocabulary inputs.

## 6. Sequence Models
* **Recurrent Neural Networks (RNNs)**:
  * Hidden state feedback loops.
  * Backpropagation Through Time sequential computation constraints.
* **Gradient Instability**:
  * Mathematical vanishing and exploding gradient thresholds during unrolled backpropagation.
  * Mitigation via gradient clipping.
* **Gated Architectures**:
  * Long Short-Term Memory (LSTM) gating mechanisms (Forget gate, Input gate, Output gate, Cell State memory line).
  * Gated Recurrent Unit (GRU) gate dynamics (Reset and Update gates).

## 7. NLP Evaluation
* **Classification Evaluation**:
  * Precision, Recall, and F1-score macro vs. micro weighting under class imbalances.
* **Sequence Generation Metrics**:
  * BLEU evaluation metrics (precision-focused n-gram overlap, brevity penalty constraints).
  * ROUGE evaluation metrics (recall-focused n-gram and longest common subsequence overlap).
  * METEOR alignment and WordNet synonym matching.
  * BERTScore contextual embedding similarities.

## 8. Production NLP
* **Inference Optimizations**:
  * Quantization (reduced precision weight mapping).
  * Inference latency profiles of tokenizers.
* **Model Deployment Challenges**:
  * Data drift, domain shifts, and specialized vocabulary failures.

