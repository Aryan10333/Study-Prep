# 00_nlp_fundamentals: Classical NLP Foundations Syllabus

## 1. Context & Alignment
* **Profile Focus:** AI Engineer / Applied AI SDE-2/SDE-3 (Aryan Chandra). Emphasizes production engineering, matrix transformations, hardware walls, and gradient dynamics over purely academic proofs.
* **Interview Frequency:** High. A core component of initial technical screens and machine learning foundations assessments.
* **Core Goal:** Master classical text representation, word embedding theory (Word2Vec, FastText), statistical sequence modeling, evaluation metrics, and recurrent networks (RNN/LSTM/GRU).

## 2. Module Chapters & Conceptual Scope
These chapters outline the sequential topics to be covered:

- **Module 01: Introduction to NLP & Classical Tasks**
  - *Key Concepts:* Definition and scope of NLP, levels of linguistic analysis (Morphology, Syntax, Semantics, Pragmatics), taxonomy of classical NLP tasks (Text Classification, Sequence Tagging, Parsing, Generation), and historical evolution (Rule-based vs. Statistical vs. Deep Learning).
  - *System Bottlenecks & Focus:* Ambiguity of language, high variance in raw text data, and representation bottlenecks of unstructured text.

- **Module 02: Text Preprocessing & Normalization**
  - *Key Concepts:* Regex-based noise cleaning, casing, stopword filtering, stemming (Porter algorithm rules), and lemmatization (morphological lookup), with clear, book-like conceptual comparisons.
  - *System Bottlenecks & Focus:* CPU-bound text processing walls, vocabulary compression vs. information loss, and serialization latency.

- **Module 03: Tokenization & Subword Algorithms**
  - *Key Concepts:* What is Tokenization, types of tokenization (Word-level, Character-level, Subword-level) with comparative pros/cons, Byte-Pair Encoding (BPE) merge loops, WordPiece, and SentencePiece architectures.
  - *System Bottlenecks & Focus:* Out-of-Vocabulary (OOV) mitigation, vocabulary footprint vs. sequence length trade-off, and tokenization/detokenization latency.

- **Module 04: Vector Space Models (Bag-of-Words & TF-IDF)**
  - *Key Concepts:* What is a Vector Space Model, count-based representation (Bag-of-Words), Term Frequency (TF) variants, Inverse Document Frequency (IDF) smoothing, L2 normalization, Cosine Similarity, **Zipf's Law (word frequency distributions)**, and the **Okapi BM25 ranking algorithm (term saturation $k_1$, length normalization $b$)**.
  - *System Bottlenecks & Focus:* High dimensionality, sparse matrix overhead, and O(V) vocabulary lookup limits.

- **Module 05: Distributed Representations (Word2Vec CBOW vs. Skip-gram)**
  - *Key Concepts:* Transition from Sparse to Dense vectors (one-hot bottlenecks), What is a Word Embedding, the Distributional Hypothesis, CBOW vs. Skip-gram architectures, Negative Sampling, and Hierarchical Softmax.
  - *System Bottlenecks & Focus:* O(V) output layer softmax computation bottleneck, negative sampling ratio selection, and memory bandwidth bounds.

- **Module 06: Subword and Matrix-Based Embeddings (GloVe & FastText)**
  - *Key Concepts:* What is a Co-occurrence Matrix, GloVe global co-occurrence matrix factorization, limitations of word-level vectors, FastText subword character n-grams, and Out-of-Vocabulary (OOV) reconstruction.
  - *System Bottlenecks & Focus:* OOV lookup recovery latency, storage footprints of subword dictionaries, and static embedding limits.

- **Module 07: Statistical Language Models**
  - *Key Concepts:* What is a Language Model, Chain Rule of Probability, N-gram approximation (Unigram, Bigram, Trigram), Markov assumption, Maximum Likelihood Estimation, Laplace smoothing, and **baseline POS Tagging & NER sequence tagging tasks**.
  - *System Bottlenecks & Focus:* Sparsity in high-order n-grams and exponential vocab scaling.

- **Module 08: Classical NLP Evaluation Metrics**
  - *Key Concepts:* Intrinsic vs. Extrinsic evaluation, Perplexity (PPL) mathematical derivation, BLEU (precision-based), ROUGE (recall-based), Word Error Rate (WER), classification evaluation metrics (Precision, Recall, F1-score, Confusion Matrix), and why Accuracy fails on imbalanced text datasets.
  - *System Bottlenecks & Focus:* Reference translation mismatch, sensitivity to tokenization rules, and n-gram overlap limitations.

- **Module 09: Recurrent Neural Networks (RNN, LSTM, GRU)**
  - *Key Concepts:* Limitations of feedforward networks for sequential text, What is a Recurrent Cell, hidden state recurrent transitions, Backpropagation Through Time (BPTT), vanishing and exploding gradients, LSTM gating gating mechanisms (forget, input, output gates), GRU gates (reset, update), and bidirectional recurrences.
  - *System Bottlenecks & Focus:* Sequential dependency preventing parallelization (memory bandwidth bound), memory footprints scaling as O(L), and vanishing gradients on sequences L > 100.

- **Module 10: Production NLP Pipelines (Data Drift & Monitoring)**
  - *Key Concepts:* Text classification serving architecture, What is Data Drift (concept, causes, types like Covariate Shift in vocabulary), Population Stability Index (PSI) drift detection, KS test, and online telemetry monitoring.
  - *System Bottlenecks & Focus:* Inference pipeline latency, emoji/slang distribution shift, and retraining schedules.
