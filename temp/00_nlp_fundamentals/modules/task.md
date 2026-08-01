# NLP Fundamentals Improvements: Checklist

This living checklist tracks the progress of the sequential, batch-based improvements mapped out in the roadmap.

---

## Batch 1: Foundational Pipeline & Preprocessing (Modules 1-2, Notebooks 1-2)
- [x] **Module 01 (NLP Introduction):**
  - [x] Add pipeline cleaning motivation & noise definitions.
  - [x] Explain word tokenization mechanics & punctuation boundaries.
  - [x] Elaborate on character tokenization attention sequence limit bottlenecks ($O(L^2)$ VRAM scaling).
- [x] **Module 02 (Text Preprocessing & Subword Tokenization):**
  - [x] Define morphological terms: *morphemes*, *prefixes*, *suffixes*, *inflectional variations*, and *canonical base forms*.
  - [x] Detail stemming (Porter heuristics) vs. lemmatization (POS lookups) algorithms.
  - [x] Add BPE merge hand-calculation on micro-corpus `{"hug": 10, "pug": 5, "hugs": 5}`.
  - [x] Add WordPiece score hand-calculation & scoring ratio explanation.
  - [x] Show NLTK comparison code block & morphological output table.
- [x] **Notebooks:**
  - [x] Refactor and execute [01_nlp_introduction.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/01_nlp_introduction.ipynb) (Wikipedia pipeline, NFC/NFD Unicode normalization checks).
  - [x] Refactor and execute [02_text_preprocessing.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/02_text_preprocessing.ipynb) (stem vs lemma latency, BPE merge loop, WordPiece score calculations).

---

## Batch 2: Retrieval & Statistical Models (Modules 3-4, Notebooks 3-4)
- [x] **Module 03 (Text Representation & Classical Retrieval Models):**
  - [x] Define document-term matrices & indexing vocabulary mappings.
  - [x] Explain $L_2$ normalizations & geometry of unit hyperspheres.
  - [x] Compare Cosine Similarity vs. Euclidean length bias.
  - [x] Add BM25 step-by-step scoring hand-calculation & parameter interpretations ($k_1$, $b$).
  - [x] Add Feature Hashing signs collision cancellation details.
  - [x] Add Scikit-Learn code block matching vector computations.
- [x] **Module 04 (Statistical Language Models & Smoothing):**
  - [x] Explain Language Model prediction targets and sequence chain rules.
  - [x] Define Markov chain orders (1st order bigram vs. 2nd order trigram).
  - [x] Detail MLE zero-probability defect & smoothing shift motivations.
  - [x] Add Laplace, Katz backoff, and perplexity hand-calculations.
  - [x] Add Numpy code block calculating transitions and perplexity.
- [x] **Notebooks:**
  - [x] Refactor and execute [03_text_representation.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/03_text_representation.ipynb) (TF-IDF from scratch, BM25 saturation, Feature Hashing simulation).
  - [x] Refactor and execute [04_statistical_language_models.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/04_statistical_language_models.ipynb) (N-gram transitions, smoothing, perplexity computations).

---

## Batch 3: Continuous Embeddings & Sequential State Buffers (Modules 5-6, Notebooks 5-6)
- [x] **Module 05 (Word Embeddings & Semantic Spaces):**
  - [x] Define dense continuous representations vs. sparse spaces.
  - [x] Prove embedding layer linear mapping as index matrix multiplication ($\mathbf{h} = \mathbf{x}^T \mathbf{W}$).
  - [x] Detail CBOW and Skip-gram architectures.
  - [x] Explain SGNS negative sampling binary logistic game and $3/4$ noise distribution.
  - [x] Add PyTorch CBOW step forward-pass matrix multiplication code & logits output logs.
- [x] **Module 06 (Sequence Models & Recurrent Architectures):**
  - [x] Explain recurrent cell loops, shared weight matrices ($W_{hh}, W_{xh}$), and variable sequences.
  - [x] Derive vanishing/exploding gradients under folded BPTT paths.
  - [x] Detail LSTM gates (Forget, Input, Candidate, Output) and the CEC linear conveyor belt.
  - [x] Explain Beam Search beam expansion pathways.
  - [x] Add PyTorch RNN/LSTMCell forward updates validation code.
- [x] **Notebooks:**
  - [x] Refactor and execute [05_word_embeddings.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/05_word_embeddings.ipynb) (PyTorch CBOW lookup, Negative Sampling SGNS custom loss implementation).
  - [x] Refactor and execute [06_sequence_models.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/06_sequence_models.ipynb) (custom RNN/LSTM state updates, forward pass checks, Beam Search pathways).

---

## Batch 4: Self-Attention, Metrics, and Maintenance (Modules 7-9, Notebooks 7-9)
- [x] **Module 07 (Attention & Transformer Prerequisites):**
  - [x] Define the Seq2Seq encoder bottleneck context decay.
  - [x] Compare Bahdanau additive vs. Luong multiplicative attention context vector weights.
  - [x] Detail Query-Key-Value projections and dot-product alignments.
  - [x] Add Scaled dot product hand-calculation & $1/\sqrt{d_k}$ Softmax saturation scaling explanation.
  - [x] Add PyTorch scaled dot-product attention calculation code.
- [x] **Module 08 (NLP Evaluation Metrics & Semantic Validation):**
  - [x] Detail confusion matrices & Precision-Recall trade-off limits.
  - [x] Explain BLEU precision clipped counts & brevity penalty.
  - [x] Detail ROUGE recall and LCS sequence matching.
  - [x] Add BLEU/ROUGE hand-calculation and BERTScore greedy alignments.
  - [x] Add NLTK evaluations code block.
- [x] **Module 09 (Production NLP & Model Maintenance):**
  - [x] Define Covariate (Data) Shift vs. Concept Drift.
  - [x] Explain Population Stability Index (PSI) equations & drift levels.
  - [x] Add PSI step-by-step computation.
  - [x] Add Python script calculating PSI and Wasserstein Distance.
- [x] **Notebooks:**
  - [x] Refactor and execute [07_attention_and_transformer_prerequisites.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/07_attention_and_transformer_prerequisites.ipynb).
  - [x] Refactor and execute [08_nlp_evaluation.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/08_nlp_evaluation.ipynb).
  - [x] Refactor and execute [09_production_nlp.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/09_production_nlp.ipynb). (PSI and Wasserstein distance drift indicators).

---

## Batch 5: High-Frequency Q&As & Compilation (Module 10, Compiler, Cheatsheet)
- [x] **Module 10 (High-Frequency Interview Question Bank):**
  - [x] Upgrade technical questions (13, 17, 18, 20, 22, 32) with detailed KaTeX derivations, evaluations, and calculations.
- [x] **Revision Cheatsheet:**
  - [x] Create source markdown for `nlp_interview_cheatsheet.md` containing variable legends, complexities, equations, and screening summaries.
- [x] **Execution & Aggregation:**
  - [x] Re-run [generate_nlp_plots.py](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/helpers/generate_nlp_plots.py) to check graphics.
  - [x] Run [compile_nlp.py](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/helpers/compile_nlp.py) to compile `nlp_master_study_guide.pdf` and generate `nlp_interview_cheatsheet.pdf` via Microsoft Edge headless print.
