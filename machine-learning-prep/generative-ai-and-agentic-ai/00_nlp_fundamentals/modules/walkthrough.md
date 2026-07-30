# Walkthrough: NLP Fundamentals Module Revisions

This walkthrough documents the step-by-step progress of the revisions applied to the NLP Fundamentals study module.

---

## Batch 1: Foundational Pipeline & Preprocessing
**Completed: 2026-07-30**

### Changes Made:
- **Module 01 ([01_nlp_introduction.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/01_nlp_introduction.md)):** Added sections on text cleaning motivations, noise classifications (markup, network items, whitespace), word tokenization (punctuation limits, contraction splits like `"don't"`), vocabulary indexing ($|V|$), and transformer VRAM scaling bottlenecks due to quadratic self-attention matrix sizes ($O(L^2)$).
- **Module 02 ([02_text_preprocessing.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/02_text_preprocessing.md)):** Added morphological terms (morphemes, affixes, inflections, lemmas). Added Porter Stemmer phase 1a rules and WordNet Lemmatization POS tags. Created BPE merge hand-calculations on `{"hug": 10, "pug": 5, "hugs": 5}` and WordPiece scoring ratio hand-calculations. Integrated comparative python benchmarking code.
- **Notebooks 01 & 02:** Refactored and executed programmatically using local python environment. Integrated scrapers, NFC/NFD Unicode normalization checks, stemmer vs lemmatizer latency plots, BPE merge iterations, and WordPiece correlation simulations.

---

## Batch 2: Retrieval & Statistical Models
**Completed: 2026-07-30**

### Changes Made:
- **Module 03 ([03_text_representation.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/03_text_representation.md)):**
  - Defined Document-Term Matrices (DTM), indexing, and sparse vector formats (One-hot, BoW, N-grams).
  - Detailed $L_2$ norm geometries (unit hyperspheres) and compared Euclidean distance length bias vs. Cosine Similarity angle matching.
  - Implemented step-by-step hand-calculations for TF-IDF vectors (`Vector_d1 = [0.8148, 0.5797, 0.0]`) and cosine similarity (`0.3361`) verifying exact consistency with scikit-learn's outputs.
  - Implemented step-by-step hand-calculations for Okapi BM25 scores (`Score_d1 = 1.6211` vs. `Score_d2 = 1.2320`) illustrating length penalization parameters ($b=0.75$, $k_1=1.2$).
  - Explained Feature Hashing collision cancellation via independent sign mapping ($\xi(w) \in \{-1, +1\}$).
- **Module 04 ([04_statistical_language_models.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/04_statistical_language_models.md)):**
  - Formulated sequence joint probabilities using the chain rule derived from conditional probabilities.
  - Explained Markov chain context history reductions (Bigram, Trigram).
  - Detailed MLE zero-probability defects and smoothing shifts (Laplace add-one, Katz backoff, interpolation).
  - Added step-by-step Laplace smoothing transition matrix counts and test sequence perplexity calculations ($\text{PPL} \approx 2.5198$), deriving perplexity as the exponentiated cross-entropy branching factor.
  - Integrated NumPy-based transition matrix validation code.
- **Notebooks 03 & 04:**
  - **[03_text_representation.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/03_text_representation.ipynb):** Computes custom TF-IDF cosine similarities, executes BM25 search rankings with length normalization penalties, and simulates signed Feature Hashing.
  - **[04_statistical_language_models.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/04_statistical_language_models.ipynb):** Builds bigram transition tables from Gutenberg text, applies Laplace smoothing, and evaluates sequence joint probabilities and perplexities.

---

## Batch 3: Continuous Embeddings & Sequential State Buffers
**Completed: 2026-07-30**

### Changes Made:
- **Module 05 ([05_word_embeddings.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/05_word_embeddings.md)):**
  - Documented the mathematical representation of embedding lookup as $\mathbf{h} = \mathbf{x}^T \mathbf{W}$ (omitted step-by-step matrix indexing proofs to keep study notes concise and interview-focused).
  - Added CBOW vs. Skip-gram conceptual intuition ("fill-in-the-blank" vs. "reverse prediction") and clean HTML/CSS architecture layout diagrams.
  - Formulated the Skip-Gram with Negative Sampling (SGNS) binary classification loss, explaining sigmoid activations, the $3/4$ scaling exponent unigram noise distribution, and step-by-step hand-calculations of SGNS loss ($\mathcal{L}_{\text{SGNS}} \approx 1.0083$).
- **Module 06 ([06_sequence_models.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/06_sequence_models.md)):**
  - Documented recurrent parameter sharing mechanisms and unfolded computational graph layouts, with an inline RNN cell flow diagram.
  - Added detailed algorithmic explanations of standard RNN hidden state update equations.
  - Explained the intuition of vanishing and exploding gradients under long sequence BPTT weight product steps ($W_{hh}^{T-t}$ decay or explosion depending on eigenvalues). Omitted intermediate matrix derivative chains.
  - Detailed LSTM cell state updates, added an inline LSTM cell internal routing diagram, and explained how the additive update gate creates a Constant Error Carousel (CEC) shortcut ($\frac{\partial C_t}{\partial C_{t-1}} \approx f_t \approx 1$) to preserve gradient flow.
  - Documented GRU variations (update and reset gate equations, parameter efficiency trade-offs).
  - Explained Bidirectional models (concatenated forward and backward streams, autoregressive generation bottleneck).
  - Documented Sequence-to-Sequence (Seq2Seq) Encoder-Decoder configurations, context vector bottleneck SVG diagram, Teacher Forcing exposure bias caveats, and Decoding strategies (Greedy vs. Beam Search).
- **Notebooks 05 & 06:**
  - **[05_word_embeddings.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/05_word_embeddings.ipynb):** Implements PyTorch matrix projection comparisons and builds a custom SGNS loss class optimizing input vector weights, asserting initial loss matches $1.0083$ exactly.
  - **[06_sequence_models.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/06_sequence_models.ipynb):** Builds manual recurrent transition update steps matching native PyTorch `RNNCell` values, and executes Beam Search decoding pathways.

---

## Batch 4: Self-Attention, Metrics, and Maintenance
**Completed: 2026-07-30**

### Changes Made:
- **Module 07 ([07_attention_and_transformer_prerequisites.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/07_attention_and_transformer_prerequisites.md)):**
  - Detailed the Seq2Seq encoder bottleneck information decay on long contexts.
  - Formulated Bahdanau additive vs. Luong multiplicative attention equations, contrasting their GPU computation properties.
  - Formulated Query-Key-Value projection matrices and dot-product alignment steps.
  - Added step-by-step hand-calculations of Scaled Dot-Product attention on a tiny 2-dimensional database, illustrating how the scaling factor $\frac{1}{\sqrt{d_k}}$ preserves Softmax gradient sensitivity and prevents vanishing gradients.
- **Module 08 ([08_nlp_evaluation.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/08_nlp_evaluation.md)):**
  - Formulated Precision, Recall, and F1 classification indicators, explaining the production threshold tradeoffs.
  - Formulated BLEU clipped precision counts and Brevity Penalty equations.
  - Embedded a **BLEU Brevity Penalty decay curve plot** (`plots/bleu_brevity_penalty.png`) with detailed semantic explanations and insights.
  - Formulated ROUGE variants (ROUGE-1, ROUGE-2, and ROUGE-L).
  - Added step-by-step hand-calculations of BLEU-2 ($\text{BLEU-2} \approx 0.3679$) and ROUGE-1 F1 ($\text{F1} \approx 0.6667$).
  - Explained semantic blind spots in overlap metrics and mapped out BERTScore contextual cosine alignments.
- **Module 09 ([09_production_nlp.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/09_production_nlp.md)):**
  - Compared batch offline inference vs. real-time latency limits.
  - Formulated Covariate Shift vs. Concept Drift.
  - Formulated Population Stability Index (PSI) and detailed stability thresholds ($<0.10$ stable, $0.10\text{--}0.25$ moderate, $>0.25$ high drift).
  - Added step-by-step hand-calculations of PSI ($\text{PSI} = 0.0875$).
  - Replaced text loop diagrams with a **premium inline SVG flowchart** visualizing the production debugging feedback loop.
- **Notebooks 07, 08, & 09:**
  - **[07_attention_and_transformer_prerequisites.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/07_attention_and_transformer_prerequisites.ipynb):** Implements scaled dot-product attention in PyTorch, asserting the context vector matches $[27.8592, 37.8592]^T$ (matching the guide).
  - **[08_nlp_evaluation.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/08_nlp_evaluation.ipynb):** Calculates BLEU scores in NLTK (asserting $=0.3679$) and ROUGE-1 recall from scratch (asserting Recall $=0.5000$ and F1 $=0.6667$).
  - **[09_production_nlp.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/notebooks/09_production_nlp.ipynb):** Computes PSI in NumPy (asserting $=0.0875$) and Wasserstein distance (EMD) in SciPy (verifying exact matching outputs at `0.2000`).

---

## Batch 5: High-Frequency Q&As & Compilation
**Completed: 2026-07-30**

### Changes Made:
- **Module 10 ([10_interview_questions.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/10_interview_questions.md)):** Upgraded technical questions (13, 17, 18, 20, 22, 32) with detailed calculations, derivations, and formulas that align exactly with the rest of the guides (including step-by-step TF-IDF unit normalizations, Laplace transition matrices, perplexity branching factors, SGNS loss derivations, LSTM Constant Error Carousel Jacobians, and BERTScore greedy cosine alignments).
- **Revision Cheatsheet ([nlp_interview_cheatsheet.md](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/modules/nlp_interview_cheatsheet.md)):** Created a highly dense, 1-page revision markdown document containing a variables legend, complexity table, master equations reference, and 5 troubleshooting steps.
- **Aggregation script ([compile_nlp.py](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/helpers/compile_nlp.py)):** Extended to compile both the master study guide and the interview cheatsheet into HTML/PDF templates, utilizing headless Microsoft Edge print.

### Verification:
- Ran `helpers/compile_nlp.py` to assert clean HTML parsing, KaTeX formula rendering, and Edge headless PDF generations. Both `nlp_master_study_guide.pdf` and `nlp_interview_cheatsheet.pdf` generated successfully.

---

## File Structure Reorganization
**Completed: 2026-07-30**

To streamline directory management and keep the module level clean, the project files have been reorganized:
- **Source Modules (`modules/`)**: Holds all Markdown modules (`01_...` to `10_...`), cheatsheet (`nlp_interview_cheatsheet.md`), improvement log (`nlp_module_improvements.md`), the roadmap checklist (`task.md`), and this `walkthrough.md`.
- **Helper Scripts (`helpers/`)**: Holds all executable Python scripts, including the notebook builders (`build_batch*.py`), the graphics generator (`generate_nlp_plots.py`), and the document aggregator (`compile_nlp.py`).
- **Jupyter Notebooks (`notebooks/`)**: Unchanged, contains executed companion Jupyter Notebooks.
- **Plots & Assets (`plots/`)**: Unchanged, contains output graph visualizers.
- **Module Level Root Deliverables**: Contains exclusively the compiled PDF and HTML final files:
  - `nlp_master_study_guide.html` / `nlp_master_study_guide.pdf`
  - `nlp_interview_cheatsheet.html` / `nlp_interview_cheatsheet.pdf`
