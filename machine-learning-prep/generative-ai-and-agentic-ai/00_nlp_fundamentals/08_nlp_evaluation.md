# Module 08: NLP Evaluation Metrics & Semantic Validation

Evaluating natural language outputs requires measuring both exact tokens and semantic meaning. This module details classification and generation metrics, highlights the limitations of n-gram overlaps, and explains semantic metrics like BERTScore.

---

## 1. Classification Metrics: Precision, Recall, and F1

For classification tasks (e.g. sentiment classification, spam detection), models are evaluated using a confusion matrix:

- **Precision**: Measures the proportion of positive predictions that were actually correct:
  $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
- **Recall**: Measures the proportion of actual positives that were correctly identified:
  $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics when dealing with class imbalances:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 2. Generation Metrics: BLEU and ROUGE

For sequence generation tasks (e.g. translation, summarization), models are evaluated against ground-truth reference texts:

- **BLEU (Bilingual Evaluation Understudy)**: Measures n-gram precision (how many generated words appear in the reference text) combined with a brevity penalty to prevent short, trivial outputs.
  - *Equation*: $\text{BLEU} = \text{Brevity Penalty} \times \exp(\sum w_n \log p_n)$
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures n-gram recall (how many reference words are captured by the generated text).
  - *ROUGE-L*: Evaluates the Longest Common Subsequence (LCS) to track sequence word order without requiring exact n-gram alignments.

---

## 3. Semantic Blind Spots of Overlap Metrics

N-gram overlap metrics (BLEU, ROUGE) measure exact match counts. Consequently, they suffer from two major **semantic blind spots**:

1. **Orthogonal Synonyms**: A generated sentence like `"A feline rested on the rug"` compared against reference `"The cat sat on the mat"` receives a BLEU score of $0$ because no words overlap, despite sharing identical meanings.
2. **Grammar & Logical Inversions**: A candidate sentence `"not bad, actually great"` compared against `"not great, actually bad"` shares identical unigrams and bigrams, but has the opposite meaning. Overlap metrics assign these sentences high scores.

---

## 4. Semantic Evaluation: Sentence-BERT and BERTScore

To resolve these blind spots, production systems use embedding-based semantic evaluation:

- **Sentence-BERT (SBERT)**: Maps entire sentences to dense, fixed-sized semantic vectors using a Siamese network. Similarity is measured using cosine distance.
- **BERTScore**: Computes token-level semantic alignments using contextual embeddings (e.g. BERT hidden states):

### BERTScore Alignment Matrix Example
```
                  Reference:   "The"     "cat"     "sat"     "on"      "the"     "mat"
Candidate:
  "A"                           0.12      0.08      0.05      0.02      0.09      0.04
  "feline"                      0.09      0.88      0.12      0.05      0.08      0.11   <-- Match found!
  "rested"                      0.04      0.11      0.82      0.10      0.05      0.07   <-- Match found!
  "on"                          0.01      0.04      0.09      0.95      0.02      0.04   <-- Match found!
  "the"                         0.08      0.07      0.05      0.03      0.98      0.06   <-- Match found!
  "rug"                         0.05      0.12      0.08      0.04      0.07      0.85   <-- Match found!
```

BERTScore aligns each candidate token to its most semantically similar reference token using cosine similarity, capturing synonyms (e.g. `"feline"` matching `"cat"` with $0.88$ score) and resolving n-gram overlap limitations.

---

> [!TIP]
> **Production Insight: Automated Metrics vs. Human Audits**
> While automated metrics like BLEU or BERTScore are excellent for Continuous Integration (CI) regression checks, they cannot replace human evaluations. In production, establish a random audit loop that routes $1\text{--}5\%$ of live system outputs to human reviewers to construct a manual "golden evaluation set" for calibration.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Quantifies the accuracy, semantic alignment, and structural quality of natural language outputs.
- **Why was it introduced?**
  Introduced to automate translation and summarization scoring, replacing expensive human evaluation loops.
- **What are its limitations?**
  BLEU/ROUGE fail to capture semantic meaning; embedding models (BERTScore) introduce computational overhead and depend on the quality of the underlying encoder.
- **Computational Complexity (Time & Memory)**
  - **BLEU Evaluation Time**: $O(L_{\text{cand}} \cdot L_{\text{ref}})$ string search.
  - **BERTScore Evaluation Time**: $O(L_{\text{cand}} \cdot L_{\text{ref}} \cdot d)$ plus the forward pass latency of the encoder network.
- **Component Variable Denotation Legend**
  - $c$: Candidate token sequence length.
  - $r$: Reference token sequence length.
  - $d$: Contextual embedding dimension size.
- **Production Use Cases**
  - Continuous Integration (CI) regression testing for translation pipelines.
  - Semantic similarity evaluations in question-answering systems.
- **Follow-up questions interviewers ask**
  - *Why does BLEU use a brevity penalty?* (Without a brevity penalty, a candidate model could output a single high-confidence word like `"the"` and receive a perfect precision score of $1.0$).
  - *How does BERTScore handle spelling variations?* (Contextual embeddings capture semantic similarities, allowing misspelled or related words to match based on their surrounding contexts).
