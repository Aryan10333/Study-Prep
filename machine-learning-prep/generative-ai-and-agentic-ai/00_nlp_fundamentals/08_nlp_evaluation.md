# Module 08: NLP Evaluation Metrics & Semantic Validation

Evaluating natural language outputs requires measuring both exact tokens and semantic meaning. This module details classification and generation metrics, derives BLEU and ROUGE scores, highlights the limitations of n-gram overlaps, and explains semantic metrics like BERTScore.

---

## 1. Classification Metrics: Precision, Recall, and F1

For classification tasks (e.g. sentiment classification, token classification), performance is evaluated using confusion matrix derivatives:

- **Precision (Positive Predictive Value)**: Measures the proportion of positive identifications that were actually correct:
  $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
- **Recall (Sensitivity)**: Measures the proportion of actual positives that were correctly identified:
  $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics when dealing with class imbalances:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 2. Generation Metrics: BLEU, ROUGE, and METEOR

For sequence generation tasks (e.g. translation, summarization), models are evaluated against ground-truth reference texts:

### Bilingual Evaluation Understudy (BLEU)
BLEU measures n-gram precision (how many generated n-grams appear in the reference text) combined with a penalty for short generations:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

Where:
- $p_n$ is the modified n-gram precision:
  $$p_n = \frac{\sum_{\text{ngram}} \text{Count}_{\text{clip}}(\text{ngram})}{\sum_{\text{ngram}} \text{Count}(\text{ngram})}$$
- $w_n$ are uniform weights (typically $0.25$ for $N=4$).
- $\text{BP}$ is the Brevity Penalty, which penalizes generations that are shorter than the reference:
  $$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$
  Where $c$ is candidate length and $r$ is reference length.

---

### Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
ROUGE measures n-gram recall (how many reference n-grams are captured by the generated text):
- **ROUGE-N**: Measures n-gram overlap:
  $$\text{ROUGE-N} = \frac{\sum_{\text{ngram}} \text{Count}_{\text{match}}(\text{ngram})}{\sum_{\text{ngram}} \text{Count}_{\text{reference}}(\text{ngram})}$$
- **ROUGE-L**: Measures the Longest Common Subsequence (LCS) between candidate and reference texts, preserving word order without requiring exact n-gram alignments.

---

## 3. Semantic Blind Spots of Overlap Metrics

N-gram overlap metrics (BLEU, ROUGE) measure exact match counts. Consequently, they suffer from two major **semantic blind spots**:

1. **Orthogonal Synonyms**: A generated sentence like `"A feline rested on the rug"` compared against reference `"The cat sat on the mat"` receives a BLEU score of $0$ because no words overlap, despite sharing identical meanings.
2. **Grammar & Logical Inversions**: A candidate sentence `"not bad, actually great"` compared against `"not great, actually bad"` shares identical unigrams and bigrams, but has the opposite meaning. Overlap metrics assign these sentences high scores.

---

## 4. Semantic Evaluation: Sentence-BERT and BERTScore

To resolve these blind spots, production systems use embedding-based semantic evaluation:

### Sentence-BERT (SBERT)
Maps entire sentences to dense, fixed-sized semantic vectors $\mathbf{u}$ and $\mathbf{v}$ using a Siamese network. Similarity is measured using cosine distance:

$$\text{Similarity} = \text{CosineSimilarity}(\mathbf{u}, \mathbf{v})$$

### BERTScore: Contextual Token Alignments
BERTScore computes token-level semantic alignments using contextual embeddings (e.g. BERT hidden states):

```
Candidate:   A   feline  rested   on  the  rug
              \    /       |       |   |   /
Reference:   The  cat     sat     on  the mat
             (Aligned via Cosine Similarity of contextual tokens)
```

1. **Represent Tokens**: Map words in candidate $C$ and reference $R$ to contextual token vectors.
2. **Compute Similarity**: Calculate a cosine similarity matrix between all candidate and reference tokens.
3. **Greedy Matching**: Match each token in candidate $C$ to its most similar token in reference $R$:
   $$\text{Recall} = \frac{1}{|R|} \sum_{r_i \in R} \max_{c_j \in C} \mathbf{r}_i^T \mathbf{c}_j$$
   $$\text{Precision} = \frac{1}{|C|} \sum_{c_j \in C} \max_{r_i \in R} \mathbf{r}_i^T \mathbf{c}_j$$
4. **F1 Calculation**: Compute the harmonic mean of the greedy precision and recall scores. This captures semantic similarity even when the words used are different.

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
