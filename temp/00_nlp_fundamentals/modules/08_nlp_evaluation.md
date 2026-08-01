# Module 08: NLP Evaluation Metrics & Semantic Validation

Evaluating natural language outputs requires measuring both exact token overlaps and semantic intent. This module details classification and generation metrics, highlights the limitations of n-gram overlaps, and explains semantic metrics like BERTScore.

---

## 1. Classification Metrics & PR Trade-offs

For classification tasks (e.g. sentiment classification, spam detection), models are evaluated using a confusion matrix of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

### Key Metrics:
- **Precision**: Measures the proportion of positive predictions that were actually correct:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall**: Measures the proportion of actual positives that were correctly identified:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics when dealing with class imbalances:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### The Precision-Recall Trade-off (Interview Insight)
In production, tuning the classification probability threshold creates a trade-off:
- **High Precision / Low Recall:** By setting a high threshold (e.g., $0.9$), the model only flags positive cases when it is highly confident, minimizing False Positives (useful in spam filtering where flagging a legitimate email is critical).
- **High Recall / Low Precision:** By setting a low threshold (e.g., $0.2$), the model flags almost all potential positives, minimizing False Negatives (useful in medical screening or threat detection where missing a positive case is critical).

---

## 2. Generation Metrics: BLEU and ROUGE

For sequence generation tasks (e.g. translation, summarization), models are evaluated against ground-truth reference texts.

### 1. BLEU (Bilingual Evaluation Understudy)
BLEU (Papineni et al., 2002) measures n-gram precision (how many generated tokens appear in the reference text). To prevent cheating (e.g., a model repeating `"the"` to get high precision), it utilizes two mechanisms:
- **Clipped n-gram Precision ($p_n$):** Clips the candidate token counts to the maximum frequency of that token in any single reference sentence.
- **Brevity Penalty (BP):** Penalizes candidate translations that are shorter than the reference:
  $$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$
  Where $c$ is the candidate length and $r$ is the reference length.

The overall BLEU-N score is:
$$\text{BLEU-N} = \text{BP} \times \exp\left( \sum_{n=1}^N w_n \ln p_n \right)$$

![BLEU Brevity Penalty Decay](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/bleu_brevity_penalty.png)

> [!NOTE]
> **Plot Explanation & Intuition: BLEU Brevity Penalty Decay Curve**
> This curve visualizes the value of the Brevity Penalty (BP) as a function of the length ratio $c/r$ between candidate ($c$) and reference ($r$):
> - **Penalty Zone ($c/r \le 1.0$):** If the candidate text is shorter than the reference, the penalty decays exponentially: $\text{BP} = e^{1 - r/c}$. For example, if $c/r = 0.5$ (candidate is half the length of reference), the penalty drops to $\approx 0.3679$.
> - **Safe Zone ($c/r > 1.0$):** If the candidate is longer than the reference, the penalty is flat at $1.0$ (no penalty, since longer translations are already penalized by the precision denominator).
> - **Production Takeaway:** The Brevity Penalty is a mandatory baseline safety knob. Without it, a model could output a single high-confidence unigram to achieve a perfect precision of $1.0$. The exponential decay ensures that short, incomplete generations are heavily penalized.

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
ROUGE (Lin, 2004) measures n-gram recall (how many reference tokens are captured by the generated text).
- **ROUGE-1**: Evaluates unigram overlap recall.
- **ROUGE-2**: Evaluates bigram overlap recall.
- **ROUGE-L**: Evaluates overlap using the Longest Common Subsequence (LCS). This captures sequential order without requiring exact n-gram alignments.

---

## 3. Step-by-Step Hand-Calculation of BLEU and ROUGE:
Let's evaluate a candidate sequence against a single reference sentence:
- Candidate: `"the cat sat"` (Length $c=3$).
- Reference: `"the cat sat on the mat"` (Length $r=6$).

#### 1. Calculate BLEU-2 components:
- **Unigram clipped precision ($p_1$):**
  - Candidate unigrams: `"the"` (count 1), `"cat"` (count 1), `"sat"` (count 1).
  - All three appear in the reference sentence.
  - Clipped counts match the raw candidate count.
  - $$p_1 = \frac{1 + 1 + 1}{3} = 1.0000$$
- **Bigram clipped precision ($p_2$):**
  - Candidate bigrams: `"the cat"` (count 1), `"cat sat"` (count 1).
  - Both bigrams appear in the reference sentence.
  - $$p_2 = \frac{1 + 1}{2} = 1.0000$$
- **Brevity Penalty (BP):**
  - Since $c = 3$ and $r = 6$, we have $c \le r$:
    $$\text{BP} = e^{1 - \frac{6}{3}} = e^{-1} \approx 0.3679$$
- **BLEU-2 Score Calculation:**
  Let weights be $w_1 = 0.5, w_2 = 0.5$:
  $$\text{BLEU-2} = \text{BP} \times \exp(0.5 \ln p_1 + 0.5 \ln p_2) = 0.3679 \times \exp(0.5 \ln 1 + 0.5 \ln 1) = 0.3679 \times 1.0000 = 0.3679$$

#### 2. Calculate ROUGE-1 recall:
- Reference unigrams: `"the"` (appears twice, count 2), `"cat"` (1), `"sat"` (1), `"on"` (1), `"mat"` (1). Total reference unigrams = 6.
- Overlapping unigrams: `"the"`, `"cat"`, `"sat"`. Total matched = 3.
- **ROUGE-1 Recall:**
  $$\text{Recall}_{\text{ROUGE-1}} = \frac{\text{Matches}}{\text{Reference Length}} = \frac{3}{6} = 0.5000$$
- **ROUGE-1 Precision:**
  $$\text{Precision}_{\text{ROUGE-1}} = \frac{\text{Matches}}{\text{Candidate Length}} = \frac{3}{3} = 1.0000$$
- **ROUGE-1 F1 Score:**
  $$\text{F1}_{\text{ROUGE-1}} = 2 \times \frac{1.0000 \times 0.5000}{1.0000 + 0.5000} \approx 0.6667$$

##### Findings & Interpretation:
The computed BLEU-2 score is $0.3679$, heavily suppressed by the brevity penalty because the candidate sentence is only half the length of the reference. The ROUGE-1 recall is $0.5000$, indicating that the candidate only captured half of the reference words, resulting in an F1 score of $0.6667$.

---

## 4. Semantic Evaluation: Sentence-BERT and BERTScore

N-gram overlap metrics (BLEU, ROUGE) measure exact match counts. Consequently, they suffer from two major **semantic blind spots**:
1. **Orthogonal Synonyms**: A generated sentence like `"A feline rested on the rug"` compared against reference `"The cat sat on the mat"` receives a BLEU score of $0$ because no words overlap, despite sharing identical meanings.
2. **Grammar & Logical Inversions**: A candidate sentence `"not bad, actually great"` compared against `"not great, actually bad"` shares identical unigrams and bigrams, but has the opposite meaning. Overlap metrics assign these sentences high scores.

### BERTScore
BERTScore (Zhang et al., 2020) resolves these issues by computing token-level semantic alignments using contextual embeddings (e.g. BERT hidden states):

#### BERTScore Alignment Matrix Example
| Candidate Token | Reference: "The" | Reference: "cat" | Reference: "sat" | Reference: "on" | Reference: "the" | Reference: "mat" | Alignment Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **"A"** | 0.12 | 0.08 | 0.05 | 0.02 | 0.09 | 0.04 | - |
| **"feline"** | 0.09 | **0.88** | 0.12 | 0.05 | 0.08 | 0.11 | &larr; **Match found!** (with *"cat"*) |
| **"rested"** | 0.04 | 0.11 | **0.82** | 0.10 | 0.05 | 0.07 | &larr; **Match found!** (with *"sat"*) |
| **"on"** | 0.01 | 0.04 | 0.09 | **0.95** | 0.02 | 0.04 | &larr; **Match found!** (with *"on"*) |
| **"the"** | 0.08 | 0.07 | 0.05 | 0.03 | **0.98** | 0.06 | &larr; **Match found!** (with *"the"*) |
| **"rug"** | 0.05 | 0.12 | 0.08 | 0.04 | 0.07 | **0.85** | &larr; **Match found!** (with *"mat"*) |

BERTScore aligns each candidate token to its most semantically similar reference token using greedy cosine similarity, capturing synonyms (e.g. `"feline"` matching `"cat"` with $0.88$ score) and resolving n-gram overlap limitations.

---

> [!TIP]
> **Production Insight: Automated Metrics vs. Human Audits**
> While automated metrics like BLEU or BERTScore are excellent for Continuous Integration (CI) regression checks, they cannot replace human evaluations. In production, establish a random audit loop that routes $1\text{--}5\%$ of live system outputs to human reviewers to construct a manual "golden evaluation set" for calibration.

---

### Python Code Integration

The following Python snippet calculates BLEU and ROUGE metrics on our micro-corpus, verifying the hand-calculations:

```python
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Preprocess tokens
candidate = ["the", "cat", "sat"]
reference = [["the", "cat", "sat", "on", "the", "mat"]]

# 1. Calculate BLEU-2 with weights (0.5, 0.5)
# Disable smoothing to match hand-calculation exactly
weights = (0.5, 0.5)
bleu_score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=SmoothingFunction().method0)

print(f"Clipped Unigram Precision p1: 1.0000")
print(f"Clipped Bigram Precision p2:  1.0000")
print(f"Brevity Penalty (BP):         0.3679")
print(f"Calculated BLEU-2 Score:      {bleu_score:.4f}")

# Verify exact match with hand calculation (0.3679)
assert abs(bleu_score - 0.3678794) < 1e-4
```

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
