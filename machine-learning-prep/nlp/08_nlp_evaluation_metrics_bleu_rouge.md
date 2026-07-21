# Module 08: NLP Evaluation Metrics: Classification, BLEU, ROUGE & Perplexity

This study guide details Precision, Recall, F1 (Macro/Micro/Weighted), BLEU-1 through BLEU-4 (with Brevity Penalty derivations), ROUGE-1, ROUGE-2, ROUGE-L (Longest Common Subsequence), Language Model Perplexity (PPL) mathematical proofs, step-by-step hand calculations, Python code, metric limitations, and interview flashcards.

> **Notebook Companion**: [08_nlp_evaluation_metrics_bleu_rouge.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/08_nlp_evaluation_metrics_bleu_rouge.ipynb)

---

## 1. Classification Metrics Taxonomy for Text

```text
Metric         Mathematical Definition                               Best Production Use Case
----------------------------------------------------------------------------------------------------------------------------------
Precision      TP / (TP + FP)                                        Minimizing False Positives (e.g. Spam detection)
Recall         TP / (TP + FN)                                        Minimizing False Negatives (e.g. Medical diagnosis)
Macro F1       Unweighted average of F1 across all classes           Evaluating rare classes in imbalanced text datasets
Micro F1       Global F1 aggregated across all instances             Overall system throughput accuracy
Perplexity     PPL = exp(CrossEntropyLoss)                           Language model generation fluency evaluation
BLEU           Brevity Penalty * exp(sum(w_n * log(p_n)))            Machine Translation evaluation
ROUGE          Recall-Oriented N-gram / LCS Match                   Text Summarization evaluation
```

---

## 2. BLEU: Bilingual Evaluation Understudy

Developed by IBM (Papineni et al., 2002), **BLEU** evaluates Machine Translation quality by comparing candidate generation $C$ against one or more reference translations $R$.

### 1. Modified N-Gram Precision ($p_n$):
To prevent candidate translations from cheating by repeating a single frequent word (e.g., `"the the the the"`), BLEU clips the count of each n-gram to the maximum frequency it appears in any single reference sentence:

$$p_n = \frac{\sum_{\text{gram}_n \in C} \text{Count}_{\text{clip}}(\text{gram}_n)}{\sum_{\text{gram}_n \in C} \text{Count}(\text{gram}_n)}$$

### 2. Brevity Penalty (BP):
Precision-only metrics favor overly short candidate translations (e.g., candidate `"The"` has $100\%$ precision). The Brevity Penalty penalizes candidates shorter than reference length $r$:

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{if } c \le r \end{cases}$$

Where $c$ is the candidate token length and $r$ is the reference target length.

### 3. Overall BLEU Score Formula:
$$\text{BLEU} = \text{BP} \times \exp\left( \sum_{n=1}^N w_n \log p_n \right)$$

Typically $N=4$ (BLEU-4) with uniform weights $w_n = 1/4 = 0.25$.

---

## 3. ROUGE: Recall-Oriented Understudy for Gisting Evaluation

Developed by Lin (2004), **ROUGE** is primarily used for Text Summarization, measuring how much of the reference summary is captured in the candidate output (Recall-oriented).

### 1. ROUGE-N (N-Gram Recall):
$$\text{ROUGE-N} = \frac{\sum_{S \in \text{Reference}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Reference}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

- **ROUGE-1**: Unigram recall (evaluates information coverage).
- **ROUGE-2**: Bigram recall (evaluates local phrase fluency).

### 2. ROUGE-L (Longest Common Subsequence):
ROUGE-L uses the Longest Common Subsequence (LCS) of in-order tokens (allowing non-contiguous gaps):

$$R_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|R|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(C, R)}{|C|}$$

$$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$

---

## 4. Perplexity (PPL): Language Model Evaluation

**Perplexity** measures how well a probability model predicts a text sequence. Intuitively, it represents the average branch factor (number of equal choices) the model faces when predicting the next token.

### Mathematical Derivation:
For sequence $W = (w_1, w_2, \dots, w_N)$, Perplexity is the reciprocal geometric mean of sequence probability:

$$\text{PPL}(W) = P(w_1, w_2, \dots, w_N)^{-1/N} = \left( \prod_{i=1}^N P(w_i | w_{<i}) \right)^{-1/N}$$

Taking the natural logarithm:

$$\log \text{PPL}(W) = -\frac{1}{N} \sum_{i=1}^N \log P(w_i | w_{<i}) = \mathcal{L}_{\text{CrossEntropy}}$$

Exponetiating both sides yields the fundamental production equivalence:

$$\mathbf{\text{PPL}(W) = \exp(\mathcal{L}_{\text{CrossEntropy}})}$$

- **Interpretation**: A Perplexity of **10** means the model is as uncertain at each token as if it were choosing uniformly at random among **10 options**. Lower perplexity indicates superior predictive confidence.

---

## 5. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we evaluate a Candidate Translation against a Reference Translation:
- **Candidate ($C$)**: `"the cat sat on mat"` (Length $c = 5$ tokens)
- **Reference ($R$)**: `"the cat sat on the mat"` (Length $r = 6$ tokens)

### 1. Calculate Modified Precision $p_1$ and $p_2$:
- **Unigrams in $C$**: `["the", "cat", "sat", "on", "mat"]` (5 tokens)
  - All 5 unigrams appear in $R$. $p_1 = 5 / 5 = \mathbf{1.0000}$
- **Bigrams in $C$**: `["the cat", "cat sat", "sat on", "on mat"]` (4 bigrams)
  - Bigrams in $R$: `["the cat", "cat sat", "sat on", "on the", "the mat"]`
  - Matches: `"the cat"`, `"cat sat"`, `"sat on"`. (3 matches out of 4)
  - $p_2 = 3 / 4 = \mathbf{0.7500}$

### 2. Calculate Brevity Penalty (BP):
- Candidate length $c = 5$, Reference length $r = 6$. Since $c \le r$:

$$\text{BP} = \exp\left(1 - \frac{6}{5}\right) = \exp(-0.20) \approx \mathbf{0.8187}$$

### 3. Compute BLEU-2 Score:
$$\text{BLEU-2} = \text{BP} \times \exp\left(0.5 \log p_1 + 0.5 \log p_2\right)$$

$$\text{BLEU-2} = 0.8187 \times \exp\left(0.5 \log(1.0) + 0.5 \log(0.75)\right) = 0.8187 \times \exp(0 + 0.5(-0.2877))$$

$$\text{BLEU-2} = 0.8187 \times \exp(-0.1438) = 0.8187 \times 0.8660 \approx \mathbf{0.7090}$$

### 4. Calculate ROUGE-1 Recall:
- Reference Unigrams = 6 tokens.
- Candidate contains 5 matching unigrams (`"the"`, `"cat"`, `"sat"`, `"on"`, `"mat"`).

$$\text{ROUGE-1 Recall} = \frac{5}{6} \approx \mathbf{0.8333}$$

### 5. Calculate Perplexity from Loss:
If cross-entropy evaluation loss $\mathcal{L}_{\text{CE}} = 2.3026$:

$$\text{PPL} = \exp(2.3026) = \mathbf{10.0000}$$

---

## 6. Production Python Implementation

```python
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 1. Candidate vs Reference Text Outputs
reference_text = "the cat sat on the mat"
candidate_text = "the cat sat on mat"

ref_tokens = reference_text.split()
cand_tokens = candidate_text.split()

# 2. Compute NLTK BLEU Score
smooth_fn = SmoothingFunction().method1
bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)

print("=== 1. BLEU Score Benchmark ===")
print(f"BLEU-1 Score: {bleu_1:.4f}")
print(f"BLEU-2 Score: {bleu_2:.4f}")

# 3. Compute ROUGE Scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_results = scorer.score(reference_text, candidate_text)

print("\n=== 2. ROUGE Score Benchmark ===")
print(f"ROUGE-1 Recall: {rouge_results['rouge1'].recall:.4f} | F1: {rouge_results['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Recall: {rouge_results['rouge2'].recall:.4f} | F1: {rouge_results['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Recall: {rouge_results['rougeL'].recall:.4f} | F1: {rouge_results['rougeL'].fmeasure:.4f}")

# 4. Compute Perplexity from Cross-Entropy Loss
eval_loss = 1.8543
perplexity = np.exp(eval_loss)
print("\n=== 3. Language Model Perplexity Benchmark ===")
print(f"Cross-Entropy Loss: {eval_loss:.4f} --> Perplexity (PPL): {perplexity:.4f}")
```

> [!NOTE]
> **Production Evaluation Alert:**
> - BLEU and ROUGE rely purely on exact surface string n-gram matching. Paraphrased candidates with identical semantic meaning (e.g. candidate `"The feline rested on the rug"`) score $0$ on BLEU/ROUGE. Modern systems supplement these with **BERTScore** or **LLM-as-a-Judge**.

---

## 7. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **BLEU Paraphrase Blindness**: BLEU heavily penalizes valid semantically correct outputs if they do not match reference n-gram surface forms.
   - *Remediation*: Use contextual embedding evaluation (BERTScore) or LLM-as-a-Judge.
2. **Length Cheating**: Generative models evaluated without Brevity Penalty can produce single-word outputs with $100\%$ precision scores.
   - *Remediation*: Always include Brevity Penalty (BP) in custom BLEU pipelines.

---

## 8. Master Interview Flashcards & Questions

#### Q1: Prove the mathematical relationship between Cross-Entropy Loss and Perplexity.
- **Answer:** Perplexity is defined as reciprocal geometric mean probability: $\text{PPL}(W) = P(w_1, \dots, w_N)^{-1/N} = \left( \prod P(w_i | w_{<i}) \right)^{-1/N}$. Taking natural log gives $\log \text{PPL}(W) = -\frac{1}{N} \sum \log P(w_i | w_{<i})$, which is exactly the definition of Cross-Entropy Loss $\mathcal{L}_{\text{CE}}$. Exponentiating both sides yields $\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$.

#### Q2: What is Brevity Penalty in BLEU, and why is it necessary?
- **Answer:** Modified n-gram precision alone rewards short candidate outputs. For example, a candidate consisting of a single token `"the"` that appears in the reference gets $100\%$ precision ($p_1 = 1.0$). Brevity Penalty (BP) penalizes candidate sentences shorter than the reference length $r$: $\text{BP} = \exp(1 - r/c)$ for $c \le r$, ensuring short sentences receive low overall BLEU scores.

#### Q3: Compare BLEU vs. ROUGE evaluation orientation.
- **Answer:** BLEU is **Precision-Oriented** (primarily used in Machine Translation to verify that generated candidate n-grams appear in the reference target). ROUGE is **Recall-Oriented** (primarily used in Text Summarization to verify that key reference information is fully captured in the generated candidate).
