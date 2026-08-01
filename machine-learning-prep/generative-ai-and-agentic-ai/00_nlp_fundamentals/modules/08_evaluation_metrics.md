# Module 08: Classical NLP Evaluation Metrics

## 1. Introduction & Intuition

### The Core Bottleneck
Building language pipelines requires standard, objective, and automated metrics to evaluate performance. In structured machine learning, we compare predictions against labels directly using accuracy or mean squared error. In NLP, however, output strings can have different lengths, word ordering, and synonyms. If a model translates a sentence as `"The cat sat on the rug"` and the reference is `"On the mat sat the cat"`, a simple token accuracy metric will score it as a failure. Evaluating text generation requires metrics that can handle semantic equivalence, sequence alignment variations, and probabilistic perplexity. The bottleneck is designing automated evaluation metrics that correlate well with human judgment without requiring expensive manual verification.

### High-Level Intuition
Think of evaluation metrics as diagnostic magnifying glasses. 
If we want to test how well a model has learned the language pattern probabilities, we measure how "surprised" it is when reading a test set of real text. If it assigns high probabilities to the real words, its surprise is low, which means its perplexity is low.
If we want to test translation quality, we count the overlap of word sequences (n-grams) between the model's translation and reference translations. If the model matches word pairs and triplets while preserving sentence length, its BLEU score is high.

---

## 2. Core Concepts & Mathematical Formulation

### Perplexity (PPL)
Perplexity measures how well a language model predicts a sample.
*   **Intuition & Practical Use:** Lower perplexity indicates the model is less surprised by the test words, meaning it has modeled the true probability distribution of the language accurately.
*   **Mathematical Formulation:**
    $$\text{PPL}(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i | w_{<i}) \right) = \exp(\text{Cross-Entropy Loss})$$

---

### BLEU (Bilingual Evaluation Understudy)
BLEU measures the precision overlap of n-grams between candidate translation and reference translations. It scales the score using a **Brevity Penalty (BP)** to prevent models from outputting short sentences containing only a few highly confident words.
*   **Intuition & Practical Use:** Evaluates machine translation quality automatically by penalizing word discrepancies and incorrect sentence lengths.
*   **Mathematical Formulation:**
    $$\text{BLEU} = \text{BP} \times \exp \left( \sum_{n=1}^N w_n \log p_n \right)$$
    $$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{if } c \le r \end{cases}$$
    Where $p_n$ is modified n-gram precision, $w_n = 1/N$, $c$ is candidate length, and $r$ is reference length.

---

### Word Error Rate (WER)
WER is the standard metric for speech-to-text transcription. It measures the minimum edit distance (Levenshtein distance) at the word level, normalizing by reference length:
*   **Mathematical Formulation:**
    $$\text{WER} = \frac{S + D + I}{N}$$
    Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference words.

---

### Classification Metrics & Class Imbalance
In text classification (e.g. Spam detection), accuracy is an insufficient metric due to class imbalance. If $99\%$ of emails are benign, a dummy classifier predicting "benign" constantly yields $99\%$ accuracy while failing completely. We evaluate:
*   **Precision:**
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
*   **Recall:**
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
*   **F1-Score:** Harmonic mean of precision and recall:
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### Hand Calculations

#### 1. BLEU Score with Brevity Penalty
Let's compute the BLEU-2 score for a candidate translation.
*   **Reference ($R$):** `"the cat sat"` (length $r = 3$)
*   **Candidate ($C$):** `"the cat"` (length $c = 2$)

*   **Step 1: Compute modified n-gram precisions**
    1.  **Unigram Precision ($p_1$):**
        *   Candidate tokens: `["the", "cat"]`. Both appear in reference. Match count = 2.
        $$p_1 = \frac{2}{2} = 1.0$$
    2.  **Bigram Precision ($p_2$):**
        *   Candidate bigrams: `["the cat"]`. Appears in reference. Match count = 1.
        $$p_2 = \frac{1}{1} = 1.0$$
*   **Step 2: Compute Brevity Penalty (BP)**
    Since candidate length $c = 2$ is less than reference length $r = 3$:
    $$\text{BP} = \exp\left(1 - \frac{3}{2}\right) = \exp(-0.5) \approx 0.6065$$
*   **Step 3: Compute BLEU-2 Score**
    Let weights $w_1 = 0.5$ and $w_2 = 0.5$:
    $$\text{BLEU-2} = \text{BP} \times \exp\left(0.5 \log p_1 + 0.5 \log p_2\right)$$
    $$\text{BLEU-2} = 0.6065 \times \exp\left(0.5 \log(1.0) + 0.5 \log(1.0)\right) = 0.6065 \times \exp(0) = 0.6065$$
The output candidate is penalized from $1.0$ down to $0.6065$ because it was too brief.

#### 2. Word Error Rate (WER)
Let's compute WER for a speech-to-text hypothesis.
*   **Reference ($R$):** `"the cat sat"` (length $N = 3$)
*   **Hypothesis ($H$):** `"the cat was sitting"` (length = 4)
*   **Step 1: Compute edit distance alignments**
    *   `"the"` $\to$ `"the"` (Match)
    *   `"cat"` $\to$ `"cat"` (Match)
    *   `"sat"` $\to$ `"was"` (Substitution, $S=1$)
    *   Insert `"sitting"` (Insertion, $I=1$). Deletions $D = 0$.
*   **Step 2: Compute WER**
    $$\text{WER} = \frac{S + D + I}{N} = \frac{1 + 0 + 1}{3} = \frac{2}{3} \approx 0.6667 \quad (66.7\%)$$

#### 3. Classification Metrics
Let's compute metrics for a spam filter on a dataset of 100 emails.
*   **Actual Labels:** 10 Spam, 90 Ham.
*   **Classifier Predictions:**
    *   True Positives (SPAM classified as SPAM): $\text{TP} = 8$.
    *   False Negatives (SPAM classified as HAM): $\text{FN} = 2$.
    *   False Positives (HAM classified as SPAM): $\text{FP} = 4$.
    *   True Negatives (HAM classified as HAM): $\text{TN} = 86$.
*   **Step 1: Compute Precision**
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{8}{8 + 4} = \frac{8}{12} \approx 0.6667$$
*   **Step 2: Compute Recall**
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{8}{8 + 2} = \frac{8}{10} = 0.8000$$
*   **Step 3: Compute F1-Score**
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.6667 \times 0.8000}{0.6667 + 0.8000} = 2 \times \frac{0.5334}{1.4667} \approx 0.7273$$

The model achieves $66.7\%$ precision, $80.0\%$ recall, and a balanced F1-score of $0.7273$.

---

#### Tensor & Shape Tracking
*   Logits tensor: `[B, L, V]`.
*   Cross-entropy loss output: `[]` (scalar float).
*   Perplexity output: `[]` (scalar float).

---

## 3. Implementation & Reference Code

Below is a Python implementation of perplexity calculation from cross-entropy loss, and Levenshtein edit distance for WER computation.

```python
import numpy as np
import torch
import torch.nn as nn

def run_evaluation_metrics():
    # 1. Compute Perplexity
    torch.manual_seed(42)
    B, L, V = 2, 4, 1000
    
    logits = torch.randn(B, L, V)
    targets = torch.randint(0, V, (B, L))
    
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, V), targets.view(-1))
    perplexity = torch.exp(loss)
    
    print(f"Cross-Entropy Loss: {loss.item():.4f}")
    print(f"Calculated Perplexity: {perplexity.item():.4f}")
    
    # 2. Compute WER
    def compute_wer(reference: str, hypothesis: str) -> float:
        r_words = reference.split()
        h_words = hypothesis.split()
        R, H = len(r_words), len(h_words)
        dp = np.zeros((R + 1, H + 1), dtype=int)
        
        for i in range(R + 1):
            dp[i, 0] = i
        for j in range(H + 1):
            dp[0, j] = j
            
        for i in range(1, R + 1):
            for j in range(1, H + 1):
                if r_words[i-1] == h_words[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = min(dp[i-1, j-1] + 1, dp[i-1, j] + 1, dp[i, j-1] + 1)
                    
        return dp[R, H] / R
        
    ref = "the cat sat on the mat"
    hyp = "the cat was sitting on the mat"
    wer = compute_wer(ref, hyp)
    print(f"\nWER Score: {wer:.4f} ({wer*100:.1f}%)")

if __name__ == "__main__":
    run_evaluation_metrics()
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Quantitative validation of generation alignment and model probabilities.
*   **Why Introduced over Legacy Approaches:** Automating overlap scoring replaced slow, expensive human evaluations, allowing real-time model validation during training epochs.
*   **Key Failure Modes & Limitations:** BLEU and ROUGE evaluate exact token overlap, completely penalizing synonyms (e.g. if candidate generates `"dog"` and reference is `"canine"`, overlap score is 0).

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Perplexity computation scales as $O(B \times L \times V)$ due to vocabulary loss calculations. Edit distance for WER scales as $O(R \times H)$ where $R$ and $H$ are word counts.
*   **Space/Memory Footprint:** DP grid for edit distance scales as $O(R \times H)$ memory.
*   **Primary Bottleneck Type:** Compute-bound during logit-softmax evaluations over large vocabulary sizes.

### 3. Production & Scalability
*   **Deployment Considerations:** For validating large-scale generation models (like LLMs), overlap metrics are often supplemented with semantic similarity lookups (like BERTScore) or LLM-based evaluations (G-Eval) to prevent synonym penalty errors.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Explain why perplexity is highly dependent on the vocabulary size ($|V|$), and why comparing perplexity values between two models with different tokenizers is mathematically invalid.
        *   *A:* Perplexity is the exponent of the cross-entropy loss over a probability distribution of size $|V|$. A model with a smaller vocabulary size (e.g. $8\text{k}$) naturally has a smaller pool of options to select from at each step compared to a model with a massive vocabulary (e.g. $128\text{k}$). The cross-entropy loss of the smaller vocab model will be lower, yielding a lower perplexity simply due to vocab size, not model capability. Comparing perplexity is only valid when the vocabulary index size and tokenizer mappings are identical.
    2.  *Q:* What are the key architectural limitations of BLEU for evaluating translation semantic quality?
        *   *A:* BLEU has three main limits: it penalizes synonyms because it matches exact string n-grams; it is insensitive to grammatical differences as long as n-gram counts match; and it does not capture sentence-level coherence or semantic intent, focusing only on local window overlaps.
