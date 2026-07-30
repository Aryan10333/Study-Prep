# Module 04: Statistical Language Models & Smoothing

Statistical Language Models (LMs) estimate probability distributions over token sequences. This module covers sequence probability chain rules, the Markov assumption, smoothing algorithms, perplexity metrics, and code implementations.

---

## 1. Language Model Fundamentals & The Chain Rule

A language model calculates the joint probability of a sequence of tokens $W = (w_1, w_2, \dots, w_m)$ appearing in a corpus. 

### Why Predict the Next Word?
In NLP, predicting the next word is the core task of generative models. By modeling the conditional probability of the next word given all preceding context words:
$$P(w_i \mid w_1, w_2, \dots, w_{i-1})$$
we can auto-regressively generate coherent text sequences.

### Derivation of the Probability Chain Rule
From basic probability theory, the conditional probability of event $B$ given event $A$ is:
$$P(A \cap B) = P(B \mid A) P(A)$$
For three events $A$, $B$, and $C$, we generalize this by grouping $A \cap B$ as a single event:
$$P(A \cap B \cap C) = P(C \mid A \cap B) P(A \cap B) = P(C \mid A, B) P(B \mid A) P(A)$$
Generalizing this step-by-step to a sequence of $m$ word tokens, the exact joint probability of a text sequence is computed as:
$$P(w_1, w_2, \dots, w_m) = P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) \dots P(w_m \mid w_1, \dots, w_{m-1}) = \prod_{i=1}^m P(w_i \mid w_1, \dots, w_{i-1})$$

---

## 2. The Markov Assumption

Calculating joint probabilities using the chain rule requires tracking long contexts. As the sequence length $m$ grows, the number of possible histories scales exponentially ($|V|^{m-1}$), which quickly becomes computationally intractable.

The Markov assumption simplifies this by assuming the probability of a word depends only on the preceding $k$ words:
- **Unigram Model** ($k=0$): Assumes word occurrences are completely independent of context:
  $$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i)$$
- **Bigram Model** ($k=1$): Assumes word depends only on the immediate predecessor (first-order Markov chain):
  $$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-1})$$
- **Trigram Model** ($k=2$): Assumes word depends on the preceding two words (second-order Markov chain):
  $$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-2}, w_{i-1})$$

---

## 3. Maximum Likelihood Estimation & Smoothing

MLE calculates probabilities by counting occurrences in a training corpus. For bigrams:

$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

Where $C(w_{i-1}, w_i)$ is the bigram count, and $C(w_{i-1})$ is the unigram count of the prefix token.

### The Zero-Probability Defect
If an n-gram never occurs in the training corpus, MLE assigns it a probability of $0$. A single zero transition causes the joint sequence probability of an entire document to collapse to 0:
$$P(W) = \prod P(w_i \mid \dots) = 0$$
Smoothing reallocates probability mass from frequent n-grams to unseen sequences to resolve this defect.

---

### Step-by-Step Hand-Calculations on a Tiny Corpus:
Let's analyze a tiny corpus of two sentences:
- Sentence 1: `"cat sat"`
- Sentence 2: `"sat cat"`
- Unique Vocabulary: `["cat", "sat"]` ($|V| = 2$)
- Total token count: 4.
- Unigram counts: $C(\text{"cat"}) = 2$, $C(\text{"sat"}) = 2$.
- Bigram counts:
  - $C(\text{"cat", "sat"}) = 1$
  - $C(\text{"sat", "cat"}) = 1$
  - $C(\text{"cat", "cat"}) = 0$
  - $C(\text{"sat", "sat"}) = 0$

#### 1. Maximum Likelihood Estimation (MLE)
Calculate the probability of sequence `"cat cat"` under MLE:
- $P_{\text{MLE}}(\text{"cat"} \mid \text{"cat"}) = \frac{C(\text{"cat", "cat"})}{C(\text{"cat"})} = \frac{0}{2} = 0.0$
- **Result:** $P_{\text{MLE}}(\text{"cat cat"}) = P(\text{"cat"}) \cdot P(\text{"cat"} \mid \text{"cat"}) = 0.5 \cdot 0 = 0.0$ (Zero-Probability Defect).

#### 2. Laplace (Add-One) Smoothing
Laplace smoothing adds $1$ to all n-gram counts, inflating the denominator by vocabulary size $|V|$:
$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$

Let's compute the smoothed probabilities for the sequence `"cat cat"`:
- $$P_{\text{Laplace}}(\text{"cat"} \mid \text{"cat"}) = \frac{C(\text{"cat", "cat"}) + 1}{C(\text{"cat"}) + |V|} = \frac{0 + 1}{2 + 2} = \frac{1}{4} = 0.2500$$
- $$P_{\text{Laplace}}(\text{"sat"} \mid \text{"cat"}) = \frac{C(\text{"cat", "sat"}) + 1}{C(\text{"cat"}) + |V|} = \frac{1 + 1}{2 + 2} = \frac{2}{4} = 0.5000$$

##### Findings & Interpretation:
Laplace smoothing successfully prevents the transition probability from collapsing to 0, assigning a non-zero probability ($0.2500$) to the unseen bigram `"cat cat"`. It achieves this by shifting probability mass away from the observed transition `"cat sat"` (whose probability dropped from $1.0$ under MLE to $0.5$ under Laplace).

#### 3. Katz Backoff & Interpolation (Intuition)
- **Katz Backoff:** If the count of a higher-order n-gram is zero, we back off to lower-order models (e.g. bigram to unigram):
  $$P_{\text{Katz}}(w_i \mid w_{i-1}) = \alpha(w_{i-1}) P(w_i) \quad \text{if } C(w_{i-1}, w_i) = 0$$
- **Linear Interpolation:** Combines scores across multiple n-gram orders:
  $$P_{\text{Interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_1 P(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_3 P(w_i)$$
  Where $\sum \lambda_j = 1$.

---

## 4. Perplexity (PPL) Derivation & Intuition

Perplexity evaluates language model performance on a test sequence.

### Mathematical Derivation from Cross-Entropy Loss
Let the cross-entropy loss $H(P, Q)$ of a sequence of length $m$ be:
$$H(P, Q) = -\frac{1}{m} \sum_{i=1}^m \log_2 P(w_i \mid w_1, \dots, w_{i-1})$$
Perplexity (PPL) is defined as the exponentiated cross-entropy loss:
$$\text{PPL}(W) = 2^{H(P, Q)} = e^{-\frac{1}{m} \sum \ln P(w_i \mid \dots)} = \left( \prod_{i=1}^m P(w_i \mid w_1, \dots, w_{i-1}) \right)^{-\frac{1}{m}}$$

### Branching Factor Intuition
Perplexity represents the **average branching factor** (the number of equally likely next words the model must choose from at each step).
- **PPL = 10:** The model is as confused as if it were choosing among 10 equally likely words (indicating high confidence and strong prediction).
- **PPL = 100:** The model has 100 equally likely choices (indicating high uncertainty).

---

### Python Code Integration

The following Python code implements Laplace smoothing and perplexity calculations on our micro-corpus, verifying the hand-calculations:

```python
import numpy as np

# Word index mappings: "cat" -> 0, "sat" -> 1
# Corpus representation
unigram_counts = np.array([2, 2]) # cat: 2, sat: 2
bigram_counts = np.array([
    [0, 1],  # cat->cat: 0, cat->sat: 1
    [1, 0]   # sat->cat: 1, sat->sat: 0
])
V = len(unigram_counts)

# Laplace-smoothed conditional transition matrix P(w_i | w_{i-1})
P_smoothed = np.zeros((V, V))
for i in range(V):
    P_smoothed[i, :] = (bigram_counts[i, :] + 1) / (unigram_counts[i] + V)

print("Laplace-Smoothed Transition Matrix P(w_j | w_i):")
print(P_smoothed)

# Calculate perplexity of a test sequence: ["cat", "cat", "sat"]
# We assume initial word probability P("cat") = 0.5
test_transitions = [
    (0, 0), # cat -> cat
    (0, 1)  # cat -> sat
]

# Joint probability: P("cat") * P("cat"|"cat") * P("sat"|"cat")
probabilities = [0.5, P_smoothed[0, 0], P_smoothed[0, 1]]
m = len(probabilities)

joint_prob = np.prod(probabilities)
log_prob_sum = np.sum(np.log(probabilities))

# Perplexity: exp(-1/m * sum(ln(P)))
perplexity = np.exp(-1/m * log_prob_sum)

print(f"\nJoint Probability of ['cat', 'cat', 'sat']: {joint_prob:.6f}")
print(f"Computed Perplexity: {perplexity:.4f}")
```

---

> [!TIP]
> **Production Insight: N-gram Memory Scaling Limits**
> Storing raw n-gram count tables in memory scales exponentially as $O(|V|^N)$. A trigram model ($N=3$) with a vocabulary $|V| = 100,000$ requires storing up to $10^{15}$ count entries, which crashes standard lookup databases. In production search auto-completes, prune n-gram hash tables by filtering entries with count frequency $< 5$, keeping memory usage minimal.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Estimates the probability distribution over text sequences, allowing models to generate coherent text.
- **Why was it introduced?**
  Introduced to score token sequences based on natural language frequencies.
- **What are its limitations?**
  - **Memory Bottleneck:** Parameter count scales exponentially as $O(|V|^N)$ as the n-gram context order $N$ increases.
  - **Zero-Probability Defect:** Fails to generate probabilities for unseen sequences without smoothing.
- **Computational Complexity (Time & Memory)**
  - **Inference Time**: $O(1)$ constant time lookup if using precomputed n-gram tables.
  - **Storage Memory**: $O(|V|^N)$ space.
- **Component Variable Denotation Legend**
  - $m$: Token count of the evaluation sequence.
  - $|V|$: Vocabulary token size.
  - $N$: N-gram context size.
  - $\lambda_i$: Interpolation coefficients.
- **Production Use Cases**
  - Text auto-complete query routing systems.
  - Basic speech recognition transcript decoding.
- **Follow-up questions interviewers ask**
  - *Why does Perplexity decrease as you increase N in an N-gram model?* (Higher-order models capture more local context, reducing uncertainty at each step, which lowers cross-entropy loss and perplexity).
  - *How do you choose lambda interpolation parameters?* (By optimizing the coefficients using expectation-maximization or grid search on a held-out validation dataset).
