# Module 04: Statistical Language Models & Smoothing

Statistical Language Models (LMs) estimate probability distributions over token sequences. This module covers sequence probability chain rules, smoothing algorithms, and perplexity metrics.

---

## 1. Sequence Probability and the Markov Assumption

A language model calculates the joint probability of a sequence of tokens $W = (w_1, w_2, \dots, w_m)$:

### The Probability Chain Rule
Using conditional probability, the exact joint probability of a text sequence is computed as:

$$P(w_1, w_2, \dots, w_m) = \prod_{i=1}^m P(w_i \mid w_1, w_2, \dots, w_{i-1})$$

### The Markov Assumption
Calculating joint probabilities requires tracking long histories, which quickly becomes computationally intractable. The Markov assumption simplifies this by assuming the probability of a word depends only on the preceding $k$ words:
- **Bigram Model** ($k=1$): Assumes word depends only on the immediate predecessor:
  $$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-1})$$

---

## 2. Maximum Likelihood Estimation (MLE)

MLE calculates probabilities by counting occurrences in a training corpus:

### Bigram MLE Formula
$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

Where $C(w_{i-1}, w_i)$ is the bigram co-occurrence count, and $C(w_{i-1})$ is the unigram count of the preceding token.

---

## 3. Smoothing Techniques and Good-Turing Intuition

If an n-gram never occurs in the training corpus, MLE assigns it a probability of $0$. A single zero count makes the joint sequence probability collapse to $0$:
$$P(W) = 0$$
Smoothing reallocates probability mass from frequent n-grams to unseen sequences.

### 1. Laplace (Add-One) Smoothing
Adds $1$ to all counts to ensure no probability evaluates to $0$:

$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$

Where $|V|$ is vocabulary size.
*Trade-off*: Assigns too much probability mass to unseen words in large vocabularies.

### 2. Backoff (Katz Backoff)
If the higher-order n-gram count is zero, the model backs off to estimate probability using lower-order n-grams (e.g. bigram $\rightarrow$ unigram):

$$P_{\text{Katz}}(w_i \mid w_{i-1}) = \alpha(w_{i-1}) P(w_i) \quad \text{if } C(w_{i-1}, w_i) = 0$$

Where $\alpha(w_{i-1})$ is a normalization factor.

### 3. Interpolation
Linearly combines probabilities from multiple n-gram orders:

$$P_{\text{Interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_1 P(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_3 P(w_i)$$

Where $\sum \lambda_j = 1$.

### 4. Good-Turing Smoothing (Intuition)
Good-Turing smoothing estimates the probability of unseen items based on the frequency of single-occurrence items (hapax legomena). 
- *Intuition*: If you read a book and find $100$ words that only occur once, it is highly likely that the next new word you encounter is similar to those single-occurrence words. It reallocates probability mass based on the "frequency of frequencies."

---

## 4. Perplexity (PPL) Intuition

![Perplexity Decay](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/perplexity_decay.png)

Perplexity evaluates language model performance on a test sequence.

### Conceptual Formulation
$$\text{PPL} = e^{\text{Cross-Entropy Loss}}$$

*Branching Factor Intuition*: Perplexity represents the average branching factor (i.e. the number of equally likely next words the model must choose from).
- **PPL = 10**: The model is choosing among 10 equally likely words at each step (indicating high confidence and strong prediction).
- **PPL = 100**: The model is choosing among 100 equally likely words (indicating high uncertainty and poor prediction).

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
  - **Memory Bottleneck**: Parameter count scales exponentially as $O(|V|^N)$ as the n-gram context order $N$ increases.
  - **Zero-Probability Defect**: Fails to generate probabilities for unseen sequences without smoothing.
- **Computational Complexity (Time & Memory)**
  - **Inference Time**: $O(1)$ constant time lookup if using precomputed n-gram tables.
  - **Storage Memory**: $O(|V|^N)$ space.
- **Component Variable Denotation Legend**
  - $m$: Token count of the evaluation sequence.
  - $|V|$: Vocabulary token size.
  - $N_c$: Number of unique n-grams appearing exactly $c$ times.
  - $\lambda_i$: Interpolation coefficients.
- **Production Use Cases**
  - Text auto-complete query routing systems.
  - Basic speech recognition transcript decoding.
- **Follow-up questions interviewers ask**
  - *Why does Perplexity decrease as you increase N in an N-gram model?* (Higher-order models capture more local context, reducing uncertainty at each step, which lowers cross-entropy loss and perplexity).
  - *How do you choose lambda interpolation parameters?* (By optimizing the coefficients using expectation-maximization or grid search on a held-out validation dataset).
