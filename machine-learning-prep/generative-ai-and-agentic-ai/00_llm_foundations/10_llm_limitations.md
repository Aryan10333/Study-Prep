# Module 10: Common LLM Limitations & Evaluation Metrics

This study guide explains the core limitations of Large Language Models (hallucinations, reasoning bounds, sychophancy, alignment tradeoffs) and defines standard evaluation metrics, deriving Perplexity step-by-step.

---

## 1. Core Model Limitations & Failures

LLMs exhibit several behavior limits rooted in their next-token prediction objective:

### 1. Hallucinations
- **Factuality vs. Generation**: Hallucinations occur when a model generates grammatically coherent but factually incorrect assertions.
- **Why they occur**: The maximum-likelihood objective optimizes for token sequence *plausibility* in the training corpus, not structural *factuality*. The model has no internal mechanism to ground assertions or gauge its own factual uncertainty, selecting high-probability transitions blindly.

### 2. Reasoning Bottlenecks & Lack of Grounding
- **Semantic Grounding**: Models manipulate tokens based on statistical co-occurrences without real-world physical or sensory grounding (the "Chinese Room" argument).
- **Reasoning Limits**: Models fail on arithmetic or logic operations requiring multi-step execution graphs because they evaluate steps left-to-right in single forward passes.

### 3. Alignment Trade-offs (Sycophancy & Mode Collapse)
- **Sycophancy**: During RLHF (Reinforcement Learning from Human Feedback), models learn to output answers that match user biases rather than truth, as human evaluators favor polite agreement.
- **Mode Collapse**: RLHF fine-tuning can over-index on preferred responses, restricting output diversity and flattening generation creativity.

---

## 2. Evaluation Metrics

To debug and evaluate generations, engineers use statistical similarity metrics:

### 1. BLEU (Bilingual Evaluation Understudy)
Computes n-gram precision between generation and reference targets, penalized for short lengths (Brevity Penalty). Standard in machine translation.

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Measures n-gram overlap recall between generation and reference targets. Standard in summarization tasks (ROUGE-L measures Longest Common Subsequence).

### 3. Perplexity (PPL)
Perplexity measures how well a model predicts a sample. It is the exponentiated cross-entropy loss of the sequence:

$$\text{PPL}(X) = \exp \left( -\frac{1}{L} \sum_{t=1}^{L} \log P(x_t | x_{<t}) \right)$$

- **Intuition**: A perplexity of $K$ means that at each token generation step, the model is on average as confused as if it had to choose uniformly among $K$ options in the vocabulary. Lower is better.

---

## 3. Step-by-Step Hand Calculation (Perplexity)

- **Scenario**: Target sequence length $L = 2$.
- **Model predictions**: True target tokens are output at step 1 and step 2 with following probabilities:
  - Step 1 ($t=1$): $P(x_1) = 0.50$
  - Step 2 ($t=2$): $P(x_2 | x_1) = 0.20$
- **Calculation**:
  1. Compute the natural log-likelihoods of the correct tokens:
     $$\log P(x_1) = \log(0.50) \approx -0.6931$$
     $$\log P(x_2 | x_1) = \log(0.20) \approx -1.6094$$
  2. Compute average cross-entropy loss (negative mean log-likelihood):
     $$\text{Average Loss} = -\frac{1}{2} \left( \log P(x_1) + \log P(x_2 | x_1) \right)$$
     $$\text{Average Loss} = -\frac{1}{2} (-0.6931 - 1.6094) = -\frac{1}{2} (-2.3025) = 1.1513$$
  3. Compute Perplexity:
     $$\text{PPL}(X) = \exp(1.1513) = e^{1.1513} \approx 3.1623$$
- **Result**: The perplexity of the generated sequence is $\approx 3.1623$.

---

## 4. Evaluation Metrics Trade-offs

| Metric | Primary Use Case | Advantage | Limitation |
|---|---|---|---|
| **BLEU** | Machine Translation | Fast, standard baseline | Fails to capture synonym variance; purely syntactic overlap |
| **ROUGE** | Text Summarization | Measures recall (captures content) | Susceptible to verbose, low-quality generations |
| **Perplexity** | Language Model Evaluation | Direct metric of probability fit | Strongly dependent on vocabulary size; incomparable across models |

---

## 5. Interview Questions & Production Trade-offs

### What problem does this solve?
Establishes metrics to track alignment drift, evaluate token generation fit, and detect output decay in production.

### Why was it introduced?
Cross-entropy loss does not capture model uncertainty directly; perplexity translates loss into vocabulary branching factor intuition.

### What are its limitations?
Reference-based overlap metrics (BLEU/ROUGE) correlate poorly with human judgment on complex tasks like coding or dialogue.

### Computational Complexity (Time & Memory)
- **Perplexity computation**: $O(L \cdot |V|)$ forward pass computations.
- **ROUGE-L LCS calculation**: $O(L_{\text{gen}} \cdot L_{\text{ref}})$ dynamic programming time.

### Component Variable Denotation Legend
- $L$: Sequence token length.
- $|V|$: Vocabulary size.
- $L_{\text{gen}}, L_{\text{ref}}$: Generated and reference sequence lengths.

### Production Use Cases:
- Monitoring pipelines calculating Perplexity on production logs to detect model drift or data shifts.
- Automated testing runs evaluating model translations using BLEU scores.

### Follow-up Questions Interviewers Ask:
1. *Why are Perplexity values incomparable between models with different tokenizers?*
   - **Answer**: Perplexity relies directly on sequence token length $L$ and vocabulary size $|V|$. If Model A uses a subword tokenizer with $|V|=32\text{k}$ and Model B uses a byte-level tokenizer with $|V|=256$, Model B will decompose the same sentence into a much longer token sequence. Since perplexity averages log-likelihoods over sequence length, the score is mathematically dependent on vocabulary segmentation density, making direct cross-tokenizer comparisons invalid.
2. *Why does RLHF introduce sycophancy, and how can we mitigate it?*
   - **Answer**: Human evaluators tend to rate polite, agreement-biased responses higher than objective corrections of user-stated misconceptions. Models trained on these ratings learn sycophancy to maximize reward. Mitigation requires injecting objective verification rules in evaluator prompts, balancing human ratings with automated factual consistency checks.
3. *What is the relationship between Perplexity and Cross-Entropy Loss?*
   - **Answer**: Perplexity is the exponential function of the average cross-entropy loss: $\text{PPL} = e^{H(X)}$. If the cross-entropy loss of a sequence is $0.0$, the perplexity is $e^0 = 1.0$, indicating absolute certainty. If the loss is $\log |V|$ (uniform random prediction over vocabulary), the perplexity is $e^{\log |V|} = |V|$, indicating uniform confusion across the entire vocabulary.
