# Module 10: Model Evaluation, Benchmark Metrics, & Alignment Evaluation

## 1. Introduction & Intuition

### The Core Bottleneck
Evaluating LLMs is a challenging engineering task. Because LLMs generate free-form natural language, standard classification metrics (like Accuracy, F1-score) are not applicable. 

In pre-training, we rely on **Perplexity (PPL)**, which measures how well a model predicts the next token. However, PPL only evaluates token probability distributions, not reasoning. 

For reasoning, we rely on benchmarks like **MMLU** (multi-task knowledge), **GPQA** (PhD-level science reasoning), and **AIME/MATH** (mathematics). 

The evaluation bottleneck is **Contamination and High Variance**:
*   **Contamination**: Pre-training datasets are so large that benchmarks often leak into training sets, rendering results meaningless.
*   **Stochasticity**: Tiny changes in prompts or parsing rules can cause wild swings in model outputs. 
Furthermore, aligning models using Reinforcement Learning requires a scoring mechanism. Deciding whether to evaluate only the final output (ORM) or each intermediate reasoning step (PRM) dictates how well a model can learn complex reasoning paths.

### High-Level Intuition
*   **Perplexity (PPL)**: Measures the model's uncertainty. Think of a multiple-choice exam. A model with low perplexity is confident and correct. A model with high perplexity is confused, guessing between many different options. Mathematically, it is the exponential of the cross-entropy loss.
*   **Benchmarks**:
    *   **MMLU**: Tests undergraduate-level general knowledge. Now saturated by top models.
    *   **GPQA**: Graduate-level science questions designed to be extremely hard for AI and easy for PhD experts.
    *   **AIME**: American Invitational Mathematics Examination problems, used to test high-level math reasoning.
*   **ORM vs. PRM**:
    *   **Outcome-Supervised (ORM)**: Like a teacher who only grades the final answer on a math test. If you copy someone else's work but get the right number, you get a 100%. This can reward models that hallucinate incorrect reasoning chains but guess the right answer.
    *   **Process-Supervised (PRM)**: Like a teacher who grades your step-by-step working. If you make a mistake in step 3, you get penalized, even if you guess the final number. This directly rewards logical reasoning and helps train reasoning models.

---

### Reward Modeling Evaluation Architectures
Below is a comparison of Outcome-Supervised (ORM) vs. Process-Supervised (PRM) reward boundaries:

<div class="eval-container" style="display: flex; gap: 20px; justify-content: center; margin: 20px 0; font-family: system-ui, -apple-system, sans-serif; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 280px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 15px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h4 style="margin: 0 0 10px 0; color: #ef4444; border-bottom: 2px solid #fee2e2; padding-bottom: 5px;">Outcome-Supervised (ORM)</h4>
    <p style="font-size: 13px; line-height: 1.6; color: #334155;">
      Evaluates <strong>only the final result</strong> of a reasoning chain.
    </p>
    <div style="font-size: 11px; background-color: #f8fafc; padding: 8px; border-radius: 4px; border: 1px dashed #cbd5e1;">
      Reasoning path: [Step 1] &rarr; [Step 2] &rarr; [Step 3] &rarr; <strong>[Final Answer]</strong> <span style="color:#ef4444; font-weight:bold;">&larr; Checked here</span>
    </div>
    <div style="font-size: 11px; color: #b91c1c; font-weight: bold; margin-top: 10px; background-color: #fef2f2; padding: 6px; border-radius: 4px;">
      Issue: Can reward correct answers arrived at via incorrect logic.
    </div>
  </div>
  <div style="flex: 1; min-width: 280px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 15px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h4 style="margin: 0 0 10px 0; color: #10b981; border-bottom: 2px solid #d1fae5; padding-bottom: 5px;">Process-Supervised (PRM)</h4>
    <p style="font-size: 13px; line-height: 1.6; color: #334155;">
      Evaluates <strong>every individual step</strong> in the reasoning trace.
    </p>
    <div style="font-size: 11px; background-color: #f8fafc; padding: 8px; border-radius: 4px; border: 1px dashed #cbd5e1;">
      Reasoning path: [Step 1] <span style="color:#10b981; font-weight:bold;">&larr; OK</span> &rarr; [Step 2] <span style="color:#10b981; font-weight:bold;">&larr; OK</span> &rarr; [Step 3] <span style="color:#ef4444; font-weight:bold;">&larr; FAIL</span>
    </div>
    <div style="font-size: 11px; color: #047857; font-weight: bold; margin-top: 10px; background-color: #ecfdf5; padding: 6px; border-radius: 4px;">
      Benefit: Directly aligns model thoughts, helping RL optimize search trees.
    </div>
  </div>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### Mathematical Formulations

#### 1. Perplexity (PPL)
$$\text{PPL}(X) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(x_i \mid x_{<i})\right)$$

*   **Purpose & High-level Intuition:** Measures model uncertainty. In practice, we evaluate the cross-entropy loss $H(X) = -\frac{1}{N} \sum \ln P(x_i \mid x_{<i})$. Perplexity is simply the exponential of cross-entropy loss:
    $$\text{PPL}(X) = e^{H(X)}$$
    A perplexity of $V$ (vocab size) represents a model selecting uniformly at random. A perplexity of 1.0 represents a model predicting the sequence with absolute confidence and 100% accuracy.

---

### Hand Calculations: Perplexity
Let's compute the perplexity for a short sequence of length $N = 2$.
Assume our model outputs the following conditional probabilities for the target tokens:
*   Probability of token 1: $P(x_1) = 0.2231$ (log probability: $\ln(0.2231) \approx -1.5$).
*   Probability of token 2: $P(x_2 \mid x_1) = 0.6065$ (log probability: $\ln(0.6065) \approx -0.5$).

#### Step 1: Compute Average Negative Log-Likelihood (Cross-Entropy Loss)
$$\begin{aligned}
\text{Loss} &= -\frac{1}{2} \left( \ln P(x_1) + \ln P(x_2 \mid x_1) \right) \\
&= -\frac{1}{2} \left( -1.5 + (-0.5) \right) \\
&= -\frac{1}{2} \left( -2.0 \right) \\
&= 1.0
\end{aligned}$$

#### Step 2: Compute Perplexity
$$\begin{aligned}
\text{PPL} &= e^{\text{Loss}} \\
&= e^{1.0} \\
&\approx 2.7183
\end{aligned}$$

*   **Conclusion:** The perplexity of the sequence is **2.72**. This means the model is, on average, as confused as if it were choosing between 2.72 options at each step.

---

### Tensor & Shape Tracking
*   **Logits Matrix**: `[B, L, V]`
*   **Targets vector**: `[B, L]` (contains token IDs).
*   **Loss vector (element-wise)**: `[B, L]`
*   **Cross-entropy reduction output**: Scalar float `loss`.
*   **PPL output**: Scalar float `ppl`.

---

## 3. Implementation & Reference Code

Below is a self-contained PyTorch implementation of Perplexity calculation from logits.

```python
import torch
import torch.nn as nn

def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # logits shape: [B, L, V]
    # targets shape: [B, L]
    B, L, V = logits.shape
    
    # 1. Reshape tensors for PyTorch CrossEntropyLoss
    # cross_entropy expects [B * L, V] inputs and [B * L] targets
    flat_logits = logits.view(-1, V)
    flat_targets = targets.view(-1)
    
    # 2. Compute average cross entropy loss
    # ignore_index is used to bypass padding tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(flat_logits, flat_targets)
    
    # 3. Take exponential of loss to obtain perplexity
    perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()

# Verification block
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, V = 2, 4, 10
    
    # Mock logits and targets
    logits = torch.randn(B, L, V)
    targets = torch.tensor([
        [3, 5, 2, -100],  # Last token is padding
        [1, 9, 7, 0]
    ])
    
    loss, ppl = calculate_perplexity(logits, targets)
    print(f"Computed Cross-Entropy Loss: {loss:.4f}")
    print(f"Computed Perplexity: {ppl:.4f}")
    
    # Check bounds
    assert ppl > 1.0, "Perplexity cannot be less than 1.0!"
    print("Perplexity calculation successfully verified!")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Establishing reliable evaluation metrics for free-form language generation, and providing reward signals for reinforcement learning.
*   **Why Introduced over Legacy Approaches:** ROUGE/BLEU (overlap-based metrics) evaluate lexical similarities but fail to capture semantic accuracy or logical steps. PRMs grade reasoning steps directly, encouraging correct logic.
*   **Key Failure Modes & Limitations:** Goodhart's Law: when a metric becomes a target, models optimize to score high on it (e.g. by exploiting benchmark multiple-choice layouts) without acquiring actual general intelligence.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Perplexity loss reduction runs in $O(B \cdot L \cdot V)$ operations.
*   **Space/Memory Footprint:** Evaluations require hosting evaluation libraries, but add negligible runtime VRAM overhead.
*   **Primary Bottleneck Type:** CPU-bound parsing routines; GPU-bound during evaluation inference loops.
*   **Variable Legend:** $B$ = Batch Size, $L$ = Sequence Length, $V$ = Vocabulary Size.

### 3. Production & Scalability
*   **Deployment Considerations:** Online telemetry monitors model perplexity over time. A sudden jump in perplexity indicates a **data drift** (e.g. users inputting queries in new languages or formats that the model was not trained on).
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does a model's perplexity decrease as we train it on more tokens, but downstream task accuracies can sometimes plateau?
        *   *A:* Perplexity measures how well the model predicts the token probability distribution. Initially, the model learns basic syntax, grammar, and sentence structures, which drastically reduces cross-entropy loss (and perplexity). However, downstream tasks often require high-level reasoning and specific knowledge, which depends on training past syntactic patterns.
    2.  *Q:* Explain how data contamination can be detected in LLM pre-training datasets.
        *   *A:* Contamination can be detected by calculating model perplexity on the benchmark test questions. If a model shows an unnaturally low perplexity (e.g. close to 1.0) on a set of test questions, it suggests that these exact sequences were present in the training set (memorization). We also run substring matching and MinHash deduplication between datasets to verify cleanliness.
