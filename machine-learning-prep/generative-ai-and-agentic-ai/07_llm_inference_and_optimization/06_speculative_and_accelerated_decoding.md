# 06. Speculative & Accelerated Decoding: Parallel Verification and Draft Models

Autoregressive token generation requires loading a model's entire weights from high-bandwidth memory (HBM) to GPU SRAM for every single token produced. Speculative decoding bypasses this memory bottleneck by generating multiple candidate tokens using a computationally cheap draft mechanism and verifying them in parallel during a single forward pass of the main target model.

---

## 1. Speculative Decoding Theory

Speculative decoding relies on two models:
1. **Draft Model ($q$)**: A small, fast model (e.g. Llama 1B) that generates tokens quickly.
2. **Target Model ($p$)**: A large, accurate model (e.g. Llama 70B) whose outputs we want to match.

At each speculative step:
1. The draft model runs autoregressively for $\gamma$ steps to generate a draft sequence:
   $$X_{\text{draft}} = (x_1, x_2, \dots, x_\gamma)$$
2. The target model evaluates the entire block of draft tokens in a single parallel forward pass, computing the target probabilities for each position:
   $$P_{\text{target}} = (p_1, p_2, \dots, p_{\gamma+1})$$
3. The engine uses **rejection sampling** to verify each draft token sequentially.

---

## 2. Lossless Distribution Preservation Proof

We must prove that the rejection sampling protocol produces outputs that strictly match the target model's probability distribution $p(x)$, ensuring no quality degradation.

### Rejection Sampling Protocol
For candidate token $x^*$ proposed by draft model distribution $q(x)$ at step $t$:
1. We accept $x^*$ with probability:
   $$\text{P}(\text{accept}) = \min\left(1, \frac{p(x^*)}{q(x^*)}\right)$$
   If accepted, we set $x_t = x^*$ and move to the next position.
2. If rejected, we evict the remaining draft tokens and sample $x_t$ from the normalized difference distribution:
   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$$

### Proof
Let $x_{\text{selected}}$ be the token selected by the protocol. The probability of selecting a specific token $x$ is the sum of the probability of proposing and accepting it, plus the probability of rejecting the draft token and selecting $x$ from the fallback distribution $p'(x)$:

$$\text{P}(x_{\text{selected}} = x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + P(\text{reject}) \cdot p'(x)$$

Let's define the total rejection mass $M$:

$$M = \sum_y \max(0, p(y) - q(y))$$

Because $\sum_y q(y) = 1$ and $\sum_y p(y) = 1$, we can show that the sum of negative differences equals the sum of positive differences:

$$\sum_y \max(0, q(y) - p(y)) = \sum_y \max(0, p(y) - q(y)) = M$$

The probability of rejection is:

$$\text{P}(\text{reject}) = 1 - \sum_y \min(q(y), p(y)) = \sum_y (q(y) - \min(q(y), p(y))) = \sum_y \max(0, q(y) - p(y)) = M$$

Now, substitute $P(\text{reject}) = M$ and the definition of $p'(x)$ back into the selection equation:

$$\text{P}(x_{\text{selected}} = x) = \min(q(x), p(x)) + M \cdot \left(\frac{\max(0, p(x) - q(x))}{M}\right)$$

$$\text{P}(x_{\text{selected}} = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x))$$

We evaluate this equation for two possible cases:

#### Case 1: $p(x) \ge q(x)$
- Here, $\min(q(x), p(x)) = q(x)$ and $\max(0, p(x) - q(x)) = p(x) - q(x)$.
- Substituting these values:
  $$\text{P}(x_{\text{selected}} = x) = q(x) + p(x) - q(x) = p(x)$$

#### Case 2: $p(x) < q(x)$
- Here, $\min(q(x), p(x)) = p(x)$ and $\max(0, p(x) - q(x)) = 0$.
- Substituting these values:
  $$\text{P}(x_{\text{selected}} = x) = p(x) + 0 = p(x)$$

In both cases, the probability of selecting token $x$ is exactly $p(x)$, matching the target model's output distribution. $\blacksquare$

---

## 3. Modern Speculative Architectures

Modern engines use advanced draft mechanisms to bypass tokenizer mismatches and dual-model VRAM overheads:

```html
<div style="display: flex; flex-direction: column; gap: 16px; font-family: 'Segoe UI', sans-serif; margin: 20px 0;">
  <!-- EAGLE -->
  <div style="border: 1px solid #3b82f6; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #1e3a8a; font-size: 14px;">1. EAGLE-2 / EAGLE-3 (Feature Speculation)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">
      Instead of tokens, EAGLE drafts hidden feature vectors at the target model's second-to-last layer. It processes candidates using a dynamic tree structures and parallel tree attention.
    </div>
  </div>

  <!-- Medusa -->
  <div style="border: 1px solid #10b981; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #065f46; font-size: 14px;">2. Medusa (Multi-Head Prediction)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">
      Appends multiple linear prediction heads to the target model's output layer. Head $k$ predicts the token $t+k$ concurrently, verifying candidates using a pre-computed validation tree mask.
    </div>
  </div>

  <!-- Prompt Lookup -->
  <div style="border: 1px solid #f59e0b; border-radius: 6px; padding: 12px; background-color: #f8fafc;">
    <h4 style="margin: 0 0 6px 0; color: #92400e; font-size: 14px;">3. Prompt-Lookup Speculation (Zero-VRAM / Zero-Train)</h4>
    <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">
      Scans prompt history for repeating N-grams. If the current sequence matches a past N-gram sequence, the engine drafts the trailing tokens directly from context history, requiring no helper draft models.
    </div>
  </div>
</div>
```

---

## 4. Expected Tokens & Speedup Math

The effectiveness of speculative decoding is governed by the **Average Acceptance Rate** ($\alpha \in [0, 1]$), representing the probability that a draft token is accepted by the target model.

If the draft length is $\gamma$, the expected number of tokens generated per target forward pass is:

$$E[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

- If $\alpha = 0.8$ and $\gamma = 5$:
  $$E[\text{tokens}] = \frac{1 - 0.8^6}{1 - 0.8} = \frac{1 - 0.262144}{0.2} = 3.689 \text{ tokens per forward pass}$$
- If $\alpha = 0.5$ and $\gamma = 5$:
  $$E[\text{tokens}] = \frac{1 - 0.5^6}{1 - 0.5} = \frac{1 - 0.015625}{0.5} = 1.968 \text{ tokens per forward pass}$$

As $\alpha$ drops, the speedup factor decreases. If the draft model overhead exceeds the gains from multi-token parallel verification, speculative decoding can run slower than raw autoregressive decoding.

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
