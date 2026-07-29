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

## 2. Rejection Sampling: The Coin-Toss Intuition

Speculative decoding is **mathematically lossless**, meaning the generated tokens are identical to what the large target model would have output on its own. It guarantees this by utilizing a corrected rejection sampling protocol.

### The Verification Protocol
For a candidate token $x^*$ proposed by the draft model distribution $q(x)$:
1. **Acceptance Probability**: We accept $x^*$ with a probability scaled by the density ratio:
   $$\alpha = \min\left(1, \frac{p(x^*)}{q(x^*)}\right)$$
2. **Acceptance**: If accepted, we keep the token and check the next drafted token.
3. **Rejection**: If rejected, we throw away all subsequent drafted tokens, and sample a replacement token from the normalized difference distribution:
   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$$

### Why This Works (The Intuition)
Think of verification as a biased coin-toss:
- **Case A: Target model likes the token *more* ($p(x^*) \ge q(x^*)$)**: The ratio is $\ge 1.0$, so we accept it 100% of the time.
- **Case B: Target model likes the token *less* ($p(x^*) < q(x^*)$)**: The ratio is $< 1.0$, so we flip a coin with an acceptance probability equal to that ratio.
- **Fallback**: If the coin comes up tails (rejection), the target model selects a token from the difference distribution, which represents the tokens the target model preferred but the draft model under-sampled.

This dynamic correction guarantees that the final selection probability is exactly $p(x)$, aligning perfectly with the target model's original distribution while bypassing autoregressive compute latency.

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

### Speculative Drafting Strategy Comparison

| Strategy | Drafting Mechanism | Pros | Cons | Production Choice |
|---|---|---|---|---|
| **Draft Model** | Small auxiliary autoregressive model (e.g. 1B) runs ahead. | • Robust; handles general conversational structures well. | • Requires loading a second model in VRAM.<br>• Tokenizer alignment mismatches. | Standard baseline; ideal if helper model is small enough to fit in remaining memory. |
| **Medusa / EAGLE** | Helper prediction heads or feature vectors inside the main model. | • Zero tokenizer issues.<br>• Minimal extra VRAM (no second model loaded). | • Requires custom training and alignment tuning for the heads. | Custom high-performance serving environments (e.g., enterprise APIs). |
| **Prompt-Lookup** | Scans prompt/context history for repeating N-grams and copies them. | • **Zero VRAM & Zero Training**: Plug-and-play compatibility.<br>• Minimal latency overhead. | • Limited to highly repetitive text styles (e.g. code generation, repetitive system logs). | Excellent for code completion servers and document-retrieval RAG bots. |

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
