# Module 07: Text Generation & Decoding Strategies

This study guide explains the engineering mechanics of text generation and token decoding strategies in LLMs, detailing greedy decoding, beam search, temperature scaling, Top-k/Top-p sampling, and repetition penalties.

---

## 1. Autoregressive Generation

In decoder-only autoregressive models, generation is a sequential loop where the model predicts the probability distribution of the next token $w_t$ given all previous tokens $w_{1:t-1}$:

$$P(w_t | w_{1:t-1}) = \text{Softmax}(\mathbf{z}_t)$$

where $\mathbf{z}_t$ is the logit vector output by the final language modeling classification head. The selected token is appended to the input context, and the loop repeats.

---

## 2. Deterministic vs. Stochastic Decoding

### Greedy Decoding
- **Concept**: Selects the token with the highest probability at every step:
  $$w_t = \arg\max_i P(w_{t,i} | w_{1:t-1})$$
- *Limitations*: Highly repetitive (stuck in local loops) and lacks creativity.

### Beam Search
- **Concept**: Maintains the top $B$ highest probability candidate sequences (beams) at each step.
- *Trade-off*: Improves search coverage but multiplies computation by $O(B)$ at every turn.

---

## 3. Stochastic Sampling Parameters

To introduce diversity, we sample from the probability distribution, modified by three control parameters:

### Temperature ($T$)
Scales the raw logit values before applying Softmax:
$$\tilde{z}_i = \frac{z_i}{T}$$
- **High $T$ ($T > 1.0$)**: flattens the distribution (increases entropy, making generation more creative/random).
- **Low $T$ ($T \to 0.0$)**: sharpens the distribution, collapsing it toward greedy selection.

### Top-k Sampling
Restricts sampling to the $K$ tokens with the highest logit values. The logits of all other tokens are set to $-\infty$, completely pruning them.

### Top-p (Nucleus) Sampling
Restricts sampling to the smallest subset of tokens whose cumulative probability exceeds threshold $p$ (e.g. $p=0.90$):
$$\sum_{i \in V_{\text{subset}}} P(w_i) \ge p$$
This dynamically scales vocabulary options based on model confidence.

### Repetition Penalty ($\theta$)
Applies a divisor or multiplier to logits of tokens that have already been generated in the output sequence to prevent repetitive loops:
$$\tilde{z}_i = \begin{cases} z_i / \theta & \text{if } z_i > 0 \\ z_i \cdot \theta & \text{if } z_i \le 0 \end{cases} \quad (\text{for } \theta \ge 1.0)$$

---

## 4. Step-by-Step Hand Calculation (Stochastic Decoding)

- **Scenario**: Vocabulary $|V| = 3$. Raw logits generated: $\mathbf{z} = [2.0, 1.0, -1.0]$.
- **Parameters**: Temperature $T = 0.5$, Top-k limit $k = 2$.
- **Repetition Penalty**: $\theta = 1.2$ applied to token index 0 (which occurred in the prompt history).
- **Calculation**:
  1. Apply Repetition Penalty to index 0:
     - Since $z_0 = 2.0 > 0$:
       $$\tilde{z}_0 = \frac{2.0}{1.2} \approx 1.6667$$
     - Others remain unchanged: $\tilde{z}_1 = 1.0$, $\tilde{z}_2 = -1.0$.
     - Adjusted logits: $\tilde{\mathbf{z}} = [1.6667, 1.0, -1.0]$
  2. Scale by Temperature $T = 0.5$ (divide all logits by $0.5$):
     $$\tilde{\mathbf{z}}_{\text{scaled}} = \left[ \frac{1.6667}{0.5}, \frac{1.0}{0.5}, \frac{-1.0}{0.5} \right] = [3.3333, 2.0, -2.0]$$
  3. Apply Top-k sampling ($k=2$, keep top 2, set rest to $-\infty$):
     - Sorted logits: index 0 ($3.3333$), index 1 ($2.0$). Index 2 is pruned.
     - Pruned logits: $\tilde{\mathbf{z}}_{\text{top2}} = [3.3333, 2.0, -\infty]$
  4. Evaluate Softmax probabilities:
     - Compute exponentials:
       $$e^{3.3333} \approx 28.0316, \quad e^{2.0} \approx 7.3891, \quad e^{-\infty} = 0.0000$$
     - Sum: $28.0316 + 7.3891 = 35.4207$
     - Probabilities:
       $$P_0 = \frac{28.0316}{35.4207} \approx 0.7914$$
       $$P_1 = \frac{7.3891}{35.4207} \approx 0.2086$$
       $$P_2 = 0.0000$$
- **Result**: The sampling probability distribution is $[0.7914, 0.2086, 0.0000]$.

---

## 5. Comparison of Decoding Strategies

| Strategy | Deterministic | Latency Profile | Best Use Case | Primary Limitation |
|---|---|---|---|---|
| **Greedy Decoding** | Yes | Low | Code generation, structured JSON extraction | Repetitive text, lacks variety |
| **Beam Search** | Yes | High ($O(B)$) | Translation, translation tasks | High computation overhead |
| **Top-k / Top-p** | No | Low | Creative writing, open dialogue agents | Prone to gibberish if parameters are too loose |

---

## 6. Interview Questions & Production Trade-offs

### What problem does this solve?
Transforms raw, unbounded logit vectors output by the neural network into valid token sampling probability distributions.

### Why was it introduced?
Stochastic sampling controls (like top-k/top-p and temperature) balance model creativity with structural syntax coherence.

### What are its limitations?
Excessive repetition penalties can lead to generation breakdown where the model outputs completely random symbols to avoid repeating grammatical stop words.

### Computational Complexity (Time & Memory)
- **Top-k sort time**: $O(|V| \cdot \log k)$ operations.
- **Top-p cumulative sum sort**: $O(|V| \cdot \log |V|)$ operations.

### Component Variable Denotation Legend
- $|V|$: Vocabulary size.
- $k$: Top-k count.
- $p$: Top-p cumulative probability.
- $\theta$: Repetition penalty parameter.
- $T$: Temperature scaling factor.

### Production Use Cases:
- Customer service bots using Greedy decoding for structured JSON schemas to ensure parser reliability.
- Story generation systems using Top-p $0.90$ and $T=0.8$ to produce varied descriptions.

### Follow-up Questions Interviewers Ask:
1. *Why is Top-p preferred over Top-k in modern conversational LLMs?*
   - **Answer**: Top-k uses a fixed limit $K$. In cases where the probability distribution is highly concentrated (e.g. predicting the next word of `"The capital of France is [Paris]"`), the correct answer might have $99\%$ probability, while the remaining $K-1$ choices have tiny probabilities, making their selection a logical failure. In cases where the distribution is flat (e.g., `"I went to the restaurant and ordered [food/steak/pizza...]"`), $K$ might prune out highly plausible choices. Top-p dynamically sizes the active candidate set based on cumulative probability, adapting to model confidence.
2. *What is the mathematical consequence of setting Temperature $T \to 0.0$ at runtime?*
   - **Answer**: As $T \to 0.0$, the difference between the maximum logit $z_{\text{max}}$ and other logits $z_j$ scaled by $1/T$ approaches infinity. During Softmax computation, the exponential value of the maximum logit dominates the sum: $\lim_{T \to 0} P(w_{\text{max}}) = 1.0$ and $P(w_j) = 0.0$ for all other tokens. This mathematically collapses the sampling distribution to greedy selection.
3. *How do you implement Repetition Penalty efficiently without adding overhead?*
   - **Answer**: Maintain a set of generated token IDs in memory. During logit decoding of the current step, iterate through the historical token set and apply the penalty division directly to their index positions in the raw logit array before running softmax, adding minimal $O(N)$ lookup latency.
