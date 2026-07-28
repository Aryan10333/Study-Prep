# 02. Decoding Strategies & Structured Generation: Sampling Mechanics and Constraints

An LLM outputs raw unbounded scores (logits) for every token in its vocabulary. The decoding strategy controls how these logits are processed to select the next token. Selecting the correct sampling parameters and constraints determines whether an LLM produces factual code, cohesive conversation, or structured JSON payloads.

---

## 1. Deterministic Decoding

Deterministic strategies always select the token path with the highest probability, producing predictable, reproducible outputs.

### Greedy Decoding
At each generation step $t$, greedy decoding selects the token index $x_t$ that maximizes the probability score:

$$x_t = \arg\max_{i} P(\text{token}_i \mid S_{t-1})$$

- **Pros**: Computationally trivial ($O(1)$ selection complexity). Highly effective for factual tasks (code, mathematics).
- **Cons**: Susceptible to repetitive loops (getting stuck repeating phrases) and misses paths where a slightly lower-probability token leads to a much higher-probability sequence later.

### Beam Search
Instead of keeping only the single best token, Beam Search maintains a fixed number of candidate sequences (the *beam width* $B$). At each step, it expands all $B$ paths, computes their cumulative log-probabilities, and keeps the top $B$ sequences:

$$\text{Score}(X_{1:t}) = \sum_{\tau=1}^{t} \log P(x_\tau \mid X_{1:\tau-1})$$

To prevent favoring shorter paths, a **Length Penalty** ($L_p$) is applied:

$$\text{Score}_{\text{normalized}}(X_{1:t}) = \frac{\sum_{\tau=1}^{t} \log P(x_\tau \mid X_{1:\tau-1})}{\left(\frac{5 + t}{6}\right)^{\alpha}}$$

where $\alpha$ is the length penalty coefficient (typically $0.5$ to $1.0$).
- **VRAM Footprint**: Keeping $B$ active states requires caching $B$ times more KV Cache memory, making Beam Search memory-prohibitive for high-concurrency online serving.

---

## 2. Stochastic Sampling & Temperature Math

Stochastic decoding introduces randomness by sampling from the probability distribution.

### Temperature Scaling
Before applying the Softmax function to convert raw logits $z_i$ into probabilities $P_i$, logits are divided by a temperature parameter $T \in (0, \infty)$:

$$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

#### Step-by-Step Hand Calculation on a 3-Token Vocabulary
Let our raw logit vector be $z = [2.0, 1.0, 0.0]$ for vocabulary tokens $[\text{"cat"}, \text{"dog"}, \text{"fish"}]$.

##### 1. Reference Temperature ($T = 1.0$)
- Scale logits: $z / 1.0 = [2.0, 1.0, 0.0]$
- Exponentiate: $e^{2.0} \approx 7.389$, $e^{1.0} \approx 2.718$, $e^{0.0} = 1.0$
- Sum of Exponents: $7.389 + 2.718 + 1.0 = 11.107$
- Probabilities:
  - $P(\text{"cat"}) = 7.389 / 11.107 \approx 0.665$
  - $P(\text{"dog"}) = 2.718 / 11.107 \approx 0.245$
  - $P(\text{"fish"}) = 1.0 / 11.107 \approx 0.090$

##### 2. Low Temperature ($T = 0.5$ - Sharpening the distribution)
- Scale logits: $z / 0.5 = [4.0, 2.0, 0.0]$
- Exponentiate: $e^{4.0} \approx 54.598$, $e^{2.0} \approx 7.389$, $e^{0.0} = 1.0$
- Sum of Exponents: $54.598 + 7.389 + 1.0 = 62.987$
- Probabilities:
  - $P(\text{"cat"}) = 54.598 / 62.987 \approx 0.867$
  - $P(\text{"dog"}) = 7.389 / 62.987 \approx 0.117$
  - $P(\text{"fish"}) = 1.0 / 62.987 \approx 0.016$
  - *Effect: The probability of the top token increases from $66.5\%$ to $86.7\%$. As $T \to 0$, sampling converges to greedy decoding.*

##### 3. High Temperature ($T = 2.0$ - Flattening the distribution)
- Scale logits: $z / 2.0 = [1.0, 0.5, 0.0]$
- Exponentiate: $e^{1.0} \approx 2.718$, $e^{0.5} \approx 1.649$, $e^{0.0} = 1.0$
- Sum of Exponents: $2.718 + 1.649 + 1.0 = 5.367$
- Probabilities:
  - $P(\text{"cat"}) = 2.718 / 5.367 \approx 0.506$
  - $P(\text{"dog"}) = 1.649 / 5.367 \approx 0.307$
  - $P(\text{"fish"}) = 1.0 / 5.367 \approx 0.186$
  - *Effect: The probability distribution becomes flatter and more uniform. As $T \to \infty$, sampling approaches a uniform random distribution.*

---

## 3. Top-K, Top-p, and Min-p Truncations

To prevent sampling highly improbable "garbage" tokens (the tail of the distribution), we truncate the vocabulary candidate list before sampling.

| Strategy | Truncation Logic | Mathematical Formulation | Failure Mode |
|---|---|---|---|
| **Top-K** | Keeps only the $K$ tokens with the highest individual logits. | $|V_{\text{candidate}}| = K$ | If the distribution is flat, it cuts off viable tokens. If it is sharp, it forces the inclusion of bad tail tokens. |
| **Top-p**<br>(Nucleus) | Sorts tokens in descending order and keeps the smallest subset whose cumulative probability exceeds threshold $p$. | $\sum_{i \in V_{\text{candidate}}} P_i \ge p$ | In highly confident settings, it still includes tail tokens if the top token is just below $p$. |
| **Min-p** | Truncates tokens whose probability is below a dynamic threshold relative to the top token's probability $p_{\text{max}}$. | $P_i < p_{\text{max}} \times p_{\text{threshold}}$ | Bypasses standard static cutoff issues, keeping only high-confidence candidates dynamically. |

---

## 4. Repetition Penalties

To prevent semantic loops, logits are modified based on token presence in the history prompt $H$:
- **Frequency Penalty**: Penalizes tokens based on how many times they have already appeared in the output history.
- **Presence Penalty**: Applies a constant penalty to any token that has appeared at least once in the history.
- **Repetition Penalty (Llama Style)**: Multiplies logits directly:
  $$z_i = \begin{cases} z_i / \theta & \text{if } z_i \ge 0 \\ z_i \cdot \theta & \text{if } z_i < 0 \end{cases}$$
  where $\theta > 1.0$ reduces the logit score of active tokens.

---

## 5. Constrained Decoding & Structured Generation

Generating structured formats like JSON or SQL requires restricting token selection at each step to match a target schema.

### Regex & CFG Logit Masking
During decoding, standard engines (such as **Outlines** or **XGrammar**) build a finite state machine (FSM) or pushdown automaton from the target regex/grammar.
1. At step $t$, the FSM determines the set of all vocabulary token strings that are valid next transitions.
2. The engine creates a binary logit mask $M$, setting the score of invalid tokens to $-\infty$:
   $$z_i = z_i + M_i, \quad M_i = \begin{cases} 0 & \text{valid} \\ -\infty & \text{invalid} \end{cases}$$
3. Softmaxing ensures that invalid tokens have exactly $0\%$ probability of being sampled.

### Mitigating Parser Overhead
Historically, calculating the logit mask at each step introduced significant CPU latency. **XGrammar** resolves this by pre-computing state transition trees in C++ and caching token masks, allowing grammar-constrained serving to run at near-native speeds without hurting TPOT.

---

## 6. Decoding Strategy Decision Matrix

| Task Domain | Optimal Temperature | Truncation parameters | Penalties | Strategy |
|---|---|---|---|---|
| **Code / Math** | $0.0$ (Greedy) | N/A | None | Greedy Decoding |
| **JSON Extraction** | $0.0$ | N/A | None | Outlines Grammar constraint |
| **Conversational Chat** | $0.7$ | Top-p = $0.90$ or Min-p = $0.05$ | Repetition = $1.05$ | Stochastic Sampling |
| **Creative Writing** | $1.2$ | Min-p = $0.10$ | Presence = $0.10$ | High Temp Stochastic |

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
