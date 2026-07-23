# Preference Optimization (Direct Alignment)

Direct preference optimization algorithms align models by bypassing the complex RL training infrastructure (PPO). By optimizing preference criteria directly through specialized loss functions, these methods improve convergence stability and reduce VRAM requirements.

---

## 1. PPO vs. Direct Alignment: Why PPO is Superseded in Production

Although RLHF via PPO is highly flexible, it presents severe challenges for production engineering:
- **High VRAM Footprint**: Loading four large networks (generator, reference, reward, critic) concurrently exceeds single-GPU capacities.
- **Training Instability**: PPO is highly sensitive to policy update clip ratios, advantage scaling, and value function estimation, often leading to sudden divergence or output mode collapse.
- **Complexity**: Synchronizing training across reward models and actors slows down iterations.

Direct alignment bypasses this by expressing the optimal policy directly in terms of SFT logits and preferences, reducing active model footprints.

---

## 2. DPO (Direct Preference Optimization)

DPO proves mathematically that the reward function $r(x, y)$ can be represented implicitly by the likelihood ratio of the aligned policy $\pi_{\theta}$ and reference SFT model $\pi_{\text{ref}}$:
$$r(x, y) = \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}$$

By substituting this representation into the Bradley-Terry preference objective, DPO derives the implicit reward loss:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

### Step-by-Step Hand Calculation: DPO Loss

Let's calculate the DPO loss for a single training step.

#### 1. Setup Likelihoods
- Prompt $x$: `"Write a function to return even numbers."`
- Winning completion $y_w$ (well-structured code).
- Losing completion $y_l$ (unformatted code with typos).
- Scale parameter: $\beta = 1.0$.

#### 2. Likelihood Outputs (Model Logits)
Suppose the reference and active policy models calculate the following token-generation probabilities:
- Preferred output probabilities:
  $$\pi_{\text{ref}}(y_w|x) = 0.20, \quad \pi_{\theta}(y_w|x) = 0.50$$
- Dispreferred output probabilities:
  $$\pi_{\text{ref}}(y_l|x) = 0.40, \quad \pi_{\theta}(y_l|x) = 0.10$$

#### 3. Calculate Ratios
- Winning ratio:
  $$\text{Ratio}_w = \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} = \frac{0.50}{0.20} = 2.50$$
- Losing ratio:
  $$\text{Ratio}_l = \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} = \frac{0.10}{0.40} = 0.25$$

#### 4. Compute Log Ratios and Scale
Using natural logs ($\log_e$):
- Winning log-ratio:
  $$\beta \log(\text{Ratio}_w) = 1.0 \times \log(2.50) \approx 0.916290$$
- Losing log-ratio:
  $$\beta \log(\text{Ratio}_l) = 1.0 \times \log(0.25) \approx -1.386294$$

#### 5. Calculate Difference
$$\Delta = \beta \log(\text{Ratio}_w) - \beta \log(\text{Ratio}_l) = 0.916290 - (-1.386294) = 2.302584$$

#### 6. Calculate Sigmoid and Final Negative Log Loss
$$\sigma(\Delta) = \frac{1}{1 + e^{-2.302584}}$$
Evaluate $e^{-2.302584} = 0.100000$:
$$\sigma(\Delta) = \frac{1}{1 + 0.1} = \frac{1}{1.1} \approx 0.909091$$
$$\mathcal{L}_{\text{DPO}} = -\log(0.909091) \approx 0.095310$$

---

## 3. ORPO (Odds Ratio Preference Optimization)

ORPO merges the Supervised Fine-Tuning (SFT) and alignment steps into a single combined loss function. This eliminates the need for a separate SFT phase and a reference model, reducing memory consumption.

ORPO utilizes the **Odds Ratio** of generating the preferred token sequence $y_w$ versus the dispreferred $y_l$:
$$\text{Odds}(y_w|x) = \frac{\pi_{\theta}(y_w|x)}{1 - \pi_{\theta}(y_w|x)}$$

$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \lambda \cdot \mathcal{L}_{\text{OR}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) - \lambda \cdot \log \sigma \left( \log \frac{\text{Odds}(y_w|x)}{\text{Odds}(y_l|x)} \right)$$

---

## 4. GRPO (Group Relative Policy Optimization)

GRPO, popularized by DeepSeek, scales reinforcement learning without requiring a separate Critic/Value network. It samples a group of $G$ outputs $\{y_1, y_2, \dots, y_G\}$ for each prompt $x$.

```
[Prompt x] -> [Active Policy Model (pi_theta)] -> Sample G outputs {y1, ..., yG}
                                                             |
                                                             v
[Group Advantage] <- [Standardize Rewards] <- [Compute Rewards {r1, ..., rG}]
```

Instead of a critic network predicting token-level baselines, GRPO standardizes the rewards within the sampled group to determine token advantages:
$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

### Step-by-Step Hand Calculation: GRPO Advantage

Let's compute the relative advantages for a group size of $G=3$.

#### 1. Setup Raw Rewards
Suppose a validator function (e.g. testing program code outputs) scores the three candidate answers:
- $r = [1.0, 2.0, 3.0]$

#### 2. Calculate Mean Reward
$$\bar{r} = \frac{1.0 + 2.0 + 3.0}{3} = 2.0$$

#### 3. Calculate Population Standard Deviation ($\sigma$)
$$\sigma = \sqrt{\frac{\sum_i (r_i - \bar{r})^2}{G}} = \sqrt{\frac{(1.0-2.0)^2 + (2.0-2.0)^2 + (3.0-2.0)^2}{3}}$$
$$\sigma = \sqrt{\frac{1.0 + 0.0 + 1.0}{3}} = \sqrt{\frac{2}{3}} \approx 0.816497$$

#### 4. Calculate Standardized Advantages
- Advantage $A_1$ (for $y_1$):
  $$A_1 = \frac{1.0 - 2.0}{0.816497} \approx -1.224744$$
- Advantage $A_2$ (for $y_2$):
  $$A_2 = \frac{2.0 - 2.0}{0.816497} = 0.000000$$
- Advantage $A_3$ (for $y_3$):
  $$A_3 = \frac{3.0 - 2.0}{0.816497} \approx 1.224744$$

Thus, output $y_3$ receives a strong positive reinforcement gradient, while output $y_1$ is strongly penalized, without calculating any critic evaluations.

---

## 5. KTO (Kahneman-Tversky Optimization)

KTO does not require pairwise preferred/dispreferred data. Instead, it utilizes unpaired binary utility indicators ($+1$ for desirable, $-1$ for undesirable outcomes), modeling human utility based on prospect theory. KTO loss is defined as:
$$\mathcal{L}_{\text{KTO}}(\theta) = -\mathbb{E}_{(x, y, y')} \left[ w(y) \log \sigma(u_\theta(x,y)) \right]$$
Where $u_\theta$ is the utility function derived from log likelihoods and $w(y)$ scales loss depending on positive or negative preferences. This significantly simplifies preference collection in production.

---

## 6. Comparative Alignment Matrix

| Dimension | PPO | DPO | ORPO | GRPO | KTO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RM Needed?** | Yes | No | No | Optional (can use rules) | No |
| **Critic Needed?**| Yes | No | No | No | No |
| **Data Style** | Pairwise | Pairwise | Pairwise | Rule / Score-based | Unpaired Binary |
| **VRAM Cost** | Very High ($4\times$) | Medium ($2\times$) | Low ($1\times$) | Medium ($2\times$) | Medium ($2\times$) |
| **Training Steps**| Two-stage | Single-stage | Single-stage | RL loop | Single-stage |

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Direct preference optimization removes the unstable actor-critic policy updates of PPO, enabling stable alignment training on standard distributed frameworks.
- **Why was it introduced?**
  To bypass PPO's high GPU requirements (loading multiple base model copies) and eliminate reward hacking issues where actor networks generate garbage outputs to maximize reward metrics.
- **What are its limitations?**
  DPO is susceptible to length bias—it tends to increase the likelihood of longer completions simply because they contain more tokens, which artificially boosts implicit reward calculations.
- **Computational Complexity (Time & Memory)**
  - **Memory Footprint**: DPO requires $2\times$ model parameters in memory (active generator + reference model). ORPO requires only $1\times$ (no reference model).
  - **Time Complexity**: Standard cross-entropy updates over paired sequences, scaling with sequence token counts.
- **Component Variable Denotation Legend**
  - $\beta$: Scaling parameter controlling the KL divergence penalty.
  - $\pi_{\theta}$: Current aligned policy log likelihood.
  - $\pi_{\text{ref}}$: Reference SFT log likelihood.
  - $G$: Sampled group size in GRPO.
  - $A_i$: Aligned advantage scaling value for index $i$.
- **Production Use Cases**
  - Aligning conversational models on pairwise preference feedback without the infrastructure complexity of RLHF.
  - Using GRPO for reasoning models where rewards are determined by verifiable rules (e.g. math correctness or compilers).
- **Follow-up questions interviewers ask**
  - *How does the implicit reward derivation of DPO relate to the Bradley-Terry probability assumption?*
  - *How would you modify the DPO loss function to penalize length bias and prevent models from outputting verbose answers?*
