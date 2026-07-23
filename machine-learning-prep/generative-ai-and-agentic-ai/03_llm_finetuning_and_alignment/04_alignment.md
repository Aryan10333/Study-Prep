# Alignment & Preference Datasets

Following Supervised Fine-Tuning, models must be aligned to ensure they generate outputs matching human preferences. This process focuses on the HHH objective—making the model Helpful, Honest, and Harmless.

---

## 1. Human Preference Data Collection & Pairwise Ranking

Unlike SFT, where training data consists of target outputs, preference training utilizes relative performance ratings.
- A prompt $x$ is sent to multiple model variants, producing candidate responses (e.g., $y_1$ and $y_2$).
- Human annotators (or LLM evaluators in RLAIF) rate the responses, establishing a preferred output ($y_w$) and a dispreferred output ($y_l$).
- This generates a preference dataset: $\mathcal{D} = \{ (x, y_w, y_l) \}$.

---

## 2. Reward Models & the Bradley-Terry Formulation

To automate preference optimization, we train a **Reward Model (RM)**. The reward model $\mathbf{r}_{\psi}(x, y)$ takes a prompt $x$ and completion $y$ as input and outputs a scalar score representing the response quality.

### Mathematical Formulation
The RM uses the **Bradley-Terry (BT) model** to define the probability that response $y_w$ is preferred over $y_l$ conditioned on prompt $x$:
$$P(y_w \succ y_l | x) = \sigma(\mathbf{r}_{\psi}(x, y_w) - \mathbf{r}_{\psi}(x, y_l)) = \frac{1}{1 + e^{-(\mathbf{r}_{\psi}(x, y_w) - \mathbf{r}_{\psi}(x, y_l))}}$$

The reward model is trained using binary cross-entropy loss over pairwise preferences:
$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(\mathbf{r}_{\psi}(x, y_w) - \mathbf{r}_{\psi}(x, y_l)) \right]$$

---

### Step-by-Step Hand Calculation: Reward Model Loss

Let's compute the loss for a single prompt $x$ with winning response $y_w$ and losing response $y_l$.

#### 1. Setup Reward Outputs
Suppose the reward model outputs the following raw scalar scores:
- Reward for preferred answer: $\mathbf{r}_{\psi}(x, y_w) = 2.5$
- Reward for dispreferred answer: $\mathbf{r}_{\psi}(x, y_l) = -0.5$

#### 2. Compute Reward Difference
$$\Delta r = \mathbf{r}_{\psi}(x, y_w) - \mathbf{r}_{\psi}(x, y_l) = 2.5 - (-0.5) = 3.0$$

#### 3. Compute Probability of Preference
Using the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$:
$$P(y_w \succ y_l | x) = \sigma(3.0) = \frac{1}{1 + e^{-3.0}}$$
Evaluate $e^{-3.0} \approx 0.049787$:
$$P(y_w \succ y_l | x) = \frac{1}{1 + 0.049787} = \frac{1}{1.049787} \approx 0.952574$$

This indicates the model assigns a $95.26\%$ probability that $y_w$ is superior to $y_l$.

#### 4. Compute Pairwise Loss
$$\mathcal{L}_{\text{RM}} = -\log(P(y_w \succ y_l | x)) = -\log(0.952574)$$
Using the natural logarithm:
$$\mathcal{L}_{\text{RM}} \approx -(-0.048590) = 0.048590$$

If the scores were reversed (i.e. model assigned $2.5$ to $y_l$ and $-0.5$ to $y_w$, rendering $\Delta r = -3.0$):
$$P(y_w \succ y_l | x) = \sigma(-3.0) \approx 0.047426$$
$$\mathcal{L}_{\text{RM}} = -\log(0.047426) \approx 3.048590$$
The higher loss penalizes the model for misaligning the rewards.

---

## 3. RLHF Pipeline Overview (PPO)

Reinforcement Learning from Human Feedback (RLHF) utilizes Proximal Policy Optimization (PPO) to align a generator model using four concurrent models in GPU memory:

```
                  +--------------------------------+
                  |         Input Prompt x         |
                  +--------------------------------+
                            /             \
                           /               \
                          v                 v
            +-------------------+     +---------------------+
            | Active Policy (pi)|     | Reference Model(ref)| [Frozen]
            +-------------------+     +---------------------+
                          \                 /
             [Tokens y]    \               / [Reference Logits]
                            v             v
                    +-----------------------------+
                    |    KL Divergence Penalty    |
                    +-----------------------------+
                                  |
                                  v
                    +-----------------------------+
                    |      Value Critic / RM      |
                    +-----------------------------+
```

1. **Active Policy Model ($\pi_{\theta}$)**: The generative model undergoing alignment.
2. **Reference Model ($\pi_{\text{ref}}$)**: A frozen copy of the pre-trained SFT model used to keep the policy from drifting too far from logical syntax.
3. **Reward Model ($r_{\psi}$)**: Evaluates completions, outputting raw scalar rewards.
4. **Value Model / Critic ($V_{\phi}$)**: Learns to estimate the expected future reward from intermediate token positions.

To prevent policy collapse (where the policy generates gibberish that exploits the reward model's boundaries), we penalize the reward using the KL-divergence of the token distributions:
$$R(x, y) = \mathbf{r}_{\psi}(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_{\theta}(y|x) \parallel \pi_{\text{ref}}(y|x))$$

---

## 4. Constitutional AI & RLAIF (AI Feedback)

In **Constitutional AI**, humans do not label preferences directly. Instead:
1. **Supervised Stage**: The model is prompted to generate critique-revision loops based on a set of rules (a "constitution").
2. **Preference Stage (RLAIF)**: A feedback model compares response pairs based on the constitution, outputting logits representing preference probabilities.
3. **Training**: An RM is trained on these automated classifications, and the generator is aligned using RL.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  RLHF mitigates the mismatch between next-token minimization (cross-entropy) and human conversational values (helpfulness, honesty, harmlessness).
- **Why was it introduced?**
  Models trained only on SFT frequently generated toxic, evasive, or unhelpful answers (e.g. refusing complex requests or hallucinating answers to please user profiles).
- **What are its limitations?**
  High VRAM overhead during training due to loading 4 copies of the model (generator, reference, reward, critic) in VRAM simultaneously, and extreme training instability in PPO.
- **Computational Complexity (Time & Memory)**
  - **Memory Complexity**: Scaling at $\sim 4 \times \text{Size of Generator}$ VRAM footprint.
  - **Time Complexity**: $O(L \cdot B)$ forward-backward steps, with multiple optimization epochs per trajectory.
- **Component Variable Denotation Legend**
  - $\pi_{\theta}$: Generative policy model parameters.
  - $\pi_{\text{ref}}$: Reference SFT model parameters.
  - $\mathbf{r}_{\psi}$: Reward model parameters.
  - $\beta$: Weight coefficient for the KL penalty.
- **Production Use Cases**
  - Safety-aligning public-facing enterprise virtual assistants.
  - Aligning conversational models to resist prompt injection and jailbreak queries.
- **Follow-up questions interviewers ask**
  - *How does the PPO value/critic network help stabilize gradient updates compared to standard REINFORCE algorithms?*
  - *If your KL penalty spikes to a high value during RLHF, what hyperparameters would you adjust to stabilize the run?*
