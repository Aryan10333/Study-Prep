# Module 05: Direct Preference Optimization (DPO), GRPO & Modern Alignment Methods

## 1. Introduction & Intuition

### The Core Bottleneck
Module 04's RLHF pipeline works, but it's operationally heavy: four models resident in memory, an inherently sequential and slow generation rollout on every training step, a separately-trained value model, and RL training dynamics (clipping, KL tuning) that are notoriously fiddly to get right. Most of that complexity exists to solve one problem — turning a *reward signal* into a *policy update* — and researchers asked whether that detour through an explicit reward model and RL loop was actually necessary at all.

### High-Level Intuition
DPO's key insight is that, for the specific KL-constrained reward-maximization objective RLHF optimizes, there is a closed-form relationship between the *optimal policy* and the *reward function* — meaning you can algebraically substitute the reward model out of the objective entirely and end up with a loss that operates directly on preference pairs, using only the policy model and a frozen reference copy. No reward model, no value model, no RL rollout loop — just a supervised-learning-style loss over (preferred, rejected) pairs, using the policy's own log-probabilities as an *implicit* reward. GRPO takes a different piece off the table: it keeps the RL formulation and reward model, but eliminates the separate value model by estimating advantage from a *group* of sampled responses to the same prompt instead of learning a value function.

---

## 2. Core Concepts & Mathematical Formulation

### DPO: The Implicit Reward Reparameterization

#### Purpose & Intuition
DPO derives a loss that directly increases the policy's relative log-probability of the preferred response over the rejected response — compared against what the frozen reference model would have assigned — without ever materializing a separate reward model. The "reward" is implicit in the ratio of policy to reference log-probabilities.

#### Mathematical Formulation
$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)$$

where $y_w$/$y_l$ are the preferred/rejected responses, $\pi_\theta$ is the policy being trained, $\pi_{\text{ref}}$ is the frozen reference (SFT) policy, and $\beta$ plays the same role as the KL penalty coefficient in RLHF — controlling how far the policy is allowed to move from the reference.

---

### Hand Calculation: DPO Loss for One Preference Pair
Suppose for a given prompt, the policy and reference models assign these log-probabilities to the preferred ($y_w$) and rejected ($y_l$) responses, with $\beta = 0.1$:

*   $\log \pi_\theta(y_w) = -2.0$, $\log \pi_{\text{ref}}(y_w) = -2.5$
*   $\log \pi_\theta(y_l) = -3.0$, $\log \pi_{\text{ref}}(y_l) = -2.8$

*   **Step 1: Compute the implicit reward for the preferred response**
    $$\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} = \log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w) = -2.0 - (-2.5) = 0.5$$

*   **Step 2: Compute the implicit reward for the rejected response**
    $$\log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} = -3.0 - (-2.8) = -0.2$$

*   **Step 3: Compute the scaled reward difference**
    $$\beta \times (0.5 - (-0.2)) = 0.1 \times 0.7 = 0.07$$

*   **Step 4: Apply sigmoid and negative log**
    $$\sigma(0.07) \approx 0.5175, \qquad \mathcal{L}_{\text{DPO}} = -\log(0.5175) \approx 0.6588$$

The policy has already increased its relative preference for $y_w$ over $y_l$ (compared to the reference), so the loss is a bit below $-\log(0.5) \approx 0.693$ (the loss at zero preference signal) — reflecting modest, correctly-directed progress.

---

### GRPO: Group-Relative Advantage Without a Value Model

#### Purpose & Intuition
PPO needs a trained value model specifically to compute the advantage estimate $\hat{A}_t$ — "how much better than expected was this outcome" — which requires training a fourth model and adds its own instability. GRPO (Group Relative Policy Optimization) sidesteps this: for a given prompt, sample a *group* of $G$ responses from the current policy, score each with the reward model, and compute each response's advantage as its reward *normalized against the group's own mean and standard deviation* — no learned value function required.

#### Mathematical Formulation
For a group of $G$ sampled responses to the same prompt with rewards $r_1, \dots, r_G$:
$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

This advantage is then used in the same clipped PPO-style objective from Module 04, just without a value model in the loop.

---

### Hand Calculation: GRPO Group-Relative Advantage
Suppose we sample $G=4$ responses to the same prompt, and the reward model scores them as: $r = [3.0,\ 5.0,\ 4.0,\ 2.0]$.

*   **Step 1: Compute the group mean**
    $$\text{mean}(r) = \frac{3.0 + 5.0 + 4.0 + 2.0}{4} = \frac{14.0}{4} = 3.5$$

*   **Step 2: Compute the group standard deviation**
    $$\text{Var}(r) = \frac{(3.0-3.5)^2 + (5.0-3.5)^2 + (4.0-3.5)^2 + (2.0-3.5)^2}{4} = \frac{0.25 + 2.25 + 0.25 + 2.25}{4} = 1.25$$
    $$\text{std}(r) = \sqrt{1.25} \approx 1.118$$

*   **Step 3: Compute each response's normalized advantage**
    $$A_1 = \frac{3.0 - 3.5}{1.118} \approx -0.447, \quad A_2 = \frac{5.0 - 3.5}{1.118} \approx 1.342$$
    $$A_3 = \frac{4.0 - 3.5}{1.118} \approx 0.447, \quad A_4 = \frac{2.0 - 3.5}{1.118} \approx -1.342$$

Response 2 (the highest-scoring of the group) gets the largest positive advantage and its probability is pushed up the most; response 4 gets pushed down. This entirely replaces what a trained value model would have estimated — the "baseline" for what counts as a good response is just the group's own average, recomputed fresh for every prompt.

---

### Why Removing the Value Model Reduces Training Overhead

GRPO's group-relative advantage removes an entire model (the value model) from the training loop — cutting the "four models in memory" of PPO down to effectively policy + reference + reward (the value model's training and inference cost disappears entirely). The trade-off is that GRPO needs to sample $G$ responses per prompt (more generation rollout per training example) to get a meaningful group statistic, rather than needing just one response plus a value-model call.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 800 200" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="400" y="24" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Model Count: PPO vs. DPO vs. GRPO</text>

  <g transform="translate(20, 45)">
    <text x="115" y="14" text-anchor="middle" font-size="12" font-weight="bold" fill="#dc2626">PPO (4 models)</text>
    <rect x="0" y="25" width="105" height="32" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3" /><text x="52" y="45" text-anchor="middle" font-size="9" fill="#1e3a8a">Policy</text>
    <rect x="120" y="25" width="105" height="32" rx="4" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="3" /><text x="172" y="45" text-anchor="middle" font-size="9" fill="#475569">Reference</text>
    <rect x="0" y="63" width="105" height="32" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="3" /><text x="52" y="83" text-anchor="middle" font-size="9" fill="#7f1d1d">Reward</text>
    <rect x="120" y="63" width="105" height="32" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.3" /><text x="172" y="83" text-anchor="middle" font-size="9" fill="#065f46">Value</text>
  </g>

  <g transform="translate(300, 45)">
    <text x="105" y="14" text-anchor="middle" font-size="12" font-weight="bold" fill="#059669">DPO (2 models)</text>
    <rect x="0" y="25" width="105" height="32" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3" /><text x="52" y="45" text-anchor="middle" font-size="9" fill="#1e3a8a">Policy</text>
    <rect x="120" y="25" width="105" height="32" rx="4" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="3" /><text x="172" y="45" text-anchor="middle" font-size="9" fill="#475569">Reference</text>
    <text x="60" y="83" font-size="9" fill="#64748b">No reward model.</text>
    <text x="60" y="97" font-size="9" fill="#64748b">No value model.</text>
  </g>

  <g transform="translate(560, 45)">
    <text x="115" y="14" text-anchor="middle" font-size="12" font-weight="bold" fill="#7c3aed">GRPO (3 models)</text>
    <rect x="0" y="25" width="105" height="32" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3" /><text x="52" y="45" text-anchor="middle" font-size="9" fill="#1e3a8a">Policy</text>
    <rect x="120" y="25" width="105" height="32" rx="4" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="3" /><text x="172" y="45" text-anchor="middle" font-size="9" fill="#475569">Reference</text>
    <rect x="0" y="63" width="105" height="32" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="3" /><text x="52" y="83" text-anchor="middle" font-size="9" fill="#7f1d1d">Reward</text>
    <text x="172" y="83" font-size="9" fill="#64748b">No value model —</text>
    <text x="172" y="97" font-size="9" fill="#64748b">group stats instead.</text>
  </g>
</svg>
</div>

---

### Tensor & Shape Tracking
*   **DPO — policy/reference log-probs for preferred & rejected responses:** `[B]` each (summed log-probability over the response sequence)
*   **DPO loss:** `[]` scalar
*   **GRPO — group of sampled responses per prompt:** `[G, L]` where $G$ is group size
*   **GRPO — reward model scores per group:** `[G]`
*   **GRPO — normalized advantages:** `[G]`

---

### A Brief Survey of Other Alignment Methods

Beyond DPO and GRPO, several related methods trade off different aspects of the same underlying problem — learning from preference data without the full RLHF pipeline:

| Method | Core Idea | When to Prefer It |
|---|---|---|
| **IPO** | Replaces DPO's sigmoid-log loss with a squared-loss objective, correcting a tendency in DPO to overfit / become overconfident on preference pairs with a clear-cut winner | When DPO shows signs of overfitting on the preference dataset |
| **KTO** | Learns from *unpaired* binary "good/bad" labels instead of requiring paired (preferred, rejected) comparisons, based on prospect theory | When only per-response quality labels are available, not pairwise comparisons |
| **ORPO** | Combines the SFT and preference-alignment objectives into a single training stage (no separate reference model needed) | When minimizing pipeline stages/complexity matters more than matching DPO's exact formulation |
| **SimPO** | Removes the reference model entirely by using length-normalized policy log-probabilities directly as the implicit reward | When reference-model memory/compute overhead is the binding constraint |

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of the DPO loss and GRPO group-relative advantage, matching the hand calculations above.

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logp_w: torch.Tensor, policy_logp_l: torch.Tensor,
             ref_logp_w: torch.Tensor, ref_logp_l: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """DPO loss for a batch of (preferred, rejected) pairs. All inputs: [B] summed log-probs."""
    implicit_reward_w = policy_logp_w - ref_logp_w  # [B]
    implicit_reward_l = policy_logp_l - ref_logp_l  # [B]
    logits = beta * (implicit_reward_w - implicit_reward_l)  # [B]
    return -F.logsigmoid(logits).mean()


def grpo_group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Group-relative advantage for GRPO. rewards: [G] scalar reward per sampled response."""
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)  # population std, matching the hand calc's variance formula
    return (rewards - mean_r) / std_r  # [G]


if __name__ == "__main__":
    # --- DPO loss, matching the hand calc ---
    policy_logp_w = torch.tensor([-2.0])
    ref_logp_w = torch.tensor([-2.5])
    policy_logp_l = torch.tensor([-3.0])
    ref_logp_l = torch.tensor([-2.8])

    loss = dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1)
    print(f"DPO loss: {loss.item():.4f}")

    # --- GRPO group-relative advantage, matching the hand calc ---
    rewards = torch.tensor([3.0, 5.0, 4.0, 2.0])
    advantages = grpo_group_advantage(rewards)
    print(f"\nGRPO rewards:     {rewards.tolist()}")
    print(f"GRPO advantages:  {[round(a, 3) for a in advantages.tolist()]}")
```

---

## Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Learning from human preference data without RLHF's operational overhead — DPO removes the reward model and value model entirely; GRPO removes just the value model while keeping an explicit reward model and RL formulation.
*   **Why Introduced over Legacy Approaches:** RLHF's 4-model pipeline and RL training instability made it expensive and fragile to run well. DPO reformulates the same underlying objective as a simple supervised loss; GRPO keeps RL's flexibility (useful when reward signal isn't naturally pairwise) while cutting one model out of the loop.
*   **Key Failure Modes & Limitations:** DPO requires paired preference data (harder to collect at scale than independent quality ratings); it can overfit confidently on preference pairs with a clear winner (motivating IPO's squared loss). GRPO's group-relative advantage becomes noisy with small group sizes $G$, and requires $G$ full generation rollouts per prompt instead of one.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** DPO needs one forward pass each through the policy and reference models per preference pair — no generation rollout required during training at all, since responses are pre-collected. GRPO still requires autoregressive generation of $G$ responses per prompt, similar cost profile to PPO's rollout but without the value-model forward pass.
*   **Space/Memory Footprint:** DPO: 2 models (policy + reference). GRPO: 3 models (policy + reference + reward), vs. PPO's 4 (adds value model).
*   **Primary Bottleneck Type:** DPO is compute-bound on ordinary forward/backward passes (no rollout) — much closer to SFT's cost profile than RLHF's. GRPO remains generation-rollout-bound like PPO, just without the value-model overhead on top.
*   **Variable Legend:** $\beta$ = DPO/KL scaling coefficient, $\pi_\theta$/$\pi_{\text{ref}}$ = policy/reference, $G$ = GRPO group size, $A_i$ = group-relative advantage for sample $i$.

### 3. Production & Scalability
*   **Deployment Considerations:** DPO's simplicity (no rollout, no reward/value models) makes it substantially cheaper to run at scale and easier to debug than PPO-based RLHF, which is a major reason it's become a common default for preference alignment; GRPO is chosen instead when the reward signal genuinely benefits from RL-style exploration (e.g., verifiable reward tasks like math/code correctness) rather than fixed offline preference pairs.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does DPO need a reference model at all, if it doesn't need a reward model?
        *   *A:* The reference model anchors the implicit reward — without it, the loss would only depend on the policy's own log-probabilities, which the policy could trivially inflate for the preferred response without any grounding in what the pretrained/SFT model considered plausible. The reference model plays the same role the KL penalty played in RLHF: keeping the policy close to a sensible starting distribution.
    2.  *Q:* When would you choose GRPO over DPO?
        *   *A:* When the reward signal is more naturally expressed as a scalar score per response than as pairwise preferences — for example, verifiable correctness rewards in math or code generation — or when you want the exploration benefits of sampling multiple candidate responses per prompt rather than relying on a fixed, pre-collected preference dataset.
