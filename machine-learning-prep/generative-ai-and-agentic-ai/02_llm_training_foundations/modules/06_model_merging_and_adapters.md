# Module 06: Model Merging & Adapter Composition

## 1. Introduction & Intuition

### The Core Bottleneck
Suppose you've fine-tuned separate copies of the same base model for three different tasks — one for coding, one for summarization, one for a specific customer-support domain. Serving three full models means three times the inference infrastructure, and none of them individually benefits from the other two's specialization. The bottleneck is: can the capabilities learned in separate fine-tuning runs be *combined* into a single model, without retraining from scratch on a combined dataset (which may not even be possible if the original training data isn't available)?

### High-Level Intuition
Because all three fine-tuned models started from the *same* pretrained weights, each one's final weights can be thought of as the base weights plus a task-specific "delta" — the direction and magnitude that fine-tuning moved the weights in. Model merging works directly on these deltas: average them, selectively combine them, or algebraically add/subtract them, producing a single set of weights that (ideally) exhibits a blend of the original models' capabilities — with zero additional inference cost, since the result is just one ordinary model.

---

## 2. Core Concepts & Mathematical Formulation

### Task Arithmetic & Weight Averaging

#### Purpose & Intuition
The simplest merge is a weighted average of multiple fine-tuned models' weights directly ("model soups"). A more general and more widely used formulation, task arithmetic, defines each model's contribution as a **task vector** — the difference between its fine-tuned weights and the shared base weights — and combines these task vectors with tunable coefficients before adding them back onto the base.

#### Mathematical Formulation
$$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{i=1}^{k} \lambda_i (\theta_i - \theta_{\text{base}})$$

where $\theta_i$ is the $i$-th fine-tuned model's weights, $\theta_i - \theta_{\text{base}}$ is its task vector, and $\lambda_i$ is a scaling coefficient controlling how strongly that task's specialization is expressed in the merged model (commonly $\lambda_i \approx 1/k$ for a simple average, or tuned per-task).

---

### Hand Calculation: Merging Two Toy Weight Vectors
Let's merge two fine-tuned models' task vectors for a single toy weight (representing one parameter), with $\theta_{\text{base}} = 1.0$.

*   **Model 1 (coding-specialized):** $\theta_1 = 1.6$, so its task vector is $\theta_1 - \theta_{\text{base}} = 1.6 - 1.0 = 0.6$
*   **Model 2 (summarization-specialized):** $\theta_2 = 0.7$, so its task vector is $\theta_2 - \theta_{\text{base}} = 0.7 - 1.0 = -0.3$

*   **Step 1: Choose merge coefficients** — equal weighting, $\lambda_1 = \lambda_2 = 0.5$

*   **Step 2: Scale each task vector**
    $$\lambda_1 (\theta_1 - \theta_{\text{base}}) = 0.5 \times 0.6 = 0.3, \qquad \lambda_2 (\theta_2 - \theta_{\text{base}}) = 0.5 \times (-0.3) = -0.15$$

*   **Step 3: Sum the scaled task vectors and add back to the base**
    $$\theta_{\text{merged}} = 1.0 + (0.3 + (-0.15)) = 1.0 + 0.15 = 1.15$$

The merged weight sits between the base (1.0) and Model 1's specialization (1.6), pulled slightly further by Model 1's larger task vector than Model 2's opposing, smaller-magnitude one — a direct numerical illustration of how conflicting task directions partially cancel in a naive average.

---

### Tensor & Shape Tracking
*   **Base model weights $\theta_{\text{base}}$:** same shape as any single model's full parameter set
*   **Task vector $\theta_i - \theta_{\text{base}}$:** identical shape to $\theta_{\text{base}}$, computed independently per fine-tuned model
*   **Merged weights $\theta_{\text{merged}}$:** identical shape to $\theta_{\text{base}}$ — merging changes values, never shapes, so the merged model serves with exactly the same architecture and inference cost as any one of the originals

---

### Beyond Naive Averaging: TIES-Merging and DARE

Naive averaging (or unweighted task arithmetic) has an obvious failure mode visible in the hand calculation above: when two task vectors point in *conflicting* directions for the same parameter, a simple sum partially cancels both, diluting both specializations rather than combining them cleanly. Two techniques address this directly:

*   **TIES-Merging** ("TrIm, Elect Sign, merge") addresses conflicting signs in three explicit steps: **(1) Trim** — zero out the smallest-magnitude entries in each task vector (they're likely noise, not meaningful specialization); **(2) Elect Sign** — for each parameter, determine which sign (positive or negative) has the greater total magnitude support across all task vectors, and only keep entries from task vectors agreeing with that elected sign; **(3) Merge** — average only the surviving, sign-agreeing entries. This directly prevents the cancellation seen in the naive hand calculation above.
*   **DARE** (Drop And REscale) randomly zeroes out (drops) a large fraction of each task vector's entries, then rescales the survivors to preserve the vector's expected magnitude — exploiting the empirical observation that task vectors are highly redundant, so most individual parameter changes can be dropped without meaningfully hurting the task's performance, which in turn reduces interference when merging multiple task vectors together.

Both are procedures applied *before* the same underlying task-arithmetic summation, not alternative formulas — they modify which entries of each task vector participate in the merge, and how, rather than changing the merge equation itself.

### Multi-Adapter Composition & Routing

An alternative to merging weights permanently is keeping multiple LoRA adapters (Module 03) *unmerged* and swapping or combining them at serving time — loading one shared base model into memory once, and attaching a different small adapter per request (or per user, or per task) with negligible additional memory cost per adapter. Some serving stacks go further and route dynamically, selecting or blending adapters per-request based on the detected task type, without ever producing a single "merged" model at all.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 780 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="390" y="24" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Task Arithmetic: Base Weights + Weighted Task Vectors</text>

  <!-- Base model, vertically centered between the two task-vector rows -->
  <rect x="20" y="90" width="140" height="45" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.4" />
  <text x="90" y="117" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">Base (&#952;_base)</text>

  <!-- Model 1 task vector (top row) -->
  <path d="M 160 100 L 250 60" stroke="#3b82f6" stroke-width="1.6" fill="none" marker-end="url(#arrow-merge)" />
  <rect x="250" y="45" width="150" height="45" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4" />
  <text x="325" y="63" text-anchor="middle" font-size="10" font-weight="bold" fill="#1e3a8a">Model 1</text>
  <text x="325" y="78" text-anchor="middle" font-size="8" fill="#1e3a8a">(&#952;_1 &#8722; &#952;_base)</text>

  <!-- Model 2 task vector (bottom row) -->
  <path d="M 160 125 L 250 165" stroke="#7c3aed" stroke-width="1.6" fill="none" marker-end="url(#arrow-merge)" />
  <rect x="250" y="150" width="150" height="45" rx="4" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.4" />
  <text x="325" y="168" text-anchor="middle" font-size="10" font-weight="bold" fill="#5b21b6">Model 2</text>
  <text x="325" y="183" text-anchor="middle" font-size="8" fill="#5b21b6">(&#952;_2 &#8722; &#952;_base)</text>

  <!-- Scaled task vectors merge into the result -->
  <path d="M 400 68 L 480 105" stroke="#3b82f6" stroke-width="1.6" fill="none" marker-end="url(#arrow-merge)" />
  <path d="M 400 172 L 480 135" stroke="#7c3aed" stroke-width="1.6" fill="none" marker-end="url(#arrow-merge)" />
  <text x="425" y="85" font-size="8" fill="#1e3a8a">&#215; &#955;_1</text>
  <text x="425" y="155" font-size="8" fill="#5b21b6">&#215; &#955;_2</text>

  <rect x="490" y="95" width="160" height="50" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.6" />
  <text x="570" y="124" text-anchor="middle" font-size="11" font-weight="bold" fill="#065f46">&#952;_merged</text>

  <text x="570" y="165" text-anchor="middle" font-size="9" fill="#64748b">One model. Same shape,</text>
  <text x="570" y="179" text-anchor="middle" font-size="9" fill="#64748b">same inference cost as any single original.</text>

  <defs>
    <marker id="arrow-merge" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of task-arithmetic merging (matching the hand calculation above) and a simplified TIES-style sign-election step.

```python
import torch

def task_arithmetic_merge(theta_base: torch.Tensor, task_vectors: list[torch.Tensor], lambdas: list[float]) -> torch.Tensor:
    """Merges multiple fine-tuned models via weighted task-vector summation.
    theta_base: [D] base model weights (flattened for illustration).
    task_vectors: list of [D] tensors, each theta_i - theta_base.
    lambdas: per-model scaling coefficients.
    """
    merged_delta = torch.zeros_like(theta_base)
    for task_vector, lam in zip(task_vectors, lambdas):
        merged_delta += lam * task_vector
    return theta_base + merged_delta


def ties_elect_sign(task_vectors: torch.Tensor) -> torch.Tensor:
    """Simplified TIES sign-election: for each parameter, determine which sign has
    greater total magnitude support across task vectors.
    task_vectors: [k, D] stacked task vectors from k fine-tuned models.
    Returns: [D] elected sign per parameter (+1 or -1).
    """
    positive_support = torch.where(task_vectors > 0, task_vectors, torch.zeros_like(task_vectors)).sum(dim=0)
    negative_support = torch.where(task_vectors < 0, task_vectors.abs(), torch.zeros_like(task_vectors)).sum(dim=0)
    return torch.where(positive_support >= negative_support, 1.0, -1.0)


if __name__ == "__main__":
    # --- Task arithmetic merge, matching the hand calc ---
    theta_base = torch.tensor([1.0])
    theta_1 = torch.tensor([1.6])
    theta_2 = torch.tensor([0.7])

    task_vectors = [theta_1 - theta_base, theta_2 - theta_base]
    merged = task_arithmetic_merge(theta_base, task_vectors, lambdas=[0.5, 0.5])
    print(f"Task vectors: {[tv.item() for tv in task_vectors]}")
    print(f"Merged weight: {merged.item():.4f}")

    # --- TIES-style sign election on a toy 4-parameter model with conflicting directions ---
    stacked_task_vectors = torch.tensor([
        [ 0.6, -0.2,  0.1, -0.5],  # Model 1's task vector
        [-0.3, -0.1,  0.4,  0.2],  # Model 2's task vector
        [ 0.5,  0.3, -0.2, -0.6],  # Model 3's task vector
    ])
    elected_signs = ties_elect_sign(stacked_task_vectors)
    print(f"\nStacked task vectors:\n{stacked_task_vectors}")
    print(f"Elected signs per parameter: {elected_signs.tolist()}")
```

---

## Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Combining capabilities from multiple independently fine-tuned models into one set of weights, without retraining on a combined dataset and without paying multi-model serving cost.
*   **Why Introduced over Legacy Approaches:** The alternative — serving separate fine-tuned models per task, or retraining from scratch on all tasks' data jointly — is far more expensive in infrastructure and data-access terms; merging reuses already-completed fine-tuning runs directly.
*   **Key Failure Modes & Limitations:** Task interference — when merged tasks require genuinely conflicting weight changes, naive averaging degrades both rather than combining them; this is precisely what TIES/DARE are designed to mitigate, not eliminate entirely.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Merging itself is a one-time, cheap elementwise operation over the full parameter set — negligible compared to the cost of the original fine-tuning runs that produced the models being merged.
*   **Space/Memory Footprint:** Merging requires holding all $k$ source models' weights simultaneously during the merge computation, but the *result* is a single model — no added inference-time memory versus any one of the originals.
*   **Primary Bottleneck Type:** The merge operation itself is memory-bandwidth-bound (reading/writing full parameter tensors); the real constraint is *quality*, not compute — how much task interference the merge introduces.
*   **Variable Legend:** $\theta_{\text{base}}$ = shared pretrained weights, $\theta_i$ = $i$-th fine-tuned model's weights, $\lambda_i$ = merge coefficient for task $i$, $k$ = number of models being merged.

### 3. Production & Scalability
*   **Deployment Considerations:** Permanent merging is preferred when the combined behavior is the actual product (one general-purpose model). Multi-adapter serving (keeping LoRA adapters unmerged and swappable) is preferred when you need to preserve per-task isolation, serve many tasks from one base model efficiently, or add/remove task specializations without re-merging.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does merging cost nothing extra at inference time, unlike serving multiple LoRA adapters?
        *   *A:* A merged model is architecturally identical to any single one of its source models — same weight shapes, same number of parameters. Serving multiple unmerged LoRA adapters, by contrast, requires either swapping adapters per-request (added latency) or holding several adapters in memory simultaneously (small but nonzero added memory per adapter).
    2.  *Q:* What specific problem does TIES-Merging's "elect sign" step solve that naive averaging doesn't?
        *   *A:* When two task vectors disagree on the sign of a parameter's needed change, naive averaging lets them partially cancel, diluting both specializations for that parameter. TIES resolves the conflict by keeping only the entries that agree with the majority-magnitude sign, avoiding that cancellation.
