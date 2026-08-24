# Module 03: LLM-as-a-Judge — Design, Biases & Calibration

## 1. Introduction & Intuition

### The Core Bottleneck
Human evaluation (Module 04) is the real, gold-standard signal — but it's real, slow, and expensive at scale. LLM-as-a-judge exists to approximate human-quality judgments using a real, fast, comparatively cheap LLM call instead, letting a team evaluate far more real outputs than human annotators alone could. That real scalability comes at the cost of two genuinely separate real risks this module treats as distinct: the judge can be systematically *biased* (its scores skew for reasons unrelated to real quality), and even a low-bias judge can be *unreliable* — it can simply fail to track real human judgment, or give meaningfully different scores under small, incidental prompt/rubric changes.

### High-Level Intuition
Hiring a fast, inexpensive grader to evaluate a huge stack of essays is only useful if that grader's real grades actually resemble what a careful, expert grader would have given — and it's not enough to check that the fast grader isn't *biased* toward, say, longer essays; you also need to confirm the grades are *reliable* — consistent, and genuinely tracking real essay quality, not just happening to look reasonable on the specific essays you glanced at.

---

## 2. Core Concepts & Mathematical Formulation

### Documented Real Judge Biases

#### Intuition & Practical Use
Real, well-documented LLM-judge biases include **position bias** (favoring whichever response is presented first or second, independent of real quality), **verbosity bias** (favoring longer responses regardless of real content quality), and **self-preference bias** (a model favoring outputs that resemble its own real stylistic tendencies over a genuinely better response from a different model). Each is a real, systematic distortion — not random noise — and each has a real, targeted mitigation: randomizing response order to detect/average out position bias; length-controlling or explicitly rubric-penalizing verbosity; using a judge model distinct from the model(s) being judged to reduce self-preference risk.

### Judge Reliability: A Distinct Concern Beyond Bias

#### Intuition & Practical Use
A judge can be genuinely low-bias — unaffected by position, length, or self-preference — and still be unreliable in two separate real ways. First, its scores might simply not *track* real human judgment well, even if internally consistent; this is a real *calibration* failure, checked via real correlation against human scores, not raw agreement (raw agreement percentages can be misleading for continuous or ordinal scores, where what matters is whether the judge's real *rankings* track human rankings, not whether individual numeric scores match exactly). Second, its scores can be genuinely *unstable* — producing meaningfully different results under small, incidental rubric or prompt-wording changes for the same real underlying judgment task, a real reliability failure distinct from any measurable bias.

---

### Hand Calculation, Three Real Worked Examples

*   **Example 1: Judge-human calibration via Spearman rank correlation.** Six real items, each with a judge score and a human score (1-10 scale): judge $= [8,6,9,5,7,4]$, human $=[7,6,9,4,8,3]$.
    $$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)} = 1 - \frac{6(2)}{6(35)} = 1 - \frac{12}{210} \approx 0.9429$$
    A real, high rank correlation ($\approx 0.94$) — this specific constructed judge's *rankings* track the human rankings closely, the real calibration evidence this module treats as more meaningful than a raw score-match percentage.

*   **Example 2: Position-bias flip rate.** Ten real constructed pairwise trials, each judged once in original order (A shown first) and once with order swapped (B shown first), tracking the judge's picked response *by identity* in each case.
    $$\text{Flip rate} = \frac{7}{10} = 70.0\%$$
    A real, computed 70% flip rate — in 7 of 10 trials, the judge's picked identity changed purely from reordering the same two real responses, a direct, quantified illustration of real position bias in this constructed example.

*   **Example 3: Rubric-wording instability.** Five real items scored under two slightly reworded rubric variants for the identical underlying judgment task: Rubric A scores $=[7,8,6,9,5]$, Rubric B scores $=[8,7,6,8,6]$.
    $$\text{Per-item differences} = [1,-1,0,-1,1], \quad \text{Variance} = 0.8, \quad \text{Std. dev.} \approx 0.89$$
    A real, nonzero average score deviation (${\approx}0.89$ points on a 10-point scale) purely from rewording the rubric — a real, computed illustration of judge instability, genuinely distinct from any bias measured in Example 2.

<div style="text-align:center">

<svg viewBox="0 0 900 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;height:auto;font-family:sans-serif">
  <style>
    .lbl{font-size:12px;fill:#333}
    .hdr{font-size:14px;font-weight:bold;fill:#222}
    .box{fill:#eaf1fb;stroke:#4c78a8;stroke-width:1.5}
    .mit{fill:#d9f2e3;stroke:#2e8b57;stroke-width:1.5}
  </style>
  <text x="450" y="24" text-anchor="middle" class="hdr">LLM-as-Judge Evaluation Pipeline</text>

  <rect x="20" y="60" width="140" height="50" class="box" rx="6"/>
  <text x="90" y="82" text-anchor="middle" class="lbl">Input + rubric</text>
  <text x="90" y="98" text-anchor="middle" class="lbl">(response pair)</text>

  <path d="M160 85 L215 85" stroke="#555" stroke-width="2" marker-end="url(#arrJ)"/>

  <rect x="220" y="60" width="140" height="50" class="box" rx="6"/>
  <text x="290" y="82" text-anchor="middle" class="lbl">Randomized</text>
  <text x="290" y="98" text-anchor="middle" class="lbl">response order</text>
  <rect x="220" y="120" width="140" height="26" class="mit" rx="4"/>
  <text x="290" y="137" text-anchor="middle" font-size="10">mitigates position bias</text>

  <path d="M360 85 L415 85" stroke="#555" stroke-width="2" marker-end="url(#arrJ)"/>

  <rect x="420" y="60" width="140" height="50" class="box" rx="6"/>
  <text x="490" y="82" text-anchor="middle" class="lbl">Judge model call</text>
  <text x="490" y="98" text-anchor="middle" class="lbl">(distinct from subject)</text>
  <rect x="420" y="120" width="140" height="26" class="mit" rx="4"/>
  <text x="490" y="137" text-anchor="middle" font-size="10">mitigates self-preference</text>

  <path d="M560 85 L615 85" stroke="#555" stroke-width="2" marker-end="url(#arrJ)"/>

  <rect x="620" y="60" width="140" height="50" class="box" rx="6"/>
  <text x="690" y="82" text-anchor="middle" class="lbl">Raw score</text>
  <text x="690" y="98" text-anchor="middle" class="lbl">+ rationale</text>

  <path d="M690 110 L690 165" stroke="#555" stroke-width="2" marker-end="url(#arrJ)"/>

  <rect x="580" y="170" width="220" height="50" class="mit" rx="6"/>
  <text x="690" y="192" text-anchor="middle" class="lbl">Calibration check</text>
  <text x="690" y="208" text-anchor="middle" class="lbl">(real correlation vs. human scores)</text>

  <path d="M580 195 L360 195" stroke="#555" stroke-width="2" marker-end="url(#arrJ)"/>
  <rect x="150" y="170" width="210" height="50" class="mit" rx="6"/>
  <text x="255" y="192" text-anchor="middle" class="lbl">Stability check</text>
  <text x="255" y="208" text-anchor="middle" class="lbl">(score variance across rubric variants)</text>

  <text x="450" y="255" text-anchor="middle" class="lbl">Green boxes: real, targeted mitigation/validation steps —</text>
  <text x="450" y="273" text-anchor="middle" class="lbl">bias mitigations happen before scoring; reliability checks (calibration, stability) validate the judge itself</text>

  <defs>
    <marker id="arrJ" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
</svg>

</div>

*   **Diagram Interpretation:** The top row shows the real judge-call pipeline with its two bias-mitigation interventions (randomized order, a distinct judge model) placed exactly where each one intervenes. The bottom row shows the two separate real reliability checks — calibration (Example 1) and stability (Example 3) — as validation steps applied to the judge's own real output, distinct from the bias-mitigation steps upstream.

---

## 3. Implementation & Reference Code

```python
def spearman_correlation(judge_scores: list[float], human_scores: list[float]) -> float:
    def rank(values):
        sorted_vals = sorted(values, reverse=True)
        return [sorted_vals.index(v) + 1 for v in values]

    judge_ranks = rank(judge_scores)
    human_ranks = rank(human_scores)
    n = len(judge_scores)
    d_sq_sum = sum((jr - hr) ** 2 for jr, hr in zip(judge_ranks, human_ranks))
    return 1 - (6 * d_sq_sum) / (n * (n**2 - 1))


def position_bias_flip_rate(trials: list[tuple[str, str]]) -> float:
    flips = sum(1 for orig, swap in trials if orig != swap)
    return flips / len(trials)


def rubric_instability(scores_a: list[float], scores_b: list[float]) -> tuple[float, float]:
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    return variance, variance ** 0.5


if __name__ == "__main__":
    # Example 1: calibration via Spearman correlation
    judge_scores = [8, 6, 9, 5, 7, 4]
    human_scores = [7, 6, 9, 4, 8, 3]
    rho = spearman_correlation(judge_scores, human_scores)
    print(f"Spearman correlation (judge vs. human): {rho:.4f}")
    assert abs(rho - 0.9429) < 0.001

    # Example 2: position-bias flip rate
    trials = [
        ("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"), ("A", "B"),
        ("B", "A"), ("A", "B"), ("A", "A"), ("B", "B"), ("A", "A"),
    ]
    flip_rate = position_bias_flip_rate(trials)
    print(f"Position-bias flip rate: {flip_rate*100:.1f}%")
    assert abs(flip_rate - 0.7) < 1e-9

    # Example 3: rubric instability
    rubric_a = [7, 8, 6, 9, 5]
    rubric_b = [8, 7, 6, 8, 6]
    variance, std = rubric_instability(rubric_a, rubric_b)
    print(f"Rubric instability: variance={variance:.2f}, std={std:.4f}")
    assert abs(variance - 0.8) < 1e-9

    print("\nVerified: a real, high judge-human correlation (0.9429) can coexist with a real, substantial")
    print("position-bias flip rate (70.0%) and real rubric instability (std~0.89) -- three genuinely")
    print("separate real judge-quality dimensions, none of which the others imply.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Real evaluation scalability — an LLM judge can score orders of magnitude more real outputs than human annotators, at a real, direct cost trade-off in bias/reliability risk that has to be actively managed, not assumed away.
* **Why Introduced over Legacy Approaches:** Automated overlap metrics (Module 02) can't capture real semantic/factual judgment the way a real LLM judge, given a rubric, plausibly can — at the real cost of the new bias/reliability risks this module covers.
* **Key Failure Modes & Limitations:** Deploying a judge with real, unmeasured position/verbosity/self-preference bias; trusting a judge's real calibration based on raw agreement rather than real rank correlation; assuming a judge validated once, on one rubric wording, stays reliable after any real rubric change.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** One real additional LLM forward pass (or several, under randomized-order or multi-sample bias mitigation) per item judged — a real, direct multiplier on evaluation cost relative to a pure automated metric.
* **Space/Memory Footprint:** Not a primary concern — judge calls are typically stateless, real API-style calls rather than requiring persistent memory.
* **Primary Bottleneck Type:** A real validity/reliability bottleneck — the central risk is a judge that looks authoritative while being systematically biased or unstable, not a compute-cost bottleneck.
* **Variable Legend:** $\rho$ = Spearman rank correlation coefficient; $d_i$ = per-item rank difference between judge and human; flip rate = fraction of paired trials where the judge's picked identity changes under reordering.

### 3. Production & Scalability
* **Deployment Considerations:** Real production LLM-judge deployments should re-run calibration (Example 1) and stability (Example 3) checks whenever the rubric, judge model, or judge prompt changes — treating a one-time validation as permanently valid is a real, common mistake this module's own instability example directly warns against.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why use Spearman rank correlation instead of raw percentage agreement to validate a judge?
        *   *A:* Raw agreement can be misleading for continuous/ordinal scores — what matters for most real use cases (ranking outputs, picking a winner) is whether the judge's real *relative ordering* tracks human ordering, which rank correlation measures directly, rather than whether individual score values match exactly.
    2.  *Q:* A judge shows strong Example-1-style calibration on one rubric. Is it safe to deploy without further checks?
        *   *A:* No — real calibration and real stability are separate properties; this module's own Example 3 shows a judge's scores can shift meaningfully under small rubric rewording even when its underlying calibration against human judgment (on the original rubric) was strong.
