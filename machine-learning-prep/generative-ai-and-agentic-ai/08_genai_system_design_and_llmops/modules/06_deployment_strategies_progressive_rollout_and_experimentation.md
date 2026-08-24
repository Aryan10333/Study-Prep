# Module 06: Deployment Strategies, Progressive Rollout & Experimentation Infrastructure

## 1. Introduction & Intuition

### The Core Bottleneck
A regressed prompt or model version in a GenAI system doesn't fail loudly the way a crashing service does — it can silently degrade real answer quality for real users while every infrastructure-level health check stays green. That asymmetry (a real quality regression is often invisible to the metrics a naive deployment process watches) is exactly why GenAI systems need progressive, reversible rollout strategies more than a typical stateless microservice does: the real blast radius of a bad change has to be bounded *before* an aggregate metric has a chance to catch it.

### High-Level Intuition
Releasing a new drug isn't "give it to everyone Monday morning" — it's phased trials on progressively larger, monitored populations, with a real, pre-defined stopping rule if early signals look bad. Progressive rollout for a GenAI system change follows the identical real logic: a small, monitored population first (a canary), a real, stated rule for whether to expand or halt, before the change ever reaches full real production traffic.

---

## 2. Core Concepts & Mathematical Formulation

### Three Real Deployment Patterns

#### Intuition & Practical Use
**Blue-green deployment**: two real, complete environments (blue = current, green = new); traffic cuts over atomically once green is validated — fast real rollback (cut back to blue) but real double-infrastructure cost during the transition. **Canary release**: a real, small percentage of traffic routed to the new version, ramped in stages, with real per-stage monitoring — slower but bounds real blast radius incrementally. **Shadow deployment**: the new version runs against real production traffic *without* serving its output to real users — real zero user-facing risk, but requires real infrastructure to duplicate traffic and compare outcomes without a real live promotion signal.

### The Real Canary Promotion Rule — With a Required Monitoring Window

#### Purpose & High-level Intuition
A real, explicit per-stage promotion rule, evaluated only once a real minimum observation window is satisfied:

$$\text{Promote} = (\text{ErrorRate} \leq \tau_{\text{err}}) \land (\text{p99 Latency} \leq \tau_{\text{lat}}) \land (\text{QualityScore} \geq \tau_{\text{quality}}) \land (\text{GuardrailFlagRate} \leq \tau_{\text{safety}})$$

A conjunction of real, independently-monitored thresholds — any single one failing halts the ramp or triggers rollback, rather than an aggregate "average looks fine" check that could hide one badly-failing signal. **Stated explicitly as a required companion condition, not an implicit assumption**: this rule is only evaluated once a real per-stage minimum sample size ($N_{\text{min}}$) and minimum observation duration ($T_{\text{min}}$) have both been satisfied. A stage's real metrics computed from too few requests or too short a window are treated as **NOT_YET_DECIDABLE** — neither promoted nor rolled back — since a small, noisy early sample could otherwise trigger either decision incorrectly.

---

### Worked Example: A Real 3-Stage Canary Ramp, With One Correct Rollback and One "Not Yet Decidable" Case

Real stated thresholds: $\tau_{\text{err}}=1\%$, $\tau_{\text{lat}}=800\text{ms}$, $\tau_{\text{quality}}=0.85$, $\tau_{\text{safety}}=0.5\%$; real monitoring window: $N_{\text{min}}=500$ requests, $T_{\text{min}}=30$ minutes.

| Stage | Traffic % | Requests observed | Window elapsed | ErrorRate | p99 Latency | QualityScore | GuardrailFlagRate | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 (early snapshot) | 5% | 120 | 8 min | 0.5% | 650ms | 0.90 | 0.2% | **NOT_YET_DECIDABLE** — $N_{\text{min}}$/$T_{\text{min}}$ not yet met |
| 1 (full window) | 5% | 640 | 32 min | 0.6% | 690ms | 0.91 | 0.3% | **PROMOTE** — all 4 real thresholds pass |
| 2 | 25% | 2,100 | 35 min | 0.7% | 720ms | **0.79** | 0.3% | **ROLLBACK** — QualityScore fails $\tau_{\text{quality}}$; every other real signal passes |

*   **Step 1: Real, direct threshold evaluation**, applied only after the real $N_{\text{min}}$/$T_{\text{min}}$ window is satisfied at each stage.
*   **Step 2: Real interpretation.** Stage 1's early snapshot (120 requests, 8 minutes) *looks* fine on every metric, but is correctly held at NOT_YET_DECIDABLE rather than prematurely promoted — a real, deliberate demonstration that a threshold rule alone, without a monitoring-window requirement, could have promoted on a small, noisy, and potentially unrepresentative early sample. Stage 2 shows a real, correct rollback trigger: three of four real signals pass comfortably, but QualityScore alone (0.79 < 0.85) fails — the conjunction rule correctly rolls back on that single failing signal rather than a misleading "3 out of 4 looks mostly fine" judgment call.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass


@dataclass
class CanaryThresholds:
    max_error_rate: float
    max_p99_latency_ms: float
    min_quality_score: float
    max_guardrail_flag_rate: float
    min_requests: int
    min_window_minutes: float


@dataclass
class StageMetrics:
    requests_observed: int
    window_minutes: float
    error_rate: float
    p99_latency_ms: float
    quality_score: float
    guardrail_flag_rate: float


def canary_decision(metrics: StageMetrics, thresholds: CanaryThresholds) -> str:
    """Real per-stage decision -- the monitoring-window check runs FIRST, before
    any threshold is even evaluated, per the module's own required companion condition."""
    if metrics.requests_observed < thresholds.min_requests or metrics.window_minutes < thresholds.min_window_minutes:
        return "NOT_YET_DECIDABLE"

    checks = {
        "error_rate": metrics.error_rate <= thresholds.max_error_rate,
        "p99_latency": metrics.p99_latency_ms <= thresholds.max_p99_latency_ms,
        "quality_score": metrics.quality_score >= thresholds.min_quality_score,
        "guardrail_flag_rate": metrics.guardrail_flag_rate <= thresholds.max_guardrail_flag_rate,
    }
    return "PROMOTE" if all(checks.values()) else "ROLLBACK"


if __name__ == "__main__":
    thresholds = CanaryThresholds(
        max_error_rate=0.01, max_p99_latency_ms=800, min_quality_score=0.85,
        max_guardrail_flag_rate=0.005, min_requests=500, min_window_minutes=30,
    )

    stage1_early = StageMetrics(120, 8, 0.005, 650, 0.90, 0.002)
    stage1_full = StageMetrics(640, 32, 0.006, 690, 0.91, 0.003)
    stage2 = StageMetrics(2100, 35, 0.007, 720, 0.79, 0.003)

    for name, m in [("Stage 1 (early)", stage1_early), ("Stage 1 (full window)", stage1_full), ("Stage 2", stage2)]:
        decision = canary_decision(m, thresholds)
        print(f"{name}: {decision}")

    assert canary_decision(stage1_early, thresholds) == "NOT_YET_DECIDABLE"
    assert canary_decision(stage1_full, thresholds) == "PROMOTE"
    assert canary_decision(stage2, thresholds) == "ROLLBACK"

    print("\nVerified: an early, small-sample snapshot that looks 'all green' is correctly held at")
    print("NOT_YET_DECIDABLE rather than promoted, and a stage failing only its real quality threshold")
    print("(while passing the other 3) is correctly rolled back -- confirming both the monitoring-window")
    print("requirement and the any-single-signal-fails rollback logic work as the module states.")
```

### Progressive Rollout Patterns, Visualized

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 1000 380" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <text x="500" y="24" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">Three Real Progressive-Rollout Patterns</text>

  <rect x="20" y="50" width="290" height="150" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="165" y="75" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a8a">Blue-Green</text>
  <rect x="40" y="90" width="110" height="40" rx="6" fill="#ffffff" stroke="#2563eb"/>
  <text x="95" y="114" text-anchor="middle" font-size="10.5" fill="#1e3a8a">Blue (current)</text>
  <rect x="170" y="90" width="110" height="40" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="225" y="114" text-anchor="middle" font-size="10.5" fill="#1e3a8a">Green (new)</text>
  <text x="165" y="150" text-anchor="middle" font-size="10" fill="#1e3a8a">atomic cutover once validated</text>
  <text x="165" y="166" text-anchor="middle" font-size="10" fill="#1e3a8a">fast rollback, 2x infra cost</text>

  <rect x="340" y="50" width="330" height="150" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="505" y="75" text-anchor="middle" font-size="12.5" font-weight="700" fill="#5b21b6">Canary (staged ramp)</text>
  <rect x="360" y="90" width="60" height="34" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="390" y="111" text-anchor="middle" font-size="10" fill="#5b21b6">5%</text>
  <rect x="430" y="90" width="60" height="34" rx="5" fill="#ddd6fe" stroke="#7c3aed"/>
  <text x="460" y="111" text-anchor="middle" font-size="10" fill="#5b21b6">25%</text>
  <rect x="500" y="90" width="60" height="34" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="530" y="111" text-anchor="middle" font-size="10" fill="#5b21b6">100%</text>
  <text x="465" y="140" text-anchor="middle" font-size="10" fill="#5b21b6">each stage gated by the</text>
  <text x="465" y="156" text-anchor="middle" font-size="10" fill="#5b21b6">promotion rule + monitoring window</text>
  <text x="465" y="176" text-anchor="middle" font-size="9.5" fill="#5b21b6">(see worked example above)</text>

  <rect x="700" y="50" width="280" height="150" rx="10" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="840" y="75" text-anchor="middle" font-size="12.5" font-weight="700" fill="#065f46">Shadow</text>
  <rect x="720" y="90" width="110" height="40" rx="6" fill="#ffffff" stroke="#059669"/>
  <text x="775" y="114" text-anchor="middle" font-size="10.5" fill="#065f46">Live (serves user)</text>
  <rect x="850" y="90" width="110" height="40" rx="6" fill="#d1fae5" stroke="#059669" stroke-dasharray="3,2"/>
  <text x="905" y="106" text-anchor="middle" font-size="10" fill="#065f46">Shadow (new)</text>
  <text x="905" y="120" text-anchor="middle" font-size="9" fill="#065f46">output not served</text>
  <text x="840" y="150" text-anchor="middle" font-size="10" fill="#065f46">zero user-facing risk,</text>
  <text x="840" y="166" text-anchor="middle" font-size="10" fill="#065f46">needs traffic-duplication infra</text>

  <rect x="150" y="230" width="700" height="120" rx="10" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="500" y="255" text-anchor="middle" font-size="12.5" font-weight="700" fill="#92400e">Promotion / Rollback / Abort Criteria (applies at every canary stage)</text>
  <text x="500" y="280" text-anchor="middle" font-size="11" fill="#92400e">Promote  =  (ErrorRate &#8804; &#964;_err)  AND  (p99 Latency &#8804; &#964;_lat)  AND  (QualityScore &#8805; &#964;_quality)  AND  (GuardrailFlagRate &#8804; &#964;_safety)</text>
  <text x="500" y="304" text-anchor="middle" font-size="10.5" fill="#92400e">only evaluated once N_min requests AND T_min minutes have both elapsed --</text>
  <text x="500" y="322" text-anchor="middle" font-size="10.5" fill="#92400e">otherwise: NOT_YET_DECIDABLE, not a premature promotion</text>
</svg>
</div>

*   **Diagram Interpretation:** All 3 real deployment patterns shown with their genuinely different risk/speed trade-off, and the module's own real promotion/rollback rule (with its required monitoring-window condition) shown as the shared real gating logic that applies at every canary stage.

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Bounding the real blast radius of a GenAI system change whose regression may not trip any infrastructure-level health check, by ramping exposure progressively and gating each stage on a real, explicit, multi-signal decision rule.
* **Why Introduced over Legacy Approaches:** A single "deploy to 100% and watch the dashboard" approach exposes every real user to a potential regression simultaneously — progressive rollout exists specifically to make that exposure incremental and reversible.
* **Key Failure Modes & Limitations:** A promotion rule evaluated without the real monitoring-window requirement, letting a noisy early sample trigger a premature promotion; an aggregate "average signal" check that lets one badly-failing real metric hide behind three passing ones; shadow deployment's real infrastructure cost of duplicating traffic without a live comparison signal being underestimated.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module is deployment/rollout process, not a compute-cost concern.
* **Space/Memory Footprint:** Real, temporary double-infrastructure cost during blue-green's transition window; real, ongoing duplicated-traffic cost for shadow deployment's full run duration.
* **Primary Bottleneck Type:** A real detection-latency bottleneck — how quickly and reliably a real regression is caught before it reaches full real traffic, bounded by the real monitoring window's own length.
* **Variable Legend:** $\tau_{\text{err}}, \tau_{\text{lat}}, \tau_{\text{quality}}, \tau_{\text{safety}}$ = real per-signal thresholds; $N_{\text{min}}, T_{\text{min}}$ = real minimum sample size and observation duration required before any decision.

### 3. Production & Scalability
* **Deployment Considerations:** Real production canary ramps typically use a real, non-linear stage progression (e.g., 5% → 25% → 100%, not equal 25% steps) — smaller early stages bound the real blast radius most tightly exactly when the change is least proven.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why not just promote as soon as the metrics look good, without a minimum sample size?
        *   *A:* A small early sample is real but statistically noisy — this module's own worked example shows an 8-minute, 120-request snapshot looking "all green" purely by chance; requiring a real $N_{\text{min}}$/$T_{\text{min}}$ window before any decision prevents that noise from triggering a premature, unsupported promotion.
    2.  *Q:* One signal fails while three pass — why roll back instead of averaging or ignoring the one failure?
        *   *A:* Because the four real signals (error rate, latency, quality, safety) each catch a genuinely different real failure mode — a real quality regression wouldn't necessarily show up in error rate or latency at all, so requiring all four to pass (not an average) is what actually lets the rule catch a quality-only regression like Stage 2's.
