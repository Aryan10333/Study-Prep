# Module 07: Prompt Evaluation, Testing & Versioning

## 1. Introduction & Intuition

### The Core Bottleneck
A prompt template in a production codebase is executable behavior — changing its wording changes what the system does, the same way changing a function's logic does. Yet prompts are routinely edited in place, deployed without a regression check, and evaluated (if at all) by a developer eyeballing a couple of outputs. This module's core claim is that a prompt change deserves the same engineering discipline as a code change: a fixed evaluation set to check against, multiple genuinely distinct quality dimensions measured together, and an explicit versioning/rollback mechanism — not because it's a nice-to-have, but because Module 05 already showed prompt selection is a measurable optimization problem, and *evaluation* is the measurement infrastructure that makes it measurable at all.

### High-Level Intuition
Shipping a prompt change without regression testing is like shipping a code change straight to production with no test suite and no ability to roll back — it might work, and when it doesn't, there's no fast, confident way to know *before* users are affected, or to revert *after* they are. Treating a prompt like versioned, tested code isn't excessive process for something "just text" — it's applying the exact same discipline that already exists for every other piece of production logic, to a piece of production logic that happens to be a string.

---

## 2. Core Concepts & Mathematical Formulation

### Prompt Regression Testing Against a Fixed Eval Set

#### Intuition & Practical Use
A fixed, representative eval set — a stable, curated collection of real or realistic inputs with known-good expected properties (not necessarily one single "correct" output, especially for open-ended generation, but checkable properties: does it stay on-topic, does it satisfy required constraints, does it pass structured-output validation) — is what makes a prompt change checkable rather than a matter of impression. Running every candidate prompt change against the *same* fixed eval set, before and after, is what makes "did this change actually help or hurt" an answerable, evidence-based question instead of a guess.

### LLM-as-Judge for Prompt-Output Quality

#### Intuition & Practical Use
For output qualities that are hard to check with a simple rule (open-ended writing quality, tone appropriateness, whether a response genuinely addresses a nuanced question), an LLM can be prompted to judge another model's output against a rubric — a scalable alternative to exhaustive human review, at real additional per-evaluation cost. The general LLM-as-judge pitfalls apply here directly: prompt sensitivity in the judge's own instructions, and judge bias (a tendency to prefer certain stylistic qualities regardless of genuine quality) — mitigated the same way as anywhere else this pattern appears: a fixed, stable judging rubric, tracked as a trend over time, periodically spot-checked against real human judgment, not treated as ground truth on its own.

### A/B Testing Prompt Variants in Production

#### Intuition & Practical Use
Offline eval-set testing (the two sections above) checks a candidate prompt against known, curated inputs before deployment; A/B testing checks it against real, live traffic after deployment — routing a fraction of real requests to a candidate variant and comparing real production outcomes (user-facing signals like task completion, or the same multi-dimensional metrics covered below) against the incumbent prompt. This catches what offline eval sets structurally can't: real input distribution drift, edge cases the eval set didn't anticipate, and genuine user-behavior signals no offline judge can substitute for — at the real cost of exposing some fraction of live traffic to an unproven variant, which is why a sound rollback mechanism (below) matters as much as the test itself.

### Prompt Versioning & Rollback

#### Intuition & Practical Use
Treating a prompt template as a versioned artifact — stored with a clear identity distinct from the "current" edit, retrievable by version, deployable/rollback-able independently of a full application deploy — is the concrete infrastructure that makes both regression testing and A/B testing actually safe to run in production: a regression or a bad A/B result is only cheap to recover from if reverting to the prior version is a fast, well-defined action, not a manual reconstruction from memory or git archaeology.

### Prompt-Change Observability

#### Intuition & Practical Use
Every production call should log *which* prompt version actually produced a given output — without this, a quality regression noticed after the fact has no way to be correlated back to the specific prompt change that caused it, turning debugging into guesswork exactly the way un-versioned code changes would. This is the same observability discipline `03_advanced_rag` Module 09 and `04_ai_agents_and_protocols` Module 08 apply to their own systems, applied here specifically to prompt version identity.

---

### Worked Example: Multi-Dimensional Prompt Evaluation, Not Regression Rate Alone
A concrete comparison of an incumbent prompt (v3) against a candidate (v4) on the *same* 50-example eval set, deliberately reporting every dimension together rather than collapsing to one number:

| Dimension | Incumbent (v3) | Candidate (v4) | Delta |
|---|---|---|---|
| Accuracy / quality score | 0.82 | 0.89 | **+0.07** (better) |
| Structured-output validity rate | 0.94 | 0.91 | **−0.03** (worse) |
| P50 latency (per call) | 620ms | 810ms | **+190ms** (worse) |
| Token cost (per call) | \$0.0041 | \$0.0058 | **+\$0.0017, +41%** (worse) |
| Robustness (accuracy variance across 5 paraphrased inputs per example) | ±0.04 | ±0.09 | **worse — more sensitive to phrasing** |
| Regression rate (examples that passed under v3, failed under v4) | — | 6 / 50 (12%) | **12% of previously-passing examples now fail** |

Read as one collapsed "quality" number, v4 looks like a clear win — accuracy improved by 7 points. Read across all six dimensions together, the real trade-off is far less obviously favorable: v4 is meaningfully slower (+190ms, roughly 31% slower), meaningfully more expensive (+41% token cost, likely from added instruction/reasoning verbosity), less reliably structured, more sensitive to phrasing (worse robustness), and — critically — the regression rate shows it's not a strict improvement even on correctness: 12% of examples that v3 handled correctly, v4 now gets wrong, even though v4's *aggregate* accuracy is higher. A production decision has to weigh accuracy gain against real latency/cost/reliability cost and the specific examples that regressed — a prompt with better aggregate accuracy but meaningfully worse cost, latency, and structured-output reliability is not automatically the better production choice, and the regression-rate figure alone would have missed most of this picture entirely.

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of the multi-dimensional evaluation comparison above, computing every dimension from a structured per-example result log rather than a single collapsed score.

```python
from dataclasses import dataclass, field


@dataclass
class ExampleResult:
    example_id: int
    correct: bool
    structured_valid: bool
    latency_ms: float
    cost: float
    paraphrase_accuracies: list[float]  # accuracy across N paraphrased variants of this example


@dataclass
class EvalRunSummary:
    accuracy: float
    structured_validity_rate: float
    p50_latency_ms: float
    avg_cost: float
    avg_robustness_stddev: float


def summarize(results: list[ExampleResult]) -> EvalRunSummary:
    import statistics

    accuracy = sum(1 for r in results if r.correct) / len(results)
    validity_rate = sum(1 for r in results if r.structured_valid) / len(results)
    latencies = sorted(r.latency_ms for r in results)
    p50_latency = latencies[len(latencies) // 2]
    avg_cost = sum(r.cost for r in results) / len(results)
    robustness_stddevs = [statistics.pstdev(r.paraphrase_accuracies) for r in results if len(r.paraphrase_accuracies) > 1]
    avg_robustness = sum(robustness_stddevs) / len(robustness_stddevs) if robustness_stddevs else 0.0

    return EvalRunSummary(
        accuracy=accuracy,
        structured_validity_rate=validity_rate,
        p50_latency_ms=p50_latency,
        avg_cost=avg_cost,
        avg_robustness_stddev=avg_robustness,
    )


def regression_rate(baseline: list[ExampleResult], candidate: list[ExampleResult]) -> float:
    """Fraction of examples that PASSED under baseline but FAIL under candidate --
    a distinct signal from aggregate accuracy delta, matching the module's point
    that a higher aggregate score can still hide real, specific regressions."""
    baseline_by_id = {r.example_id: r.correct for r in baseline}
    regressed = 0
    compared = 0
    for r in candidate:
        if r.example_id in baseline_by_id:
            compared += 1
            if baseline_by_id[r.example_id] and not r.correct:
                regressed += 1
    return regressed / compared if compared else 0.0


if __name__ == "__main__":
    import random
    random.seed(11)

    def make_results(n: int, correct_ids: set[int], base_latency: float, base_cost: float,
                      base_validity: float, base_robustness_range: float) -> list[ExampleResult]:
        """Correctness is assigned by an EXPLICIT id set, not a probabilistic draw --
        this is a constructed worked example illustrating that aggregate accuracy and
        regression rate are genuinely different signals, not a claim about real-world
        prompt-comparison statistics, so it's built deterministically on purpose."""
        out = []
        for i in range(n):
            out.append(ExampleResult(
                example_id=i,
                correct=i in correct_ids,
                structured_valid=random.random() < base_validity,
                latency_ms=base_latency + random.uniform(-30, 30),
                cost=base_cost + random.uniform(-0.0003, 0.0003),
                paraphrase_accuracies=[0.85 + random.uniform(-base_robustness_range, base_robustness_range) for _ in range(5)],
            ))
        return out

    # v3: correct on ids 0-40 (41/50 = 0.82 accuracy).
    v3_correct_ids = set(range(0, 41))
    # v4: REGRESSES ids 0-5 (6 examples v3 got right, v4 now gets wrong), but FIXES
    # 8 of v3's 9 wrong examples (ids 41-48) -- net higher aggregate accuracy (43/50 = 0.86)
    # while still containing a real, nonzero regression rate. This is the exact
    # mechanism the module's worked example describes, made explicit and deterministic.
    v4_correct_ids = (v3_correct_ids - set(range(0, 6))) | set(range(41, 49))

    v3_results = make_results(50, correct_ids=v3_correct_ids, base_latency=620, base_cost=0.0041,
                                base_validity=0.94, base_robustness_range=0.04)
    v4_results = make_results(50, correct_ids=v4_correct_ids, base_latency=810, base_cost=0.0058,
                                base_validity=0.91, base_robustness_range=0.09)

    v3_summary = summarize(v3_results)
    v4_summary = summarize(v4_results)

    print(f"v3: accuracy={v3_summary.accuracy:.2f}, validity={v3_summary.structured_validity_rate:.2f}, "
          f"p50_latency={v3_summary.p50_latency_ms:.0f}ms, cost=${v3_summary.avg_cost:.4f}, robustness_std={v3_summary.avg_robustness_stddev:.3f}")
    print(f"v4: accuracy={v4_summary.accuracy:.2f}, validity={v4_summary.structured_validity_rate:.2f}, "
          f"p50_latency={v4_summary.p50_latency_ms:.0f}ms, cost=${v4_summary.avg_cost:.4f}, robustness_std={v4_summary.avg_robustness_stddev:.3f}")

    rate = regression_rate(v3_results, v4_results)
    print(f"\nRegression rate (v3 passed, v4 failed): {rate:.2%}")

    # Verify the core point: v4 can show higher aggregate accuracy AND a nonzero
    # regression rate at the same time -- these are genuinely different signals.
    assert v4_summary.accuracy > v3_summary.accuracy, "v4 should show higher aggregate accuracy in this constructed example"
    assert rate > 0.0, "v4 must show a nonzero regression rate despite higher aggregate accuracy -- the whole point of tracking it separately"
    assert v4_summary.p50_latency_ms > v3_summary.p50_latency_ms
    assert v4_summary.avg_cost > v3_summary.avg_cost
    print("\nVerified: v4's higher aggregate accuracy coexists with a real, nonzero regression rate and worse latency/cost --")
    print("confirming aggregate accuracy alone would have hidden all three of these real trade-offs.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Turning "does this prompt change actually help" from an unmeasured impression into an evidence-based, multi-dimensional, repeatable check — the same rigor already expected of any other production code change.
* **Why Introduced over Legacy Approaches:** Editing a prompt in place with no eval set, no versioning, and no regression check has no mechanism to catch a real quality regression before it reaches users, or to quickly recover once one is found.
* **Key Failure Modes & Limitations:** Collapsing evaluation to a single aggregate quality number, hiding real cost/latency/robustness/regression trade-offs (this module's worked example); relying on LLM-as-judge output as unquestioned ground truth without periodic human spot-checks; deploying a prompt change with no versioning, making rollback slow or manual.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Regression testing cost scales with eval-set size × number of dimensions checked (most dimensions are cheap to compute from already-collected call metadata, except LLM-as-judge dimensions, which cost a real additional LLM call per judged example).
* **Space/Memory Footprint:** Requires storing the eval set, per-example results across versions (for regression-rate computation), and prompt version history — all real, if modest, storage relative to the LLM-call costs involved.
* **Primary Bottleneck Type:** Not compute-bound — the real bottleneck is engineering discipline: maintaining a genuinely representative eval set over time, and actually gating deployment on the full multi-dimensional check rather than skipping it under time pressure.
* **Variable Legend:** No closed-form formula in this module, per the Non-Goals constraint on advanced statistical derivations — all quantities in the worked example are direct, computable rates/deltas from per-example results.

### 3. Production & Scalability
* **Deployment Considerations:** Gate every prompt deployment on the full multi-dimensional comparison (accuracy, structured-output validity, latency, cost, robustness, regression rate), not accuracy alone; version every deployed prompt with a fast, well-defined rollback path; log the specific prompt version behind every production call for observability.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* A candidate prompt shows higher aggregate accuracy on your eval set. Would you ship it?
        *   *A:* Not without checking the other dimensions first — this module's worked example shows a real case where higher aggregate accuracy coexists with a real regression rate, worse latency, and worse cost; the full picture, not one number, should drive the decision.
    2.  *Q:* How do you keep an LLM-as-judge evaluation trustworthy over time?
        *   *A:* Hold the judge's own prompt/rubric fixed across comparisons (never compare scores produced by two different judge-prompt versions), and periodically spot-check its judgments against real human review to catch judge drift or bias before it silently skews every downstream comparison.
