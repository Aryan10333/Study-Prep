# Module 09: Production Evaluation Pipelines — Continuous Evaluation, Regression Testing & Drift Detection

## 1. Introduction & Intuition

### The Core Bottleneck
Every technique in Modules 02-08 is a real, one-time or periodic check unless it's turned into a continuously-running, automated real process. This module covers exactly that automation — CI/CD-style continuous evaluation for model/system-level changes (distinct from `05_prompt_engineering_and_structured_generation` Module 07's prompt-template-level A/B testing). But automating evaluation introduces a genuinely new real risk this module treats as its own, distinct failure mode: the evaluation *pipeline itself* can silently change — its dataset, its evaluator, its configuration — producing an apparent quality shift that has nothing to do with any real underlying system change.

### High-Level Intuition
A factory's automated quality-control line is only trustworthy if its own measuring instruments stay calibrated and consistent run to run — if someone quietly swaps the measuring gauge for a slightly different one between production runs, an apparent quality change on the report might be entirely an artifact of the new gauge, not the product. Continuous evaluation has exactly this same real risk, and this module treats it as a genuinely separate failure mode from the product (system) actually changing.

---

## 2. Core Concepts & Mathematical Formulation

### Evaluation-Set/Configuration Versioning and Reproducibility

#### Intuition & Practical Use
A continuous-evaluation result is only trustworthy if the real evaluation dataset, model version, prompt/configuration, and evaluator (metric or judge) version are all tracked together, as a real, versioned unit. An unversioned change in any single one of these — the eval set gets updated references, the judge model gets silently upgraded, a config default changes — can produce a real, misleading apparent quality shift with zero connection to the actual system change under test. This is a real prerequisite for trusting continuous evaluation at all, not an optional nicety.

### Real Drift, Explicitly Split Into Distinct Types

#### Intuition & Practical Use
Once evaluation-pipeline versioning is under control, genuine real system-quality drift still needs to be diagnosed correctly — and "drift" is not one uniform phenomenon. **Input/data drift**: the real distribution of incoming requests changes (e.g., users start asking longer, differently-shaped queries). **Output/quality drift**: real output quality degrades while the input distribution stays stable — pointing at the system itself, not what's being asked of it. **Model/behavior drift**: the underlying real model's behavior changes, e.g., after a silent provider-side update, detectable via a fixed regression suite shifting even when nothing in the team's own configuration changed. Each real type has a genuinely different real detection method and remediation — conflating them leads to fixing the wrong thing.

---

### Worked Example 1: Evaluation-Pipeline Drift/Versioning Failure — A Distinct Real Failure Mode
The **identical** real model output, scored against two different real versions of an evaluation set's reference answers (Eval Set V1's originals vs. Eval Set V2's updated references for the same 10 items).

$$\text{Accuracy}_{\text{V1}} = \frac{8}{10} = 0.80 \qquad \text{Accuracy}_{\text{V2}} = \frac{6}{10} = 0.60 \quad (\text{SAME model output both times})$$

A real, apparent 20-percentage-point quality drop — produced entirely by the real evaluation-set reference update, with the model's real output never changing at all between the two scoring runs. **This is explicitly labeled evaluation-pipeline drift/versioning failure — a distinct real failure mode from genuine system-quality drift** (Worked Example 2, below), not folded into the same category. A team that didn't track the eval-set version change here would wrongly conclude their real system regressed.

### Worked Example 2: Three Genuine System-Quality Drift Types, Each Requiring Different Remediation
Conceptually distinct from Worked Example 1's versioning-failure mode — these three scenarios assume the evaluation pipeline itself is correctly versioned and stable, and diagnose real, genuine system-quality change instead.

| Drift type | Real signal | Baseline | Current | Real change | Remediation |
|---|---|---|---|---|---|
| **Input/data drift** | Mean query length | 12 words | 19 words | $+58.3\%$ | Re-tune retrieval/context budget (Module 06 of `06_llm_inference_and_optimization`'s own batching content, referenced not re-derived) for the new real input profile — not necessarily a model problem |
| **Output/quality drift** | Faithfulness (Module 05), stable input dist. | 0.85 | 0.68 | $-20.0\%$ | Investigate real system components (retrieval index staleness, model degradation) since input itself hasn't shifted |
| **Model/behavior drift** | Fixed regression-suite pass rate, identical prompts | 0.95 | 0.80 | $-15.8\%$ | Pin/verify real model version, investigate real provider-side changes — an abrupt shift tied to the model itself, not a gradual quality slide |

Each real remediation is genuinely different — re-tuning for a new input profile doesn't fix a genuine model-behavior regression, and pinning a model version doesn't address an input-distribution shift. Correctly attributing *which* type of real drift occurred is the real, necessary first step before choosing a remediation.

<div style="text-align:center">

<svg viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;height:auto;font-family:sans-serif">
  <style>
    .lbl{font-size:11px;fill:#333}
    .hdr{font-size:14px;font-weight:bold;fill:#222}
    .ver{fill:#eaf1fb;stroke:#4c78a8;stroke-width:1.5}
    .gate{fill:#fdeedd;stroke:#c96;stroke-width:1.5}
    .out{fill:#d9f2e3;stroke:#2e8b57;stroke-width:1.5}
  </style>
  <text x="450" y="22" text-anchor="middle" class="hdr">Continuous Evaluation Pipeline</text>

  <rect x="20" y="50" width="150" height="30" class="ver" rx="4"/>
  <text x="95" y="70" text-anchor="middle" class="lbl">Eval dataset (versioned)</text>
  <rect x="20" y="90" width="150" height="30" class="ver" rx="4"/>
  <text x="95" y="110" text-anchor="middle" class="lbl">Model version (versioned)</text>
  <rect x="20" y="130" width="150" height="30" class="ver" rx="4"/>
  <text x="95" y="150" text-anchor="middle" class="lbl">Prompt/config (versioned)</text>
  <rect x="20" y="170" width="150" height="30" class="ver" rx="4"/>
  <text x="95" y="190" text-anchor="middle" class="lbl">Evaluator (versioned)</text>

  <text x="95" y="220" text-anchor="middle" class="lbl" fill="#b05a3a">An unversioned change here =</text>
  <text x="95" y="235" text-anchor="middle" class="lbl" fill="#b05a3a">Worked Example 1's failure mode</text>

  <path d="M175 120 L225 120" stroke="#555" stroke-width="2" marker-end="url(#arrP)"/>

  <rect x="230" y="90" width="160" height="60" class="gate" rx="6"/>
  <text x="310" y="115" text-anchor="middle" class="lbl" font-weight="bold">Automated eval run</text>
  <text x="310" y="132" text-anchor="middle" font-size="10">scores against all 4</text>
  <text x="310" y="145" text-anchor="middle" font-size="10">versioned inputs above</text>

  <path d="M390 120 L445 120" stroke="#555" stroke-width="2" marker-end="url(#arrP)"/>

  <rect x="450" y="90" width="160" height="60" class="gate" rx="6"/>
  <text x="530" y="115" text-anchor="middle" class="lbl" font-weight="bold">Drift-type diagnosis</text>
  <text x="530" y="132" text-anchor="middle" font-size="10">input / output / model</text>
  <text x="530" y="145" text-anchor="middle" font-size="10">(Worked Example 2)</text>

  <path d="M610 120 L665 120" stroke="#555" stroke-width="2" marker-end="url(#arrP)"/>

  <rect x="670" y="90" width="200" height="60" class="out" rx="6"/>
  <text x="770" y="112" text-anchor="middle" class="lbl" font-weight="bold">Gate decision</text>
  <text x="770" y="130" text-anchor="middle" font-size="10">pass / canary release /</text>
  <text x="770" y="145" text-anchor="middle" font-size="10">rollback trigger</text>

  <defs>
    <marker id="arrP" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
</svg>

</div>

*   **Diagram Interpretation:** The four versioned inputs feed the automated eval run — an unversioned change to any one of them is exactly where Worked Example 1's failure mode enters the pipeline. The drift-type diagnosis step, downstream and conceptually separate, applies Worked Example 2's three-way distinction only once the inputs above are confirmed stable and correctly versioned.

---

## 3. Implementation & Reference Code

```python
def accuracy(correct: int, total: int) -> float:
    return correct / total


def diagnose_versioning_failure(acc_v1: float, acc_v2: float, model_output_changed: bool) -> str:
    if not model_output_changed and acc_v1 != acc_v2:
        return "evaluation-pipeline versioning failure (NOT a real system-quality change)"
    return "real system-quality change (or no change)"


def diagnose_drift_type(input_changed: bool, output_changed_given_stable_input: bool, fixed_regression_changed: bool) -> str:
    if fixed_regression_changed:
        return "model/behavior drift"
    if input_changed:
        return "input/data drift"
    if output_changed_given_stable_input:
        return "output/quality drift"
    return "no drift detected"


if __name__ == "__main__":
    # Worked Example 1: same model output, different eval-set reference versions
    acc_v1 = accuracy(8, 10)
    acc_v2 = accuracy(6, 10)
    print(f"Eval-set V1 accuracy: {acc_v1}, V2 accuracy (SAME model output): {acc_v2}")
    diagnosis = diagnose_versioning_failure(acc_v1, acc_v2, model_output_changed=False)
    print(f"Diagnosis: {diagnosis}")
    assert abs((acc_v1 - acc_v2) - 0.2) < 1e-9  # float arithmetic: 0.8 - 0.6 != exactly 0.2
    assert "versioning failure" in diagnosis

    # Worked Example 2: three distinct drift types
    scenarios = [
        {"name": "input_drift", "baseline": 12, "current": 19, "type": diagnose_drift_type(True, False, False)},
        {"name": "output_drift", "baseline": 0.85, "current": 0.68, "type": diagnose_drift_type(False, True, False)},
        {"name": "model_drift", "baseline": 0.95, "current": 0.80, "type": diagnose_drift_type(False, False, True)},
    ]
    for s in scenarios:
        change_pct = (s["current"] - s["baseline"]) / s["baseline"] * 100
        print(f"{s['name']}: {s['baseline']} -> {s['current']} ({change_pct:+.1f}%), diagnosed as: {s['type']}")

    assert scenarios[0]["type"] == "input/data drift"
    assert scenarios[1]["type"] == "output/quality drift"
    assert scenarios[2]["type"] == "model/behavior drift"
    assert len({s["type"] for s in scenarios}) == 3, "All three scenarios must be diagnosed as genuinely distinct drift types"

    print("\nVerified: a real 20-percentage-point apparent quality change was correctly attributed to eval-set")
    print("versioning, not system drift; and three distinct real drift scenarios were each correctly diagnosed")
    print("as genuinely different types, each implying a different real remediation.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Turning every earlier module's evaluation technique into a real, automated, continuously-running production process — while correctly distinguishing an evaluation-pipeline artifact from genuine system drift, and correctly attributing genuine drift to its real, specific type.
* **Why Introduced over Legacy Approaches:** A one-time offline evaluation run, however rigorous, says nothing about real ongoing production quality — continuous evaluation is what catches a real regression *after* deployment, not just before it.
* **Key Failure Modes & Limitations:** Trusting a continuous-evaluation quality shift without first ruling out Worked Example 1's versioning-failure mode; treating all real "quality degraded" signals identically instead of diagnosing which of the three real drift types (Worked Example 2) actually occurred, leading to the wrong remediation being applied.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Real, direct multiplier proportional to how frequently continuous evaluation runs and how large the real evaluation set is — a genuine, real trade-off between catching drift quickly and the real compute/cost of running evaluation continuously.
* **Space/Memory Footprint:** Real, growing storage for versioned datasets/configs/evaluator artifacts across every real evaluation run — necessary for the reproducibility this module's own Worked Example 1 depends on.
* **Primary Bottleneck Type:** A real trustworthiness bottleneck — an unversioned or misdiagnosed continuous-evaluation pipeline can be actively misleading, worse than having no continuous evaluation at all if its false signals drive the wrong real engineering response.
* **Variable Legend:** Versioning = tracking real dataset/model/config/evaluator identity together as one unit; drift type = which of the three real, distinct system-quality-change categories (input/output/model) a genuine regression belongs to.

### 3. Production & Scalability
* **Deployment Considerations:** Real production continuous-evaluation pipelines should gate deployments via canary releases and real, automated rollback triggers (visualized in this module's own diagram) — but only after Worked Example 1's versioning check passes and Worked Example 2's drift-type diagnosis correctly attributes any real detected change.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* A continuous-evaluation dashboard shows a sudden quality drop after a routine eval-set update. What would you check first?
        *   *A:* Whether the real model's actual output changed at all, or whether — as in this module's own Worked Example 1 — the eval set's real reference answers changed instead, which alone can produce an apparent quality drop with zero real underlying system change.
    2.  *Q:* Why does it matter whether a real regression is classified as input drift vs. model drift, if the quality metric dropped either way?
        *   *A:* Because the real, correct remediation is genuinely different for each — re-tuning for a new real input profile (input drift) does nothing to fix an actual model-behavior regression (model drift), and vice versa; misdiagnosing the type wastes real engineering effort on the wrong fix.
