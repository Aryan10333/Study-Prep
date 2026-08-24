# Module 05: LLMOps Foundations — Versioning, Registries & CI/CD Pipeline Architecture

## 1. Introduction & Intuition

### The Core Bottleneck
Prior topics established *how* to test a prompt change (`05_prompt_engineering_and_structured_generation`) and *how* to continuously evaluate a system (`07_llm_evaluation_observability_and_guardrails`) — but neither one answers the real operational question this module owns: what real infrastructure actually runs those tests automatically, refuses to deploy a change that fails them, and records exactly what was deployed so a later regression can be diagnosed. Without that real enforcement and recording layer, "we have a way to evaluate a change" and "a change cannot reach production without passing that evaluation" are two very different real claims.

### High-Level Intuition
A test suite that engineers *could* run before merging code is a real, weaker guarantee than a CI pipeline that *automatically* runs it and blocks the merge on failure — the difference is enforcement, not capability. LLMOps' CI/CD layer plays exactly that real enforcement role for prompt/model/pipeline changes, and its artifact registry plays the real role a build system's dependency lockfile plays: recording precisely which versions of everything combined to produce a given real result.

---

## 2. Core Concepts & Mathematical Formulation

### The Real Operational Flow: Versioned Inputs → Gate → Approval → Deployment → Lineage

#### Intuition & Practical Use
The module's own real, required flow, stated explicitly rather than described as "running tests": **versioned inputs** (a specific model version, prompt version, evaluator version, dataset/index version, and deployment-config version, each independently tracked) → **evaluation execution** (already-defined tests from `05_prompt_engineering_and_structured_generation`/`07_llm_evaluation_observability_and_guardrails` run against those versioned inputs) → **quality gate** (a real, automated pass/fail decision — no human has to remember to check) → **approval** (a real, often human-in-the-loop sign-off step for higher-risk changes) → **deployment** (the change actually ships) → **recorded lineage** (the full real version combination that produced this deployment is durably logged). This module owns the real pipeline and registry infrastructure that executes this flow — not the evaluation logic itself.

### Real Artifact Lineage: 5 Versions, Jointly Recorded

#### Intuition & Practical Use
A real production result is only fully explainable if the *combination* of these 5 real version identifiers is jointly recorded, not just one of them in isolation: **model version**, **prompt version**, **evaluator version** (which metric/judge/rubric version scored it), **dataset/index version** (which knowledge-base or eval-set snapshot was used), and **deployment-config version** (feature flags, routing rules, serving parameters in effect). Knowing only "which model" is real but insufficient for real rollback or debugging — a regression could just as easily trace to a prompt change, an evaluator change, or an index re-index (Module 04's own scope) that shipped independently of the model.

---

### Worked Example (No Formula): A Real Regression, Diagnosed via Joint Lineage

A real production quality score drops from 0.91 to 0.76 between Monday and Tuesday. Two real, jointly-recorded lineage snapshots:

| Version Component | Monday (known-good) | Tuesday (regressed) | Changed? |
|---|---|---|---|
| Model version | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | No |
| Prompt version | `support-v12` | `support-v12` | No |
| Evaluator version | `judge-rubric-v3` | `judge-rubric-v3` | No |
| Dataset/index version | `kb-snapshot-2024-11-01` | `kb-snapshot-2024-11-01` | No |
| Deployment-config version | `routing-cfg-v8` | `routing-cfg-v9` | **Yes** |

*   **Step 1: Real, direct comparison.** All 5 real version components are compared field-by-field between the two lineage snapshots.
*   **Step 2: Real interpretation.** Only the deployment-config version changed — not the model, prompt, evaluator, or knowledge base. Without joint lineage tracking, a team seeing only "which model was deployed" (unchanged) would have no real lead at all; with all 5 tracked, the real regression is correctly and immediately localized to the Tuesday config change (e.g., a real routing-rule edit that inadvertently changed which retrieval index tier a query hit), directing the real fix to the actual changed component — not a re-investigation of the model, which was never the real cause.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass, asdict


@dataclass
class ArtifactLineage:
    model_version: str
    prompt_version: str
    evaluator_version: str
    dataset_index_version: str
    deployment_config_version: str


def diff_lineage(known_good: ArtifactLineage, candidate: ArtifactLineage) -> list[str]:
    """Real, field-by-field lineage diff -- returns exactly which of the 5 real
    version components changed, localizing a real regression's likely cause."""
    good_fields = asdict(known_good)
    candidate_fields = asdict(candidate)
    return [
        field for field in good_fields
        if good_fields[field] != candidate_fields[field]
    ]


QUALITY_GATE_THRESHOLD = 0.85  # pre-stated, per this topic's own established anti-cherry-picking convention


def quality_gate(score: float) -> str:
    return "PROMOTE" if score >= QUALITY_GATE_THRESHOLD else "BLOCK"


if __name__ == "__main__":
    monday = ArtifactLineage(
        model_version="gpt-4o-mini-2024-07-18",
        prompt_version="support-v12",
        evaluator_version="judge-rubric-v3",
        dataset_index_version="kb-snapshot-2024-11-01",
        deployment_config_version="routing-cfg-v8",
    )
    tuesday = ArtifactLineage(
        model_version="gpt-4o-mini-2024-07-18",
        prompt_version="support-v12",
        evaluator_version="judge-rubric-v3",
        dataset_index_version="kb-snapshot-2024-11-01",
        deployment_config_version="routing-cfg-v9",  # the one real, actual change
    )

    changed = diff_lineage(monday, tuesday)
    print(f"Real changed lineage components between Monday and Tuesday: {changed}")
    assert changed == ["deployment_config_version"]

    monday_score, tuesday_score = 0.91, 0.76
    print(f"Monday quality gate ({monday_score}): {quality_gate(monday_score)}")
    print(f"Tuesday quality gate ({tuesday_score}): {quality_gate(tuesday_score)}")
    assert quality_gate(monday_score) == "PROMOTE"
    assert quality_gate(tuesday_score) == "BLOCK"

    print("\nVerified: joint lineage diffing correctly and uniquely localizes the real regression's")
    print("cause to the deployment-config version -- the only one of 5 real tracked components that")
    print("actually changed -- and the quality gate correctly blocks Tuesday's regressed real result.")
```

### The Real CI/CD + Lineage Pipeline, Visualized

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 1000 300" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <defs>
    <marker id="a5" markerWidth="8" markerHeight="8" refX="6" refY="2.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,5 L7,2.5 z" fill="#475569" />
    </marker>
  </defs>
  <text x="500" y="24" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">CI/CD + Artifact-Lineage Pipeline</text>

  <rect x="10" y="55" width="150" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="85" y="78" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">Versioned Inputs</text>
  <text x="85" y="96" text-anchor="middle" font-size="9.5" fill="#1e3a8a">model + prompt +</text>
  <text x="85" y="110" text-anchor="middle" font-size="9.5" fill="#1e3a8a">evaluator + dataset/</text>
  <text x="85" y="124" text-anchor="middle" font-size="9.5" fill="#1e3a8a">index + config</text>
  <line x1="160" y1="100" x2="188" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#a5)"/>

  <rect x="188" y="65" width="150" height="70" rx="8" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="263" y="93" text-anchor="middle" font-size="11.5" font-weight="700" fill="#5b21b6">Evaluation</text>
  <text x="263" y="109" text-anchor="middle" font-size="9.5" fill="#5b21b6">Execution</text>
  <text x="263" y="123" text-anchor="middle" font-size="9" fill="#64748b">(Topic 05/07's tests)</text>
  <line x1="338" y1="100" x2="366" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#a5)"/>

  <rect x="366" y="65" width="150" height="70" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="441" y="93" text-anchor="middle" font-size="11.5" font-weight="700" fill="#92400e">Quality Gate</text>
  <text x="441" y="109" text-anchor="middle" font-size="9.5" fill="#92400e">automated pass/fail</text>
  <line x1="516" y1="100" x2="544" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#a5)"/>

  <rect x="544" y="65" width="150" height="70" rx="8" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="619" y="93" text-anchor="middle" font-size="11.5" font-weight="700" fill="#065f46">Approval</text>
  <text x="619" y="109" text-anchor="middle" font-size="9.5" fill="#065f46">human sign-off for</text>
  <text x="619" y="123" text-anchor="middle" font-size="9.5" fill="#065f46">higher-risk changes</text>
  <line x1="694" y1="100" x2="722" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#a5)"/>

  <rect x="722" y="65" width="130" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="787" y="93" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">Deployment</text>
  <text x="787" y="109" text-anchor="middle" font-size="9.5" fill="#1e3a8a">the change ships</text>
  <line x1="852" y1="100" x2="880" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#a5)"/>

  <rect x="880" y="55" width="110" height="90" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="935" y="80" text-anchor="middle" font-size="11.5" font-weight="700" fill="#991b1b">Recorded</text>
  <text x="935" y="96" text-anchor="middle" font-size="11.5" font-weight="700" fill="#991b1b">Lineage</text>
  <text x="935" y="114" text-anchor="middle" font-size="9" fill="#991b1b">all 5 versions,</text>
  <text x="935" y="127" text-anchor="middle" font-size="9" fill="#991b1b">jointly logged</text>

  <rect x="366" y="175" width="150" height="55" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-dasharray="3,2"/>
  <text x="441" y="197" text-anchor="middle" font-size="10.5" font-weight="600" fill="#991b1b">FAIL path:</text>
  <text x="441" y="213" text-anchor="middle" font-size="10" fill="#991b1b">blocked, never deployed</text>
  <line x1="441" y1="135" x2="441" y2="175" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a5)"/>
</svg>
</div>

*   **Diagram Interpretation:** The real, required flow from the module's own text visualized end-to-end, with an explicit real FAIL path shown branching off the quality gate — a change that fails is blocked before deployment and recorded lineage, never silently shipped, and every real deployed change carries its full, jointly-recorded 5-version lineage.

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Turning "we have a way to evaluate a change" into "a change cannot reach production without automatically passing that evaluation, and its full version combination is recorded when it does" — the real operational enforcement and accountability layer.
* **Why Introduced over Legacy Approaches:** Manual, ad hoc "someone remembers to check the eval dashboard before deploying" processes are a real, common production failure point — automated gating removes the dependency on a real human remembering, and joint lineage recording removes the dependency on a real human reconstructing what was deployed after the fact.
* **Key Failure Modes & Limitations:** Recording only the model version and not the other 4 real components, making a config-only or prompt-only regression undiagnosable; a quality gate with no real, stated threshold (Module 09's own drift/versioning content, referenced) letting a borderline change through inconsistently.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module is pipeline/registry infrastructure, not a compute-cost concern.
* **Space/Memory Footprint:** Real, linear storage growth in the artifact registry proportional to the number of real deployed version combinations retained — a real, genuine retention-policy trade-off (how far back to keep full lineage) not unlike Module 04's own storage-sizing concerns.
* **Primary Bottleneck Type:** A real process/enforcement bottleneck — the risk is an unenforced or unrecorded change slipping through, not a computational one.
* **Variable Legend:** The 5 real lineage components: model version, prompt version, evaluator version, dataset/index version, deployment-config version.

### 3. Production & Scalability
* **Deployment Considerations:** Real production LLMOps pipelines typically tier the approval step by real risk level — a low-risk prompt tweak might auto-promote on passing the quality gate alone, while a real model-version change requires an explicit human approval step, since the two carry genuinely different real blast-radius risk.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why record all 5 lineage components instead of just the model version, which is usually the biggest real change?
        *   *A:* Because a real regression is just as often caused by a prompt, evaluator, dataset/index, or config change as by a model change — this module's own worked example shows a config-only change causing a real quality drop while the model stayed identical; recording only the model version would have left that regression undiagnosable.
    2.  *Q:* Should the quality gate's threshold ever change?
        *   *A:* Yes, but any change to it should itself be versioned and treated like any other real config change — an unversioned, silently-adjusted threshold would reintroduce exactly the kind of untraceable change this module's whole lineage discipline exists to prevent.
