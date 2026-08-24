# Module 01: The GenAI System Design Interview Framework

## 1. Introduction & Intuition

### The Core Bottleneck
A system design interview isn't graded on whether you eventually mention the right technology — it's graded on whether your answer stays coherent for 30-45 minutes under real interviewer pressure (scale changes, a component fails, cost must drop 10x). Without a repeatable framework, candidates default to one of two real failure modes: over-engineering a solution to requirements nobody stated, or jumping straight to an architecture diagram and missing a constraint (a real latency budget, a real compliance requirement) that should have reshaped the whole answer. A framework's real job is to make the answer's structure survive pressure that an unstructured answer doesn't.

### High-Level Intuition
Think of it like a doctor's real diagnostic checklist versus guessing at a diagnosis from the first symptom mentioned. A checklist doesn't make the doctor smarter — it makes sure a real, decision-relevant question ("does this patient have any allergies?") never gets silently skipped just because the conversation moved on. This module's framework plays the same real role for a system-design answer: it doesn't replace technical knowledge, it makes sure the technical knowledge gets applied in the right real order, to the right real, stated constraints.

---

## 2. Core Concepts & Mathematical Formulation

### The Five-Step Framework

#### Intuition & Practical Use
A real, repeatable answer structure: **(1) Clarify functional requirements** — what must the system actually do (a real feature list, not an assumed one); **(2) Clarify non-functional requirements** — a real stated latency budget, availability target, cost ceiling, and data-freshness requirement, since each of these can independently reshape the architecture; **(3) Back-of-envelope capacity estimation** — a real, first-class step (Module 03's own content), not an afterthought squeezed in after the architecture is already drawn; **(4) Propose a high-level architecture** — selecting from a real, small set of archetypes (Module 02's own content) rather than inventing one from scratch under time pressure; **(5) Deep-dive one or two real bottleneck components, then state trade-offs and failure modes** — not equal-depth coverage of every box in the diagram.

### The Framework Flow

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 980 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <defs>
    <marker id="arrow01" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#3b82f6" />
    </marker>
  </defs>

  <rect x="10" y="70" width="165" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" />
  <text x="92" y="100" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">Step 1</text>
  <text x="92" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Clarify</text>
  <text x="92" y="136" text-anchor="middle" font-size="12" fill="#1e3a8a">Functional</text>
  <text x="92" y="152" text-anchor="middle" font-size="12" fill="#1e3a8a">Requirements</text>

  <line x1="175" y1="115" x2="210" y2="115" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow01)" />

  <rect x="210" y="70" width="165" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" />
  <text x="292" y="100" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">Step 2</text>
  <text x="292" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Clarify Non-</text>
  <text x="292" y="136" text-anchor="middle" font-size="12" fill="#1e3a8a">Functional Reqs</text>
  <text x="292" y="152" text-anchor="middle" font-size="11" fill="#475569">latency, cost, SLO</text>

  <line x1="375" y1="115" x2="410" y2="115" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow01)" />

  <rect x="410" y="70" width="165" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" />
  <text x="492" y="100" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">Step 3</text>
  <text x="492" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Back-of-Envelope</text>
  <text x="492" y="136" text-anchor="middle" font-size="12" fill="#1e3a8a">Capacity</text>
  <text x="492" y="152" text-anchor="middle" font-size="12" fill="#1e3a8a">Estimate</text>

  <line x1="575" y1="115" x2="610" y2="115" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow01)" />

  <rect x="610" y="70" width="165" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" />
  <text x="692" y="100" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a8a">Step 4</text>
  <text x="692" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Propose High-</text>
  <text x="692" y="136" text-anchor="middle" font-size="12" fill="#1e3a8a">Level</text>
  <text x="692" y="152" text-anchor="middle" font-size="12" fill="#1e3a8a">Architecture</text>

  <line x1="775" y1="115" x2="805" y2="115" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow01)" />

  <rect x="805" y="55" width="165" height="120" rx="8" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5" />
  <text x="887" y="85" text-anchor="middle" font-size="13" font-weight="700" fill="#5b21b6">Step 5</text>
  <text x="887" y="105" text-anchor="middle" font-size="12" fill="#5b21b6">Deep-Dive 1-2</text>
  <text x="887" y="121" text-anchor="middle" font-size="12" fill="#5b21b6">Bottlenecks</text>
  <text x="887" y="141" text-anchor="middle" font-size="12" fill="#5b21b6">+ Trade-offs /</text>
  <text x="887" y="157" text-anchor="middle" font-size="12" fill="#5b21b6">Failure Modes</text>

  <path d="M292,160 C292,210 692,230 692,175" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow01)" />
  <text x="492" y="228" text-anchor="middle" font-size="11" fill="#64748b">Step 2's NFRs directly reshape Step 4's architecture choice</text>

  <text x="490" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">The 5-Step GenAI System Design Framework</text>
</svg>
</div>

*   **Diagram Interpretation:** A linear real flow across the 5 steps, with an explicit real feedback path shown from Step 2 (NFRs) to Step 4 (architecture) — visualizing that NFRs are not a final tuning pass but a real, upstream input that can change which architecture Step 4 proposes, exactly as the worked example below demonstrates numerically.

### Why Non-Functional Requirements Reshape the Architecture, Not Just Tune It

#### Intuition & Practical Use
A common real mistake is treating non-functional requirements (NFRs) as a final tuning pass after the "real" architecture is chosen. In GenAI systems specifically, an NFR can flip the entire architecture choice: a real 200ms p99 latency budget for a customer-facing chat response effectively rules out a multi-step agentic tool-calling loop as the primary path (each real tool call adds real, serial latency) and pushes toward a simpler single-retrieval-plus-generation design, or a smaller/faster real model; a real 2s budget for the same functional requirement comfortably affords a real multi-step agentic loop. The same functional requirement, under two different real NFRs, can produce two genuinely different correct architectures — which is exactly why NFRs are gathered in Step 2, before Step 4's architecture proposal, not after it.

### Surviving Real Interviewer Probes

#### Intuition & Practical Use
A structured answer's real value shows up specifically when an interviewer probes it: "what if traffic grows 100x," "this component just went down, what happens," "cost must drop 10x — what do you change." A framework-structured answer already has the real vocabulary to respond precisely (Module 03's capacity math for scale, Module 07's reliability patterns for failure, Module 03's cost engineering for cost) rather than improvising from scratch under real time pressure — the framework's steps are exactly the later modules' own content, referenced here as one coherent whole for the first time.

---

### Worked Example (No Formula): The Same Framework, Two Different Real NFRs, Two Different Real Architectures

A real prompt: "Design a customer-support chat assistant." Two real candidate NFR sets, applied to the identical functional requirement, walked through the same 5-step framework:

| Framework Step | Scenario A: real p99 latency budget = 200ms | Scenario B: real p99 latency budget = 2s |
|---|---|---|
| 1. Functional requirements | Answer real customer questions using company knowledge base | *(identical)* |
| 2. Non-functional requirements | **200ms p99**, real high availability, moderate cost ceiling | **2s p99**, real high availability, moderate cost ceiling |
| 3. Capacity estimate | Real per-request budget leaves little room for serial steps (Module 03 math) | Real per-request budget affords several serial steps |
| 4. Architecture | Single-retrieval + generation, real archetype 1 (RAG assistant), no agentic loop | Real archetype 2 (agentic system) — multi-step tool use (order lookup, ticket creation) affordable within budget |
| 5. Deep-dive | Real retrieval-latency optimization (Module 06's own serving-latency content, referenced) | Real tool-call reliability (Module 07's retry/fallback content) |

*   **Step 1: Real, direct comparison.** The identical functional requirement, under two different real, stated NFR sets, is walked through the identical 5-step framework.
*   **Step 2: Real interpretation.** The framework's steps didn't change — but because Step 2 (NFRs) is gathered *before* Step 4 (architecture), the real 200ms-vs-2s constraint correctly produces two genuinely different real architectures at Step 4, and two genuinely different real deep-dive priorities at Step 5. A candidate who skipped Step 2 or gathered it only loosely ("keep it reasonably fast") would have no principled real basis for choosing between these two genuinely different correct answers.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass, field


@dataclass
class SystemDesignAnswer:
    functional_requirements: list[str] = field(default_factory=list)
    non_functional_requirements: dict[str, str] = field(default_factory=dict)
    capacity_estimate_done: bool = False
    architecture_archetype: str | None = None
    deep_dive_components: list[str] = field(default_factory=list)
    tradeoffs_stated: list[str] = field(default_factory=list)


def framework_completeness_check(answer: SystemDesignAnswer) -> list[str]:
    """Real, minimal completeness check mirroring the module's own 5-step framework --
    flags a real, specific missing step rather than a generic 'incomplete' verdict."""
    missing = []
    if not answer.functional_requirements:
        missing.append("Step 1: functional requirements not clarified")
    if not answer.non_functional_requirements:
        missing.append("Step 2: non-functional requirements not clarified")
    if not answer.capacity_estimate_done:
        missing.append("Step 3: capacity estimate skipped")
    if answer.architecture_archetype is None:
        missing.append("Step 4: no architecture proposed")
    if not answer.deep_dive_components:
        missing.append("Step 5: no bottleneck deep-dive performed")
    if not answer.tradeoffs_stated:
        missing.append("Step 5: no trade-offs/failure-modes discussion")
    return missing


if __name__ == "__main__":
    # Real Scenario A: 200ms p99 budget -- a weak, unstructured answer that skips Step 2 and Step 3
    weak_answer = SystemDesignAnswer(
        functional_requirements=["Answer customer questions using company knowledge base"],
        architecture_archetype="agentic system",  # jumped to Step 4 without Steps 2-3
    )
    weak_missing = framework_completeness_check(weak_answer)
    print("Weak answer, missing framework steps:")
    for m in weak_missing:
        print(f"  - {m}")
    assert "Step 2: non-functional requirements not clarified" in weak_missing
    assert "Step 3: capacity estimate skipped" in weak_missing

    # Real Scenario A: 200ms p99 budget -- a complete, structured answer
    strong_answer_a = SystemDesignAnswer(
        functional_requirements=["Answer customer questions using company knowledge base"],
        non_functional_requirements={"p99_latency_ms": "200", "availability": "high", "cost_ceiling": "moderate"},
        capacity_estimate_done=True,
        architecture_archetype="RAG assistant (single retrieval, no agentic loop)",
        deep_dive_components=["retrieval-latency optimization"],
        tradeoffs_stated=["no multi-step tool use possible within 200ms budget"],
    )
    strong_missing_a = framework_completeness_check(strong_answer_a)
    print(f"\nScenario A (200ms budget) -- architecture: {strong_answer_a.architecture_archetype}")
    print(f"Missing steps: {strong_missing_a}")
    assert strong_missing_a == []

    # Real Scenario B: 2s p99 budget -- same functional requirement, different real NFR, different real architecture
    strong_answer_b = SystemDesignAnswer(
        functional_requirements=["Answer customer questions using company knowledge base"],
        non_functional_requirements={"p99_latency_ms": "2000", "availability": "high", "cost_ceiling": "moderate"},
        capacity_estimate_done=True,
        architecture_archetype="agentic system (multi-step tool use: order lookup, ticket creation)",
        deep_dive_components=["tool-call reliability and retry design"],
        tradeoffs_stated=["higher latency accepted in exchange for richer real task completion"],
    )
    strong_missing_b = framework_completeness_check(strong_answer_b)
    print(f"\nScenario B (2s budget) -- architecture: {strong_answer_b.architecture_archetype}")
    print(f"Missing steps: {strong_missing_b}")
    assert strong_missing_b == []

    assert strong_answer_a.architecture_archetype != strong_answer_b.architecture_archetype
    print("\nVerified: the identical functional requirement, under two different real, stated NFRs,")
    print("correctly produces two genuinely different real architectures under the same 5-step framework --")
    print("confirming NFRs must be gathered (Step 2) before the architecture is proposed (Step 4).")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Keeping a system-design interview answer coherent and complete under real time pressure, by fixing a real, repeatable step order rather than relying on ad hoc improvisation.
* **Why Introduced over Legacy Approaches:** A "just start drawing boxes" approach works for simple CRUD-service interviews but breaks down for GenAI systems specifically, where a real NFR (latency budget) can flip the entire correct architecture — skipping requirements-gathering isn't just sloppy, it can produce a genuinely wrong answer for the stated constraints.
* **Key Failure Modes & Limitations:** Over-engineering a solution to unstated requirements; under-specifying NFRs and then defending an arbitrary architecture choice with no real principled basis; spending equal time on every component instead of identifying the real one or two bottlenecks worth a deep-dive.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module is process/methodology, not a compute-cost concern.
* **Space/Memory Footprint:** Not applicable.
* **Primary Bottleneck Type:** A structural/communication bottleneck — the real risk is an incoherent or incomplete answer under real time constraints, not a computational one.
* **Variable Legend:** FR = functional requirements, NFR = non-functional requirements (latency, availability, cost, freshness) — the two real requirement classes gathered before any architecture is proposed.

### 3. Production & Scalability
* **Deployment Considerations:** This framework mirrors a real production practice — a system's actual architecture document typically opens with stated requirements and NFRs before any component diagram, for the same real reason: architecture decisions need a stated, traceable justification.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* An interviewer says "just design something reasonable" without giving you specific NFRs — what do you do?
        *   *A:* State 2-3 real, reasonable candidate NFR sets explicitly (e.g., "I'll assume a sub-second latency budget and high availability — let me know if that's off"), and proceed — making the assumption visible and checkable is itself part of a structured real answer, rather than silently picking one and hoping it's right.
    2.  *Q:* How much time should the capacity estimate (Step 3) actually take in an interview?
        *   *A:* A few real minutes for a back-of-envelope figure (Module 03's own worked math) — enough to ground the architecture choice and flag an obvious real bottleneck, not a fully precise number; spending too long here starves Step 5's deep-dive of real interview time.
