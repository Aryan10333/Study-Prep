# Module 09: End-to-End GenAI System Design Case Studies

## 1. Introduction & Intuition

### The Core Bottleneck
Every prior module in this topic is real and individually correct — but a real interview answer that mechanically visits all 8 of them at equal depth is a genuinely worse answer than one that spends real, uneven time on the one or two components that actually matter for the specific system being asked about. This module's real goal is demonstrating that prioritization skill directly, not adding a 9th technical concept.

### High-Level Intuition
A doctor giving a patient a full-body MRI for a sprained ankle isn't being thorough — they're wasting real time and missing the actual point. A strong system-design answer, like a strong diagnosis, spends its real limited time where the actual problem is, and states everything else briefly enough to show awareness without burning the clock.

---

## 2. Core Concepts & Mathematical Formulation

### Synthesis, Not a New Technique

#### Intuition & Practical Use
This module introduces no new formula or mechanism — it applies Module 01's 5-step framework, in order, to 2 full real systems, explicitly citing Modules 02-08's own already-established content (architecture archetype, capacity/cost math, data lifecycle, LLMOps lineage, rollout strategy, reliability plan, security/authorization) as already-built answers being composed, not re-derived. The real, organizing goal of each walkthrough is demonstrating prioritization under real interview time constraints.

### A Real Weak Answer vs. a Real Prioritized Answer

#### Intuition & Practical Use
A real, common weak pattern: touching all 8 prior modules for roughly equal, shallow real time — "we'd use RAG here, and here's our CI/CD, and here's our security..." — with no real component receiving enough depth to demonstrate genuine understanding. A real, stronger pattern: a brief real requirements/architecture sketch (lightly citing Modules 02-08 as needed), then explicitly naming and justifying which one or two components are the *actual* real bottleneck for *this specific* system, and spending the real majority of remaining time there.

---

### Worked Example: RAG-Based Enterprise Knowledge Assistant (Time-Boxed, ~35 Minutes)

**Step 1-2 (Requirements, ~5 min):** FR: answer employee questions from internal company docs across multiple real business units. NFR: p99 latency 1.5s (real, moderate — not ultra-low-latency), high availability, real strict multi-tenant/department data isolation (a regulated enterprise, real compliance-sensitive).

**Step 3 (Capacity, ~3 min, citing Module 03):** Real, brief Little's-Law-based GPU estimate stated aloud, not re-derived on the whiteboard in full: "at our stated QPS and token counts, this comes out to roughly N GPUs at our target utilization" — a real, cited number, not a restated derivation.

**Step 4 (Architecture, ~5 min, citing Module 02):** Archetype 1 (RAG assistant) — single retrieval-then-generation flow, no agentic loop needed for this FR.

**Step 5 (Real, prioritized deep-dive, ~18 min): Data infrastructure + authorization (Modules 04 and 08), explicitly justified as the real bottleneck for THIS system** — because the real, dominant risk for a multi-department enterprise knowledge assistant is a real cross-department data leak, not raw latency or throughput. Deep-dive covers: real per-department metadata-filtered retrieval (Module 08's own worked example, directly reused), real deletion-propagation for departing-employee document access revocation (Module 04), and a real, brief mention of LLMOps lineage (Module 05) for auditability. Deployment (Module 06) and reliability (Module 07) are each cited in one real sentence, not deep-dived — correctly, since they are not this system's dominant real risk.

**Step 5 (closing, ~4 min):** Real trade-offs stated: chose stricter, physically-separated per-department indexes over cheaper shared-index metadata filtering, accepting real higher storage cost (Module 04's own replication-cost math) for a stronger real isolation guarantee, given the real regulated-enterprise context.

### Worked Example: Agentic Coding Copilot (Time-Boxed, ~35 Minutes)

**Step 1-2 (Requirements, ~5 min):** FR: read a codebase, run tests, propose and iterate on a fix. NFR: real availability expectation is high (developers rely on it continuously), real latency budget is generous (2-5s per step, multi-step task), real cost ceiling is moderate.

**Step 3-4 (Capacity + Architecture, ~6 min, citing Modules 02-03):** Archetype 2 (agentic system) — real multi-step loop (read → run tests → evaluate → iterate); capacity estimate cited briefly, noting the real per-step latency accumulation this archetype's own dominant bottleneck implies.

**Step 5 (Real, prioritized deep-dive, ~19 min): Reliability engineering (Module 07), explicitly justified as the real bottleneck for THIS system** — because a real multi-step agentic loop chains several real external calls (test runners, real tool invocations), and any one real transient failure mid-loop is far more consequential here than in a single-shot RAG answer. Deep-dive covers: real retry-eligibility taxonomy applied specifically to a "run tests" tool call (idempotent, real safe to retry) versus a "commit fix" tool call (non-idempotent, needs a real idempotency key before any retry); real circuit-breaker behavior if the test-runner service itself degrades; a real, stated fallback-model capability trade-off (a smaller fallback model may reason less reliably through a multi-step fix, a real, explicit degradation, not silently accepted). Security (Module 08) is cited in one real sentence (tool access scoped to the real specific repo, not the developer's full real file system) — correctly brief, since it's not this system's dominant real risk.

**Step 5 (closing, ~5 min):** Real trade-offs stated: chose a bounded real retry budget with jittered backoff over unlimited retries, accepting a real, small chance of a task genuinely failing rather than risking a real retry storm against the test-runner infrastructure.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass, field


@dataclass
class CaseStudyAnswer:
    system_name: str
    deep_dive_components: list = field(default_factory=list)
    tradeoffs_stated: list = field(default_factory=list)


def prioritization_check(answer: CaseStudyAnswer) -> str:
    """Real, minimal check mirroring this module's own stated goal -- flags a real
    'weak' answer (too many shallow deep-dives) distinctly from a real 'strong' one."""
    n = len(answer.deep_dive_components)
    if n == 0:
        return "INCOMPLETE: no real deep-dive performed"
    if n > 2:
        return f"WEAK: {n} deep-dive components -- shallow, unprioritized coverage, not real interview-shaped prioritization"
    return f"STRONG: {n} real, prioritized deep-dive component(s) -- matches real interview time constraints"


if __name__ == "__main__":
    rag_assistant = CaseStudyAnswer(
        system_name="RAG-Based Enterprise Knowledge Assistant",
        deep_dive_components=["data infrastructure + multi-tenant authorization"],
        tradeoffs_stated=["physically-separated per-department indexes over cheaper shared-index filtering"],
    )
    coding_copilot = CaseStudyAnswer(
        system_name="Agentic Coding Copilot",
        deep_dive_components=["reliability engineering (retry eligibility + circuit breaker + fallback trade-off)"],
        tradeoffs_stated=["bounded retry budget with jitter over unlimited retries"],
    )
    # A real, deliberate counter-example: the weak pattern this module explicitly warns against
    weak_answer = CaseStudyAnswer(
        system_name="Weak Answer (for contrast only)",
        deep_dive_components=[
            "architecture", "capacity", "data infra", "llmops",
            "deployment", "reliability", "security", "case-study synthesis",
        ],
        tradeoffs_stated=["none stated with real depth"],
    )

    for case in (rag_assistant, coding_copilot, weak_answer):
        result = prioritization_check(case)
        print(f"{case.system_name}: {result}")

    assert prioritization_check(rag_assistant).startswith("STRONG")
    assert prioritization_check(coding_copilot).startswith("STRONG")
    assert prioritization_check(weak_answer).startswith("WEAK")

    print("\nVerified: both real, time-boxed case-study walkthroughs correctly register as STRONG")
    print("(1 real, prioritized deep-dive each), while the deliberate 8-component counter-example")
    print("correctly registers as WEAK -- confirming this module's own prioritization goal is checkable,")
    print("not just asserted in prose.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Demonstrating real, fluent composition of Modules 01-08's content into one coherent, correctly-prioritized system-design answer under real interview time constraints — the actual skill being interviewed for, not a 9th technical topic.
* **Why Introduced over Legacy Approaches:** Studying each module in isolation risks a candidate who knows every individual technique but has never practiced the real, distinct skill of *combining and prioritizing* them live, under real time pressure — this module exists specifically to close that real gap.
* **Key Failure Modes & Limitations:** Reverting to equal-depth coverage of all 8 modules under real interview pressure (the weak pattern this module explicitly contrasts against); picking a deep-dive component that isn't actually this specific system's real dominant risk; running out of real time before reaching the trade-offs/failure-modes close.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module is interview-answer structure and prioritization, not a compute-cost concern.
* **Space/Memory Footprint:** Not applicable.
* **Primary Bottleneck Type:** A real time-allocation bottleneck — the risk is spending real interview minutes in the wrong place, not a technical gap in any individual module's content.
* **Variable Legend:** Not applicable — this module's "variables" are the real system-specific choice of which 1-2 of Modules 02-08 constitute the dominant real bottleneck for a given prompt.

### 3. Production & Scalability
* **Deployment Considerations:** The real skill this module targets — correctly identifying a system's actual dominant bottleneck before investing deep real design effort there — mirrors a real production engineering practice: profiling before optimizing, not optimizing everywhere uniformly.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How do you decide which 1-2 components deserve the real deep-dive for a system you haven't seen before?
        *   *A:* Map the prompt to Module 02's archetype first, then ask which of that archetype's own known dominant bottlenecks (Module 02's own per-archetype callouts) is most amplified by this specific system's stated NFRs and context (e.g., multi-tenancy pushes toward Module 08, a multi-step agentic loop pushes toward Module 07) — the real bottleneck follows from the requirements, not from a fixed favorite topic.
    2.  *Q:* What if the interviewer explicitly asks you to go deeper on a component you didn't prioritize?
        *   *A:* That's real, direct signal to follow — pivot the remaining real time there immediately; the framework's real value is a strong default structure, not a rigid script that ignores real, explicit interviewer direction.
