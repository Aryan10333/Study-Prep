# Module 02: Reference Architectures for Common GenAI System Design Questions

## 1. Introduction & Intuition

### The Core Bottleneck
Most real GenAI system design questions are variations on a small, recurring set of underlying patterns — not genuinely novel architectures invented from scratch each time. Recognizing which real archetype (or archetype-plus-variant) a given prompt maps to is a distinct, real skill from knowing how any one component works; a candidate who deeply understands retrieval (`03_advanced_rag`) and agents (`04_ai_agents_and_protocols`) individually can still stumble in an interview if they can't quickly recognize which of those already-built components a specific prompt actually calls for, and in what combination.

### High-Level Intuition
A structural engineer doesn't invent a new bridge design for every river — they recognize whether the real span, load, and terrain call for a beam bridge, an arch, or a suspension design, each a known real pattern with known real trade-offs, then adapts the specific dimensions. This module's 4 archetypes play the same real role for GenAI systems: a small, memorizable set of known patterns, each composed from already-built components (prior topics), that covers the large majority of real system-design prompts once a candidate learns to recognize which one (or which combination) applies.

---

## 2. Core Concepts & Mathematical Formulation

### The Four Core Archetypes

#### Intuition & Practical Use
Deliberately kept to four, not more — a real, tight set an interview candidate can actually hold in working memory and apply under time pressure, rather than a long taxonomy that risks becoming shallow memorization:

1.  **RAG-based Q&A/search assistant** — a real user query triggers retrieval against a knowledge base, then generation grounded in the retrieved content. Dominant real bottleneck: retrieval quality and latency (`03_advanced_rag`'s own content, referenced not re-derived).
2.  **Agentic task-completion system** — a real user goal triggers a multi-step loop of real tool calls, intermediate reasoning, and generation, until the task is complete or a stopping condition is hit. Dominant real bottleneck: multi-step reliability and real per-step latency accumulation (`04_ai_agents_and_protocols`'s own content, referenced not re-derived).
3.  **Real-time interactive LLM service** — a real, low-latency conversational or completion service (chat, code-completion) where response time dominates the real user experience. Dominant real bottleneck: inference-serving latency and real concurrency (`06_llm_inference_and_optimization`'s own content, referenced not re-derived).
4.  **Batch/offline generation pipeline** — a real, large volume of generation work (bulk content generation, bulk summarization, bulk data labeling) run without a real per-request user waiting synchronously. Dominant real bottleneck: real throughput and cost efficiency, not per-request latency.

### The Four Archetypes, Consolidated

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 1000 640" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="2.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,5 L7,2.5 z" fill="#475569" />
    </marker>
  </defs>
  <text x="500" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#0f172a">Four Core GenAI System Archetypes</text>

  <rect x="20" y="55" width="460" height="255" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" />
  <text x="250" y="80" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">Archetype 1: RAG-Based Q&amp;A / Search Assistant</text>

  <rect x="45" y="100" width="110" height="46" rx="6" fill="#ffffff" stroke="#2563eb" />
  <text x="100" y="128" text-anchor="middle" font-size="12" fill="#1e3a8a">User Query</text>
  <line x1="155" y1="123" x2="185" y2="123" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="185" y="100" width="110" height="46" rx="6" fill="#ffffff" stroke="#2563eb" />
  <text x="240" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Retrieve</text>
  <text x="240" y="135" text-anchor="middle" font-size="10" fill="#64748b">(03_advanced_rag)</text>
  <line x1="295" y1="123" x2="325" y2="123" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="325" y="100" width="110" height="46" rx="6" fill="#ffffff" stroke="#2563eb" />
  <text x="380" y="128" text-anchor="middle" font-size="12" fill="#1e3a8a">Generate</text>

  <rect x="45" y="175" width="390" height="55" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-dasharray="3,2"/>
  <text x="240" y="197" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1e3a8a">Dominant Bottleneck:</text>
  <text x="240" y="215" text-anchor="middle" font-size="11" fill="#1e3a8a">Retrieval quality &amp; latency</text>

  <text x="45" y="255" font-size="10.5" fill="#475569">Single retrieval step, no</text>
  <text x="45" y="270" font-size="10.5" fill="#475569">multi-step planning loop.</text>

  <rect x="520" y="55" width="460" height="255" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5" />
  <text x="750" y="80" text-anchor="middle" font-size="13.5" font-weight="700" fill="#5b21b6">Archetype 2: Agentic Task-Completion System</text>

  <rect x="545" y="100" width="95" height="46" rx="6" fill="#ffffff" stroke="#7c3aed" />
  <text x="592" y="128" text-anchor="middle" font-size="12" fill="#5b21b6">User Goal</text>
  <line x1="640" y1="123" x2="665" y2="123" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="665" y="90" width="290" height="66" rx="6" fill="#ffffff" stroke="#7c3aed" stroke-dasharray="3,2" />
  <text x="810" y="106" text-anchor="middle" font-size="10.5" fill="#5b21b6">Loop: Plan -&gt; Tool Call -&gt; Observe</text>
  <text x="810" y="122" text-anchor="middle" font-size="10" fill="#64748b">(04_ai_agents_and_protocols)</text>
  <text x="810" y="140" text-anchor="middle" font-size="10" fill="#64748b">repeats until goal met / stop condition</text>

  <rect x="545" y="175" width="390" height="55" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-dasharray="3,2"/>
  <text x="740" y="197" text-anchor="middle" font-size="11.5" font-weight="600" fill="#5b21b6">Dominant Bottleneck:</text>
  <text x="740" y="215" text-anchor="middle" font-size="11" fill="#5b21b6">Multi-step reliability &amp; latency accumulation</text>

  <text x="545" y="255" font-size="10.5" fill="#475569">Real per-step latency stacks</text>
  <text x="545" y="270" font-size="10.5" fill="#475569">serially across the loop.</text>

  <rect x="20" y="335" width="460" height="255" rx="10" fill="#ecfdf5" stroke="#059669" stroke-width="1.5" />
  <text x="250" y="360" text-anchor="middle" font-size="13.5" font-weight="700" fill="#065f46">Archetype 3: Real-Time Interactive LLM Service</text>

  <rect x="45" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#059669" />
  <text x="100" y="408" text-anchor="middle" font-size="12" fill="#065f46">Request</text>
  <line x1="155" y1="403" x2="185" y2="403" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="185" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#059669" />
  <text x="240" y="400" text-anchor="middle" font-size="12" fill="#065f46">Serve</text>
  <text x="240" y="415" text-anchor="middle" font-size="10" fill="#64748b">(06_llm_inference)</text>
  <line x1="295" y1="403" x2="325" y2="403" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="325" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#059669" />
  <text x="380" y="408" text-anchor="middle" font-size="12" fill="#065f46">Response</text>

  <rect x="45" y="450" width="390" height="30" rx="5" fill="#ffffff" stroke="#059669" stroke-dasharray="2,2" />
  <text x="240" y="470" text-anchor="middle" font-size="10" fill="#065f46">Variant overlay: on-device fallback for simple requests</text>

  <rect x="45" y="495" width="390" height="55" rx="6" fill="#d1fae5" stroke="#059669" stroke-dasharray="3,2"/>
  <text x="240" y="517" text-anchor="middle" font-size="11.5" font-weight="600" fill="#065f46">Dominant Bottleneck:</text>
  <text x="240" y="535" text-anchor="middle" font-size="11" fill="#065f46">Serving latency &amp; concurrency</text>

  <rect x="520" y="335" width="460" height="255" rx="10" fill="#fffbeb" stroke="#d97706" stroke-width="1.5" />
  <text x="750" y="360" text-anchor="middle" font-size="13.5" font-weight="700" fill="#92400e">Archetype 4: Batch / Offline Generation Pipeline</text>

  <rect x="545" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#d97706" />
  <text x="600" y="408" text-anchor="middle" font-size="12" fill="#92400e">Job Queue</text>
  <line x1="655" y1="403" x2="685" y2="403" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="685" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#d97706" />
  <text x="740" y="408" text-anchor="middle" font-size="12" fill="#92400e">Batch Generate</text>
  <line x1="795" y1="403" x2="825" y2="403" stroke="#475569" stroke-width="1.5" marker-end="url(#a2)" />

  <rect x="825" y="380" width="110" height="46" rx="6" fill="#ffffff" stroke="#d97706" />
  <text x="880" y="408" text-anchor="middle" font-size="12" fill="#92400e">Store Results</text>

  <rect x="545" y="495" width="390" height="55" rx="6" fill="#fef3c7" stroke="#d97706" stroke-dasharray="3,2"/>
  <text x="740" y="517" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">Dominant Bottleneck:</text>
  <text x="740" y="535" text-anchor="middle" font-size="11" fill="#92400e">Throughput &amp; cost efficiency</text>

  <text x="545" y="465" font-size="10.5" fill="#475569">No real user waiting synchronously --</text>
  <text x="545" y="480" font-size="10.5" fill="#475569">per-request latency is not the concern.</text>
</svg>
</div>

*   **Diagram Interpretation:** All 4 archetypes shown with their real component chain and dominant bottleneck; each component chain names the specific already-built prior topic it composes (`03_advanced_rag`, `04_ai_agents_and_protocols`, `06_llm_inference_and_optimization`), and Archetype 3 shows the on-device/cloud-hybrid pattern as a real dashed variant overlay, not a separate box outside the 4-archetype grid.

### On-Device/Cloud-Hybrid as a Variant, Not a Fifth Archetype

#### Intuition & Practical Use
A real on-device/cloud-hybrid deployment (e.g., a small real on-device model handling simple queries, falling back to a real cloud model for complex ones) is a genuine, real pattern — but it's a *deployment-topology variant* applied on top of one of the 4 archetypes above (most often archetype 3, the real-time interactive service), not a structurally distinct 5th archetype with its own separate component set. Treating it as a variant, per the signed-off syllabus's own explicit framing, keeps the core archetype set tight while still covering this real, recurring pattern when a prompt calls for it.

### Recognizing Which Archetype a Prompt Maps To

#### Intuition & Practical Use
The real, practiced skill this module targets is archetype *selection*, not just archetype *description*. Some real prompts are unambiguous ("design a customer support chatbot that answers from our docs" → archetype 1). Others are deceptively worded — a prompt mentioning "an assistant that looks things up and takes actions" sounds agentic (archetype 2) on the surface, but if it's actually a single real retrieval-then-answer flow with no real multi-step tool use or intermediate planning, it's genuinely archetype 1, not archetype 2. Misclassifying the archetype early costs real interview time correcting course later.

---

### Worked Example (No Formula): Classifying Three Real, Deliberately Ambiguous Prompts

| Prompt (as given by an interviewer) | Surface-level read | Real, correct archetype | Why |
|---|---|---|---|
| "Design an assistant that looks up order status and answers shipping questions from our FAQ." | Sounds agentic ("looks up") | **Archetype 1 (RAG assistant)** | A real, single lookup-then-answer flow — no real multi-step planning or chained tool calls; "looks up order status" is one real retrieval/API call, not an agentic loop. |
| "Design a coding assistant that can read a codebase, run tests, and iterate on a fix." | Sounds like a simple real-time service (it's "a coding assistant") | **Archetype 2 (agentic system)** | The real requirement is a multi-step loop — read, run, evaluate, iterate — a genuine real agentic control flow, not a single-shot completion. |
| "Design a system that generates weekly summary reports for 10,000 real enterprise accounts overnight." | Sounds real-time ("generates reports") | **Archetype 4 (batch/offline pipeline)** | No real user is waiting synchronously — "overnight" and "10,000 accounts" are the real signal that throughput/cost, not per-request latency, is the dominant real concern. |

*   **Step 1: Real, direct classification.** Each prompt is read for its real, stated (or implied) synchronicity and control-flow requirement, not just its surface vocabulary.
*   **Step 2: Real interpretation.** All three prompts contain at least one surface-level word that could mislead a fast classification ("looks up," "assistant," "generates") — the real, correct classification depends on the actual control-flow and latency requirement underneath the wording, which is exactly the skill this module is built to practice, not a vocabulary-matching exercise.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass


@dataclass
class ArchetypeSignal:
    multi_step_tool_use: bool
    synchronous_user_waiting: bool
    real_time_latency_sensitive: bool


def classify_archetype(signal: ArchetypeSignal) -> str:
    """Real, minimal archetype classifier based on the module's own real, stated
    control-flow/synchronicity signals -- not surface vocabulary matching."""
    if not signal.synchronous_user_waiting:
        return "Archetype 4: Batch/Offline Generation Pipeline"
    if signal.multi_step_tool_use:
        return "Archetype 2: Agentic Task-Completion System"
    if signal.real_time_latency_sensitive:
        return "Archetype 3: Real-Time Interactive LLM Service"
    return "Archetype 1: RAG-Based Q&A/Search Assistant"


if __name__ == "__main__":
    prompts = {
        "Order status + FAQ assistant": ArchetypeSignal(
            multi_step_tool_use=False, synchronous_user_waiting=True, real_time_latency_sensitive=False
        ),
        "Coding assistant that reads/runs/iterates": ArchetypeSignal(
            multi_step_tool_use=True, synchronous_user_waiting=True, real_time_latency_sensitive=False
        ),
        "Overnight report generation for 10,000 accounts": ArchetypeSignal(
            multi_step_tool_use=False, synchronous_user_waiting=False, real_time_latency_sensitive=False
        ),
    }

    for prompt_name, signal in prompts.items():
        result = classify_archetype(signal)
        print(f"{prompt_name!r}")
        print(f"  -> {result}")

    assert classify_archetype(prompts["Order status + FAQ assistant"]) == "Archetype 1: RAG-Based Q&A/Search Assistant"
    assert classify_archetype(prompts["Coding assistant that reads/runs/iterates"]) == "Archetype 2: Agentic Task-Completion System"
    assert classify_archetype(prompts["Overnight report generation for 10,000 accounts"]) == "Archetype 4: Batch/Offline Generation Pipeline"
    print("\nVerified: all 3 deliberately ambiguous prompts classify correctly using real")
    print("control-flow/synchronicity signals, not surface vocabulary -- confirming the module's")
    print("own point that archetype selection requires reading past misleading surface wording.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Fast, correct recognition of which real, already-known architecture pattern a given system-design prompt calls for, so interview time goes to applying and deep-diving the right pattern rather than inventing one from scratch or misclassifying it.
* **Why Introduced over Legacy Approaches:** A "start from a blank page every time" approach wastes real interview time re-deriving a pattern that has a known, real, recurring shape — a small, memorized archetype set with a real classification skill is faster and more reliable under time pressure.
* **Key Failure Modes & Limitations:** Misclassifying a prompt based on surface vocabulary rather than its real control-flow/synchronicity requirements; treating the on-device/cloud-hybrid variant as a structurally separate 5th archetype rather than a topology overlay on one of the 4; forcing a real prompt into an archetype it doesn't cleanly fit instead of naming it as a genuine hybrid of two archetypes.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module is architectural composition, not a compute-cost concern (compute cost is owned by each archetype's own already-built serving/retrieval/agent components).
* **Space/Memory Footprint:** Not applicable at this module's level of abstraction.
* **Primary Bottleneck Type:** Varies by archetype and is the real point of the module: retrieval-latency-bound (archetype 1), multi-step-reliability-and-latency-bound (archetype 2), serving-latency/concurrency-bound (archetype 3), throughput/cost-bound (archetype 4) — a candidate should be able to state which bottleneck type applies before deep-diving.
* **Variable Legend:** FR/NFR from Module 01 determine which archetype's dominant bottleneck actually matters for a given real prompt.

### 3. Production & Scalability
* **Deployment Considerations:** Real production systems frequently combine archetypes (e.g., a real-time interactive service, archetype 3, whose backend is itself a RAG assistant, archetype 1) — recognizing a composite/hybrid case and naming both contributing archetypes explicitly is a stronger real answer than forcing a single label.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* A prompt doesn't cleanly fit any single archetype — what do you do?
        *   *A:* Name it explicitly as a real hybrid (e.g., "this is fundamentally a real-time service, archetype 3, with a RAG-assistant, archetype 1, backend") rather than forcing an artificial single classification — the real skill is recognizing composition, not rigid categorization.
    2.  *Q:* Why does it matter which archetype you pick before Module 01's Step 3 (capacity estimation)?
        *   *A:* Each archetype has a genuinely different real bottleneck and therefore a genuinely different capacity-estimation approach (Module 03) — e.g., archetype 4's batch pipeline is sized on real total throughput/turnaround time, while archetype 3's real-time service is sized on real concurrent in-flight requests (Little's Law) — picking the wrong archetype produces the wrong capacity math downstream.
