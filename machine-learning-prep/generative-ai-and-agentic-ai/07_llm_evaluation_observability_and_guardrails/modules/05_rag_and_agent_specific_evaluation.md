# Module 05: RAG & Agent-Specific Evaluation

## 1. Introduction & Intuition

### The Core Bottleneck
Modules 02-04 covered evaluation for a single-turn, single-output generation. RAG and agentic systems break that assumption on two real fronts: RAG's output depends on real retrieved context that itself needs evaluating (did the answer actually follow from what was retrieved?), and agentic systems execute real multi-step trajectories where a single final "success/fail" label hides how that success was actually reached. Both real system types need evaluation dimensions single-turn metrics don't capture.

### High-Level Intuition
Grading a research paper's *conclusion* alone misses whether the paper's cited sources actually support that conclusion — that's RAG's real faithfulness problem. Grading whether a project *ultimately succeeded* alone misses whether it took three efficient, well-chosen steps or fifteen wasteful, backtracking ones to get there — that's the real agent-efficiency problem this module treats as inseparable from task success.

---

## 2. Core Concepts & Mathematical Formulation

### RAG Faithfulness/Groundedness

#### Intuition & Practical Use
Faithfulness asks a real, specific question distinct from "is the answer correct": does the answer's content actually follow from the real retrieved context, or does it include real, unsupported claims the retrieved content never stated? A real answer can be factually true in the world yet unfaithful to its own retrieved context (fabricating support it didn't actually have) — a genuinely different failure mode from Module 06's broader hallucination-detection scope, scoped here specifically to the RAG answer-vs-context relationship.

$$\text{Faithfulness} = \frac{\text{Supported claims}}{\text{Total claims}}$$

### Context Precision and Recall, Defined Explicitly

#### Intuition & Practical Use
The RAG-evaluation ecosystem uses multiple, genuinely different real definitions for "context precision/recall" in practice — this module fixes one concrete, stated convention rather than treating the terms as self-evident. **Context precision**'s real denominator is the *retrieved* set — of what was retrieved, how much was actually relevant? **Context recall**'s real denominator is the *total relevant* set in the corpus for that query — of everything that was actually relevant, how much did retrieval actually surface? "Relevant" itself means a chunk judged (by a stated real rubric — human or LLM-judge label) as containing information necessary to support the real correct answer — one specific, chosen convention, not the only possible one in the wider ecosystem.

$$\text{Context precision} = \frac{\text{Retrieved chunks judged relevant}}{\text{Total retrieved chunks}} \qquad \text{Context recall} = \frac{\text{Retrieved chunks judged relevant}}{\text{Total relevant chunks in the corpus}}$$

### Agent Trajectory Correctness and Real Efficiency, Evaluated Together

#### Intuition & Practical Use
An agentic system's real task-completion rate is necessary but insufficient on its own — two agent runs can both genuinely succeed at the identical task while consuming wildly different real resources to get there. Real agent efficiency metrics — tool-call count, token usage, wall-clock latency, and real cost per completed task — are tracked as a genuinely separate axis *alongside* trajectory/task-success correctness, not folded into a single blended score, since a production system cares about both independently.

---

### Hand Calculation, Three Real Worked Examples

*   **Example 1: Faithfulness.** A real answer makes 5 distinct claims; checking each against the retrieved context finds 4 genuinely supported and 1 unsupported (fabricated).
    $$\text{Faithfulness} = \frac{4}{5} = 0.80$$

*   **Example 2: Context precision and recall**, under this module's own stated definitions. A real query has 6 total relevant chunks in the corpus; the retriever returns 5 chunks, of which 3 are judged relevant.
    $$\text{Context precision} = \frac{3}{5} = 0.60 \qquad \text{Context recall} = \frac{3}{6} \approx 0.50$$
    Precision and recall diverge here precisely because their real denominators differ (retrieved-set size vs. total-relevant-set size) — a direct, computed illustration of why the two need separately stated definitions rather than being used interchangeably.

*   **Example 3: Agent efficiency delta between two equally successful runs.** Two real agent runs both complete the identical task successfully:

    | Metric | Run A | Run B | Delta |
    |---|---|---|---|
    | Tool calls | 3 | 7 | +4 (133.3% more) |
    | Tokens | 1,200 | 3,100 | +1,900 (158.3% more) |
    | Latency | 4.2s | 9.8s | +5.6s (133.3% more) |
    | Cost | $0.008 | $0.021 | +$0.013 (162.5% more) |

    Both runs would score identically (1.0) on task-success rate alone — the real efficiency deltas above, computed directly, are exactly what a success-rate-only evaluation would miss entirely.

<div style="text-align:center">

<svg viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;height:auto;font-family:sans-serif">
  <style>
    .lbl{font-size:11px;fill:#333}
    .hdr{font-size:14px;font-weight:bold;fill:#222}
    .step{fill:#eaf1fb;stroke:#4c78a8;stroke-width:1.5}
    .correct{fill:#d9f2e3;stroke:#2e8b57;stroke-width:1}
    .eff{fill:#fdeedd;stroke:#c96;stroke-width:1}
  </style>
  <text x="450" y="22" text-anchor="middle" class="hdr">Agent Trajectory: Correctness AND Efficiency, Tracked Together</text>

  <g>
    <rect x="30" y="50" width="150" height="40" class="step" rx="6"/>
    <text x="105" y="74" text-anchor="middle" class="lbl">Step 1: search_docs()</text>
    <rect x="30" y="95" width="70" height="20" class="correct" rx="3"/>
    <text x="65" y="109" text-anchor="middle" font-size="10">correct ✓</text>
    <rect x="110" y="95" width="70" height="20" class="eff" rx="3"/>
    <text x="145" y="109" text-anchor="middle" font-size="10">210 tok</text>

    <path d="M180 70 L235 70" stroke="#555" stroke-width="2" marker-end="url(#arrA)"/>

    <rect x="240" y="50" width="150" height="40" class="step" rx="6"/>
    <text x="315" y="74" text-anchor="middle" class="lbl">Step 2: fetch_page()</text>
    <rect x="240" y="95" width="70" height="20" class="correct" rx="3"/>
    <text x="275" y="109" text-anchor="middle" font-size="10">correct ✓</text>
    <rect x="320" y="95" width="70" height="20" class="eff" rx="3"/>
    <text x="355" y="109" text-anchor="middle" font-size="10">340 tok</text>

    <path d="M390 70 L445 70" stroke="#555" stroke-width="2" marker-end="url(#arrA)"/>

    <rect x="450" y="50" width="150" height="40" class="step" rx="6"/>
    <text x="525" y="74" text-anchor="middle" class="lbl">Step 3: fetch_page()</text>
    <rect x="450" y="95" width="70" height="20" fill="#f6c6c6" stroke="#b23" rx="3"/>
    <text x="485" y="109" text-anchor="middle" font-size="10">redundant ✗</text>
    <rect x="530" y="95" width="70" height="20" class="eff" rx="3"/>
    <text x="565" y="109" text-anchor="middle" font-size="10">340 tok</text>

    <path d="M600 70 L655 70" stroke="#555" stroke-width="2" marker-end="url(#arrA)"/>

    <rect x="660" y="50" width="150" height="40" class="step" rx="6"/>
    <text x="735" y="74" text-anchor="middle" class="lbl">Step 4: final_answer()</text>
    <rect x="660" y="95" width="70" height="20" class="correct" rx="3"/>
    <text x="695" y="109" text-anchor="middle" font-size="10">correct ✓</text>
    <rect x="740" y="95" width="70" height="20" class="eff" rx="3"/>
    <text x="775" y="109" text-anchor="middle" font-size="10">180 tok</text>
  </g>

  <text x="450" y="150" text-anchor="middle" class="lbl">Task-success rate: 1.0 (final answer correct) — hides Step 3's real, unnecessary redundant call</text>
  <text x="450" y="170" text-anchor="middle" class="lbl">Real total: 4 tool calls, 1,070 tokens — 1 call (25%) contributed nothing to the correct real outcome</text>
  <text x="450" y="200" text-anchor="middle" class="hdr" fill="#b05a3a">Correctness alone (1.0) would miss this real, quantifiable efficiency waste</text>

  <defs>
    <marker id="arrA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#555"/>
    </marker>
  </defs>
</svg>

</div>

*   **Diagram Interpretation:** A real, constructed 4-step agent trajectory where the final task succeeds (green correctness markers on steps 1, 2, 4) but step 3 is a real, redundant repeated call contributing nothing — visible only by tracking per-step correctness *and* efficiency (orange token counts) together, exactly the kind of real waste a single trajectory-success label alone would hide.

---

## 3. Implementation & Reference Code

```python
def faithfulness_score(supported_claims: int, total_claims: int) -> float:
    return supported_claims / total_claims


def context_precision_recall(retrieved_relevant: int, total_retrieved: int, total_relevant_in_corpus: int) -> dict:
    precision = retrieved_relevant / total_retrieved
    recall = retrieved_relevant / total_relevant_in_corpus
    return {"precision": precision, "recall": recall}


def agent_efficiency_delta(run_a: dict, run_b: dict) -> dict:
    return {
        key: {"delta": run_b[key] - run_a[key], "pct_more": (run_b[key] - run_a[key]) / run_a[key] * 100}
        for key in run_a
    }


if __name__ == "__main__":
    # Example 1: faithfulness
    faithfulness = faithfulness_score(supported_claims=4, total_claims=5)
    print(f"Faithfulness: {faithfulness}")
    assert faithfulness == 0.8

    # Example 2: context precision/recall
    ctx = context_precision_recall(retrieved_relevant=3, total_retrieved=5, total_relevant_in_corpus=6)
    print(f"Context precision: {ctx['precision']}, Context recall: {ctx['recall']:.4f}")
    assert ctx["precision"] == 0.6
    assert abs(ctx["recall"] - 0.5) < 1e-9

    # Example 3: agent efficiency delta, both runs equally successful
    run_a = {"tool_calls": 3, "tokens": 1200, "latency_s": 4.2, "cost": 0.008}
    run_b = {"tool_calls": 7, "tokens": 3100, "latency_s": 9.8, "cost": 0.021}
    deltas = agent_efficiency_delta(run_a, run_b)
    for key, d in deltas.items():
        print(f"{key}: delta=+{d['delta']:.3f} ({d['pct_more']:.1f}% more)")

    assert abs(deltas["tool_calls"]["pct_more"] - 133.3) < 0.1
    assert abs(deltas["cost"]["pct_more"] - 162.5) < 0.1

    print("\nVerified: both runs reach the identical successful task outcome (task-success rate = 1.0 for each),")
    print("yet Run B consumes substantially more real tool calls/tokens/latency/cost -- exactly the real")
    print("efficiency signal a success-rate-only evaluation would completely miss.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Evaluating multi-step, retrieval-grounded, or agentic real system output along dimensions single-turn metrics (Modules 02-04) don't cover — real answer-vs-context faithfulness, real retrieval quality, and real trajectory efficiency.
* **Why Introduced over Legacy Approaches:** A single-turn correctness label is real and necessary but structurally blind to *how* a multi-step real system reached its output — this module's dimensions make that "how" measurable.
* **Key Failure Modes & Limitations:** Using "context precision/recall" without stating which real convention is meant (this module's own explicit-definition discipline exists precisely because the ecosystem doesn't have one universal definition); evaluating agents purely on task-success rate and missing real, substantial efficiency waste (this module's own trajectory diagram).

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable in the traditional sense — the real cost here is evaluation *labeling* effort (claim-checking for faithfulness, relevance-judging for context precision/recall), not model FLOPs.
* **Space/Memory Footprint:** Not a primary concern for this module's evaluation techniques themselves.
* **Primary Bottleneck Type:** A real evaluation-completeness bottleneck — the risk of declaring a multi-step system "good" based on an incomplete set of real evaluation dimensions.
* **Variable Legend:** Faithfulness's claims = discrete factual assertions extracted from a real answer; "relevant" (context precision/recall) = a chunk judged, under a stated rubric, as necessary to support the real correct answer.

### 3. Production & Scalability
* **Deployment Considerations:** Real production RAG/agent evaluation dashboards should track faithfulness, context precision/recall, task-success rate, and the real efficiency metrics (Example 3) as genuinely separate real signals — collapsing them into one blended score hides exactly which dimension is failing when a real regression occurs.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Can a RAG answer be faithful to its retrieved context but still factually wrong?
        *   *A:* Yes — faithfulness only checks whether claims follow from what was retrieved, not whether the retrieved content itself was correct; a real answer can be perfectly faithful to genuinely wrong or outdated retrieved context.
    2.  *Q:* Why track agent efficiency separately from task-success rate rather than combining them into one score?
        *   *A:* They answer genuinely different real questions a production team needs separately — did the agent succeed, and at what real resource cost — collapsing them into one blended number would hide which one degraded when a real regression happens, exactly as this module's own worked example demonstrates.
