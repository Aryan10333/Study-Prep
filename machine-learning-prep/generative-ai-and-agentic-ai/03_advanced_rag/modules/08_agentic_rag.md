# Module 08: Agentic RAG & Self-Correcting Retrieval Loops

## 1. Introduction & Intuition

### The Core Bottleneck
Every retrieval strategy in Modules 02-07 is a **fixed pipeline**: retrieve once (however cleverly), generate once, done — regardless of whether the retrieved context actually turned out to be good enough to answer the question. When retrieval genuinely misses (a bad first attempt, an ambiguous query, a corpus gap), a fixed pipeline has no way to notice and recover; it just generates the best answer it can from whatever was retrieved, wrong or not. Agentic RAG's core idea is letting the model itself decide, dynamically, whether to retrieve, whether what it retrieved is good enough, and whether to retrieve again — trading a fixed, predictable cost for the ability to recover from a bad first attempt.

### High-Level Intuition
A fixed RAG pipeline is a student who looks up one page in the textbook and writes their answer from whatever's on that page, no matter how relevant it turns out to be. Agentic RAG is a student who looks at the page, asks themselves "does this actually answer the question," and — if not — goes back to look up a different page before answering. This self-correction is genuinely powerful, but it's not free: that student takes longer and does more work per question, and an ill-behaved version of that student could loop back to the textbook indefinitely without ever being satisfied, which is exactly why explicit loop-termination guards matter in production.

**Scope note:** this module covers **retrieval-specific agent-loop mechanics only** — how an agent decides to retrieve, evaluates retrieval quality, and re-retrieves. General agent architecture, the Model Context Protocol (MCP) for tool-calling, and multi-agent orchestration patterns are owned by the dedicated AI Agents topic and are cross-referenced here, not re-taught, matching Module 09's own scope-discipline principle.

---

## 2. Core Concepts & Mathematical Formulation

This module is architectural/procedural throughout — Agentic RAG's tool-calling loop, Self-RAG/Corrective RAG's self-critique-and-re-retrieve pattern, and multi-agent retrieval don't reduce to a single core formula, consistent with how prior modules treated other pure-architecture techniques.

### Agentic RAG: Retrieval as a Tool-Calling Decision

#### Intuition & Practical Use
Instead of a hardcoded "always retrieve, then always generate" pipeline, Agentic RAG exposes retrieval as a **tool** the model can choose to call — the model can decide *whether* retrieval is needed at all (a simple greeting doesn't need a document lookup), *what* to search for (potentially reformulating the query itself, connecting back to Module 06), and *whether* to call retrieval again after seeing the first result set. This is the retrieval-specific application of general LLM tool-calling (the mechanics of which — function schemas, tool-call parsing, multi-turn tool loops — are covered in depth in the dedicated AI Agents topic).

### Self-RAG & Corrective RAG (CRAG): Self-Critique and Re-Retrieval

#### Intuition & Practical Use
Self-RAG and Corrective RAG both add an explicit **quality check** on retrieved content before generation commits to it, differing mainly in *what* triggers a retry:
*   **Self-RAG** trains (or prompts) the model to emit explicit "reflection" signals — is this retrieved passage relevant, is it supported, is the generated answer actually grounded in it — and can trigger re-retrieval or regeneration based on those self-assessed signals.
*   **Corrective RAG (CRAG)** adds a separate, lighter-weight relevance evaluator (not necessarily the generator model itself) that scores retrieved documents as correct/ambiguous/incorrect, and routes accordingly — correct documents proceed to generation as-is, ambiguous ones get supplemented (e.g., with a web search fallback), and incorrect ones trigger a full re-retrieval with a reformulated query.

Both patterns share the same core loop shape: **retrieve → generate (or evaluate) → self-critique → conditionally re-retrieve**, repeated until the quality check passes or a termination guard is hit.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 760 280" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="380" y="16" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Self-Correcting Retrieval Loop (Self-RAG / CRAG shape)</text>

  <!-- Retry loop, drawn first so it sits behind the boxes -->
  <path d="M495,75 Q495,40 200,40 Q60,40 60,73" fill="none" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4,2" marker-end="url(#arrow08b)"/>
  <defs>
    <marker id="arrow08b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#dc2626"/>
    </marker>
  </defs>
  <text x="270" y="36" text-anchor="middle" font-size="9" fill="#991b1b" font-weight="600">retry: not relevant / not grounded (reformulated query)</text>

  <rect x="40" y="75" width="130" height="50" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="105" y="104" text-anchor="middle" font-size="10.5" fill="#1e3a8a" font-weight="600">Retrieve</text>

  <rect x="230" y="75" width="130" height="50" rx="6" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="295" y="104" text-anchor="middle" font-size="10.5" fill="#065f46" font-weight="600">Generate</text>

  <rect x="420" y="75" width="150" height="50" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="495" y="98" text-anchor="middle" font-size="10.5" fill="#854d0e" font-weight="600">Self-Critique</text>
  <text x="495" y="113" text-anchor="middle" font-size="8.5" fill="#854d0e">(relevance / groundedness)</text>

  <rect x="630" y="75" width="110" height="50" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <text x="685" y="104" text-anchor="middle" font-size="10.5" fill="#14532d" font-weight="600">Answer</text>

  <g stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow08a)">
    <line x1="170" y1="100" x2="228" y2="100"/>
    <line x1="360" y1="100" x2="418" y2="100"/>
    <line x1="570" y1="100" x2="628" y2="100"/>
  </g>
  <defs>
    <marker id="arrow08a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <!-- Loop termination guard -->
  <rect x="420" y="180" width="230" height="55" rx="6" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="535" y="202" text-anchor="middle" font-size="10" fill="#991b1b" font-weight="600">Loop-Termination Guard</text>
  <text x="535" y="218" text-anchor="middle" font-size="8.5" fill="#991b1b">max_retries reached OR budget exceeded</text>
  <text x="535" y="230" text-anchor="middle" font-size="8.5" fill="#991b1b">-&gt; force-answer with best available context</text>
  <line x1="535" y1="125" x2="535" y2="178" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="3,2" marker-end="url(#arrow08b)"/>
</svg>
</div>

### Multi-Agent Retrieval

#### Intuition & Practical Use
For corpora spanning genuinely distinct domains (legal, engineering, customer support), a single generalist retrieval agent may perform worse than a set of **specialized retriever agents**, each tuned (different embedding model, different chunking strategy, different reranking criteria) to its own domain, with a coordinating step deciding which specialist(s) to invoke per query — conceptually similar to Module 06's semantic routing, but with each destination being a full retrieval agent rather than just a different index.

### When NOT to Use Agentic RAG

#### Intuition & Practical Use
The self-correction loop's entire value proposition is *recovering from a bad first retrieval attempt* — which means it earns its added latency/cost specifically on queries where the first attempt is plausibly wrong or incomplete and a second attempt can measurably fix it. For queries a single well-tuned retrieve-then-generate pass (Modules 02-05) already answers correctly and consistently, the agentic loop's extra self-critique and possible re-retrieval rounds add real latency and cost with no corresponding quality benefit — it should be reserved for query types with a demonstrated first-attempt failure rate, not applied unconditionally as a default.

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of the self-correcting retrieval loop, with an explicit loop-termination guard (the concrete fix for the "unbounded latency/cost" failure mode).

```python
from dataclasses import dataclass
from enum import Enum


class RelevanceVerdict(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class RetrievalAttempt:
    query: str
    retrieved_docs: list[str]
    verdict: RelevanceVerdict


def evaluate_relevance(retrieved_docs: list[str], query: str) -> RelevanceVerdict:
    """Placeholder for a real relevance evaluator (Self-RAG's reflection tokens or a
    CRAG-style lightweight scorer). Real implementations call an LLM/classifier here."""
    if not retrieved_docs:
        return RelevanceVerdict.INCORRECT
    if len(retrieved_docs) < 2:
        return RelevanceVerdict.AMBIGUOUS
    return RelevanceVerdict.CORRECT


def reformulate_query(original_query: str, attempt_number: int) -> str:
    """Placeholder for real query reformulation (Module 06 techniques applied on retry)."""
    return f"{original_query} (reformulated, attempt {attempt_number})"


def agentic_retrieve(
    query: str,
    retrieve_fn,
    max_retries: int = 3,
) -> tuple[list[str], list[RetrievalAttempt]]:
    """Self-correcting retrieval loop with an explicit termination guard --
    the concrete fix for Agentic RAG's unbounded-cost failure mode."""
    history: list[RetrievalAttempt] = []
    current_query = query

    for attempt_number in range(1, max_retries + 1):
        docs = retrieve_fn(current_query)
        verdict = evaluate_relevance(docs, current_query)
        history.append(RetrievalAttempt(current_query, docs, verdict))

        if verdict == RelevanceVerdict.CORRECT:
            return docs, history

        if attempt_number == max_retries:
            # Termination guard: force-answer with the best available context rather
            # than looping indefinitely -- an explicit, bounded exit, not a silent one.
            return docs, history

        current_query = reformulate_query(query, attempt_number + 1)

    return [], history  # unreachable, but keeps the type checker honest


if __name__ == "__main__":
    # A retriever that "improves" on each reformulated query, simulating a real re-retrieval fix
    call_log = []

    def fake_retrieve(q: str) -> list[str]:
        call_log.append(q)
        if "reformulated" in q:
            return ["relevant_doc_1", "relevant_doc_2"]  # the retry succeeds
        return []  # first attempt genuinely finds nothing

    docs, history = agentic_retrieve("What is our Q3 refund policy?", fake_retrieve, max_retries=3)
    print(f"Final docs: {docs}")
    print(f"Attempts made: {len(history)}")
    for i, attempt in enumerate(history, 1):
        print(f"  Attempt {i}: query={attempt.query!r} -> verdict={attempt.verdict.value}")

    assert history[0].verdict == RelevanceVerdict.INCORRECT  # first attempt found nothing
    assert history[-1].verdict == RelevanceVerdict.CORRECT   # retry succeeded
    assert len(docs) == 2

    # Verify the termination guard: a retriever that NEVER succeeds still stops at max_retries
    def always_fails(q: str) -> list[str]:
        return []

    _, exhausted_history = agentic_retrieve("Unanswerable query", always_fails, max_retries=3)
    assert len(exhausted_history) == 3, "Loop must stop at max_retries, not loop forever"
    print(f"\nTermination guard verified: stopped after {len(exhausted_history)} attempts, not indefinitely")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Fixed retrieve-once pipelines have no mechanism to recover from a bad first retrieval attempt; Agentic RAG lets the system detect that failure and try again with a reformulated query, at the cost of variable, potentially higher latency per request.
* **Why Introduced over Legacy Approaches:** A fixed pipeline's failure mode is silent — a bad retrieval simply produces a bad (or hallucinated) answer with no signal anything went wrong; self-critique makes retrieval quality an explicit, checkable step in the loop rather than an unvalidated assumption.
* **Key Failure Modes & Limitations:** Unbounded latency/cost without an explicit termination guard (the loop could retry indefinitely on a genuinely unanswerable query); the self-critique step itself can be wrong (a confidently-incorrect relevance judgment either stops a loop that should have continued, or continues one that should have stopped).

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Cost scales with the number of retrieval-generation-critique rounds actually taken, bounded by `max_retries` — worst case is `max_retries`× a single fixed-pipeline pass's cost, not unbounded, provided a termination guard is actually implemented.
* **Space/Memory Footprint:** Minimal beyond tracking attempt history for the current request (as in `RetrievalAttempt` above) — no persistent index-side storage cost, unlike Modules 02-04's ingestion-time decisions.
* **Primary Bottleneck Type:** Latency-bound, specifically on the *variable* number of LLM calls per request (retrieval query formulation, generation, self-critique, each potentially repeated) — the defining operational difference from a fixed pipeline's constant, predictable latency.
* **Variable Legend:** `max_retries` = the termination guard's bound; no additional closed-form formula variables, per this module's prose/procedural scope.

### 3. Production & Scalability
* **Deployment Considerations:** Always implement an explicit `max_retries`/budget guard (as in the reference code) — an agentic loop with no bound on retries is a real production incident waiting to happen on any genuinely unanswerable or malformed query; log attempt history for every request to make loop behavior debuggable (directly connects to Module 09's debugging methodology).
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How would you prevent an agentic RAG loop from running forever on a bad query?
        *   *A:* An explicit, hard-coded `max_retries` or total-latency-budget guard that forces a final answer (with the best available context, clearly caveated if quality is uncertain) once the bound is hit — never leave loop termination as an implicit assumption the model's own judgment is responsible for.
    2.  *Q:* How is Self-RAG different from Corrective RAG (CRAG)?
        *   *A:* Self-RAG has the generator model itself emit reflection signals about relevance/groundedness as part of its own generation process; CRAG uses a separate, typically lighter-weight relevance evaluator to score retrieved documents and route accordingly (proceed, supplement, or fully re-retrieve) — CRAG decouples the quality check from the generator, which can be cheaper and more consistent than relying on the generator's own self-assessment.
