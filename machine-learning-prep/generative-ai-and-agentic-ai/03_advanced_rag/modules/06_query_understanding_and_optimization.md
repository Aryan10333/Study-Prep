# Module 06: Query Understanding, Transformation & Optimization

## 1. Introduction & Intuition

### The Core Bottleneck
Real user queries are often short, ambiguous, or phrased nothing like the source documents that actually contain the answer — "what's our refund policy" won't lexically or even semantically match a document titled "Section 4.2: Return and Reimbursement Procedures" as closely as it should. Retrieval quality is capped not just by how good the chunking/embedding/indexing pipeline is (Modules 02-04), but by how well the *query itself* is shaped before it's ever sent to the retriever. Query transformation techniques exist specifically to close that gap — but every one of them adds an extra LLM call to the critical path, so the real skill is knowing when that added latency/cost is actually buying a meaningful retrieval improvement.

### High-Level Intuition
A vague or oddly-phrased question to a librarian gets a vague or oddly-matched answer; a librarian who first asks a clarifying question, rephrases your question in more searchable terms, or splits a compound question into its parts before searching will generally do better — at the cost of that extra back-and-forth taking more time. Every technique in this module is a different way of doing that "let me rephrase/split/route your question before I search" step automatically.

---

## 2. Core Concepts & Mathematical Formulation

This module is architectural/procedural throughout — none of these techniques reduce to a single core formula worth a dedicated hand calculation (consistent with how `02_llm_training_foundations` treated its own technique-survey modules). Each is presented with what it does, when it helps, and — just as importantly — **when it's not worth the added latency/cost**.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 780 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="390" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Query Transformation: Original Query Branching Into Techniques</text>

  <rect x="320" y="45" width="140" height="45" rx="6" fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>
  <text x="390" y="72" text-anchor="middle" font-size="11" fill="#0f172a" font-weight="600">Original Query</text>

  <g font-size="10">
    <rect x="30" y="140" width="140" height="45" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4"/>
    <text x="100" y="166" text-anchor="middle" fill="#1e3a8a" font-weight="600">Rewrite</text>

    <rect x="210" y="140" width="140" height="45" rx="5" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.4"/>
    <text x="280" y="166" text-anchor="middle" fill="#5b21b6" font-weight="600">HyDE</text>

    <rect x="390" y="140" width="140" height="45" rx="5" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.4"/>
    <text x="460" y="166" text-anchor="middle" fill="#854d0e" font-weight="600">Decompose</text>

    <rect x="570" y="140" width="180" height="45" rx="5" fill="#fed7aa" stroke="#c2410c" stroke-width="1.4"/>
    <text x="660" y="160" text-anchor="middle" fill="#7c2d12" font-weight="600">Multi-Query</text>
    <text x="660" y="174" text-anchor="middle" font-size="8.5" fill="#7c2d12">+ semantic routing</text>
  </g>

  <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#arrow06a)">
    <path d="M340,90 Q200,110 100,138"/>
    <path d="M370,90 Q320,110 280,138"/>
    <path d="M410,90 Q440,110 460,138"/>
    <path d="M440,90 Q550,110 660,138"/>
  </g>
  <defs>
    <marker id="arrow06a" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <text x="390" y="220" text-anchor="middle" font-size="9" fill="#334155" font-weight="600">Each branch adds a full LLM call before retrieval even starts.</text>
  <text x="390" y="238" text-anchor="middle" font-size="9" fill="#64748b">A well-formed, unambiguous, single-hop query skips all four branches entirely.</text>
</svg>
</div>

### Technique Comparison

| Technique | What it does | When it helps | When NOT to use it |
|---|---|---|---|
| **Query expansion / rewriting** | An LLM rewrites the user's query into one or more alternative phrasings closer to how the corpus's language is written | Queries phrased very differently from source document vocabulary (jargon mismatch, casual phrasing vs. formal documents) | Queries that already retrieve well unmodified — the extra LLM call adds a full round-trip of latency for no measurable retrieval gain |
| **HyDE (Hypothetical Document Embeddings)** | An LLM generates a *hypothetical* answer to the query, and that hypothetical answer's embedding — not the query's own embedding — is used to search the index (a plausible answer's phrasing tends to resemble real answer documents more than the terse question does) | Sparse/short queries where the gap between "how a question is phrased" and "how an answer is phrased" is large | Well-formed, already-answer-like queries; also risky when the hypothetical answer is confidently wrong about a fact — its embedding can then point retrieval in a plausible-sounding but incorrect direction |
| **Query decomposition** | Splits a compound/multi-hop question ("compare X's Q1 and Q2 revenue") into independent sub-questions, retrieves for each separately, and combines results | Genuinely multi-hop or multi-part questions a single retrieval pass can't satisfy at once | Single-hop questions — decomposing an already-simple question just adds LLM-call overhead and multiple retrieval round-trips for no benefit |
| **Multi-query retrieval** | Generates several *paraphrased* variants of the same query and retrieves for each, then merges/deduplicates results (conceptually similar to RRF fusion, Module 05, but fusing across query variants rather than across retrieval methods) | Queries with genuine ambiguity in phrasing where different paraphrases surface different relevant documents | Queries with one clear, unambiguous phrasing — generating variants of an already-precise query mostly just adds noise and retrieval cost |
| **Semantic routing** | Classifies (via embedding similarity or a lightweight classifier) which of several retrieval sources/indexes a query should be directed to, before searching any of them | Systems with multiple distinct corpora/indexes (e.g., separate legal, HR, and engineering document stores) where searching all of them per query is wasteful | A system with only one corpus/index — routing has nothing to route between |

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of a query-transformation dispatcher, illustrating the decision logic for *when* each technique fires (matching the "when NOT to use it" column above) rather than the LLM-call mechanics themselves, which are provider-specific.

```python
from dataclasses import dataclass
from enum import Enum


class QueryTransform(Enum):
    NONE = "none"                    # query is already well-formed, retrieve directly
    REWRITE = "rewrite"              # vocabulary mismatch suspected
    HYDE = "hyde"                    # very short/terse query, answer-shaped embedding likely to help
    DECOMPOSE = "decompose"          # compound/multi-hop query detected
    MULTI_QUERY = "multi_query"      # ambiguous phrasing, multiple interpretations plausible


@dataclass
class QueryAnalysis:
    token_count: int
    has_compound_markers: bool  # e.g., "and", "compare", "vs" detected
    is_ambiguous: bool          # e.g., multiple plausible interpretations flagged upstream


def choose_transform(analysis: QueryAnalysis, min_tokens_for_hyde: int = 6) -> QueryTransform:
    """Decision logic for which (if any) query transform to apply -- each one skipped
    by default unless there's a concrete signal it's worth its added latency/cost."""
    if analysis.has_compound_markers:
        return QueryTransform.DECOMPOSE
    if analysis.token_count < min_tokens_for_hyde:
        return QueryTransform.HYDE
    if analysis.is_ambiguous:
        return QueryTransform.MULTI_QUERY
    return QueryTransform.NONE  # well-formed, single-hop, unambiguous -- retrieve directly, no extra LLM call


def route_to_index(query_embedding, index_centroids: dict[str, list[float]]) -> str:
    """Semantic routing: pick the index whose centroid is closest to the query embedding.
    Only meaningful when more than one index exists."""
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)

    return max(index_centroids, key=lambda name: cosine(query_embedding, index_centroids[name]))


if __name__ == "__main__":
    # Well-formed, single-hop query -> no transform needed, save the extra LLM call
    well_formed = QueryAnalysis(token_count=9, has_compound_markers=False, is_ambiguous=False)
    assert choose_transform(well_formed) == QueryTransform.NONE

    # Terse query -> HyDE
    terse = QueryAnalysis(token_count=3, has_compound_markers=False, is_ambiguous=False)
    assert choose_transform(terse) == QueryTransform.HYDE

    # Compound query -> decomposition, regardless of length
    compound = QueryAnalysis(token_count=12, has_compound_markers=True, is_ambiguous=False)
    assert choose_transform(compound) == QueryTransform.DECOMPOSE

    # Ambiguous but well-formed -> multi-query
    ambiguous = QueryAnalysis(token_count=10, has_compound_markers=False, is_ambiguous=True)
    assert choose_transform(ambiguous) == QueryTransform.MULTI_QUERY

    print("All query-transform routing decisions verified.")

    # Semantic routing over two toy indexes
    centroids = {
        "legal_docs": [0.9, 0.1, 0.0],
        "engineering_docs": [0.0, 0.2, 0.9],
    }
    query_vec = [0.85, 0.15, 0.05]  # closer to legal_docs
    chosen = route_to_index(query_vec, centroids)
    print(f"Routed to: {chosen}")
    assert chosen == "legal_docs"
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Bridging the gap between how users actually phrase questions and how relevant content is actually written/indexed, so retrieval isn't limited by surface-level phrasing mismatch.
* **Why Introduced over Legacy Approaches:** Sending the raw user query straight to retrieval (no transformation) is the cheapest option but leaves retrieval quality capped by however well (or poorly) the user happened to phrase their question — query transformation techniques trade added latency/cost for closing that gap on the queries that actually need it.
* **Key Failure Modes & Limitations:** Query rewriting/HyDE can drift from genuine user intent (rewriting introduces the LLM's own assumptions about what the user meant); HyDE specifically risks anchoring retrieval on a confidently-wrong hypothetical answer; decomposition can over-split a question that was actually answerable in one retrieval pass, adding unnecessary latency.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Each transformation technique adds one (or, for multi-query/decomposition, several) full LLM generation calls to the critical path *before* retrieval even starts — a direct, often-dominant addition to end-to-end query latency compared to untransformed retrieval.
* **Space/Memory Footprint:** Negligible persistent footprint — these are query-time-only transformations with no index-side storage cost (unlike Modules 02-04's ingestion-time decisions).
* **Primary Bottleneck Type:** Latency-bound on the transformation LLM call itself, which is typically a larger, slower model call than the retrieval step it precedes — meaning an unnecessary query transformation can easily become the single largest contributor to end-to-end query latency.
* **Variable Legend:** No dedicated formula variables for this module — decisions are threshold/heuristic-driven (query length, compound-question detection, ambiguity signals) rather than closed-form quantities.

### 3. Production & Scalability
* **Deployment Considerations:** Gate every transformation technique behind a cheap, fast upstream classifier (as in the reference code's `choose_transform`) rather than applying it unconditionally to every query — the majority of production queries are often well-formed enough to skip transformation entirely, and paying the extra LLM-call latency on every single query when only a minority need it is a common, avoidable cost.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How would you evaluate whether a query transformation technique is actually helping in production?
        *   *A:* A/B test it directly against Module 09's retrieval metrics (Recall@k, MRR, NDCG) on a held-out query set representative of real production traffic, comparing transformed vs. untransformed retrieval — and weigh any measured quality gain against the added latency, since a small quality improvement may not justify a large latency cost.
    2.  *Q:* What's the risk of always applying HyDE unconditionally?
        *   *A:* Beyond the added latency of every query needing an extra generation call, HyDE's hypothetical answer can be confidently wrong for factual/specific queries, and searching against a wrong hypothetical answer's embedding can retrieve worse results than just searching the original query directly — it's not a strictly-better default, it's a targeted fix for a specific failure mode (terse, answer-shape-mismatched queries).
