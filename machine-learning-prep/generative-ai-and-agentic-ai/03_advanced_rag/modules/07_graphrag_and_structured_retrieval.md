# Module 07: GraphRAG & Structured/Knowledge-Graph Retrieval

## 1. Introduction & Intuition

### The Core Bottleneck
Flat vector retrieval (Modules 03-05) treats every chunk as an independent, isolated unit — it has no notion that "the CEO mentioned in document A" and "the same person mentioned in document C's org chart" are the same entity, or that answering a "global" question like "what are the main themes across this entire corpus" requires synthesizing across *many* documents rather than retrieving a handful of individually-similar chunks. GraphRAG addresses exactly this class of question by building an explicit knowledge graph of entities and relationships, enabling retrieval that reasons about connections between documents, not just similarity within them.

### High-Level Intuition
Flat vector retrieval is like searching a pile of index cards for the ones that look most similar to your question. GraphRAG is like first building a map of how everything in the pile *relates* to everything else — this person works at this company, which was acquired by that company, whose CEO said this — and then answering questions by walking that map. This is powerful specifically for questions a similarity search can't answer by finding "similar" text alone (multi-hop reasoning, corpus-wide summarization), but building and maintaining that map is real, ongoing work — which is exactly why it's a targeted technique for a specific class of question, not a default replacement for flat retrieval.

---

## 2. Core Concepts & Mathematical Formulation

This module stays architectural/conceptual throughout — per this repo's "Strictly Prohibit Academic Formalisms" rule, community-detection algorithm internals (e.g., the modularity-optimization math behind Louvain-style clustering) are explicitly out of scope. What matters for interview readiness is the *architecture* and *when to reach for it*, not re-deriving graph-clustering objective functions.

### Knowledge Graph Construction

#### Intuition & Practical Use
Building a knowledge graph from unstructured text starts with **entity extraction** (identifying people, organizations, products, concepts as graph nodes) and **relation extraction** (identifying and labeling the edges between them — "acquired," "works at," "reports to"), typically performed by an LLM prompted to extract structured (entity, relation, entity) triples from each document/chunk. The resulting graph is then enriched with **community detection** — clustering densely-connected groups of entities/relations into higher-level "communities," each of which can be pre-summarized by an LLM into a short natural-language description (this is the core mechanism behind Microsoft's GraphRAG approach specifically).

### Graph-Based Retrieval: Local vs. Global Queries

#### Intuition & Practical Use
GraphRAG retrieval splits into two distinct query patterns:
*   **Local (entity-level) queries** — "what products does Company X sell" — traverse the graph starting from a specific matched entity, pulling in its directly connected neighbors and relations. This resembles flat retrieval's precision but with explicit relational structure instead of just semantic similarity.
*   **Global (corpus-level) queries** — "what are the major themes across this document collection" — retrieve pre-computed community summaries instead of walking individual entities, since no single chunk (or even entity neighborhood) contains a "global" answer; it has to be synthesized from many parts of the graph at once, which is exactly what the community summaries were built in advance to answer cheaply at query time.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 820 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="410" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Knowledge Graph: Local (Entity) vs. Global (Community) Retrieval</text>

  <!-- Graph nodes -->
  <g font-size="9">
    <circle cx="150" cy="90" r="16" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="150" y="93" text-anchor="middle" fill="#1e3a8a">CEO</text>
    <circle cx="260" cy="60" r="16" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="260" y="63" text-anchor="middle" fill="#1e3a8a">Co. X</text>
    <circle cx="260" cy="130" r="16" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="260" y="133" text-anchor="middle" fill="#1e3a8a">Prod A</text>
    <circle cx="370" cy="90" r="16" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="370" y="93" text-anchor="middle" fill="#1e3a8a">Co. Y</text>

    <circle cx="520" cy="60" r="15" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3"/>
    <text x="520" y="63" text-anchor="middle" fill="#1e3a8a" font-size="8">Prod B</text>
    <circle cx="600" cy="90" r="15" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3"/>
    <text x="600" y="93" text-anchor="middle" fill="#1e3a8a" font-size="8">Co. Z</text>
    <circle cx="560" cy="150" r="15" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.3"/>
    <text x="560" y="153" text-anchor="middle" fill="#1e3a8a" font-size="8">CFO</text>
  </g>
  <g stroke="#93c5fd" stroke-width="1.3">
    <line x1="165" y1="85" x2="248" y2="65"/>
    <line x1="165" y1="97" x2="248" y2="122"/>
    <line x1="276" y1="65" x2="356" y2="85"/>
    <line x1="533" y1="68" x2="588" y2="85"/>
    <line x1="588" y1="98" x2="572" y2="140"/>
    <line x1="536" y1="70" x2="548" y2="140"/>
  </g>

  <!-- Community boundary for global query -->
  <ellipse cx="560" cy="100" rx="90" ry="75" fill="none" stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="560" y="195" text-anchor="middle" font-size="9" fill="#5b21b6" font-weight="600">Community 2 (pre-summarized)</text>

  <!-- Local query annotation -->
  <text x="150" y="55" text-anchor="middle" font-size="9" fill="#991b1b" font-weight="600">Local query starts here</text>
  <line x1="150" y1="60" x2="150" y2="72" stroke="#dc2626" stroke-width="1.3" marker-end="url(#arrow07a)"/>
  <defs>
    <marker id="arrow07a" markerWidth="7" markerHeight="7" refX="3.5" refY="5" orient="auto">
      <path d="M0,0 L7,0 L3.5,7 Z" fill="#dc2626"/>
    </marker>
  </defs>

  <text x="410" y="230" text-anchor="middle" font-size="9.5" fill="#64748b">Local: traverse from a matched entity. Global: retrieve a pre-computed community summary, no traversal needed.</text>
</svg>
</div>

---

### When NOT to Use GraphRAG

#### Intuition & Practical Use
Graph construction (entity/relation extraction across the entire corpus, community detection, community summarization) is a real, non-trivial upfront and ongoing cost — every new/updated document potentially requires re-extracting entities and relations, and community structure can shift as the graph grows, requiring re-summarization. This makes GraphRAG a poor fit for:
*   **Small corpora**, where flat retrieval already performs well and graph construction overhead isn't justified by the corpus size.
*   **High-update-frequency content**, where the graph would need near-constant re-extraction/re-summarization to stay current, and the resulting staleness risk may exceed what a simpler flat index would have.
*   **Single-hop-lookup-dominated workloads**, where most real queries are answerable directly from one relevant chunk — Module 05's flat hybrid retrieval already handles this well at a fraction of GraphRAG's setup and maintenance cost, and paying the graph-construction cost buys nothing for query types that never needed relational reasoning in the first place.

GraphRAG earns its cost specifically on **multi-hop** questions (reasoning across multiple connected entities) and **global/corpus-summary** questions (that no single chunk or entity neighborhood can answer alone) — not as a general-purpose retrieval upgrade.

---

## 3. Implementation & Reference Code

Below is a self-contained (deliberately small, illustrative) implementation of local vs. global graph query routing, matching the local/global distinction above. Real entity/relation extraction and community detection are LLM-call- and clustering-library-driven respectively, and are out of scope for this illustrative reference.

```python
from dataclasses import dataclass, field


@dataclass
class KnowledgeGraph:
    """Minimal toy graph: entities and (source, relation, target) triples, plus
    precomputed community summaries -- illustrating the local vs. global query split."""
    entities: set[str] = field(default_factory=set)
    triples: list[tuple[str, str, str]] = field(default_factory=list)  # (source, relation, target)
    community_summaries: dict[str, str] = field(default_factory=dict)  # community_id -> summary text
    entity_to_community: dict[str, str] = field(default_factory=dict)

    def add_triple(self, source: str, relation: str, target: str) -> None:
        self.entities.update([source, target])
        self.triples.append((source, relation, target))

    def local_query(self, entity: str) -> list[tuple[str, str, str]]:
        """Traverse from a matched entity: return its directly connected relations."""
        return [t for t in self.triples if t[0] == entity or t[2] == entity]

    def global_query(self, community_id: str) -> str:
        """Retrieve a precomputed community summary -- no traversal, no per-query synthesis cost."""
        return self.community_summaries.get(community_id, "No summary available for this community.")


if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.add_triple("CEO", "leads", "Company X")
    kg.add_triple("Company X", "sells", "Product A")
    kg.add_triple("Company X", "acquired", "Company Y")
    kg.add_triple("Product B", "competes_with", "Product A")
    kg.add_triple("Company Z", "makes", "Product B")

    kg.entity_to_community.update({"Product B": "community_2", "Company Z": "community_2", "CFO": "community_2"})
    kg.community_summaries["community_2"] = (
        "Community 2 covers Company Z's product line, centered on Product B, "
        "which directly competes with Company X's Product A."
    )

    # Local query: "what does Company X do?"
    local_results = kg.local_query("Company X")
    print("Local query results for 'Company X':")
    for triple in local_results:
        print(f"  {triple}")
    assert len(local_results) == 3  # leads, sells, acquired all touch Company X

    # Global query: "what are the major themes in this corpus?" -> retrieve precomputed summary, no traversal
    global_result = kg.global_query("community_2")
    print(f"\nGlobal query result: {global_result}")
    assert "Product B" in global_result and "Product A" in global_result
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Answering multi-hop and corpus-wide "global" questions that flat vector similarity search structurally cannot answer, since no single chunk contains the synthesized answer a graph traversal or community summary can provide.
* **Why Introduced over Legacy Approaches:** Flat retrieval (Modules 03-05) treats every chunk as independent and has no notion of cross-document entity relationships or corpus-wide themes; GraphRAG makes those relationships explicit and queryable, at the cost of a real graph-construction and maintenance pipeline flat retrieval doesn't need.
* **Key Failure Modes & Limitations:** Entity/relation extraction errors compound into a noisy or incorrect graph; community structure and summaries go stale as the corpus updates without a corresponding re-extraction/re-summarization pass; graph construction cost scales with corpus size and update frequency, making it a poor fit for small or rapidly-changing corpora.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Graph construction is $O(N_{\text{chunks}})$ LLM extraction calls (one or more per chunk) plus a clustering pass over the resulting graph for community detection — a substantial one-time (and recurring, on updates) cost compared to flat indexing's simpler embed-and-store pipeline.
* **Space/Memory Footprint:** Stores both the graph structure (entities, relations) and precomputed community summaries, on top of (not instead of) whatever flat vector index the entity/chunk text is also indexed in.
* **Primary Bottleneck Type:** Graph construction is LLM-extraction-latency-bound and scales with corpus size; local queries are graph-traversal-bound (typically fast); global queries are bound by however many community summaries need to be retrieved/synthesized, but are cheap precisely because that synthesis happened at index time, not query time.
* **Variable Legend:** $N_{\text{chunks}}$ = corpus size for extraction cost purposes; no additional formula-specific variables, per this module's prose-only scope.

### 3. Production & Scalability
* **Deployment Considerations:** Treat graph construction as a recurring pipeline stage tied to the document lifecycle (Module 02) — new/edited documents need entity/relation re-extraction, and community structure needs periodic re-computation as the graph grows, not a one-time build assumed to stay valid indefinitely.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How would you keep a knowledge graph in sync as documents are added or edited?
        *   *A:* Tie graph updates directly into Module 02's document lifecycle hooks — re-run entity/relation extraction on the changed document's chunks specifically (not the whole corpus), and periodically (not on every single edit) re-run community detection, since community structure is a corpus-wide property that doesn't need to shift on every individual document change.
    2.  *Q:* When would you choose GraphRAG over a simpler hybrid retrieval setup?
        *   *A:* When the actual production query distribution includes a meaningful share of multi-hop or corpus-wide "global" questions that flat retrieval demonstrably fails on — not by default, since GraphRAG's construction/maintenance cost is only justified by query types that genuinely need relational or corpus-wide reasoning.
