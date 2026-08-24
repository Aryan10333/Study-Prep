# Module 04: Data & Knowledge Infrastructure at Scale

## 1. Introduction & Intuition

### The Core Bottleneck
A RAG system's retrieval *algorithm* (`03_advanced_rag`'s own scope) can be excellent and still sit on top of a knowledge base that is operationally broken — stale, missing deleted documents, or leaking one tenant's data into another's results. This module owns the real infrastructure *operations* question: how does the knowledge base stay correct, current, deletable, and tenant-isolated as real data and traffic both grow, a genuinely separate concern from how well the retrieval algorithm ranks what it's given.

### High-Level Intuition
A library's card catalog being well-organized (the retrieval algorithm) doesn't help if the library never removes cards for books it no longer owns, never adds cards for new arrivals, and lets patrons from one branch browse another branch's restricted archive. This module is about running the library's real, ongoing operations — not about how good the catalog's search index is.

---

## 2. Core Concepts & Mathematical Formulation

### Vector Storage Sizing — Replication and Index Overhead Kept as Two Distinct Real Steps

#### Purpose & High-level Intuition
A real, common oversimplification blends replication cost and index-structure overhead into one factor, obscuring which real driver is responsible for how much storage. This module keeps them as two sequential, real, separately-attributable steps.

**Step 1 — real raw vector storage:**
$$\text{Storage}_{\text{raw}} = N_{\text{vectors}} \times d_{\text{embed}} \times \text{bytes}_{\text{per-float}}$$

**Step 2 — real replication (a genuinely multiplicative redundancy cost):**
$$\text{Storage}_{\text{replicated}} = \text{Storage}_{\text{raw}} \times R$$

Where $R$ is a real, stated replication factor — each replica is a full additional real copy, purchased for real availability, not an incidental overhead.

**Step 3 — real index/metadata overhead (a separate, distinct multiplicative factor, applied per-replica):**
$$\text{Storage}_{\text{total}} = \text{Storage}_{\text{replicated}} \times (1 + \text{overhead}_{\text{index}})$$

Where $\text{overhead}_{\text{index}}$ is a real, separately-stated fraction (e.g., a real HNSW graph's own additional structure) — kept as its own real term so the real cost attributable to redundancy (Step 2) is never confused with the real cost attributable to index structure (Step 3).

### The Full Real Knowledge-Base Lifecycle

#### Intuition & Practical Use
Ingestion and freshness are necessary but not sufficient real lifecycle stages — this module's own syllabus-mandated scope requires the full real lifecycle: **ingestion/ETL** (getting real new/changed data in), **deletion propagation** (a real, often-missed requirement — a document deleted or erased at the source must actually be removed from every real index replica, not just the source system), **re-indexing and versioning** (a real index version is a point-in-time real snapshot; a bad re-index needs a real rollback path to the previous version, not a one-way door), and **multi-tenant isolation** (a real, hard guarantee that one tenant's documents are never retrievable by another tenant's queries — enforced via real index-partitioning or metadata-filtering, not assumed).

### The Full Real Lifecycle, Visualized

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 1000 470" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <defs>
    <marker id="a4" markerWidth="8" markerHeight="8" refX="6" refY="2.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,5 L7,2.5 z" fill="#475569" />
    </marker>
  </defs>
  <text x="500" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">Vector-DB / Knowledge-Infrastructure Lifecycle</text>

  <rect x="20" y="55" width="180" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="110" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1e3a8a">Ingestion / ETL</text>
  <text x="110" y="98" text-anchor="middle" font-size="10.5" fill="#475569">new / changed docs</text>
  <line x1="200" y1="85" x2="230" y2="85" stroke="#475569" stroke-width="1.5" marker-end="url(#a4)"/>

  <rect x="230" y="55" width="180" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="320" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1e3a8a">Indexing</text>
  <text x="320" y="98" text-anchor="middle" font-size="10.5" fill="#475569">embed + build index</text>
  <line x1="410" y1="85" x2="440" y2="85" stroke="#475569" stroke-width="1.5" marker-end="url(#a4)"/>

  <rect x="440" y="55" width="220" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="550" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1e3a8a">Sharding &amp; Replication</text>
  <text x="550" y="98" text-anchor="middle" font-size="10.5" fill="#475569">primary + R replicas</text>

  <rect x="700" y="45" width="270" height="90" rx="8" fill="#ffffff" stroke="#2563eb" stroke-dasharray="3,2"/>
  <text x="835" y="63" text-anchor="middle" font-size="11" font-weight="600" fill="#1e3a8a">Index Replicas</text>
  <rect x="712" y="72" width="70" height="26" rx="4" fill="#dbeafe" stroke="#2563eb"/>
  <text x="747" y="89" text-anchor="middle" font-size="9.5" fill="#1e3a8a">primary</text>
  <rect x="792" y="72" width="80" height="26" rx="4" fill="#dbeafe" stroke="#2563eb"/>
  <text x="832" y="89" text-anchor="middle" font-size="9.5" fill="#1e3a8a">replica 1</text>
  <rect x="882" y="72" width="80" height="26" rx="4" fill="#dbeafe" stroke="#2563eb"/>
  <text x="922" y="89" text-anchor="middle" font-size="9.5" fill="#1e3a8a">replica 2</text>
  <line x1="660" y1="85" x2="695" y2="85" stroke="#475569" stroke-width="1.5" marker-end="url(#a4)"/>

  <path d="M550,115 C550,150 300,150 300,140" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a4)"/>
  <text x="420" y="172" text-anchor="middle" font-size="10.5" fill="#991b1b">Deletion propagation: primary AND every replica</text>

  <rect x="230" y="200" width="220" height="60" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="340" y="225" text-anchor="middle" font-size="12.5" font-weight="600" fill="#92400e">Re-Index (new version)</text>
  <text x="340" y="243" text-anchor="middle" font-size="10.5" fill="#92400e">point-in-time snapshot</text>

  <path d="M320,115 L320,200" fill="none" stroke="#475569" stroke-width="1.5" marker-end="url(#a4)"/>

  <rect x="230" y="290" width="220" height="55" rx="8" fill="#fef3c7" stroke="#d97706" stroke-dasharray="3,2"/>
  <text x="340" y="315" text-anchor="middle" font-size="12" font-weight="600" fill="#92400e">Rollback to Prior Version</text>
  <text x="340" y="332" text-anchor="middle" font-size="10.5" fill="#92400e">if re-index degrades quality</text>

  <path d="M340,260 L340,290" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a4)"/>
  <text x="340" y="278" text-anchor="middle" font-size="10" fill="#92400e">a bad re-index is not a one-way door</text>

  <rect x="530" y="200" width="410" height="145" rx="10" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="735" y="222" text-anchor="middle" font-size="12.5" font-weight="700" fill="#065f46">Multi-Tenant Isolation Boundary</text>

  <rect x="550" y="240" width="170" height="50" rx="6" fill="#ffffff" stroke="#059669"/>
  <text x="635" y="260" text-anchor="middle" font-size="11" fill="#065f46">Tenant A query</text>
  <text x="635" y="276" text-anchor="middle" font-size="10" fill="#475569">sees only Tenant A docs</text>

  <rect x="750" y="240" width="170" height="50" rx="6" fill="#ffffff" stroke="#059669"/>
  <text x="835" y="260" text-anchor="middle" font-size="11" fill="#065f46">Tenant B query</text>
  <text x="835" y="276" text-anchor="middle" font-size="10" fill="#475569">sees only Tenant B docs</text>

  <text x="735" y="315" text-anchor="middle" font-size="10.5" fill="#065f46">enforced via metadata filtering or</text>
  <text x="735" y="330" text-anchor="middle" font-size="10.5" fill="#065f46">physically separate per-tenant indexes</text>
</svg>
</div>

*   **Diagram Interpretation:** The real ingestion→indexing→sharding/replication flow across the top, a real deletion-propagation path shown explicitly reaching every replica (not just the primary), a real re-index/rollback pair showing re-indexing is not a one-way door, and a real multi-tenant isolation boundary shown as its own distinct region — all 4 real lifecycle stages from this module's own syllabus-mandated scope visualized together.

---

### Worked Example: Real Storage Sizing, Replication and Index Overhead Kept Separate

Real stated assumptions: $N_{\text{vectors}} = 10{,}000{,}000$, $d_{\text{embed}} = 1{,}536$ (a real, common embedding dimension), $\text{bytes}_{\text{per-float}} = 4$ (FP32), $R = 3$ (real 3x replication for availability), $\text{overhead}_{\text{index}} = 0.20$ (a real, stated 20% HNSW graph overhead).

*   **Step 1 — real raw storage:**
    $$\text{Storage}_{\text{raw}} = 10{,}000{,}000 \times 1{,}536 \times 4 = 61.44 \text{ GB}$$

*   **Step 2 — real replicated storage:**
    $$\text{Storage}_{\text{replicated}} = 61.44 \times 3 = 184.32 \text{ GB}$$

*   **Step 3 — real total storage (with index overhead):**
    $$\text{Storage}_{\text{total}} = 184.32 \times 1.20 = 221.184 \text{ GB}$$

*   **Real interpretation.** Replication alone (Step 2) triples the real footprint (61.44 GB → 184.32 GB, a real +122.88 GB cost attributable specifically to availability/redundancy); index overhead (Step 3) then adds a further real +36.864 GB on top, attributable specifically to the index structure itself — the two real cost drivers stay individually visible instead of being buried in one blended multiplier.

### Worked Example (No Formula): Real Deletion Propagation

A real user invokes their data-erasure right on a document ingested 6 months ago. Real, correct propagation path: (1) the document is marked deleted in the real source system; (2) a real deletion event is emitted to the ingestion pipeline; (3) the pipeline removes the document's real chunks/vectors from the *primary* index; (4) the same real removal is propagated to *every* real replica (Step 2's $R=3$ copies) — not just the primary — since a query routed to a stale replica would otherwise still surface the deleted content; (5) the real deletion is logged for audit purposes (Module 08's own compliance-logging scope, referenced here as a downstream consumer, not re-derived). A real, common failure mode this walkthrough exposes: deleting from the primary index alone while replicas silently retain the deleted content until their next real re-index cycle — a genuine, real compliance gap if that cycle is infrequent.

---

## 3. Implementation & Reference Code

```python
from dataclasses import dataclass, field


def storage_bytes(n_vectors: int, dim: int, bytes_per_float: int, replication: int, index_overhead: float) -> dict:
    raw = n_vectors * dim * bytes_per_float
    replicated = raw * replication
    total = replicated * (1 + index_overhead)
    return {"raw_bytes": raw, "replicated_bytes": replicated, "total_bytes": total}


@dataclass
class IndexReplica:
    name: str
    documents: set = field(default_factory=set)


def propagate_deletion(replicas: list[IndexReplica], doc_id: str) -> list[str]:
    """Real deletion propagation across every real replica, not just the primary --
    returns the list of replicas a real, correct propagation actually touched."""
    touched = []
    for replica in replicas:
        if doc_id in replica.documents:
            replica.documents.remove(doc_id)
            touched.append(replica.name)
    return touched


if __name__ == "__main__":
    sizing = storage_bytes(
        n_vectors=10_000_000, dim=1536, bytes_per_float=4, replication=3, index_overhead=0.20
    )
    raw_gb = sizing["raw_bytes"] / 1e9
    replicated_gb = sizing["replicated_bytes"] / 1e9
    total_gb = sizing["total_bytes"] / 1e9
    print(f"Real raw storage: {raw_gb:.2f} GB")
    print(f"Real replicated storage (R=3): {replicated_gb:.2f} GB")
    print(f"Real total storage (+20% index overhead): {total_gb:.3f} GB")
    assert abs(raw_gb - 61.44) < 1e-6
    assert abs(replicated_gb - 184.32) < 1e-6
    assert abs(total_gb - 221.184) < 1e-3

    replicas = [
        IndexReplica("primary", documents={"doc_A", "doc_B", "doc_C"}),
        IndexReplica("replica_1", documents={"doc_A", "doc_B", "doc_C"}),
        IndexReplica("replica_2", documents={"doc_A", "doc_B", "doc_C"}),
    ]
    touched = propagate_deletion(replicas, "doc_B")
    print(f"\nReal deletion of 'doc_B' propagated to: {touched}")
    assert touched == ["primary", "replica_1", "replica_2"]
    assert all("doc_B" not in r.documents for r in replicas)

    print("\nVerified: real storage cost from replication (Step 2) and index overhead (Step 3) stay")
    print("individually attributable, and real deletion propagation correctly reaches every replica,")
    print("not just the primary index -- confirming the module's own real lifecycle requirements.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Keeping a real knowledge base correct, current, deletable, and tenant-isolated as data and traffic scale — the real operational layer beneath an already-good retrieval algorithm.
* **Why Introduced over Legacy Approaches:** Treating a vector store as a "write-once, query-forever" store ignores real production requirements (deletion/erasure obligations, index staleness, multi-tenant safety) that a purely retrieval-quality-focused design would miss entirely.
* **Key Failure Modes & Limitations:** Deleting from the primary index while stale replicas silently retain deleted content; a bad re-index with no real rollback path, forcing a full real rebuild under production pressure; relying on retrieval-time filtering alone for tenant isolation without a real, independent guarantee (e.g., separate physical indexes per tenant) for the highest-sensitivity real deployments.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module's cost model is real storage/operations, not model compute.
* **Space/Memory Footprint:** $\mathcal{O}(N_{\text{vectors}} \times d_{\text{embed}} \times R)$ — real storage scales linearly in vector count, embedding dimension, and replication factor, then a further real constant-factor increase from index overhead.
* **Primary Bottleneck Type:** A real storage-cost and operational-correctness bottleneck (not a compute bottleneck) — the real risk is stale/incorrect/leaked data, not slow arithmetic.
* **Variable Legend:** $N_{\text{vectors}}$ = real corpus vector count, $d_{\text{embed}}$ = real embedding dimension, $R$ = real replication factor, $\text{overhead}_{\text{index}}$ = real index-structure overhead fraction.

### 3. Production & Scalability
* **Deployment Considerations:** Real production systems typically schedule periodic real re-index cycles (to absorb accumulated updates/deletions efficiently) alongside real incremental updates for lower-latency freshness — a genuine real trade-off between re-index cost and staleness window, distinct from the storage-sizing question above.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* A user invokes a data-deletion request — how do you guarantee it's actually gone from every replica?
        *   *A:* Propagate the real deletion event to every replica explicitly (not just the primary), and treat a replica's deletion-propagation lag as a real, monitored SLA — a deletion isn't complete until every real replica confirms it, which is exactly the gap this module's own worked example exposes.
    2.  *Q:* How would you guarantee real multi-tenant isolation beyond metadata filtering at query time?
        *   *A:* For the highest-sensitivity real deployments, use physically separate indexes per tenant (or per tenant-tier) rather than relying solely on a metadata filter that could fail open under a real bug — metadata filtering is real, useful defense-in-depth, but a hard physical boundary is the stronger real guarantee.
