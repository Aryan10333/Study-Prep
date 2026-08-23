# Module 02: Document Processing, Chunking & Document Lifecycle Management

## 1. Introduction & Intuition

### The Core Bottleneck
Retrieval can only ever be as good as the units it retrieves. If a document is chunked so a fact is split across a chunk boundary, no embedding model or ANN index can recover what was lost — the chunk that gets retrieved simply doesn't contain the whole answer. And in production, documents aren't a static, one-time upload: they get added, edited, and deleted continuously, and every one of those changes has to propagate into the index correctly or the system quietly starts answering from stale, deleted, or duplicated content. Chunking strategy and index lifecycle management are the two least glamorous parts of a RAG system and the two most responsible for silent production quality regressions.

### High-Level Intuition
Chunking is choosing how to cut up a book into index cards. Cut too coarsely (huge chunks) and each card is packed with irrelevant text diluting the one relevant sentence a query actually needs. Cut too finely (tiny chunks) and a single card in isolation loses the surrounding context needed to make sense of it — "the CEO announced this in Q3" means little as a chunk if the sentence naming *which* CEO and *which company* was on the previous page. Every chunking strategy in this module is a different answer to the same tension: preserve enough surrounding context to be meaningful, without diluting the chunk so much that the exact relevant fact gets buried.

Document lifecycle management is the second half of the same problem, but over *time* instead of *space*: a perfectly-chunked index on day one silently rots if nothing updates it when the source document changes on day thirty.

---

## 2. Core Concepts & Mathematical Formulation

### Document Parsing & Chunking Strategies

#### Intuition & Practical Use
Before chunking, raw documents (PDFs, HTML, Word docs, scanned images) need to be parsed into clean text, ideally *layout-aware* — a PDF parser that understands tables, headers, and columns produces far cleaner chunks than one that just concatenates every character it finds in reading order, which can interleave unrelated columns or lose table structure entirely. Once parsed, several chunking strategies trade off differently:

| Strategy | How it works | Best for | Weakness |
|---|---|---|---|
| **Fixed-size** | Split every $N$ tokens/characters, regardless of content | Simple, fast, uniform corpora | Cuts mid-sentence/mid-fact with no regard for meaning |
| **Recursive** | Split on a priority list of separators (paragraph → sentence → word) until chunks fit a size budget | General-purpose default | Still size-driven, not meaning-driven |
| **Semantic** | Split at points where embedding similarity between adjacent sentences drops (a topic shift) | Content with clear topic boundaries | Extra embedding calls at ingestion time; sensitive to embedding model quality |
| **Parent-child (small-to-big)** | Index small chunks for precise retrieval, but return their larger *parent* chunk (or the whole section) to the generator | Balancing retrieval precision with generation context | Extra bookkeeping to track parent-child relationships |
| **Late Chunking** | Embed the *entire* document first with a long-context embedding model, *then* pool the resulting token-level representations into per-span vectors | Documents with strong cross-chunk dependencies (pronouns, references spanning sections) | Requires a long-context-capable embedding model; more expensive per-document embedding pass |

**Late Chunking, explained further:** standard chunking embeds each chunk *in isolation* — the embedding model never sees the rest of the document while encoding chunk 3, so it has no representation of what chunk 2 or chunk 4 said. Late Chunking inverts the order: run the whole document through a long-context embedding model first (so every token's contextual representation already reflects the full document), and only *afterward* pool spans of those token representations into per-chunk vectors. The resulting chunk embeddings carry information from the *entire* document's context, not just their own local text — directly fixing the "pronoun with no antecedent" and "fact referencing an earlier section" problems standard chunking structurally cannot solve, since standard chunking never gives the embedding model a chance to see beyond chunk boundaries in the first place. This needs no deep mathematical treatment: the mechanism is entirely about *when* pooling happens relative to *when* the model sees full-document context, not a new formula.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 860 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="430" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Standard Chunking vs. Late Chunking</text>

  <!-- Standard chunking -->
  <text x="20" y="55" font-size="11" font-weight="600" fill="#991b1b">Standard: chunk first, embed each in isolation</text>
  <g font-size="9">
    <rect x="20" y="65" width="120" height="40" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1.2"/>
    <text x="80" y="88" text-anchor="middle" fill="#991b1b">Chunk 1 -&gt; Embed</text>
    <rect x="160" y="65" width="120" height="40" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1.2"/>
    <text x="220" y="88" text-anchor="middle" fill="#991b1b">Chunk 2 -&gt; Embed</text>
    <rect x="300" y="65" width="120" height="40" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1.2"/>
    <text x="360" y="88" text-anchor="middle" fill="#991b1b">Chunk 3 -&gt; Embed</text>
  </g>
  <text x="220" y="122" text-anchor="middle" font-size="8.5" fill="#991b1b">Each embedding sees ONLY its own chunk -- no cross-chunk context</text>

  <!-- Late chunking -->
  <text x="20" y="160" font-size="11" font-weight="600" fill="#065f46">Late Chunking: embed full document, then pool per span</text>
  <rect x="20" y="170" width="400" height="35" rx="4" fill="#ecfdf5" stroke="#059669" stroke-width="1.2"/>
  <text x="220" y="192" text-anchor="middle" font-size="9.5" fill="#065f46">Full document -&gt; long-context embedding model (one pass)</text>

  <g stroke="#059669" stroke-width="1.3" fill="none" marker-end="url(#arrow02)">
    <line x1="90" y1="205" x2="90" y2="222"/>
    <line x1="220" y1="205" x2="220" y2="222"/>
    <line x1="350" y1="205" x2="350" y2="222"/>
  </g>
  <defs>
    <marker id="arrow02" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#059669"/>
    </marker>
  </defs>
  <g font-size="8.5" fill="#065f46">
    <rect x="45" y="225" width="90" height="24" rx="3" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="90" y="240" text-anchor="middle">Pool -&gt; span 1</text>
    <rect x="175" y="225" width="90" height="24" rx="3" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="220" y="240" text-anchor="middle">Pool -&gt; span 2</text>
    <rect x="305" y="225" width="90" height="24" rx="3" fill="#d1fae5" stroke="#059669" stroke-width="1"/>
    <text x="350" y="240" text-anchor="middle">Pool -&gt; span 3</text>
  </g>

  <text x="640" y="100" font-size="9" fill="#334155" width="200">
    <tspan x="640" dy="0">Each pooled span vector was</tspan>
    <tspan x="640" dy="14">computed from token representations</tspan>
    <tspan x="640" dy="14">that already attended to the</tspan>
    <tspan x="640" dy="14">*entire* document -- cross-chunk</tspan>
    <tspan x="640" dy="14">references are preserved.</tspan>
  </text>
</svg>
</div>

---

### Chunk Count & Storage Overhead from Overlap

#### Purpose & Intuition
Chunks are usually created with a sliding overlap (each chunk repeats the tail of the previous one) specifically to reduce the "fact split across a boundary" failure mode — but overlap isn't free, it multiplies stored/embedded tokens. Knowing this formula lets you predict both retrieval-unit count and index storage cost directly from a document's length before ingesting it.

#### Mathematical Formulation
For a document of $L_{\text{doc}}$ tokens, a chosen chunk size, and an overlap between consecutive chunks:
$$N_{\text{chunks}} = \left\lceil \frac{L_{\text{doc}} - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \right\rceil$$

### Hand Calculation: Chunk Count for a 2,000-Token Document
With $L_{\text{doc}}=2{,}000$, `chunk_size`=400, `overlap`=50:

*   **Step 1: Compute the stride (effective new tokens per chunk)**
    $$\text{stride} = \text{chunk\_size} - \text{overlap} = 400 - 50 = 350$$

*   **Step 2: Apply the chunk-count formula**
    $$N_{\text{chunks}} = \left\lceil \frac{2{,}000 - 50}{350} \right\rceil = \left\lceil 5.571 \right\rceil = 6$$

*   **Step 3: Estimate the storage overhead from overlap**
    $$\text{overhead\_tokens} \approx (N_{\text{chunks}} - 1) \times \text{overlap} = 5 \times 50 = 250 \text{ tokens} \;(\approx 12.5\% \text{ of the original document})$$

Six chunks are needed to cover a 2,000-token document with this configuration, at a real, quantifiable storage/embedding cost of roughly 12.5% more tokens than the raw document length — the overlap-vs-boundary-safety trade-off made concrete instead of an arbitrary default.

---

### Metadata Extraction & Enrichment

#### Intuition & Practical Use
Beyond the chunk's raw text, storing structured metadata (source document, section title, page number, author, last-modified date, access-control tags) alongside each chunk's vector enables *filtered* retrieval — restricting a search to a specific document set, date range, or permission level *before* (or alongside) the similarity search, rather than retrieving broadly and hoping post-hoc filtering doesn't discard the one relevant result. This is a procedural/systems concern, not a mathematical one: metadata schema design and how a given vector database supports combined vector+filter queries (Module 04).

---

### Production Document Lifecycle Management

#### Intuition & Practical Use
An index is never really "done" in production — source documents get added, edited, and removed continuously, and each of those events needs a corresponding, correct index-side action:

| Lifecycle Event | Required Index Action | Failure Mode if Skipped |
|---|---|---|
| **New document added** | Chunk, embed, and insert incrementally — *without* a full corpus re-index | Full re-indexing at scale is slow/expensive; doing it per-document keeps ingestion cheap and fast |
| **Document edited** | Re-chunk/re-embed the changed spans, increment a **version** identifier, and replace (not just add to) the old chunks | Old and new versions both remain retrievable, and the system may cite outdated information alongside current information for the same query |
| **Document deleted** | Remove or **tombstone** (mark deleted, filter out at query time, physically purge later) its chunks | Deleted content stays retrievable and citable indefinitely — a real compliance/correctness problem, not just a quality one |
| **Stale embedding detection** | Periodically check whether a chunk's stored content hash/timestamp still matches its live source | An index entry silently drifts out of sync with its source document with no automatic signal that anything is wrong |

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 820 230" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="410" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Document Lifecycle State Flow</text>

  <!-- Loop back from stale-detected to edited/versioned (re-index), drawn first so it sits behind the boxes -->
  <path d="M505,90 Q505,44 295,44 Q170,44 170,88" fill="none" stroke="#059669" stroke-width="1.4" stroke-dasharray="4,2" marker-end="url(#arrow02c)"/>
  <defs>
    <marker id="arrow02c" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#059669"/>
    </marker>
  </defs>
  <text x="340" y="38" text-anchor="middle" font-size="8.5" fill="#065f46">re-index once drift is caught</text>

  <rect x="20" y="90" width="150" height="50" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="95" y="119" text-anchor="middle" font-size="11" fill="#1e3a8a" font-weight="600">Added</text>

  <rect x="220" y="90" width="150" height="50" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="295" y="113" text-anchor="middle" font-size="11" fill="#854d0e" font-weight="600">Edited / Versioned</text>
  <text x="295" y="128" text-anchor="middle" font-size="8.5" fill="#854d0e">(re-chunk changed spans)</text>

  <rect x="420" y="90" width="170" height="50" rx="6" fill="#fed7aa" stroke="#c2410c" stroke-width="1.5"/>
  <text x="505" y="113" text-anchor="middle" font-size="11" fill="#7c2d12" font-weight="600">Stale-Detected</text>
  <text x="505" y="128" text-anchor="middle" font-size="8.5" fill="#7c2d12">(hash mismatch, missed edit)</text>

  <rect x="640" y="90" width="150" height="50" rx="6" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="715" y="113" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="600">Deleted /</text>
  <text x="715" y="128" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="600">Tombstoned</text>

  <g stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow02b)">
    <line x1="170" y1="115" x2="218" y2="115"/>
    <line x1="370" y1="115" x2="418" y2="115"/>
    <line x1="590" y1="115" x2="638" y2="115"/>
  </g>
  <defs>
    <marker id="arrow02b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <text x="95" y="160" text-anchor="middle" font-size="8.5" fill="#334155">Incremental insert,</text>
  <text x="95" y="172" text-anchor="middle" font-size="8.5" fill="#334155">no full re-index</text>
  <text x="715" y="160" text-anchor="middle" font-size="8.5" fill="#334155">Excluded from queries</text>
  <text x="715" y="172" text-anchor="middle" font-size="8.5" fill="#334155">immediately, purged later</text>
</svg>
</div>

**Worked end-to-end lifecycle example.** Trace a single policy document through its full life in the index:
1.  **Added (Day 0):** The document is chunked (6 chunks, per the calculation above), embedded, and inserted with `doc_id=POLICY-42`, `version=1`.
2.  **Incrementally indexed:** No other document's chunks are touched — insertion is additive, not a corpus-wide rebuild.
3.  **Edited (Day 10):** Section 3 changes. Only the chunks derived from Section 3 are re-chunked and re-embedded; `version` increments to `2` for those chunks (and, depending on the versioning granularity chosen, optionally for the whole document).
4.  **Re-indexed:** The new `version=2` chunks are inserted; the old `version=1` chunks for that section are marked replaced, not left retrievable alongside the new ones.
5.  **Stale detection (Day 40):** A periodic job compares each indexed chunk's stored content hash against the live document. If Section 5 changed on Day 35 but nothing re-indexed it, this check is what surfaces the drift — without it, the index would silently keep serving Day-0 content for Section 5 indefinitely.
6.  **Deleted (Day 60):** The policy is retired. Its chunks are tombstoned immediately (excluded from query results via a `deleted=true` filter) and physically purged from the index in a later batch cleanup pass, rather than being deleted synchronously in the request path.

This sequence is the concrete version of the abstract table above — the same four lifecycle actions, demonstrated in the order and dependency they actually occur in production, not just defined in isolation.

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of the chunk-count/overlap calculation and a minimal document lifecycle state tracker matching the worked example above.

```python
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum


def compute_chunk_count(doc_len_tokens: int, chunk_size: int, overlap: int) -> int:
    """Chunk count from document length, chunk size, and overlap. Matches the hand calculation above."""
    assert overlap < chunk_size, "overlap must be smaller than chunk_size or the stride is non-positive"
    stride = chunk_size - overlap
    return math.ceil((doc_len_tokens - overlap) / stride)


def estimate_overlap_overhead_tokens(n_chunks: int, overlap: int) -> int:
    """Approximate extra stored tokens from overlap (interior chunks only)."""
    return max(0, n_chunks - 1) * overlap


class LifecycleState(Enum):
    ACTIVE = "active"
    STALE = "stale"          # content hash mismatch detected, needs re-indexing
    TOMBSTONED = "tombstoned"  # deleted, excluded from queries, pending physical purge


@dataclass
class IndexedChunk:
    doc_id: str
    chunk_id: str
    version: int
    content_hash: str
    state: LifecycleState = LifecycleState.ACTIVE


@dataclass
class DocumentIndex:
    """Minimal lifecycle tracker matching the Module 02 worked example (add -> edit/version -> stale-detect -> delete)."""
    chunks: dict = field(default_factory=dict)  # chunk_id -> IndexedChunk

    def add_document(self, doc_id: str, chunk_texts: list[str]) -> list[str]:
        """New document: incremental insert, version=1, no full re-index of anything else."""
        chunk_ids = []
        for i, text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}::v1::{i}"
            self.chunks[chunk_id] = IndexedChunk(doc_id, chunk_id, version=1, content_hash=_hash(text))
            chunk_ids.append(chunk_id)
        return chunk_ids

    def update_section(self, doc_id: str, old_chunk_ids: list[str], new_chunk_texts: list[str], new_version: int) -> list[str]:
        """Edited document: replace only the affected chunks, bump version -- old chunks are NOT left retrievable."""
        for cid in old_chunk_ids:
            self.chunks[cid].state = LifecycleState.TOMBSTONED
        new_ids = []
        for i, text in enumerate(new_chunk_texts):
            chunk_id = f"{doc_id}::v{new_version}::{i}"
            self.chunks[chunk_id] = IndexedChunk(doc_id, chunk_id, version=new_version, content_hash=_hash(text))
            new_ids.append(chunk_id)
        return new_ids

    def detect_stale(self, chunk_id: str, live_text: str) -> bool:
        """Periodic drift check: does the stored hash still match the live source?"""
        chunk = self.chunks[chunk_id]
        if chunk.state == LifecycleState.ACTIVE and chunk.content_hash != _hash(live_text):
            chunk.state = LifecycleState.STALE
            return True
        return False

    def tombstone_document(self, doc_id: str) -> int:
        """Deleted document: mark excluded from queries immediately; physical purge happens in a later batch job."""
        count = 0
        for chunk in self.chunks.values():
            if chunk.doc_id == doc_id and chunk.state != LifecycleState.TOMBSTONED:
                chunk.state = LifecycleState.TOMBSTONED
                count += 1
        return count

    def active_chunks_for(self, doc_id: str) -> list[str]:
        return [c.chunk_id for c in self.chunks.values() if c.doc_id == doc_id and c.state == LifecycleState.ACTIVE]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    # Verify the chunk-count hand calculation
    n_chunks = compute_chunk_count(doc_len_tokens=2000, chunk_size=400, overlap=50)
    overhead = estimate_overlap_overhead_tokens(n_chunks, overlap=50)
    print(f"Chunk count: {n_chunks}, overlap overhead: {overhead} tokens")
    assert n_chunks == 6
    assert overhead == 250

    # Walk the worked lifecycle example
    index = DocumentIndex()
    v1_ids = index.add_document("POLICY-42", [f"section {i} original text" for i in range(6)])
    assert len(index.active_chunks_for("POLICY-42")) == 6

    # Edit: Section 3 (index 3) changes -> re-chunk/re-embed just that chunk, bump version
    v2_ids = index.update_section("POLICY-42", [v1_ids[3]], ["section 3 EDITED text"], new_version=2)
    active = index.active_chunks_for("POLICY-42")
    assert v1_ids[3] not in active and v2_ids[0] in active
    print(f"After edit: {len(active)} active chunks (old v1 section 3 tombstoned, new v2 chunk active)")

    # Stale detection: Section 5 changed live but was never re-indexed
    is_stale = index.detect_stale(v1_ids[5], live_text="section 5 CHANGED but not re-indexed")
    assert is_stale
    print(f"Stale detected on {v1_ids[5]}: {is_stale}")

    # Delete: tombstone the whole document
    tombstoned = index.tombstone_document("POLICY-42")
    assert len(index.active_chunks_for("POLICY-42")) == 0
    print(f"Tombstoned {tombstoned} remaining chunks; 0 active chunks remain for POLICY-42")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Splitting documents into retrievable units that preserve enough context to be meaningful without diluting the specific fact a query needs, and keeping the index correctly synchronized with a constantly-changing source corpus over time.
* **Why Introduced over Legacy Approaches:** Naive fixed-size chunking treats every document identically regardless of content structure; a one-time ingestion pipeline with no lifecycle handling silently accumulates stale, duplicated, or deleted-but-still-retrievable content as the corpus evolves.
* **Key Failure Modes & Limitations:** Chunk-boundary fact splitting, cross-chunk context loss (the problem Late Chunking specifically targets), overlap's storage/embedding cost multiplier, and — on the lifecycle side — stale entries, version collisions (old and new content both retrievable), and non-purged tombstones accumulating index bloat.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Standard chunking is $O(L_{\text{doc}})$ to split; semantic chunking adds embedding calls per candidate boundary; Late Chunking requires one long-context embedding pass over the *full* document (more expensive per-document than embedding each small chunk independently, but still $O(L_{\text{doc}})$, not worse asymptotically).
* **Space/Memory Footprint:** Total stored chunk tokens scale as $N_{\text{chunks}} \times \text{chunk\_size}$, inflated by the overlap-overhead term above; lifecycle metadata (version, content hash, tombstone flag) adds a small constant overhead per chunk.
* **Primary Bottleneck Type:** Ingestion-pipeline throughput is typically I/O- and embedding-API-latency-bound at corpus scale, not compute-bound on the chunking logic itself; stale-detection jobs are bound by how frequently they can afford to re-scan the corpus for drift.
* **Variable Legend:** $L_{\text{doc}}$ = document length in tokens, $N_{\text{chunks}}$ = resulting chunk count, `chunk_size`/`overlap` = the two tunable sizing parameters.

### 3. Production & Scalability
* **Deployment Considerations:** Incremental indexing (never a full corpus rebuild for a single document change) is a hard requirement past a modest corpus size; tombstoning-then-batch-purge (rather than synchronous hard-delete) keeps the delete path fast and safe even under high write concurrency.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How would you detect a stale index entry without re-embedding the entire corpus on every check?
        *   *A:* Store a cheap content hash (not the full embedding) per chunk at index time, and periodically re-hash the live source, comparing hashes — only re-chunk/re-embed the specific chunks whose hash actually changed, not the whole corpus.
    2.  *Q:* Why not just hard-delete a document's chunks immediately when it's deleted?
        *   *A:* Tombstoning (mark-and-filter) keeps the delete path fast and lets query-time filtering exclude it instantly, while the more expensive physical purge (freeing index space, rebalancing shards) can run asynchronously in a batch job without blocking the delete request itself.
