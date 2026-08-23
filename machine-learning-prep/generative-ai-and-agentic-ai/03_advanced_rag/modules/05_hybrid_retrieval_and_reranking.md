# Module 05: Hybrid Retrieval & Reranking

## 1. Introduction & Intuition

### The Core Bottleneck
Dense vector search (Modules 03-04) is excellent at matching *meaning* but can miss exact keyword/entity matches that a sparse method like BM25 (`00_nlp_fundamentals`) catches trivially — a query for a specific product SKU, error code, or proper noun can retrieve poorly from pure dense search if that exact token happens to sit in an awkward part of embedding space. Conversely, BM25 misses paraphrases and synonyms dense search handles naturally. Neither retrieval method is strictly better than the other; they fail on different query types, which is exactly why combining them (hybrid retrieval) consistently outperforms either alone in production.

### High-Level Intuition
Think of BM25 as a literal-minded librarian who finds every book containing your exact search terms, and dense vector search as an intuitive librarian who understands what you *mean* even if you don't use the right words, but sometimes misses the obviously-relevant book sitting right in front of them because it doesn't "feel" similar in the abstract. Hybrid retrieval asks both librarians and merges their recommendations. Reranking then adds a third, slower-but-more-careful librarian who actually reads each shortlisted book closely (a cross-encoder, Module 03) before handing you the final short list — too expensive to have read every book in the library, but perfectly affordable once the list is down to a handful of candidates.

---

## 2. Core Concepts & Mathematical Formulation

### Reciprocal Rank Fusion (RRF)

#### Purpose & Intuition
Given two (or more) separately-ranked result lists — say, BM25's ranking and a dense vector search's ranking — RRF combines them into a single fused ranking using *only* each document's rank position in each list, not its raw score (BM25 scores and cosine similarities live on entirely different, incomparable scales, so naively averaging raw scores would be meaningless). A document that ranks reasonably well across *both* lists gets a higher fused score than a document that ranks #1 in only one list and poorly in the other — RRF specifically rewards consistency across retrieval methods.

#### Mathematical Formulation
For a document $d$ appearing at rank $\text{rank}_i(d)$ in retrieval list $i$, with constant $k$ (commonly 60):
$$\text{RRF}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}$$

### Hand Calculation: Fusing BM25 and Vector Search Rankings
Five documents are ranked differently by BM25 and dense vector search:

| Document | BM25 rank | Vector rank |
|---|---|---|
| A | 1 | 2 |
| B | 2 | 4 |
| C | 3 | 1 |
| D | 4 | 5 |
| E | 5 | 3 |

With $k=60$:
$$\text{RRF}(A) = \frac{1}{61} + \frac{1}{62} \approx 0.032522, \qquad \text{RRF}(C) = \frac{1}{63} + \frac{1}{61} \approx 0.032266$$
$$\text{RRF}(B) = \frac{1}{62} + \frac{1}{64} \approx 0.031754, \qquad \text{RRF}(E) = \frac{1}{65} + \frac{1}{63} \approx 0.031258, \qquad \text{RRF}(D) = \frac{1}{64} + \frac{1}{65} \approx 0.031010$$

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 780 220" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="390" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Hybrid Retrieval + Reranking Pipeline</text>

  <rect x="20" y="50" width="140" height="45" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="90" y="76" text-anchor="middle" font-size="10.5" fill="#1e3a8a" font-weight="600">BM25 ranks</text>

  <rect x="20" y="110" width="140" height="45" rx="5" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.4"/>
  <text x="90" y="136" text-anchor="middle" font-size="10.5" fill="#5b21b6" font-weight="600">Vector ranks</text>

  <rect x="220" y="80" width="140" height="45" rx="5" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.4"/>
  <text x="290" y="106" text-anchor="middle" font-size="10.5" fill="#854d0e" font-weight="600">RRF Fusion</text>

  <rect x="420" y="80" width="150" height="45" rx="5" fill="#fed7aa" stroke="#c2410c" stroke-width="1.4"/>
  <text x="495" y="102" text-anchor="middle" font-size="10.5" fill="#7c2d12" font-weight="600">Cross-Encoder</text>
  <text x="495" y="115" text-anchor="middle" font-size="8.5" fill="#7c2d12">Rerank</text>

  <rect x="630" y="80" width="130" height="45" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="1.4"/>
  <text x="695" y="106" text-anchor="middle" font-size="10.5" fill="#14532d" font-weight="600">Top-N Final</text>

  <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#arrow05a)">
    <line x1="160" y1="72" x2="235" y2="98"/>
    <line x1="160" y1="132" x2="235" y2="108"/>
    <line x1="360" y1="102" x2="418" y2="102"/>
    <line x1="570" y1="102" x2="628" y2="102"/>
  </g>
  <defs>
    <marker id="arrow05a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <text x="390" y="180" text-anchor="middle" font-size="9" fill="#64748b">Two independently-ranked lists fused by rank position (RRF), then the fused Top-N is reranked by a slower, more accurate cross-encoder.</text>
</svg>
</div>

**Fused ranking: A > C > B > E > D.** The key insight is that **A wins the fused ranking despite C being ranked #1 by vector search** — because A ranks consistently well in *both* lists (1st and 2nd) while C's #1 vector rank is dragged down by only being 3rd in BM25. RRF specifically rewards documents both retrieval methods agree on, rather than being dominated by either single list's top result.

---

### Candidate-Set Sizing: The Retrieval Funnel

#### Intuition & Practical Use
Production hybrid retrieval isn't one search followed by one answer — it's a narrowing funnel: retrieve a generous **Top-K** candidate set from each retrieval method, fuse them with RRF, **rerank** the fused list down to a smaller **Top-N** using a more expensive cross-encoder, and finally assemble only the best **Top-M** of those into the context actually sent to the generator.
$$\text{Top-K (initial retrieval)} \rightarrow \text{RRF fusion} \rightarrow \text{Top-N (reranked)} \rightarrow \text{Top-M (final context)}$$

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 700 190" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="350" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Candidate-Set Sizing Funnel</text>

  <!-- Funnel trapezoid shapes, narrowing left to right -->
  <polygon points="30,50 230,50 210,110 50,110" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="130" y="75" text-anchor="middle" font-size="11" fill="#1e3a8a" font-weight="600">Top-K = 50</text>
  <text x="130" y="92" text-anchor="middle" font-size="8.5" fill="#1e3a8a">initial retrieval</text>

  <polygon points="250,58 430,58 415,102 265,102" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.4"/>
  <text x="340" y="80" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">Top-N = 10</text>
  <text x="340" y="94" text-anchor="middle" font-size="8.5" fill="#5b21b6">after rerank</text>

  <polygon points="450,66 590,66 580,94 460,94" fill="#dcfce7" stroke="#16a34a" stroke-width="1.4"/>
  <text x="525" y="83" text-anchor="middle" font-size="11" fill="#14532d" font-weight="600">Top-M = 5</text>

  <g stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#arrow05b)">
    <line x1="230" y1="80" x2="248" y2="80"/>
    <line x1="430" y1="80" x2="448" y2="80"/>
  </g>
  <defs>
    <marker id="arrow05b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <text x="350" y="140" text-anchor="middle" font-size="9" fill="#334155" font-weight="600">Wider stage = higher odds the relevant doc survives, at directly higher latency/cost.</text>
  <text x="350" y="158" text-anchor="middle" font-size="9" fill="#64748b">K controls retrieval recall ceiling; N bounds reranking cost; M bounds final context size/cost.</text>
</svg>
</div>

### Worked Example: $K=50 \to N=10 \to M=5$
*   **Top-K=50:** Cast a wide net — retrieve the 50 best candidates from *each* of BM25 and vector search (cheap per-candidate cost, so a generous $K$ costs little). A wide $K$ maximizes the odds the truly relevant document is *somewhere* in the candidate pool before anything narrows it.
*   **RRF fusion:** Merge the two Top-50 lists into one ranked list using the RRF formula above — still cheap, since RRF is just arithmetic over already-computed ranks.
*   **Top-N=10 (reranked):** Run the more expensive cross-encoder (Module 03) over only the fused list's top 10 — narrow enough to keep reranking cost bounded, wide enough that a document ranked, say, 7th by the cheap fusion step still gets a chance to be correctly promoted to 1st by the more accurate reranker.
*   **Top-M=5 (final context):** Only the reranker's top 5 are actually assembled into the prompt sent to the generator — keeping context small and focused (avoiding the "lost in the middle" problem from Module 01) rather than dumping all 10 reranked candidates into context.

**Why widening any stage trades recall for latency/cost.** Increasing $K$ improves the odds a genuinely relevant document survives into the fused list at all — but doing so directly increases the number of documents that must be embedded/scored at retrieval time. Increasing $N$ improves the odds the reranker gets a chance to correctly promote a document the cheap fusion step under-ranked — but every additional candidate in $N$ is another full cross-encoder forward pass (Module 03's most expensive per-candidate operation). Increasing $M$ gives the generator more potentially-relevant material — but every additional token in $M$ both costs more (Module 01's per-token pricing) and risks diluting the generator's attention across more context.

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of RRF fusion matching the hand calculation above, plus the candidate-set funnel structure.

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuses multiple ranked lists (each a list of doc IDs, best first) via RRF.
    Matches the hand-calculated formula: RRF(d) = sum_i 1 / (k + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for position, doc_id in enumerate(ranked_list, start=1):  # 1-indexed rank
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def retrieval_funnel(bm25_ranked: list[str], vector_ranked: list[str], top_k: int, top_n: int, top_m: int,
                      rerank_fn) -> list[str]:
    """Top-K retrieval -> RRF fusion -> Top-N rerank -> Top-M final context, matching the funnel above."""
    bm25_topk = bm25_ranked[:top_k]
    vector_topk = vector_ranked[:top_k]

    fused = reciprocal_rank_fusion([bm25_topk, vector_topk])
    fused_topn_ids = [doc_id for doc_id, _ in fused[:top_n]]

    reranked_ids = rerank_fn(fused_topn_ids)  # cross-encoder scoring, most expensive stage
    return reranked_ids[:top_m]


if __name__ == "__main__":
    # Hand calc: BM25 ranks [A,B,C,D,E], vector ranks [C,A,E,B,D]
    bm25_ranked = ["A", "B", "C", "D", "E"]
    vector_ranked = ["C", "A", "E", "B", "D"]

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
    print("Fused RRF ranking:")
    for doc_id, score in fused:
        print(f"  {doc_id}: {score:.6f}")

    fused_order = [doc_id for doc_id, _ in fused]
    assert fused_order == ["A", "C", "B", "E", "D"], "Fused order should match the hand calculation"
    assert abs(fused[0][1] - 0.032522) < 1e-5

    # Candidate-set funnel: K=50 -> N=10 -> M=5
    def fake_rerank(candidate_ids):
        # A real reranker would score with a cross-encoder; here we just simulate keeping input order.
        return candidate_ids

    bm25_50 = [f"doc_{i}" for i in range(50)]
    vector_50 = [f"doc_{(i * 7) % 50}" for i in range(50)]  # a differently-ordered candidate pool
    final_context = retrieval_funnel(bm25_50, vector_50, top_k=50, top_n=10, top_m=5, rerank_fn=fake_rerank)
    print(f"\nFinal Top-M context (M=5): {final_context}")
    assert len(final_context) == 5
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Neither pure sparse (BM25) nor pure dense (embedding) retrieval alone reliably handles both exact-match and semantic-match query types; hybrid fusion plus reranking combines their strengths while using an expensive, high-precision reranker only where it's affordable — a small candidate set, not the whole corpus.
* **Why Introduced over Legacy Approaches:** Single-method retrieval systematically misses one entire class of queries (dense-only misses exact keyword/entity matches, sparse-only misses paraphrases/synonyms); RRF specifically avoids the pitfall of naively combining incomparable raw scores from different retrieval methods by fusing on rank position instead.
* **Key Failure Modes & Limitations:** RRF's $k$ constant flattens score differences among top-ranked documents (a deliberate design choice, but one that can under-differentiate strong candidates); reranking cost grows linearly with $N$, making an overly generous rerank stage a real latency/cost problem at scale.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Retrieval (BM25 + dense ANN) is roughly $O(K)$ per method; RRF fusion is $O(K \log K)$ for the sort; reranking is $O(N)$ full cross-encoder forward passes, the single most expensive stage in the funnel per candidate.
* **Space/Memory Footprint:** No persistent storage cost beyond the underlying BM25 index and vector index themselves (Modules 04 covers the vector side); the funnel's memory footprint is transient, scoped to one query's candidate lists.
* **Primary Bottleneck Type:** Reranking is compute-bound (cross-encoder inference, Module 03) and is almost always the funnel's latency bottleneck; retrieval itself is typically memory-bandwidth/index-latency-bound (Module 04).
* **Variable Legend:** $k$ = RRF's rank-damping constant, $K/N/M$ = the three funnel stage widths (initial retrieval / reranked / final context).

### 3. Production & Scalability
* **Deployment Considerations:** Tune $K$, $N$, $M$ against measured latency budgets and Module 09's Recall@k/NDCG metrics, not by intuition — the funnel's stage widths are the single biggest latency/cost lever in a hybrid retrieval pipeline, and are worth A/B testing directly against end-to-end answer quality.
* **When Reranking Isn't Worth It:** Skip the reranking stage (or use a much smaller $N$) when the candidate set is already small, when the latency budget is tight enough that a full cross-encoder pass over even 10 candidates is unaffordable, or when the first-stage retriever (a well-tuned hybrid fusion) is already precise enough on the target query distribution that the added cross-encoder pass buys negligible additional precision — reranking's cost is only justified when it measurably improves the final Top-M's relevance over skipping it.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does RRF use rank position instead of raw similarity/BM25 scores?
        *   *A:* BM25 scores and cosine similarities live on entirely different, non-comparable scales (BM25 is an unbounded term-frequency-weighted score; cosine similarity is bounded in $[-1,1]$) — naively averaging or summing them would let whichever score happens to have a larger numeric range dominate the fusion regardless of actual relevance; rank position is scale-free and directly comparable across any retrieval method.
    2.  *Q:* How would you decide the right $K$, $N$, $M$ for a production system?
        *   *A:* Start from the latency budget (reranking cost scales with $N$, so work backward from "how much reranking time can I afford"), then sweep $K$/$N$/$M$ against Module 09's retrieval metrics on a held-out evaluation set to find the smallest funnel widths that don't measurably hurt Recall@k/NDCG — widening further only adds cost without a corresponding quality gain.
