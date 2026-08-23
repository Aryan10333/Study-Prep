# Implementation Plan: Track 3 — Interview Q&A (`03_advanced_rag`)

Scope: this plan covers only the standalone Interview Q&A cheatsheet, per `interview_qa_generator/SKILL.md`. Track 1 (9 study-guide modules) and Track 2 (6 companion notebooks, all executed with real data/models/live LLM calls) are both complete; this track does not modify either.

---

## 1. Target File Paths

| Artifact | Path |
|---|---|
| Source Q&A markdown | `modules/10_advanced_rag_interview_questions.md` |
| Compiled standalone cheatsheet HTML | `advanced_rag_interview_cheatsheet.html` |
| Compiled standalone cheatsheet PDF | `advanced_rag_interview_cheatsheet.pdf` |
| Compiler | `helpers/compile_advanced_rag.py` — add a second `compile_document()` call in `main()` at the existing placeholder comment (line 666), mirroring the master-guide call's pattern but pointing at the single Q&A module, producing a separate standalone cheatsheet (not appended into the master study guide) |

## 2. Reference Sources

No external question banks. Every question is derived directly from this topic's own 9 study-guide modules (`modules/01_...md` through `modules/09_...md`) and their real, verified hand-calcs (Late Chunking, chunk-count/overlap, HNSW/IVF/IVF-PQ parameter math, RRF fusion, retrieval metrics) — consistent with this repo's existing Q&A tracks.

**A differentiator from prior Q&A tracks**: Track 2's 6 companion notebooks produced real, executed, measured results (not just theory), several of which surfaced genuine, sometimes counter-intuitive findings — e.g., RRF-fused retrieval beating both single-signal baselines, IVF-PQ's recall ceiling traced to undertrained centroids, GraphRAG's cross-document connectivity gap without entity resolution, and a self-correction loop whose remedy didn't match its diagnosed failure type. Where a question's real notebook result adds genuine interview value (a concrete number, a real surprising finding, a defensible production judgment call), the **Production Perspective** or **Common Mistakes** sections will cite it explicitly (e.g., "measured on the full SciFact corpus in this repo's own Notebook 04...") rather than staying purely theoretical — grounding the answer in something the candidate actually ran, not just read.

## 3. Proposed Question List (59 questions, grouped by module)

Revised per user feedback: replaced Q12 (a stale-embedding-detection question already substantially covered by Q11's lifecycle walkthrough) with an explicit deletion/update-consistency guarantee question, and replaced Q57 (LLM-as-judge pitfalls, thematically overlapping with Q53's RAGAS/faithfulness metrics) with an explicit prompt-injection and cross-tenant data-leakage question — closing a real gap, since the syllabus's Module 09 scope explicitly calls out RAG-specific security but the original 59 had no question testing it directly. Total question count unchanged at 59.

**Module 01 — RAG Fundamentals & Production Architecture (6)**
1. When would you choose RAG over long-context, and when would you choose long-context (or both)?
2. Walk through the naive RAG pipeline (ingest → chunk → embed → index → retrieve → generate) and its characteristic failure modes.
3. Given a corpus size and query volume, how would you estimate the cost crossover between a RAG system and stuffing full documents into a long-context window?
4. What distinguishes "Advanced RAG" architecture (pre-retrieval / retrieval / post-retrieval stages) from the naive pipeline?
5. How would you break down the end-to-end latency budget across a production RAG pipeline?
6. Beyond cost, what criteria (freshness, query frequency, latency budget, reasoning requirements) belong in a RAG-vs-long-context decision checklist?

**Module 02 — Document Processing, Chunking & Lifecycle Management (7)**
7. Fixed-size vs. recursive vs. semantic chunking — what are the real trade-offs?
8. What is Late Chunking, and why does it preserve cross-chunk semantics that standard chunk-then-embed pipelines structurally lose?
9. Given a document's token length, chunk size, and overlap, how would you calculate the resulting chunk count and storage overhead?
10. What is parent-child (small-to-big) chunking, and when does it outperform flat chunking?
11. Walk through a document's full production lifecycle: added → indexed → edited/versioned → re-indexed → detected stale → deleted/tombstoned.
12. How would you guarantee that deleted or updated documents are never returned by retrieval — what consistency gap can exist between a source-corpus change and the live index, and how do you close it?
13. What metadata would you extract and enrich during ingestion to enable filtered retrieval later?

**Module 03 — Embeddings for Retrieval & Vector Representations (6)**
14. When do cosine similarity and raw dot product diverge in ranking, and why does that matter for a real retrieval system?
15. Bi-encoders vs. cross-encoders — what's the fundamental architectural difference, and what does it cost you at query time?
16. What is Matryoshka Representation Learning, and what does truncating an embedding actually trade away?
17. When would you fine-tune an embedding model for domain adaptation instead of using an off-the-shelf model?
18. What causes embedding drift, and how would you detect it in production?
19. How does embedding dimensionality trade off against index storage and query latency?

**Module 04 — Vector Indexing, ANN Search & Vector Database Internals (8)**
20. Walk through HNSW's graph-based search mechanics — what do `M`, `efConstruction`, and `efSearch` each control?
21. Given `nlist` and `nprobe` for an IVF index, how do you reason about the recall/latency trade-off?
22. Given a 768-dim embedding split into `m` subvectors at `bits`-per-code, how do you calculate IVF-PQ's compression ratio and bytes/vector?
23. When is exact (brute-force) search still the right production choice over any ANN structure?
24. If you double `nprobe`, what happens to p99 latency and recall — and where are the diminishing returns?
25. How do production vector databases (Pinecone, Weaviate, Qdrant, Milvus) handle sharding and hybrid metadata filtering?
26. Why might an IVF-PQ index still fall short of exact-search recall even when `nprobe` scans 100% of clusters?
27. How would you choose between HNSW and IVF-PQ for a given corpus size and memory budget?

**Module 05 — Hybrid Retrieval & Reranking (6)**
28. How does Reciprocal Rank Fusion combine a BM25 ranking and a dense ranking into one score?
29. Walk through the candidate-set sizing funnel: initial Top-K retrieval → RRF fusion → rerank to Top-N → final Top-M context.
30. When is cross-encoder reranking not worth its added latency?
31. What is ColBERT-style late-interaction retrieval, and how does it sit between bi-encoder and cross-encoder in the cost/quality spectrum?
32. Why can RRF-fused retrieval outperform both of its input signals, even when one signal is clearly the stronger of the two alone?
33. How would you size K, N, and M in a retrieve-then-rerank pipeline under a tight latency budget?

**Module 06 — Query Understanding, Transformation & Optimization (6)**
34. What is HyDE, and why does embedding a hypothetical generated document sometimes retrieve better than embedding the raw query?
35. How does query decomposition help with multi-hop questions — and is that benefit guaranteed?
36. What is semantic routing across multiple retrieval sources or indexes?
37. When should you avoid query transformation techniques like HyDE or query rewriting altogether?
38. What failure modes can query rewriting introduce if the rewritten query drifts from user intent?
39. How does multi-query retrieval differ from query decomposition?

**Module 07 — GraphRAG & Structured/Knowledge-Graph Retrieval (6)**
40. How is a knowledge graph constructed from unstructured text for GraphRAG?
41. What's the difference between local (entity-level) and global (corpus-summary) queries in GraphRAG, and how does community detection support the global case?
42. When should you not reach for GraphRAG?
43. Why does genuine cross-document, multi-hop GraphRAG value depend on entity resolution — and what happens to the graph without it?
44. How does graph construction cost and staleness compare to maintaining a flat vector index?
45. When does graph-based retrieval concretely outperform flat vector/hybrid retrieval?

**Module 08 — Agentic RAG & Self-Correcting Retrieval Loops (6)**
46. What is Agentic RAG, and how does it differ from a fixed retrieve-then-generate pipeline?
47. How do Self-RAG and Corrective RAG (CRAG) perform self-critique and trigger re-retrieval?
48. When should you not reach for Agentic RAG?
49. How would you design loop termination and budget guards for an iterative retrieval agent?
50. How does multi-agent retrieval (specialized retriever agents per source/domain) differ from a single agentic loop?
51. If a self-correction loop's remediation doesn't fix a diagnosed retrieval failure, what does that tell you about matching the remedy to the failure type?

**Module 09 — RAG Evaluation, Debugging & Production Hardening (8, including 2 cross-module synthesis questions)**
52. How do Recall@k, MRR, and NDCG differ, and what does each one actually tell you that the others don't?
53. What do RAGAS-style generation metrics (faithfulness, answer relevancy, context precision/recall) each catch that retrieval metrics can't?
54. Walk through a systematic stage-isolation methodology for debugging a bad RAG answer, from query through generation.
55. Given a retrieval failure, how do you distinguish a "ranking problem" (the right document exists but is ranked too low) from a "representation problem" (the embedding itself is far from the query)?
56. What retrieval-observability telemetry would you log in production, and why does each field matter for debugging?
57. How would you prevent prompt injection via retrieved content and cross-tenant data leakage in a production RAG system?
58. *(synthesis)* How would you design the full production RAG pipeline end-to-end — chunking → embedding → indexing → hybrid retrieval → reranking → generation → evaluation/observability — and where would you deliberately cut corners for an MVP vs. a mature system?
59. *(synthesis)* A production RAG system's answer quality has degraded over the last month with no code changes. Walk through your systematic debugging approach from symptom to root cause.

---

## Status: Written — Awaiting Compilation & Final Review

All 59 questions written to `modules/10_advanced_rag_interview_questions.md` in 5 batches (Modules 01-02, 03-04, 05-06, 07-08, 09 + Final Revision Sheet), each grounded in this topic's own study modules' formulas/hand-calcs and, where genuinely valuable, real measured results from Track 2's 6 executed notebooks (Late Chunking's +0.25 similarity gain, RRF beating both single-signal baselines, IVF-PQ's undertrained-codebook recall ceiling, HyDE's +5-point recall gain, GraphRAG's cross-document connectivity gap, and the self-correction remedy-mismatch finding).

Mandatory structural compliance check (`interview_qa_generator/SKILL.md` Section 5) passed: 59/59 question blocks with all 10 required sub-headings each, no derivation chains found, Final Revision Sheet present with all 3 required subsections (Quick-Recall Takeaways Table, Essential Formula Cheat Sheet, Top Follow-up Q&As).

`helpers/compile_advanced_rag.py` updated with a second `compile_document()` call producing the standalone `advanced_rag_interview_cheatsheet.html`/`.pdf` (119 pages, 1,352,863 bytes). Compilation succeeded on the first attempt. Verified: 0 `file:///` leaks, 0 unresolved `MATHPLACEHOLDER` leaks, 118 `q-card` divs (59 questions × 2 follow-up Q&As each, matching exactly), 59 `follow-up-section` divs (one per question, matching exactly).

Note: the master study guide recompile in the same `main()` run failed with a `PermissionError` on the existing `advanced_rag_master_study_guide.pdf` (locked by another process, unrelated to this track's changes — the master guide's own content was not modified). Worked around by invoking `compile_document()` directly for only the new Q&A file, bypassing the locked master PDF entirely. The master guide itself remains unchanged and does not need recompiling for this track.

Awaiting the user's final review per `AGENTS.md`'s Track 3 checkpoint before considering this track complete.
