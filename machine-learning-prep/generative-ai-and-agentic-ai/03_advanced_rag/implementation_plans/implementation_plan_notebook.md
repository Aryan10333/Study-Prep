# Implementation Plan: Track 2 — Companion Notebooks (`03_advanced_rag`)

Scope: this plan covers only the `notebooks/` companion notebooks, per `notebook_generator/SKILL.md`. Track 1 (9 study-guide modules) is complete and signed off; Track 3 (Interview Q&A) is separate and not covered here.

---

## 0. Environment Reality Check (affects every notebook below)

| Check | Result |
|---|---|
| GPU present | **Yes** — NVIDIA GeForce RTX 4060 Laptop GPU, 8.6GB VRAM |
| PyTorch build | **CUDA-enabled** (`2.13.0+cu126`), already installed from the previous topic's Track 2 work |
| `sentence-transformers` | Installed (5.6.1) — covers bi-encoder, cross-encoder, and Matryoshka-capable models |
| `faiss` | Installed (1.14.3) — real HNSW, IVF, and IVF-PQ index construction (Module 04's exact ANN structures) |
| `rank_bm25` | Installed — real sparse BM25 retrieval for hybrid fusion (Module 05) |
| `networkx` | Installed (3.6.1) — real graph construction and community detection for GraphRAG (Module 07) |
| `transformers` / `datasets` | Installed |
| `HF_TOKEN` / `OPENAI_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` | All set in `.env` |
| `chromadb` / `qdrant_client` | **Not installed, staying uninstalled** — `faiss` alone covers every quantitative claim (recall/latency/compression) the Track 1 modules make, and `scifact` has no rich per-document metadata to filter on anyway, so a metadata-filtering demo would be synthetic regardless of which library ran it |
| `einops` | **Installed during Track 2 planning** — a real, previously-undiscovered missing dependency for `nomic-ai/nomic-embed-text-v1.5`'s custom modeling code, caught by verifying the model loads *before* committing the plan around it (see Section 2 note below) |

Unlike the previous topic, no new heavy installs (CUDA PyTorch, peft/trl) were needed — the environment was already almost fully provisioned for this topic's notebooks, modulo the one missing `einops` dependency caught during verification.

**Resource discipline across all 6 notebooks**: never hold more than one embedding model, one reranker, and one LLM-API client's worth of GPU memory live at once — explicitly `del` a model and call `torch.cuda.empty_cache()` before loading the next one within a notebook, rather than accumulating multiple loaded models across cells. On an 8.6GB laptop GPU this isn't optional convenience, it's what keeps every notebook from risking an OOM as the topic's models (bi-encoder, cross-encoder, and any local models) stack up across a single session.

---

## 1. Notebook List & Target File Paths

| # | File path | Maps to Module(s) |
|---|---|---|
| 01 | `notebooks/01_document_processing_chunking_and_lifecycle.ipynb` | 02 (Document Processing, Chunking & Lifecycle) |
| 02 | `notebooks/02_embeddings_for_retrieval.ipynb` | 03 (Embeddings for Retrieval) |
| 03 | `notebooks/03_vector_indexing_and_ann_search.ipynb` | 04 (Vector Indexing, ANN Search) |
| 04 | `notebooks/04_hybrid_retrieval_and_reranking.ipynb` | 05 (Hybrid Retrieval & Reranking) |
| 05 | `notebooks/05_query_transformation_and_graphrag.ipynb` | 06 + 07 (Query Understanding, GraphRAG) |
| 06 | `notebooks/06_agentic_rag_and_evaluation.ipynb` | 08 + 09 (Agentic RAG, Evaluation/Debugging) |

**Module 01 (RAG Fundamentals) is intentionally excluded** — its content is a cost-model calculator already fully demonstrated as runnable code in the Track 1 module itself, with no additional real-world engineering pipeline to build; giving it a dedicated notebook would just re-run the same formula, not add a new real system to build.

Each notebook is built and executed **one at a time, sequentially**, per `notebook_generator/SKILL.md` — not in a batch run.

---

## 2. Real-World Datasets & APIs Per Notebook

| # | Datasets / APIs | Why this one |
|---|---|---|
| 01 | `Salesforce/wikitext` (`wikitext-2-raw-v1`, real Wikipedia article text, already used successfully in the previous topic) + `nomic-ai/nomic-embed-text-v1.5` (real long-context embedding model, 8192-token context) | A real narrative Wikipedia article has genuine cross-references (pronouns, "the company," earlier-section callbacks) that make Late Chunking's benefit concretely demonstrable — the model was specifically chosen because it supports both the long context Late Chunking needs *and* the Matryoshka truncation Notebook 02 needs, avoiding a mismatched "claims a property the model doesn't have" pitfall. |
| 02 | Same `nomic-ai/nomic-embed-text-v1.5` (bi-encoder + real Matryoshka truncation) + `cross-encoder/ms-marco-MiniLM-L-6-v2` (real, widely-used cross-encoder) + a small real query/passage set from `BeIR/scifact` (see below) | Real bi-encoder vs. cross-encoder timing/quality comparison, and real (not simulated) Matryoshka truncation quality degradation curve. |
| 03 | `BeIR/scifact` (real scientific-claim retrieval benchmark: real corpus + real queries + real relevance judgments, small enough for a laptop GPU) | Provides a real corpus large enough for a genuine `faiss` HNSW/IVF/PQ index, with real relevance judgments so recall isn't just assumed — it's measured against real ground truth. This is also the notebook that can produce *real* recall-vs-latency and compression data, as a direct real-experiment counterpart to Track 1's clearly-labeled illustrative Module 04 plots. |
| 04 | Same `BeIR/scifact` corpus/queries (real BM25 + real dense retrieval + real cross-encoder rerank on the same real ground truth) | Keeps hybrid fusion evaluable against the same real relevance judgments as Notebook 03, so RRF's real effect on real Recall@k is directly measurable, not asserted. |
| 05 | Live Groq/OpenAI API calls (HyDE generation, query decomposition) + a small set of real Wikipedia paragraphs about real companies/executives (`Salesforce/wikitext`-derived or a few real article excerpts) for GraphRAG entity/relation extraction | Real LLM-driven query transformation and real LLM-driven entity/relation extraction — both genuinely require a live model call, not something fakeable locally. |
| 06 | Live Groq/OpenAI API calls (self-critique/relevance judgment in the agentic loop) + reuses `BeIR/scifact`'s real queries/relevance judgments for the evaluation half | Real self-correcting retrieval loop needs a real LLM judgment call to be genuine (not simulated); reusing Notebook 03/04's real ground truth keeps the Recall@k/MRR/NDCG numbers computed on real data, matching Module 09's hand-calc pattern at real scale. |

**Graceful API-failure handling (Notebooks 05 & 06):** every live Groq/OpenAI call is wrapped so a rate-limit, timeout, or outage doesn't crash the whole notebook — on failure, the cell prints a clearly labeled `[API UNAVAILABLE — FALLBACK]` message, falls back to a fixed, explicitly-labeled canned example for that one step, and the notebook continues executing the remaining cells. The fallback path is never silently indistinguishable from a real live response — every fallback-triggered output is visibly marked as such in both the printed output and the paired Output Explanation cell, consistent with Refinement 9's real-vs-illustrative labeling discipline.

Base embedding stack across notebooks: **`nomic-ai/nomic-embed-text-v1.5`** (bi-encoder, long-context, Matryoshka) + **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (reranking) — consistent models let numbers be compared notebook-to-notebook, matching the "one consistent base" pattern from the previous topic's Track 2.

**Pre-flight model capability verification (done before finalizing this plan, not assumed):** loaded `nomic-ai/nomic-embed-text-v1.5` directly on the real GPU and confirmed both required properties with real numbers, not documentation claims:
- **Long-context (Late Chunking, Notebook 01):** real `max_seq_length = 8192`; a genuinely long ~4,500-word test document tokenized to `4,502` real tokens and embedded in one pass with no silent truncation (`(1, 768)` output).
- **Matryoshka truncation (Notebook 02):** truncating a real 768-dim embedding to 128 dims and renormalizing produces a unit-norm vector (`norm = 1.0`) that **still correctly ranks a genuinely related document above an unrelated one** — `sim(query, related_doc) = 0.672 > sim(query, unrelated_doc) = 0.483` at 128 dims, mirroring the same correct ordering seen at the full 768 dims (`0.602 > 0.474`). This is real evidence the model's truncated embeddings remain meaningful, not just a documented claim taken on faith.

This verification also caught one real, previously-undiscovered issue: the model's custom modeling code requires `einops`, which was not installed — fixed by installing it, documented in Section 0's environment table above.

---

## 3. Engineering Pipelines, Hardware Assertions & Profiling Steps

- **Notebook 01**: Real chunking pipeline (fixed-size, recursive, semantic-boundary via embedding similarity drop, parent-child, and Late Chunking via `nomic-embed-text-v1.5`'s long-context pass + span pooling) run against a real Wikitext article; validates Module 02's chunk-count formula against the real chunker's actual output count. Real document-lifecycle tracker (add/edit/stale-detect/delete) demonstrated against the same real chunked document, reusing Module 02's content-hash logic.
- **Notebook 02**: Real bi-encoder vs. cross-encoder timing comparison (`time.perf_counter()`) over a real query set; real cosine-vs-dot-product ranking divergence check on real (not toy 4-dim) embeddings; real Matryoshka truncation sweep (768 → 256 → 128 → 64 dims) measuring real retrieval quality degradation.
- **Notebook 03**: Build real `faiss.IndexHNSWFlat` and real `faiss.IndexIVFPQ` over `BeIR/scifact`'s real corpus. **Sweeps are deliberately kept to a small, representative set of settings, not exhaustive grid search** — e.g. `efSearch` ∈ {10, 50, 200} and `nprobe` ∈ {a low/medium/high fraction of `nlist`} — enough points to show the real recall-vs-latency-vs-memory trade-off shape without turning the notebook into a benchmarking harness. Measures real recall-vs-latency (`time.perf_counter()` per query) and real compression ratio (`index.sa_code_size()` or raw memory footprint) against the real ground-truth relevance judgments — the real-experiment counterpart to Track 1's illustrative Module 04 plots.
- **Notebook 04**: Real `rank_bm25` BM25 index + real dense `faiss` retrieval over the same real corpus; real RRF fusion (Module 05's exact formula) of the two real ranked lists; real cross-encoder rerank of the fused Top-N. **Explicitly measures and reports quality (Recall@k, NDCG) and latency at every funnel stage** — Dense/BM25 (individually) → RRF fusion → Reranking → Final Top-M — as a single comparison table/plot per query set, not just an end-to-end before/after number, so each stage's real marginal contribution to quality and cost is visible on its own.
- **Notebook 05**: Real HyDE (live LLM generates a hypothetical answer, embedded and compared against real query-embedding retrieval on the same real query); real query decomposition (live LLM splits a genuine multi-hop question, retrieves per sub-question); real GraphRAG entity/relation extraction (live LLM call producing real structured triples from real text), built into a real `networkx` graph with a real local traversal query and a real `networkx`-based community detection pass for a global-query demonstration. All live-LLM steps wrapped with the fallback handling described above.
- **Notebook 06**: Real self-correcting retrieval loop (live LLM relevance judgment triggers real re-retrieval with a reformulated query, bounded by a real `max_retries` termination guard — directly reusing Module 08's reference implementation pattern), with fallback handling on the live-LLM judgment call; real Recall@k/MRR/NDCG computed over `BeIR/scifact`'s real relevance judgments at real system scale (not the Module 09 hand-calc's 5-document toy example). **Deterministic failure-mode experiment**: deliberately construct one *known* failure with a *predicted* expected diagnosis before running the check — e.g., manually truncate a chunk mid-fact so the correct answer is provably split across a boundary — then run Module 09's stage-isolation methodology against it and verify the diagnosis lands on the predicted stage (Chunking), not just "some stage fails." This is a pass/fail assertion on the methodology itself, not an open-ended exploration.

All notebooks include explicit `assert` statements on tensor/index shapes and numeric bounds, and headless execution uses `matplotlib.use('Agg')` with `%matplotlib inline` + `plt.show()` for any inline plots, per the skill.

**Real vs. illustrative labeling discipline**: every plot, table, and printed result in every notebook is generated from real execution on real data by construction (per this skill's production-scope rule) — but any place a *toy or synthetic* input is deliberately used (e.g., Notebook 06's deliberately-corrupted chunk for the failure-mode test, or an API-fallback canned example) is explicitly labeled as such in both the code's printed output and the paired Output Explanation cell, so a reader can never mistake a deliberately-constructed test case for a naturally-occurring real-world result.

---

## 4. Design Decisions (Confirmed)

All four original open questions are now resolved by explicit user sign-off:

1. **`chromadb`: skipped.** `faiss` alone covers every quantitative claim the Track 1 modules make, and `scifact` has no rich per-document metadata to filter on anyway — a metadata-filtering demo would be synthetic regardless of which library ran it.
2. **`BeIR/scifact` confirmed** as the shared real corpus/queries/relevance-judgments backbone across Notebooks 03, 04, and 06, for consistent, directly-comparable numbers across the ANN, hybrid-retrieval, and evaluation notebooks.
3. **GraphRAG scope confirmed unchanged**: `networkx`'s built-in `greedy_modularity_communities` for the global-query community-detection step, no hand-rolled algorithm — matches Module 07's theory scope exactly.
4. **Live API calls confirmed** kept to a small, fixed number of examples per notebook (not a sweep), now additionally required to fail gracefully with a clearly-labeled fallback rather than crashing the notebook (Section 3 above).

Six additional refinements folded into the plan above: graceful API-failure fallback (Section 2/3), pre-flight embedding-model capability verification (Section 2 — already run, caught a real missing `einops` dependency), focused/representative rather than exhaustive ANN parameter sweeps (Notebook 03), explicit per-stage funnel metrics (Notebook 04), a deterministic pass/fail failure-injection test for the debugging methodology (Notebook 06), and an explicit real-vs-illustrative labeling discipline for any deliberately-constructed test case (Section 3). No additional notebooks needed — the six-notebook structure stands.

---

## Status: Complete

All 6 notebooks built, executed, and verified (0 unexecuted cells, 0 errors, 0 pending explanation placeholders across every notebook):

1. `01_document_processing_chunking_and_lifecycle.ipynb` — real chunking strategies, real Late Chunking (+0.2517 similarity gain), real document lifecycle tracking.
2. `02_embeddings_for_retrieval.ipynb` — real bi- vs. cross-encoder timing, real cosine/dot-product divergence, real Matryoshka truncation sweep (zero quality loss to 128 dims).
3. `03_vector_indexing_and_ann_search.ipynb` — real HNSW and IVF-PQ sweeps on the full 5,183-doc corpus; HNSW matches the exact-recall ceiling at 3.7x lower latency; IVF-PQ's real recall ceiling (0.7188) traced to a genuine undertrained-PQ-centroids limitation.
4. `04_hybrid_retrieval_and_reranking.ipynb` — real BM25 + dense + RRF fusion + cross-encoder reranking funnel, monotonically improving Recall@10 (0.7557 -> 0.8173 -> 0.8207 -> 0.8313).
5. `05_query_transformation_and_graphrag.ipynb` — real live-LLM HyDE (+5pt recall), real query decomposition (honest non-improvement on its test case), real GraphRAG triple extraction with an honestly-surfaced cross-document connectivity limitation.
6. `06_agentic_rag_and_evaluation.ipynb` — real system-scale Recall@k/MRR/NDCG, real stage-isolation debugging (ranking vs. representation problem), real self-correcting retrieval loop with deterministic failure-mode injection (diagnosis confirmed by a live critic; remediation honestly did not fix this specific failure, itself an informative negative result).

Track 2 complete.
