# Implementation Plan: Track 1 — Study Notes Modules (`03_advanced_rag`)

Scope: this plan covers **only** the 9 theory modules in `modules/` and their compiled `advanced_rag_master_study_guide.{html,pdf}`, per `study_guide_generator/SKILL.md`. Notebooks (Track 2) and the Interview Q&A module (Track 3) are separate tracks with their own implementation-plan checkpoints, planned after this track is signed off and written.

---

## 1. Module List & Target File Paths

| # | File path | Title |
|---|---|---|
| 01 | `modules/01_rag_fundamentals_and_architecture.md` | RAG Fundamentals & Production Architecture Patterns |
| 02 | `modules/02_document_processing_chunking_and_lifecycle.md` | Document Processing, Chunking & Document Lifecycle Management |
| 03 | `modules/03_embeddings_for_retrieval.md` | Embeddings for Retrieval & Vector Representations |
| 04 | `modules/04_vector_indexing_and_ann_search.md` | Vector Indexing, ANN Search & Vector Database Internals |
| 05 | `modules/05_hybrid_retrieval_and_reranking.md` | Hybrid Retrieval & Reranking |
| 06 | `modules/06_query_understanding_and_optimization.md` | Query Understanding, Transformation & Optimization |
| 07 | `modules/07_graphrag_and_structured_retrieval.md` | GraphRAG & Structured/Knowledge-Graph Retrieval |
| 08 | `modules/08_agentic_rag.md` | Agentic RAG & Self-Correcting Retrieval Loops |
| 09 | `modules/09_rag_evaluation_debugging_and_production.md` | RAG Evaluation, Debugging & Production Hardening |

Compiled deliverables (Track 1 only, via a new `helpers/compile_advanced_rag.py` built directly from the hardened `pdf_compiler` pattern — portable `BASE_DIR`, base64 image embedding, `check=True` + retry loop, and the now-standard `--virtual-time-budget` scaled up for any module with heavy KaTeX density): `advanced_rag_master_study_guide.html` / `.pdf`.

---

## 2. Formulas to Retain (with Hand-Calc Plan) vs. Prose-Only

Per `study_guide_generator/SKILL.md`'s Formula Selection Constraint: only core, frequently-asked concepts get a formula block + step-by-step hand calculation on small numbers; everything else is explained in prose/intuition only, with comparison tables where useful.

### Module 01 — RAG Fundamentals & Production Architecture
- **Core (formula + hand calc):** The **Long Context vs. RAG cost crossover** — total cost of re-processing a full corpus as context on every query ($\text{Cost}_{\text{LC}} = N_{\text{queries}} \times N_{\text{tokens,corpus}} \times \text{price}_{\text{token}}$) vs. RAG's one-time embedding cost plus a small per-query retrieval+generation cost ($\text{Cost}_{\text{RAG}} = \text{Cost}_{\text{embed,once}} + N_{\text{queries}} \times (\text{price}_{\text{retrieval}} + N_{\text{tokens,context}} \times \text{price}_{\text{token}})$). Hand calc: a concrete small corpus (e.g., 50K tokens) and query volume, computing the break-even query count where RAG becomes cheaper. The quantified crossover is then paired with a **full decision-criteria checklist** (corpus size, freshness/update frequency, query frequency/volume, latency budget, cost, required retrieval accuracy, and reasoning requirements — single-hop vs. multi-hop/multi-document synthesis) as a GFM comparison table, so the module gives both the number and the qualitative judgment calls a cost formula alone can't capture (e.g., a cheap-by-the-formula long-context approach can still lose on latency or on multi-hop reasoning quality).
- **Prose-only:** The naive RAG pipeline and its failure modes (described procedurally with the pipeline diagram), Advanced RAG's pre-retrieval/retrieval/post-retrieval staging (architectural, not mathematical).

### Module 02 — Document Processing, Chunking & Document Lifecycle
- **Core (formula + hand calc):** Chunk count from document length, chunk size, and overlap: $N_{\text{chunks}} = \left\lceil \dfrac{L_{\text{doc}} - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \right\rceil$. Hand calc: a concrete document token length with a chosen chunk size/overlap, computing the resulting chunk count and total stored-token overhead from overlap.
- **Prose-only:** Fixed-size vs. recursive vs. semantic chunking (comparison table), parent-child chunking (architecture description). **Late Chunking** as a first-class technique — pooling-after-embedding mechanism described conceptually (no formula, since the mechanism is "embed once with full-document context, pool per span afterward," not a derivable quantity), with explicit emphasis on the *why*: a chunk embedded in isolation has no representation of the text before/after its boundary, while a late-chunked vector is pooled from token representations that already attended to the full surrounding document, directly preserving cross-chunk semantic linkage standard pre-chunk-then-embed pipelines lose. **Document lifecycle** (incremental updates, deletion/tombstoning, re-indexing, versioning, stale-embedding detection) — procedural, described with a lifecycle-state diagram, *and* one concrete worked example tracing a single document through every lifecycle stage end-to-end (added → incrementally indexed → edited/versioned → re-indexed → detected stale if missed → deleted/tombstoned), so the concepts are demonstrated in sequence rather than only defined individually.

### Module 03 — Embeddings for Retrieval & Vector Representations
- **Core (formula + hand calc):** Cosine similarity and dot-product similarity between two embedding vectors — explicitly named in the skill as a "fundamental vector similarity metric." Hand calc: two tiny toy embedding vectors (e.g., $d=4$), computing cosine similarity by hand and contrasting it with the unnormalized dot product to show why normalization matters for retrieval ranking.
- **Prose-only:** Bi-encoder vs. cross-encoder architecture (conceptual + diagram), Matryoshka Representation Learning / embedding truncation (intuition-level — the "why truncation still works" explanation, not the training objective's derivation), embedding fine-tuning for domain adaptation (procedural).

### Module 04 — Vector Indexing, ANN Search & Vector Database Internals
- **Core (formula + hand calc):**
  - Distance metrics (cosine, dot product, Euclidean/L2) already planned per the original scope — hand calc on a tiny 2D/3D toy vector pair for each metric, showing they can rank differently.
  - **IVF `nlist`/`nprobe` recall-latency trade-off**: fraction of the index actually scanned per query, $\text{fraction\_scanned} = n_{\text{probe}} / n_{\text{list}}$, with latency scaling roughly linearly in that fraction. Hand calc: a concrete $n_{\text{list}}=100$, $n_{\text{probe}}=8$ example, computing the scanned fraction and the resulting rough speedup vs. brute-force.
  - **IVF-PQ compression ratio**: $\text{bytes}_{\text{raw}} = d \times 4$ vs. $\text{bytes}_{\text{PQ}} = m \times \frac{\log_2(k)}{8}$ for $m$ subvectors and a $k$-entry codebook per subvector. Hand calc: a concrete $d=768$ embedding compressed with $m=96$ subvectors and $k=256$ centroids (1 byte/subvector), computing the exact compression ratio.
- **Worked numeric intuition, no formula block (Concept Simplification applies — HNSW's search cost isn't a single closed-form quantity):** HNSW graph construction/search mechanics — a concrete small example (e.g., "$M=16$ means each node keeps roughly 16 neighbor edges per layer; `efSearch=50` explores up to 50 candidates during query time") illustrating how $M$/`efConstruction`/`efSearch` trade off graph memory, build time, and recall, without pretending there's a single derivable formula the way there is for IVF/PQ.
- **Closing parameter-to-effect drill (GFM table, not a new formula):** A single consolidated table mapping `nlist`, `nprobe`, $M$, `efConstruction`, and `efSearch` each against "what it controls" and "effect of increasing it on recall / latency / memory" — deliberately placed at the end of the module as an interview-drill artifact distinct from the individual hand-calcs above, since the ask specifically was reasoning fluency ("if I double `nprobe`, what happens to p99 latency and recall") rather than another isolated definition.
- **Prose-only:** Vector database architecture (Pinecone, Weaviate, Qdrant, Milvus) — a comparison table of sharding/filtering/hybrid-query support, not vendor pricing or deep internals of any one product.

### Module 05 — Hybrid Retrieval & Reranking
- **Core (formula + hand calc):** Reciprocal Rank Fusion (RRF): $\text{RRF}(d) = \sum_{i} \dfrac{1}{k + \text{rank}_i(d)}$ across the sparse (BM25) and dense retrieval result lists. Hand calc: a small toy example with 4-5 documents ranked differently by BM25 vs. vector search, computing each document's fused RRF score and the resulting merged ranking by hand — the single most commonly-asked hybrid-search formula in interviews.
- **Core (worked funnel example, not a formula):** The **candidate-set sizing funnel** — Initial retrieval Top-K → RRF fusion → rerank down to Top-N → final assembled context Top-M — walked through with one concrete numeric example (e.g., $K=50 \to N=10 \to M=5$), explicitly reasoning at each narrowing step about why a wider stage improves the odds a truly relevant document survives to the next stage, paired with the direct latency/cost cost of fetching/scoring/feeding more documents at that stage.
- **Prose-only:** Cross-encoder reranking (architecture + why it's a second, more expensive stage — including an explicit **"when reranking isn't worth it"** subsection: small candidate sets, tight latency budgets, or a first-stage retriever already precise enough that the added pass buys negligible precision), ColBERT-style late-interaction retrieval (conceptual MaxSim intuition, not the full scoring derivation), multi-stage retrieve-then-rerank pipeline design (procedural, with a pipeline diagram).

### Module 06 — Query Understanding, Transformation & Optimization
- **No core formulas** — this module is architectural/procedural throughout (query expansion/rewriting, HyDE, query decomposition for multi-hop questions, multi-query retrieval, semantic routing), matching the pattern of `02_llm_training_foundations`' Module 06 (DPO variants survey) and Module 08 (failure-mode catalog): a technique-vs-when-to-use-it comparison table rather than individual formula blocks, since none of these are single-formula concepts. Each technique's entry in that table includes an explicit **"when NOT to use this"** column/note — e.g., HyDE/query rewriting add a full extra LLM call to the critical path, rarely justified when queries already retrieve well unmodified.

### Module 07 — GraphRAG & Structured/Knowledge-Graph Retrieval
- **No core formulas** — per "Strictly Prohibit Academic Formalisms," community-detection algorithm internals (e.g., Louvain modularity optimization) stay out of scope entirely; knowledge graph construction (entity/relation extraction) and graph-based retrieval/summarization are described architecturally with a diagram, and the local (entity-level) vs. global (corpus-summary) query trade-off is a comparison table, not math. Closes with an explicit **"when NOT to use GraphRAG"** subsection — small/low-update corpora and single-hop-lookup-dominated workloads are a poor fit given graph construction/maintenance cost, and are better served by Module 05's flat hybrid retrieval.

### Module 08 — Agentic RAG & Self-Correcting Retrieval Loops
- **No core formulas** — Agentic RAG's tool-calling loop, Self-RAG/Corrective RAG's self-critique-and-re-retrieve pattern, and multi-agent retrieval are all architectural/procedural, described with a loop diagram and a technique comparison table (matching how `02_llm_training_foundations` treated GRPO's *concept* prose-side vs. its one retained formula — here, none of Module 08's techniques reduce to a single core formula). Closes with an explicit **"when NOT to use Agentic RAG"** subsection (queries a single well-tuned retrieve-then-generate pass already answers correctly don't justify the loop's unbounded latency/cost). **Scope note:** content stays limited to retrieval-specific agent-loop mechanics — general agent architecture, MCP protocol internals, and multi-agent orchestration patterns are cross-referenced from the dedicated AI Agents topic, not re-taught here, mirroring Module 09's own scope-discipline treatment.

### Module 09 — RAG Evaluation, Debugging & Production Hardening
- **Core (formula + hand calc):** Retrieval metrics — Recall@k, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG) — explicitly named in the skill as "core evaluation statistics." Hand calc: a small toy retrieved-ranking list against a known set of relevant documents, computing Recall@k, MRR, and NDCG by hand for that one query (mirroring how `00_nlp_fundamentals` Module 08 treated BLEU/ROUGE/PPL).
- **Prose-only:** RAGAS-style generation metrics (faithfulness, answer relevancy, context precision/recall — LLM-judge-based, no closed-form formula), the **RAG debugging methodology** (query → chunking → embedding → retrieval → reranking → context → generation stage isolation, presented as a diagnostic flowchart + procedure, not math), **retrieval observability** (telemetry design — what to log and why: the query itself, retrieved document/chunk IDs, retrieval scores, reranker scores, Top-K, retrieval latency, and empty/low-quality-retrieval rates, procedural), semantic caching, RAG-specific security, multi-tenant isolation, and production scaling (all comparison-table/procedural, consistent with the *Scope Discipline* note in the syllabus keeping this module RAG-specific rather than re-deriving general LLM-eval or LLMOps math).

---

## 3. Visual Diagrams & Plots

Matplotlib PNGs go in `plots/`, generated via a new `helpers/generate_advanced_rag_plots.py` (Agg backend, `.venv` execution, following the label-clearance rules in `study_guide_generator/SKILL.md` Section 6 point 7 for any box-and-arrow diagrams). Flowcharts/architecture diagrams use inline responsive SVG directly in the module markdown per the same skill's Section 3 (no Mermaid/ASCII), each rendered and visually checked for label/arrow collisions before the module is considered complete (per the skill's explicit "actually render and view it" rule).

| Module | Visual | Type |
|---|---|---|
| 01 | Naive RAG pipeline flow (ingest → chunk → embed → index → retrieve → generate), annotated with pre-retrieval/retrieval/post-retrieval stage groupings | Inline SVG |
| 01 | Long Context vs. RAG cost crossover chart: total cost vs. query volume, with the break-even point marked | Matplotlib PNG |
| 02 | Chunking strategies comparison diagram: fixed-size, recursive, semantic, parent-child, and Late Chunking shown side-by-side against the same source text | Inline SVG |
| 02 | Document lifecycle state-flow diagram: add → update/version → stale-detection → delete/tombstone | Inline SVG |
| 03 | Bi-encoder vs. cross-encoder architecture comparison (separate vs. joint encoding, with the resulting latency/quality trade-off labeled) | Inline SVG |
| 03 | Embedding dimensionality vs. index storage / query latency trade-off chart | Matplotlib PNG |
| 04 | Small toy HNSW multi-layer graph illustration (layers, entry point, greedy search path) | Inline SVG |
| 04 | Recall vs. latency curve across ANN configurations (varying `efSearch`/`nprobe`) | Matplotlib PNG |
| 04 | PQ compression ratio bar chart: raw float storage vs. PQ-compressed storage for the Module 04 hand-calc's concrete example | Matplotlib PNG |
| 05 | Hybrid retrieval + reranking pipeline: BM25 ranks + vector ranks → RRF fusion → cross-encoder rerank → top-N | Inline SVG |
| 05 | Candidate-set sizing funnel: Top-K → RRF → Top-N → Top-M, narrowing left-to-right with the Module 05 hand-calc's concrete numbers labeled at each stage | Inline SVG |
| 06 | Query transformation techniques diagram: original query branching into HyDE / decomposition / multi-query / routing paths | Inline SVG |
| 07 | Knowledge graph construction + local vs. global query routing diagram | Inline SVG |
| 08 | Agentic RAG / Self-RAG / Corrective RAG loop diagram (retrieve → generate → self-critique → re-retrieve, with an explicit loop-termination guard labeled) | Inline SVG |
| 09 | **RAG debugging stage-isolation flowchart**: query → chunking → embedding → retrieval → reranking → context → generation, each stage paired with its diagnostic check | Inline SVG |
| 09 | Illustrative Recall@k / MRR / NDCG comparison bar chart for the Module 09 hand-calc's toy example | Matplotlib PNG |

All plots are illustrative of a mathematical relationship or architecture (consistent with existing topics' plots) — none present fabricated numbers as if they were real benchmark results from a specific named vector database or embedding model.

---

## 4. Open Design Questions / Dependencies

1. **Cross-referencing, not duplicating:** Modules will link back to `00_nlp_fundamentals` for BM25 (Module 05) and classical embeddings (Module 03's boundary note), rather than re-deriving them — same pattern already used and now corrected in the syllabus (BM25 is Module 05's baseline, not Module 04's).
2. **GraphRAG depth check:** Per the "Strictly Prohibit Academic Formalisms" rule, Module 07 stays at the architectural/conceptual level for community detection (no Louvain modularity math) — flagging this explicitly so it's easy to push back on if more algorithmic depth is wanted.
3. **Vector database vendor coverage:** Module 04's Pinecone/Weaviate/Qdrant/Milvus comparison stays at the architecture-pattern level (sharding, filtering, hybrid queries) — not vendor-specific pricing, API syntax, or deep internals of any single product, consistent with this repo's general preference for durable concepts over tool-specific trivia.
4. **Module 09 debugging methodology depth:** The stage-isolation methodology (query → chunking → embedding → retrieval → reranking → context → generation) is planned as a procedural flowchart + diagnostic-check-per-stage description at study-guide depth; concrete runnable debugging code/notebooks are deferred to Track 2, not duplicated here.
5. **Compiler script:** Plan assumes a new `helpers/compile_advanced_rag.py`, written directly from the already-hardened pattern (portable paths, base64 images, retry logic, and a KaTeX-density-aware `--virtual-time-budget`) — no open question, just noting it avoids repeating known mistakes from earlier topics.
6. **Notebooks and Q&A are out of scope for this plan** and will each get their own implementation-plan checkpoint once Track 1 is complete, per `notebook_generator/SKILL.md` and `interview_qa_generator/SKILL.md`.

---

## Status: Complete

Signed off across two revision passes (see `README.md`'s Status note for the full point-by-point list). All 9 modules written to `modules/`, each hand-calc verified by direct execution (cost crossover, chunk-count/overlap, cosine vs. dot product, IVF fraction-scanned, PQ compression ratio, RRF fusion, Recall@k/MRR/NDCG), every code block executed and asserted correct, all 11 inline SVGs rendered and visually checked (3 real defects found — a text overflow past the canvas edge and two title/label collisions — and fixed), and 5 matplotlib plots generated via `helpers/generate_advanced_rag_plots.py` and visually verified. Structural compliance confirmed across all 9 modules (required section headings, code blocks). Compiled to `advanced_rag_master_study_guide.{html,pdf}` (67 pages) via `helpers/compile_advanced_rag.py`, verified with 0 `file:///` leaks and 0 unresolved math placeholders.

Along the way, PDF compilation hit the documented "orphaned msedge.exe processes" failure mode (`pdf_compiler/SKILL.md`) — repeated SVG-verification screenshots during this session had accumulated stuck Edge processes that silently blocked every subsequent headless print, including previously-working PDFs from other topics. Confirmed via a sanity re-test on an already-successful file, then cleared with the user's explicit sign-off (`taskkill /F /IM msedge.exe`), after which compilation succeeded immediately.
