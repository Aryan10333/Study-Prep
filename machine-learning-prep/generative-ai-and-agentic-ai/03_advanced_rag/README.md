# 03_advanced_rag: Advanced Retrieval-Augmented Generation (RAG) Syllabus

## 1. Context & Alignment
* **Profile Focus:** AI Engineer / Applied AI SDE-2/SDE-3 (Aryan Chandra). The candidate has already shipped production RAG (hybrid BM25 + vector retrieval, chunking, reranking, metadata extraction) at Jio Platforms, so this topic is scoped to go *beyond* that baseline — deep internals of indexing/retrieval trade-offs, GraphRAG, Agentic RAG, formal evaluation, and the production hardening (caching, security, multi-tenancy) that differentiates a senior-level RAG system from a first working prototype.
* **Interview Frequency:** Extremely High. RAG system design is one of the single most common deep-dives across AI Engineer, Applied AI Engineer, MLE, and AI Platform Engineer screens — both as a dedicated system-design round and as a recurring thread through behavioral/project-deep-dive rounds.
* **Core Goal:** Master the full production RAG lifecycle — document processing and chunking, embedding and vector indexing internals, hybrid retrieval and reranking, query understanding, GraphRAG, Agentic/self-correcting RAG loops, formal evaluation frameworks (RAGAS-style), and the caching/security/multi-tenancy/scalability concerns of running RAG as a real product, not a notebook demo.

## 2. Module Chapters & Conceptual Scope
These chapters outline the sequential topics to be covered:

- **Module 01: RAG Fundamentals & Production Architecture Patterns**
  - *Key Concepts:* Why RAG (vs. fine-tuning vs. long-context, and when to combine them), the naive RAG pipeline (ingest → chunk → embed → index → retrieve → generate) and its failure modes, Advanced RAG architecture patterns (pre-retrieval / retrieval / post-retrieval stages), and RAG vs. fine-tuning vs. long-context cost/latency/freshness trade-offs.
  - *System Bottlenecks & Focus:* End-to-end latency budget across the pipeline, staleness vs. cost of re-indexing, and the "garbage in, garbage out" failure surface of naive RAG.

- **Module 02: Document Processing & Chunking Strategies**
  - *Key Concepts:* Document parsing (PDF/HTML/structured docs, layout-aware parsing), fixed-size vs. recursive vs. semantic chunking, parent-child (small-to-big) chunking, sliding-window overlap, and metadata extraction/enrichment for filtered retrieval.
  - *System Bottlenecks & Focus:* Chunk-size vs. retrieval-precision trade-off, context fragmentation across chunk boundaries, and ingestion pipeline throughput at document-corpus scale.

- **Module 03: Embeddings for Retrieval & Vector Representations**
  - *Key Concepts:* Dense embedding models (bi-encoders) vs. cross-encoders, embedding model selection (dimensionality, domain fit, multilingual support), Matryoshka Representation Learning and embedding truncation, and embedding fine-tuning for domain adaptation. Builds directly on `00_nlp_fundamentals`' classical embeddings (Word2Vec/GloVe/FastText, not re-covered here) and BM25 (Module 04's sparse-side baseline).
  - *System Bottlenecks & Focus:* Embedding dimensionality vs. index storage/query-latency trade-off, and embedding drift when the domain shifts from the model's training distribution.

- **Module 04: Vector Indexing, ANN Search & Vector Database Internals**
  - *Key Concepts:* Approximate Nearest Neighbor (ANN) search, HNSW graph construction and search mechanics, IVF and IVF-PQ (product quantization) for large-scale compression, exact vs. approximate recall trade-offs, and vector database architecture (Pinecone, Weaviate, Qdrant, Milvus) — sharding, filtering, and hybrid metadata queries.
  - *System Bottlenecks & Focus:* Recall-vs-latency-vs-memory trade-off curve, index build time at scale, and the compute/storage cost of re-indexing on corpus updates.

- **Module 05: Hybrid Retrieval & Reranking**
  - *Key Concepts:* Sparse (BM25, from `00_nlp_fundamentals`) + dense hybrid fusion (Reciprocal Rank Fusion), cross-encoder reranking, ColBERT-style late-interaction retrieval, and multi-stage retrieve-then-rerank pipelines.
  - *System Bottlenecks & Focus:* Reranking latency cost vs. precision gain, candidate-set size tuning (top-k retrieved → top-n reranked), and when hybrid search outperforms dense-only retrieval.

- **Module 06: Query Understanding, Transformation & Optimization**
  - *Key Concepts:* Query expansion and rewriting, HyDE (Hypothetical Document Embeddings), query decomposition for multi-hop questions, multi-query retrieval, and semantic routing across multiple retrieval sources/indexes.
  - *System Bottlenecks & Focus:* Added latency/cost of query-transformation LLM calls, and failure modes where query rewriting drifts from user intent.

- **Module 07: GraphRAG & Structured/Knowledge-Graph Retrieval**
  - *Key Concepts:* Knowledge graph construction from unstructured text (entity/relation extraction), graph-based retrieval and community detection/summarization (Microsoft GraphRAG-style), and when graph retrieval outperforms flat vector retrieval (multi-hop, "global" corpus-level questions).
  - *System Bottlenecks & Focus:* Graph construction cost and staleness, and the local (entity-level) vs. global (corpus-summary) query trade-off.

- **Module 08: Agentic RAG & Self-Correcting Retrieval Loops**
  - *Key Concepts:* Agentic RAG (LLM-driven retrieval tool-calling instead of a fixed pipeline), Self-RAG and Corrective RAG (retrieval-quality self-critique and re-retrieval), iterative retrieve-generate-verify loops, and multi-agent retrieval (specialized retriever agents per source/domain).
  - *System Bottlenecks & Focus:* Unbounded latency/cost from iterative agent loops, and the need for explicit loop termination/budget guards in production.

- **Module 09: RAG Evaluation, Production Hardening & Scalability**
  - *Key Concepts:* Retrieval metrics (Recall@k, MRR, NDCG), generation metrics (faithfulness, answer relevancy, context precision/recall — RAGAS-style), LLM-as-judge evaluation pitfalls, semantic caching, RAG-specific security (prompt injection via retrieved content, data leakage across tenants), multi-tenant index isolation, and production scaling (sharding, read replicas, cost/latency SLOs).
  - *System Bottlenecks & Focus:* Evaluation metric noise/bias, cache-hit-rate vs. staleness trade-off, and the blast radius of a single-tenant data leak in a shared vector index.

Module 10 (Interview Q&A track, generated separately per `interview_qa_generator/SKILL.md`) will cover standalone screening questions across all nine modules above.
