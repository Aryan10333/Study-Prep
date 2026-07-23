# Module 13: Hands-on Vector Databases: FAISS, Elasticsearch, Milvus & pgvector

This guide details enterprise vector databases (FAISS, Elasticsearch/OpenSearch, Milvus/Qdrant, `pgvector`), Product Quantization (PQ-64) memory compression math ($96\text{x}$ compression), and LangChain integration code.

> **Notebook Companion**: [13_hands_on_vector_databases_faiss_pgvector_elastic.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/13_hands_on_vector_databases_faiss_pgvector_elastic.ipynb)

---

## 1. Vector Database Selection Matrix

```text
Vector DB       Architecture          Primary Index Types     Best Role
----------------------------------------------------------------------------------------------------------------------
FAISS           In-Memory C++ / GPU   HNSW, IVF-PQ, Flat      Local prototyping & GPU benchmarking
pgvector        PostgreSQL Extension  HNSW, IVFFlat           Relational DB + Vector in single Postgres instance
Elasticsearch   Distributed Java      HNSW, BM25 Keyword      Hybrid Search with existing Elastic ecosystem
Milvus / Qdrant Cloud-Native C++/Rust HNSW, DiskANN           Scaling 100M+ vectors with distributed sharding
```

---

## 2. Product Quantization (PQ-64) Memory Math (Andrew Ng Style)

For $N = 1,000,000$ vectors of dimension $d = 1,536$:
- FP32 Memory: $1,000,000 \times 1,536 \times 4 \text{ bytes} = 6.144\text{ GB}$
- PQ-64 Memory ($m=64$ sub-vectors, 1 byte codebook index): $1,000,000 \times 64 \times 1 \text{ bytes} = 0.064\text{ GB}$
- Compression Factor: $\frac{6.144}{0.064} = \mathbf{96\text{x} \text{ RAM Reduction}}!$

---

## 3. Production Python Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

docs = ["Microservice A communicates with Database B via gRPC over TLS 1.3."]

if os.getenv("OPENAI_API_KEY"):
    # FAISS In-Memory Vector Store Execution
    faiss_store = FAISS.from_texts(docs, OpenAIEmbeddings())
    results = faiss_store.similarity_search("gRPC protocol", k=1)
    print("=== FAISS Vector Search Result ===")
    print(results[0].page_content)
```