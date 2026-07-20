# Module 11: Scaling Enterprise RAG & Security

This guide details vector DB sharding across 100M+ documents, HNSW index tuning (`ef_construction`, `M`), Multi-Tenant ACL metadata payload filtering, and Hot/Cold storage strategies.

> **Notebook Companion**: [11_scaling_enterprise_rag_and_security.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/11_scaling_enterprise_rag_and_security.ipynb)

---

## 1. Scaling to 100M+ Vector Documents: Sharding & Partitioning

Scaling vector databases to tens of millions of documents requires distributed clustering strategies:
- **Tenant-Based Partitioning**: Isolates document vectors per enterprise customer into separate database collections.
- **Consistent Hashing Sharding**: Distributes vector index shards across multiple database nodes based on document ID hash.
- **Hot vs Cold Storage Tiering**: Keeps high-frequency recent document vectors in RAM (HNSW index) while offloading historical archives to disk (DiskANN / IVF-PQ on SSD).

---

## 2. HNSW Graph Tuning Parameters

- **`M` (Max Graph Edges per Node)**: Controls graph connectivity ($M = 16 - 64$). Higher $M$ improves recall at the cost of higher index RAM footprint.
- **`ef_construction`**: Size of dynamic candidate list during graph index construction. Higher values slow down indexing build speed but increase search recall.
- **`ef_search`**: Size of candidate list during runtime query execution.

---

## 3. HNSW Index Memory Footprint Math (Andrew Ng Style)

An HNSW graph index stores $N = 10,000,000$ ($10\text{M}$) vectors of dimension $d = 1,536$ with max graph edges $M = 16$.

$$\text{Memory Footprint} = N \times (d \times 4 + M \times 8) \text{ bytes}$$

$$\text{Memory} = 10,000,000 \times (1536 \times 4 + 16 \times 8) = 10,000,000 \times (6144 + 128) = 10,000,000 \times 6272 = \mathbf{62.72\text{ GB RAM}}$$

---

## 4. Multi-Tenancy & Access Control Lists (ACL Security Filtering)

Enterprise security requires strict Role-Based Access Control (RBAC):
- **Payload Metadata ACLs**: Store user authorization tags (`"allowed_roles": ["Engineering", "Manager"]`) inside vector payload metadata.
- **Pre-Filtering Query Assertion**: Vector search queries MUST append payload filters (`filter={"allowed_roles": user_role}`) to ensure users never retrieve unauthorized documents.

---

## 5. Production LangChain Security Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

docs = [
    "Public Company Handbook: Employees receive 20 days PTO.",
    "Manager Compensation Guide: Managers receive stock option grants.",
    "Executive Acquisition Spec: Project Titan buyout terms."
]
metadatas = [
    {"role": "Public"},
    {"role": "Manager"},
    {"role": "Executive"}
]

if os.getenv("OPENAI_API_KEY"):
    vectorstore = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    
    # User with 'Manager' role queries vector store
    results = vectorstore.similarity_search("Handbook Compensation", k=2, filter={"role": "Manager"})
    print("=== Multi-Tenant ACL Payload Filtered Results (Role = Manager) ===")
    for r in results:
        print(f"Role: {r.metadata['role']} | Content: {r.page_content}")
```