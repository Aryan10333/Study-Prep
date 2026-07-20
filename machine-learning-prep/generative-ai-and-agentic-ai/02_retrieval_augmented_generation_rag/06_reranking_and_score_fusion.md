# Module 06: Re-ranking & Score Fusion Mechanics

This guide details Bi-Encoder candidate retrieval vs. Cross-Encoder reranking precision, ColBERT Late Interaction matrix MaxSim scoring, Score Fusion algorithms, step-by-step hand calculations, and LangChain `ContextualCompressionRetriever` execution pipelines.

> **Notebook Companion**: [06_reranking_and_score_fusion.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/06_reranking_and_score_fusion.ipynb)

---

## 1. Bi-Encoder vs Cross-Encoder Architecture

```text
Architecture    Input Layer                  Compute Complexity   Precision                     Best Role
----------------------------------------------------------------------------------------------------------------------
Bi-Encoder      Embed Query & Doc separately Fast O(1) Vector Search Moderate (Recall Generator) Candidate Retrieval (Top-100)
Cross-Encoder   Joint Query + Doc Attention  Heavy O(N) Computation High (Precision Reranker) Candidate Reranking (Top-20)
ColBERT         Token-level Late Interaction Moderate O(N*M) MaxSim High                          Middle ground reranker
```

---

## 2. ColBERT Late Interaction Matrix MaxSim Scoring

ColBERT (Contextualized Late Interaction over BERT) retains fine-grained token embeddings for queries $E_q = [q_1, \dots, q_N]$ and documents $E_d = [d_1, \dots, d_M]$.

The MaxSim score between query $q$ and document $d$ is:

$$\text{Score}_{\text{ColBERT}}(q, d) = \sum_{i=1}^N \max_{j=1}^M \left( q_i \cdot d_j^T \right)$$

---

## 3. Score Fusion Methods & Normalization Algorithms

When merging candidate lists from disparate search engines:
1. **Min-Max Normalization**: Scales bounded scores into $[0, 1]$ interval:
   $$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
2. **Linear Weighted Combination**: Blends normalized sparse and dense scores:
   $$S_{\text{hybrid}} = \alpha \cdot S_{\text{dense, norm}} + (1 - \alpha) \cdot S_{\text{sparse, norm}} \quad (\alpha = 0.7)$$
3. **Rank-Based Fusion (RRF)**: Merges rank positions directly without needing score scale alignment.

---

## 4. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose top-3 candidate chunks retrieved by Bi-Encoder produce Cosine similarities:
- Candidate 1 (`"Datadog logs retention"`): Bi-Encoder Score = **0.85**
- Candidate 2 (`"Microservice A uses gRPC over TLS 1.3"`): Bi-Encoder Score = **0.81**

Query: `"What protocol connects Microservice A?"`

1. **Bi-Encoder Ranking**: Candidate 1 ranks #1 ($0.85 > 0.81$) due to broad keyword overlap ("logs", "metrics").
2. **Cross-Encoder Reranking**: Processes joint self-attention $f(Q, D)$:
   - Candidate 1 Cross-Encoder Logit Score: **-2.40** (Irrelevant to protocol query)
   - Candidate 2 Cross-Encoder Logit Score: **+4.85** (Direct factual match: gRPC over TLS 1.3)
3. **Outcome**: Cross-Encoder correctly promotes Candidate 2 to **Rank #1**, eliminating the false-positive Bi-Encoder candidate.

---

## 5. Production LangChain Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

docs = [
    "Microservice A logs metrics directly to Datadog with trace IDs.",
    "Microservice A communicates with Database B via gRPC over TLS 1.3.",
    "Datadog retains telemetry logs for 30 days."
]

if os.getenv("OPENAI_API_KEY"):
    vectorstore = FAISS.from_texts(docs, OpenAIEmbeddings())
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    reranked_docs = compression_retriever.invoke("gRPC protocol TLS")
    print("=== Cross-Encoder Precision Reranked Output ===")
    for idx, doc in enumerate(reranked_docs, 1):
        print(f"Rank #{idx}: {doc.page_content}")
```