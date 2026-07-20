# Module 04: Embedding Generation & Multi-Vector Representations

This guide details Embedding Model Selection, Domain-Specific Fine-tuning, InfoNCE Contrastive Loss mechanics, Matryoshka Representation Learning (MRL) dimension slicing, Batch Embedding Caching, Vector Migration, and Multi-Vector Document Summaries, complete with step-by-step hand calculations, LangChain code, and production trade-offs.

> **Notebook Companion**: [04_embedding_generation_and_multi_vector.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/04_embedding_generation_and_multi_vector.ipynb)

---

## 1. Choosing Embedding Models & Dimension Trade-Offs

```text
Embedding Model             Vector Dimension (d)   MRL Slicing Support  Primary Benchmark (MTEB)  Best Suited For
----------------------------------------------------------------------------------------------------------------------
text-embedding-3-small      1,536                  Yes (Down to 256)    62.3%                     High-speed, low-cost API
text-embedding-3-large      3,072                  Yes (Down to 256)    64.6%                     High-accuracy enterprise RAG
bge-large-en-v1.5           1,024                  No                   64.2%                     Open-source self-hosted GPU
e5-mistral-7b               4,096                  No                   66.6%                     State-of-the-art retrieval
```

---

## 2. Mathematical Precision: InfoNCE Contrastive Loss (Andrew Ng Style)

Fine-tuning embedding models on domain-specific corpora uses the **InfoNCE Contrastive Loss**:

$$\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\exp(\text{sim}(q, p^+) / \tau) + \sum_{j=1}^K \exp(\text{sim}(q, p_j^-) / \tau)}$$

Where:
- $q$ is the query embedding vector.
- $p^+$ is the positive (relevant) document chunk embedding.
- $p_j^-$ are negative (irrelevant) document chunk embeddings.
- $\tau$ is the temperature scaling parameter (typically $\tau = 0.07$).

### Step-by-Step Hand Calculation Example:
Suppose query vector similarity scores with positive and negative documents are:
- $\text{sim}(q, p^+) = 0.80$
- $\text{sim}(q, p^-) = 0.10$
- Temperature $\tau = 0.10$

1. **Calculate Exponentials:**
   - Positive term: $\exp(0.80 / 0.10) = \exp(8.0) \approx 2,980.95$
   - Negative term: $\exp(0.10 / 0.10) = \exp(1.0) \approx 2.72$
2. **Calculate Softmax Fraction:**
   $$\frac{2,980.95}{2,980.95 + 2.72} = \frac{2,980.95}{2,983.67} \approx 0.999088$$
3. **Calculate Negative Log Likelihood Loss:**
   $$\mathcal{L}_{\text{InfoNCE}} = - \log(0.999088) = \mathbf{0.000912}$$

---

## 3. Matryoshka Representation Learning (MRL) Slicing Math

MRL embedding models (e.g. OpenAI `text-embedding-3-small/large`) structure vector dimensions nestedly. Truncating vector dimensions (e.g. $1,536 \rightarrow 256$) retains $>98\%$ of retrieval accuracy while reducing memory footprint by $6\text{x}$.

### Dimension Slicing Equation:
$$v_{\text{sliced}} = \frac{v[:k]}{\|v[:k]\|_2}$$

---

## 4. Multi-Vector Summaries & Proposition Indexing

Instead of embedding raw 1,000-token chunks directly, Multi-Vector indexing generates:
1. **Document Summaries**: LLM summarizes chunks into 1-2 sentence abstracts.
2. **Propositions**: LLM breaks chunks into atomic factual assertions.
3. **Vector Mapping**: Embedded proposition vectors link directly to the full parent chunk text.

---

## 5. Production LangChain Code Implementation

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    embeddings_256 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
    vec = embeddings_256.embed_query("Enterprise Microservice Architecture Specification")
    print("=== OpenAI MRL Sliced Embedding ===")
    print(f"Vector Dimension: {len(vec)}")
    print(f"First 5 Dimensions: {vec[:5]}")
```