# RAG Evaluation & Benchmarking

Evaluating RAG systems requires profiling both the **retrieval quality** (did we fetch the correct chunks?) and the **generation quality** (did the LLM synthesize a correct response from those chunks?).

---

## 1. Retrieval Metrics: Math & Worked Traces

Retrieval performance evaluates search ranking quality before formatting the context prompt.

### 1. Hit Rate@K
Checks whether the relevant document is present in the top $K$ retrieved chunks. Returns $1$ if present, $0$ otherwise.

### 2. Mean Reciprocal Rank (MRR)
Measures the rank position of the first relevant retrieved document:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank position of the first relevant document for query $i$. If no relevant document is found, reciprocal rank is $0$.

#### Worked MRR Hand Calculation:
Evaluate a query batch of size $|Q| = 3$:
- Query 1: first relevant chunk is found at rank 2. $\text{RR}_1 = \frac{1}{2} = 0.5$
- Query 2: first relevant chunk is found at rank 1. $\text{RR}_2 = \frac{1}{1} = 1.0$
- Query 3: first relevant chunk is found at rank 4. $\text{RR}_3 = \frac{1}{4} = 0.25$

$$\text{MRR} = \frac{0.5 + 1.0 + 0.25}{3} = \frac{1.75}{3} \approx 0.5833$$

---

### 3. Normalized Discounted Cumulative Gain (NDCG)
NDCG measures retrieval ranking positions weighted by document relevance scores ($rel_i$).
$$\text{DCG}_p = \sum_{i=1}^p \frac{rel_i}{\log_2(i+1)}$$
$$\text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$$
Where $\text{IDCG}_p$ is the Ideal DCG—the DCG score of the retrieved documents sorted in descending order of relevance.

#### Worked NDCG Hand Calculation (p=3):
Assume retrieved document relevance scores are: $[2, 3, 0]$ (where $3$ is highly relevant, $2$ is partially relevant, $0$ is irrelevant).
- Actual retrieved order relevance: $rel_1 = 2$, $rel_2 = 3$, $rel_3 = 0$.

1. **Calculate Actual DCG@3**:
   $$\text{DCG}_3 = \frac{2}{\log_2(2)} + \frac{3}{\log_2(3)} + \frac{0}{\log_2(4)}$$
   $$\text{Math}: \log_2(2) = 1.0, \quad \log_2(3) \approx 1.585, \quad \log_2(4) = 2.0$$
   $$\text{DCG}_3 = \frac{2}{1.0} + \frac{3}{1.585} + 0 = 2 + 1.8927 \approx 3.8927$$

2. **Calculate Ideal DCG@3 (IDCG@3)**:
   The ideal relevance order is sorted descending: $[3, 2, 0]$.
   $$\text{IDCG}_3 = \frac{3}{\log_2(2)} + \frac{2}{\log_2(3)} + \frac{0}{\log_2(4)}$$
   $$\text{IDCG}_3 = \frac{3}{1.0} + \frac{2}{1.585} + 0 = 3 + 1.2618 \approx 4.2618$$

3. **Calculate NDCG@3**:
   $$\text{NDCG}_3 = \frac{\text{DCG}_3}{\text{IDCG}_3} = \frac{3.8927}{4.2618} \approx 0.9134$$

---

## 2. Generation Metrics: Ragas Triad

Generation metrics assess whether the model uses the retrieved context cleanly without introducing hallucinations.

### 1. Faithfulness (Groundedness)
Measures if all statements generated in the response are grounded in the retrieved context.
- **Method**: The LLM extracts distinct statements from the generated response, then checks whether each statement is supported by the context.
$$\text{Faithfulness Score} = \frac{\text{Number of statements supported by context}}{\text{Total number of generated statements}}$$

### 2. Answer Relevance
Measures whether the generated response directly answers the user query.
- **Method**: The LLM generates $N$ variations of queries matching the generated response, then measures the average cosine similarity between those generated query vectors and the user query.

### 3. Context Recall
Measures whether all relevant details present in the ground-truth answer are successfully retrieved in the context.
- **Method**: The LLM splits the ground truth answer into distinct facts, then checks whether each fact is present in the retrieved context.

---

## 3. Synthetic Golden Dataset Generation

To run robust offline regression testing, we must construct a **Synthetic Golden Dataset** (typically 100 to 500 query-context-ground_truth triplets):
1. **Source Parsing**: Extract high-entropy paragraphs from the document index.
2. **LLM Query Synthesis**: Prompt a teacher LLM to generate questions matching the paragraph. Specify styles (e.g. rewrite as colloquial query, add spelling mistakes, form as query expansion).
3. **Ground Truth Compilation**: Prompt the LLM to write a comprehensive, factual answer based *only* on the text paragraph.
4. **Regression Run**: Run the candidate RAG pipeline over the generated queries, and calculate Ragas metrics to establish baseline performance bounds.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  RAG evaluation metrics solve the optimization blind-spot, allowing developers to quantitatively measure whether pipeline changes improve retrieval and generation accuracy.
- **Why was it introduced?**
  Because manual inspection of model completions does not scale, and standard NLP metrics (BLEU, ROUGE) measure token overlap rather than semantic correctness.
- **What are its limitations?**
  LLM-as-a-judge evaluations introduce validation variance and high token API costs.
- **Computational Complexity (Time & Memory)**
  - **MRR Calculation**: $O(Q \cdot K)$ where $Q$ is query volume and $K$ is top-K rank positions.
  - **NDCG Calculation**: $O(Q \cdot K \cdot \log K)$ sorting search operations.
- **Component Variable Denotation Legend**
  - $Q$: Total query evaluation dataset size.
  - $p$: Target rank position evaluated ($K$).
- **Production Use Cases**
  - Running Golden Dataset regression tests before pushing vector index updates.
- **Follow-up questions interviewers ask**
  - *If your LLM-as-a-judge starts exhibiting length bias (grading long answers higher), how do you resolve it?*
  - *What is the difference between Context Precision and Context Recall?*
