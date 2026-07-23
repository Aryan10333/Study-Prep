# Advanced & Agentic RAG Architectures

Enterprise RAG requirements often stretch beyond simple query-response matching. Advanced systems incorporate active reflection loops, external fallbacks, Knowledge Graph synthesis, and agentic tool-calling pipelines.

---

## 1. Self-RAG: Self-Reflection Tokens

**Self-RAG (Self-Reflective Retrieval-Augmented Generation)** trains an LLM to generate special **reflection tokens** to decide dynamically when to retrieve documents, evaluate retrieved relevance, and criticize generation quality.

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Self-RAG Self-Reflection Token Execution Loop</div>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Step 1 -->
        <div style="flex: 1; min-width: 140px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #64748b;">
            <div style="color: #475569; font-weight: bold; font-size: 11px;">1. Generate Text</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Predicts next tokens</div>
        </div>
        <!-- Arrow -->
        <div style="align-self: center; color: #94a3b8; font-weight: bold;">→</div>
        <!-- Step 2 -->
        <div style="flex: 1; min-width: 140px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #2563eb;">
            <div style="color: #1e40af; font-weight: bold; font-size: 11px;">2. Emit [Retrieve]</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Decides if retrieval needed</div>
        </div>
        <!-- Arrow -->
        <div style="align-self: center; color: #94a3b8; font-weight: bold;">→</div>
        <!-- Step 3 -->
        <div style="flex: 1; min-width: 140px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #059669;">
            <div style="color: #065f46; font-weight: bold; font-size: 11px;">3. Grade Relevance</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Parses [IsRel] & [IsSup]</div>
        </div>
    </div>
</div>

### Self-RAG Reflection Token Schema

During post-training, special tokens are added to the model's vocabulary and trained using standard causal loss masking:

| Reflection Token | Type | Possible Predicted Values | Description |
| :--- | :--- | :--- | :--- |
| **`[Retrieve]`** | Retrieval Trigger | `[NoRetrieval]`, `[Retrieve]`, `[Continue]` | Decides whether the prompt requires querying the vector database. |
| **`[IsRel]`** | Relevance Grader | `[Relevant]`, `[Irrelevant]` | Evaluates whether a retrieved candidate chunk contains information matching the query. |
| **`[IsSup]`** | Grounding Grader | `[FullySupported]`, `[PartiallySupported]`, `[NoSupport]` | Evaluates whether the generated sentence is fully supported by the retrieved context. |
| **`[IsUse]`** | Utility Grader | `[Utility:5]`, `[Utility:4]`, `[Utility:3]`, `[Utility:2]`, `[Utility:1]` | Evaluates the overall helpfulness and relevance of the final response to the user. |

---

## 2. Corrective RAG (CRAG)

Corrective RAG adds a **heuristic evaluation step** between retrieval and generation:
1. **Retrieve**: Pull the top candidate documents from the vector database.
2. **Confidence Grading**: Pass the retrieved chunks through a lightweight evaluator model to calculate relevance confidence.
   - **Correct**: If confidence is high, proceed to Context Assembly.
   - **Incorrect**: If confidence is very low, discard the chunks and trigger a Web Search API (e.g. Tavily / Serper) to gather open-domain facts.
   - **Ambiguous**: Combine retrieved vector chunks and search results to synthesize the prompt.

---

## 3. GraphRAG: Knowledge Graphs & Community Summaries

Standard vector search models match isolated chunks semantically. However, they struggle with global synthesis queries like "Summarize all patents relating to semiconductor cooling." 

**GraphRAG** (popularized by Microsoft Research) structures unstructured documents into a Knowledge Graph:
1. **Entity Extraction**: An LLM parses text to extract entities (e.g., people, organizations) and relationships (edges).
2. **Hierarchical Clustering**: Graphs are clustered using algorithms (like Leiden clustering) to group related entities.
3. **Community Summary**: An LLM generates a text summary for each cluster community at multiple levels of abstraction.
4. **Global Search Querying**: Instead of searching single chunk vectors, GraphRAG queries the community summaries to compile global answers.

---

## 4. Agentic RAG: MCP Tool Servers vs. Passive Tools

Active retrieval moves the decision control to the LLM agent:

- **Traditional Tool-Calling**: The LLM output matches a function name schema (e.g. `query_database(query="Intel profits")`). The system intercepts the call, queries a local database, and returns the text.
- **MCP Server tool integration**: The LLM queries standardized Model Context Protocol (MCP) servers. The servers run autonomously, exposing live file paths, git hooks, SSH shells, or database schemas directly to the agent over JSON-RPC.
- **Benefit**: The model determines its own retrieval trajectory sequentially (e.g., first reading a log file, finding a database ID, then querying SQL via the MCP DB server tool) rather than relying on a static pre-computed vector match.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Advanced architectures solve global-lookup search errors (GraphRAG), retrieval quality verification (Self-RAG/CRAG), and live interactive queries (MCP).
- **Why was it introduced?**
  To enable models to evaluate their own output logic and query live active databases dynamically.
- **What are its limitations?**
  Graph construction compute costs are extremely high ($10\times$ more LLM API calls). Agentic loops can result in infinite tool call cycles and high latency.
- **Computational Complexity (Time & Memory)**
  - **Graph Clustering (Leiden)**: $O(E \cdot \log V)$ where $E$ is relationships and $V$ is entities.
  - **Self-RAG Generation**: Scales with step token evaluations.
- **Component Variable Denotation Legend**
  - $V$: Number of entity nodes in the Knowledge Graph.
  - $E$: Number of relationship edges in the Knowledge Graph.
- **Production Use Cases**
  - GraphRAG analysis over medical journals databases.
  - MCP agent loops for repository debugging tasks.
- **Follow-up questions interviewers ask**
  - *If a query requires exact keyword matching, why would GraphRAG be less efficient than hybrid BM25 search?*
  - *How do you prevent loops when an agentic tool call returns an error?*
