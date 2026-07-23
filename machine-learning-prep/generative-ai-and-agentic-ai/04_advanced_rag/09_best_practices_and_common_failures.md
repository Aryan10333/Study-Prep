# Best Practices, Failure Modes & Decision Frameworks

Designing production RAG systems requires anticipating structural failure patterns and navigating architectural decision trade-offs based on cost, scale, latency, and accuracy.

---

## 1. Common Failure Modes & Debugging Checklists

### 1. The "Lost-in-the-Middle" Phenomenon
- **Definition**: LLMs are best at retrieving information from the beginning and end of long input prompts. Chunks placed in the middle of long contexts are often ignored.
- **Diagnostics**: Graph recall accuracy against chunk position indices.
- **Resolution**: Use Cross-Encoder rerankers to compress candidate pools, keeping context payloads small and relevant.

### 2. Retrieval Drift
- **Definition**: Over time, changes in terminology, naming conventions, or product models cause user query vectors to miss older database documents.
- **Resolution**: Implement hybrid search (BM25 + Dense) and continuous index auditing.

### 3. Context Window Saturation
- **Definition**: Formatting high counts of candidate documents causes token limit saturation, increasing generation costs and causing generation truncation.
- **Resolution**: Enforce dynamic top-K selection based on score distance distributions.

---

## 2. Decision Frameworks for Production GenAI

### Framework A: Adaptation Strategies Selection

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;">Adaptation Decision Flow</div>
    <!-- Root -->
    <div style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px 18px; border-radius: 6px; text-align: center; border-top: 3px solid #64748b; font-weight: bold; font-size: 12px; color: #1e293b;">
        Adaptation Requirement
    </div>
    <!-- Split connector line -->
    <div style="width: 60%; height: 2px; background-color: #cbd5e1; position: relative; margin: 5px 0;">
        <div style="position: absolute; left: 50%; top: -5px; width: 2px; height: 10px; background-color: #cbd5e1;"></div>
        <div style="position: absolute; left: 0; top: 0; width: 2px; height: 10px; background-color: #cbd5e1;"></div>
        <div style="position: absolute; right: 0; top: 0; width: 2px; height: 10px; background-color: #cbd5e1;"></div>
    </div>
    <!-- Children -->
    <div style="display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Left: RAG -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center; border-top: 3px solid #2563eb;">
            <div style="color: #1e40af; font-weight: bold; font-size: 11px; text-transform: uppercase;">Factual & Grounding</div>
            <div style="align-self: center; color: #94a3b8; font-weight: bold; font-size: 12px; margin: 2px 0;">↓</div>
            <div style="background-color: #eff6ff; color: #1e40af; border: 1px solid #2563eb; padding: 6px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; width: 85%; text-align: center;">
                RAG Indexing
            </div>
        </div>
        <!-- Right: SFT -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center; border-top: 3px solid #7c3aed;">
            <div style="color: #5b21b6; font-weight: bold; font-size: 11px; text-transform: uppercase;">Behavior, Style & Format</div>
            <div style="align-self: center; color: #94a3b8; font-weight: bold; font-size: 12px; margin: 2px 0;">↓</div>
            <div style="background-color: #f5f3ff; color: #5b21b6; border: 1px solid #7c3aed; padding: 6px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; width: 85%; text-align: center;">
                Supervised Fine-Tuning (SFT)
            </div>
        </div>
    </div>
</div>


- **RAG**: Choose when you need to ground generation in verifiable, real-time files, retrieve specific codes, or maintain data lineage.
- **Fine-Tuning (SFT)**: Choose when you need to enforce structured output formatting, match a strict persona style, or adapt to hardware VRAM limitations.
- **Prompt Engineering**: Choose for fast prototyping, simple formatting, or low-volume task routing.

---

### Framework B: Long-Context Prompts vs. Vector RAG

With Claude 3.5 and Gemini Pro supporting 1M+ contexts, stuffing whole documentation files directly is a valid approach.

| Adaptability Dimension | Long-Context Prompt (Full stuffing) | Vector-Based RAG |
| :--- | :--- | :--- |
| **Operational Setup** | Low (no database, no indexing pipelines) | High (Vector DB, ingestion sync, chunking) |
| **Retrieval Accuracy** | High (captures cross-document relations) | Variable (depends on search recall) |
| **Query Cost** | Very High (paying for full document text per call) | Low (paying only for matching top-K chunks) |
| **Query Latency** | High (seconds to compile full inputs) | Low (fast execution) |

*Production Choice*: Use **Long-Context** for low-volume complex queries (e.g. legal document synthesis). Use **Vector RAG** for high-volume, low-latency, and cost-constrained production APIs.

---

### Framework C: GraphRAG vs. Traditional Vector RAG
- **GraphRAG**: Choose when queries require global synthesis over entire collections (e.g. "What are the common research trends?").
- **Vector RAG**: Choose for query-centric lookups (e.g. "What was Intel's Q3 revenue?").

---

### Framework D: Passive RAG vs. MCP Active Tool-Calling
- **Passive RAG**: Choose for static documentation search (FAQs, manuals).
- **MCP Active Tool-Calling**: Choose when the agent needs to query live databases, execute remote APIs, or dynamically choose search strategies based on intermediate responses.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Decision frameworks and failure playbooks prevent developers from implementing incorrect architectures, avoiding system rewrites.
- **Why was it introduced?**
  To guide system designers through balancing the trade-offs of cost, latency, complexity, and accuracy.
- **What are its limitations?**
  Heuristic decisions must be validated against continuous offline regression evaluation suites.
- **Production Use Cases**
  - Selecting RAG + Prompt Caching to balance cost-accuracy budgets for enterprise customer support agents.
- **Follow-up questions interviewers ask**
  - *If your system encounters the Lost-in-the-Middle phenomenon, why would increasing context windows fail to resolve it?*
  - *Under what pricing constraints does Long-Context prompt caching become cheaper than Vector RAG?*
