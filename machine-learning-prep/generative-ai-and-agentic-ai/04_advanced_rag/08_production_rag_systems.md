# Production RAG Systems & Infrastructure

Moving RAG from prototype to production at enterprise scale requires designing microservices, implementing strict multi-tenant isolation, ensuring compliance (PII redaction, RBAC), and adding caching frameworks to optimize latency and costs.

---

## 1. Enterprise Streaming: SSE & WebSockets

 RAG query loops (retrieval, reranking, and generation) take time. To avoid long user wait times, production services stream responses token-by-token:
- **Server-Sent Events (SSE)**: Standard HTTP protocol where the server sends a continuous, unidirectional stream of text events (`text/event-stream`) to the client. This is the preferred method for chatbot completions.
- **WebSockets**: Bidirectional, full-duplex TCP communication channels. Used when the client needs to stream audio or continuous metadata queries back to the server in real-time.

---

## 2. Multi-Tenant Isolation Architectures

Enterprises must guarantee that tenant data remains isolated. If Tenant A queries the RAG system, they must never retrieve documents belonging to Tenant B.

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Multi-Tenant Isolation Models</div>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Model A -->
        <div style="flex: 1; min-width: 220px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; text-align: center; border-top: 3px solid #64748b;">
            <div style="color: #475569; font-weight: bold; font-size: 11px;">1. Metadata Namespace Filtering</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Shared index collection</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 8px;">Filter: tenant_id == 'A'</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 4px;">Pros: Cheap, highly scalable</div>
        </div>
        <!-- Model B -->
        <div style="flex: 1; min-width: 220px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; text-align: center; border-top: 3px solid #2563eb;">
            <div style="color: #1e40af; font-weight: bold; font-size: 11px;">2. Schema / Cluster Isolation</div>
            <div style="font-size: 9px; color: #64748b; margin-top: 4px;">Dedicated index per tenant</div>
            <div style="background-color: #eff6ff; color: #1e40af; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 8px;">Physical separation</div>
            <div style="background-color: #eff6ff; color: #1e40af; padding: 4px 8px; border-radius: 4px; font-size: 10px; margin-top: 4px;">Pros: Strict compliance, no leaks</div>
        </div>
    </div>
</div>

### Multi-Tenancy Trade-offs

| Isolation Pattern | Implementation | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Metadata Namespace Filtering** | All tenant vectors load to a single index. Queries append filter (e.g. `tenant_id == 'A'`). | Low cost, low operational overhead, simple updates. | Risk of data leakage if filters fail or vector database bugs bypass filter checks. |
| **Schema / Index Isolation** | Dedicated collection/table index per tenant. | High security, compliance-friendly, allows custom schemas. | Higher database cost, harder to manage schema migrations across thousands of tenants. |
| **Physical Cluster Isolation** | Separate database clusters per tenant. | Absolute isolation, custom configurations, optimal security. | Prohibitively expensive for standard multi-tenant SaaS. |

---

## 3. Security & Compliance (PII, RBAC, Injection)

1. **PII Redaction**: Raw texts pass through redaction engines (like Microsoft Presidio) to mask phone numbers, names, and credit cards before indexing or sending to external LLM endpoints.
2. **Role-Based Access Control (RBAC)**: Documents are tagged with security access groups (e.g., `HR-Admin`, `Finance-Read`). Queries pass user access groups to the vector database's metadata filter, preventing unauthorized retrieval.
3. **Prompt Injection Guardrails**: Strict input schemas and verification models check retrieved chunks and user prompts before execution to block hidden malicious instructions.

---

## 4. Observability & Tracing: Langfuse / LangSmith

Debugging RAG requires structured trace logging:
- **Trace Boundaries**: Log retrieval latency, database search queries, candidate chunks returned, reranker scores, dynamic prompt construction, token counts, and generation completion time.
- **Trace Context**: Track session IDs to review complete multi-turn conversational trajectories, making it simple to diagnose where errors occur (e.g., was it a retrieval error or a generation failure?).

---

## 5. Caching Frameworks: Result vs. Semantic Cache

To minimize LLM generation latency and token API cost:
- **Result Caching**: Checks for exact query matches. If the query exists in Redis, returns the saved completion instantly.
- **Semantic Caching (GPTCache)**: Embeds the user query and queries a cache database. If a cached query has a similarity score above a strict threshold (e.g. $> 0.96$), returns its cached response.
  - *Gotcha*: Cache invalidation. If database documents are updated, semantic cached responses must be cleared or updated immediately to prevent stale information lookup.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Production infrastructure solves security leakage (isolation/RBAC), latency bottlenecks (SSE, caches), and debugging visibility (observability).
- **Why was it introduced?**
  To meet enterprise security standards and maintain interactive user experiences during long generation steps.
- **What are its limitations?**
  Metadata filtering is vulnerable to software logic bugs. Schema isolation incurs high database management costs. Caches introduce stale data risks.
- **Computational Complexity (Time & Memory)**
  - **Metadata Filter Matching**: $O(K \cdot F)$ where $K$ is retrieved nodes and $F$ is filter operations.
  - **Semantic Cache Search**: $O(\log N_{\text{cache}})$.
- **Component Variable Denotation Legend**
  - $N_{\text{cache}}$: Number of entries in the semantic cache.
- **Production Use Cases**
  - Storing customer vectors inpgvector using namespace isolation.
- **Follow-up questions interviewers ask**
  - *How would you implement secure RBAC when your vector database does not support metadata pre-filtering?*
  - *If a user updates their document permissions, how quickly does your index filter reflect this shift?*
