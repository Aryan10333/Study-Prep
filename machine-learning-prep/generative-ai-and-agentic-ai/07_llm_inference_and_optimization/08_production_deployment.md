# 08. Production Deployment: Architecture, Routing, and Scalability

Deploying LLMs at scale requires wraping inference engines within enterprise-grade deployment topologies. Serving platforms must manage rate limits, coordinate failover strategies, partition hardware resources across multiple tenants, and roll out model updates without causing system downtime or violating latency SLAs.

---

## 1. Enterprise Serving Topologies

Production workloads are split into two execution topologies:

### Online Real-Time Serving
- **Workload**: Dynamic, user-facing applications (chatbots, real-time typing assistance) requiring low latencies.
- **Protocol**: Server-Sent Events (SSE) or WebSockets to stream generated tokens to the client as they are produced, minimizing perceived latency (TTFT).
- **Target SLA**: Lower TTFT ($<200\text{ ms}$) and comfortable TPOT ($<30\text{ ms}$).

### Offline Batch Processing
- **Workload**: High-volume, non-interactive tasks (document classification, database summarization, embedding generation).
- **Protocol**: REST batch APIs or background queues (e.g. Celery). Schedulers combine inputs into massive batches to maximize GPU memory throughput.
- **Target SLA**: Tokens per Second (TPS) throughput, ignoring individual request latency.

---

## 2. Gateway Router & Provider Fallbacks

To ensure high availability, enterprise gateways abstract self-hosted engines (vLLM/SGLang clusters) and commercial APIs behind a unified API layer:

```html
<div style="font-family: 'Segoe UI', sans-serif; border: 1.5px solid #64748b; border-radius: 8px; background-color: #f8fafc; padding: 16px; margin: 20px 0;">
  <!-- Gateway Router Header -->
  <div style="background-color: #1e293b; color: white; padding: 8px; border-radius: 4px; font-weight: bold; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; text-align: center; margin-bottom: 16px;">
    Unified Gateway Router (API Gateway)
  </div>

  <div style="display: flex; justify-content: space-between; align-items: stretch; gap: 12px; font-size: 11px;">
    <!-- Local Stack -->
    <div style="flex: 1; border: 1px solid #2563eb; background-color: #eff6ff; padding: 10px; border-radius: 6px; text-align: center;">
      <div style="font-weight: 700; color: #1e40af; margin-bottom: 6px; text-transform: uppercase;">Primary Route (Self-Hosted)</div>
      <div style="color: #1e3a8a; margin-bottom: 4px; font-weight: 600;">vLLM / SGLang Cluster</div>
      <div style="color: #64748b;">Lowest unit cost, secure data boundary</div>
    </div>

    <!-- Switch Logic -->
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100px; text-align: center;">
      <span style="font-size: 9px; font-weight: bold; color: #e11d48; margin-bottom: 4px;">If 429 / Timeout</span>
      <div style="width: 100%; height: 3px; background-color: #cbd5e1; position: relative;">
        <div style="position: absolute; right: 0; top: -3px; border-top: 4px solid transparent; border-bottom: 4px solid transparent; border-left: 6px solid #64748b;"></div>
      </div>
    </div>

    <!-- Fallback Stack -->
    <div style="flex: 1; border: 1px solid #10b981; background-color: #ecfdf5; padding: 10px; border-radius: 6px; text-align: center;">
      <div style="font-weight: 700; color: #065f46; margin-bottom: 6px; text-transform: uppercase;">Fallback Route (Commercial)</div>
      <div style="color: #047857; margin-bottom: 4px; font-weight: 600;">OpenAI / Gemini / Anthropic</div>
      <div style="color: #64748b;">High reliability, pay-per-token limits</div>
    </div>
  </div>
</div>
```

- **Failover Logic**: If the self-hosted engine throws a 503 (overloaded) or 429 (rate limit), the router catches the exception and redirects the query to a fallback commercial endpoint.
- **Model Redirection**: Queries can be routed dynamically based on prompt classification: complex reasoning prompts go to expensive target models, while simple classification queries go to smaller, quantized self-hosted models.

---

## 3. Multi-Tenant Resource Isolation

Sharing a single GPU cluster across multiple business units requires enforcing resource boundaries to prevent one tenant's queries from degrading the performance of others:
- **Rate-Limiting Semaphores**: Limit the number of concurrent active prefill and decode slots a single tenant can occupy.
- **VRAM Cache Quota Allocations**: Restrict the maximum KV Cache block allocations a single tenant's requests can hold in the active block table.
- **Priority Queues**: Place requests into prioritized tiers. Under heavy load, the scheduler evicts or preempts low-priority decoding iterations to allocate block capacity to high-priority requests.

---

## 4. Blue/Green and Canary Deployments

Model updates require swapping weights on active GPU memories without dropping connections.

### Blue/Green Deployments
Two identical hardware environments are maintained. The **Blue** cluster runs the active production model (e.g. Model V1). The **Green** cluster is updated with the new model (e.g. Model V2). 
Once Green passes verification, the load balancer shifts traffic to the Green cluster, and Blue is decommissioned.
- **Cons**: Requires doubling the active GPU allocation during transition.

### Canary Deployments
Instead of an instant swap, a small fraction of production traffic (e.g. $5\%$) is routed to the new model instance.
- **Shadow Traffic**: Production requests are duplicated at the gateway. The duplicate request is processed by the new model in the background, comparing its outputs, latency, and error rates against production without returning its responses to the user.
- **Regression Triggers**: The gateway monitors output metrics (perplexity drift, TTFT spikes, guardrail violations). If any metric violates safety bounds, traffic is automatically routed back to the production instance.

---

### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
