# Module 03: Memory Systems & State Persistence

Production agent systems require state persistence. This module details the memory taxonomy of agents, the lifecycle management of context windows, memory retrieval strategies (linking to RAG concepts), and state serialization mechanisms.

---

## 1. Agent Memory Taxonomy

In agent architectures, memory is categorized by access speed, persistence tier, and semantic scope:

```
                            ┌────────────────────────┐
                            │     Working Memory     │
                            │ (Local execution state)│
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   Short-Term Memory    │
                            │    (Context window)    │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │    Long-Term Memory    │
                            │(Vector DB / SQL State) │
                            └────────────────────────┘
```

- **Working Memory**: In-flight variables local to the current reasoning step (e.g. intermediate tool outputs, current loop indexes). Exists only during execution.
- **Short-Term Memory**: The model's context window. Contains the conversation history, system prompt, and reasoning trajectory.
- **Long-Term Memory**: Persistent store across sessions.
  - *Episodic Memory*: Narrative history of past user interactions and task trajectories.
  - *Semantic Memory*: Fact lookups and database records (retrieved via vector search).
  - *Procedural Memory*: Rules of operation and agent capability descriptions.

---

## 2. Memory Lifecycle Management: Worked Tracing

As an agent runs a multi-step trajectory, the context window scales. We must manage this token volume using summarization and sliding window buffers.

### Hand-Calculation: Summarization Buffer & Token Decay
Let's define a system with a strict token budget:
- **Maximum Context Limit ($C_{\text{limit}}$)**: $1,000\text{ tokens}$
- **System Prompt ($P_{\text{sys}}$)**: $300\text{ tokens}$ (Fixed)
- **Summarization Trigger Threshold ($H_{\text{trigger}}$)**: $85\%$ of $C_{\text{limit}} = 850\text{ tokens}$

We trace the step-by-step token accumulation:

#### Step 1: First Interaction
- **User input**: $150\text{ tokens}$
- **Agent response**: $200\text{ tokens}$
- **Current Token Count ($C_1$)**:
  $$C_1 = P_{\text{sys}} + \text{Input}_1 + \text{Response}_1 = 300 + 150 + 200 = 650\text{ tokens}$$
  Status: $650 < 850$ (No action required).

#### Step 2: Second Interaction
- **User input**: $200\text{ tokens}$
- **Agent response**: $300\text{ tokens}$
- **Current Token Count ($C_2$)**:
  $$C_2 = C_1 + \text{Input}_2 + \text{Response}_2 = 650 + 200 + 300 = 1,150\text{ tokens}$$
  Status: $1,150 > 850$ (Threshold exceeded! Trigger summarization loop).

#### Step 3: Summarization Compression
- The agent calls a summary model to compress the oldest interaction block ($\text{Input}_1 + \text{Response}_1 = 350\text{ tokens}$) into a concise semantic recap.
- **Summary Result**: "User query about tax computed to $16k." ($50\text{ tokens}$)
- **New Compact Context Count ($C_{\text{new}}$)**:
  $$C_{\text{new}} = P_{\text{sys}} + \text{Summary} + \text{Input}_2 + \text{Response}_2 = 300 + 50 + 200 + 300 = 850\text{ tokens}$$
  Status: Context compressed below the hard limit, preserving execution state.

---

## 3. Memory Retrieval Strategies (Vector & SQL Mappings)

To retrieve long-term episodic or semantic memories without overloading the prompt, agents use retrieval mechanisms built on top of RAG principles:

1. **Semantic Recency Weighting**:
   When querying vector databases for memories, scores are computed using a combined cosine similarity and temporal decay calculation:
   $$\text{Memory Score} = \text{CosineSimilarity}(Q, M) \times \lambda^{\Delta t}$$
   where $\lambda \in [0, 1]$ is a temporal decay parameter, and $\Delta t$ is the time elapsed since the memory was written. This guarantees the model retrieves relevant yet recent events.
2. **Metadata Key-Value Routing**:
   Querying memories explicitly matching user attributes (e.g. `user_id = 456`) before running semantic vector calculations to isolate memory leak risks across distinct sessions.

---

## 4. State Persistence: Threads vs. Global Enterprise State

- **Thread-Scoped Session State**: Isolation of conversation history within a single channel (e.g. Chat session UUID). Stored in lightweight document caches (Redis, Postgres JSONB).
- **Global Enterprise State**: Shared variables accessible to all agent workers in an organization (e.g. global inventory numbers, product directories). Requires strong locking policies (optimistic concurrency locks) to prevent race conditions during updates.

---

## 5. Memory Failure Modes

- **Context Saturation**: The prompt accumulates too many intermediate details, causing the model to neglect critical rules due to Lost-in-the-Middle positioning bias.
- **Memory Explosion**: High step counts write massive redundant logs to storage, increasing database scaling costs.
- **Prompt Poisoning**: Malicious instructions retrieved from episodic memory structures (stored from previous malicious runs) inject injection payloads back into the context window.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Memory systems allow agents to maintain continuity of state, parameters, and history across multiple reasoning steps and different chat sessions.
- **Why was it introduced?**
  Introduced to solve the memoryless design of raw transformer models, which possess no innate persistence capability between individual stateless API calls.
- **What are its limitations?**
  - Prompt overhead cost increases.
  - Risk of memory retrieval latency.
  - Risk of memory drift (recalling outdated facts).
- **Computational Complexity (Time & Memory)**
  - **Time**: Memory retrieval via vector database is $O(d)$ for approximate search (where $d$ is embedding dimensions). Summarization costs $O(L_{\text{old}})$ execution tokens.
  - **Memory**: Database memory requires $O(N \cdot d)$ floating point numbers where $N$ is the number of stored memories.
- **Component Variable Denotation Legend**
  - $C_{\text{limit}}$: Maximum token budget of the context window.
  - $\lambda$: Memory temporal decay coefficient.
  - $\Delta t$: Time elapsed since memory creation.
  - $N$: Number of stored memories.
- **Production Use Cases**
  - CRM agents recalling a customer's preference profile across multi-month session intervals.
  - Development agents maintaining persistent workspace file change buffers across system crashes.
- **Follow-up questions interviewers ask**
  - *How do you prevent optimistic locking issues when two agent workers update the same thread state?* (Use database-level version tokens or serialize all state updates through a message queue).
  - *What is the trade-off of memory summarization?* (Summarization reduces token usage but discards raw syntactic details, leading to potential loss of precise parameters).
