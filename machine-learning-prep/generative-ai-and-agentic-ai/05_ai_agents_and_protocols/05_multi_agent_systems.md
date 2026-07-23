# Module 05: Multi-Agent Systems & Coordination

Complex production tasks (like compiling software or managing customer support workflows) often degrade in quality when handled by a single, unconstrained agent. This module details multi-agent topologies, communication patterns, handoff dynamics, deadlock prevention, and Human-in-the-Loop coordination.

---

## 1. Multi-Agent Topologies

Dividing tasks across specialized agents isolates context scopes and improves execution reliability:

- **Supervisor/Worker**: A central coordinator receives the user query, breaks it down, invokes specialized sub-agents, gathers their outputs, and synthesizes the final response.
- **Hierarchical Teams**: Multiple Supervisor/Worker nested structures where a master supervisor delegates to sub-supervisors, managing execution scopes cleanly.
- **Swarm / Handoffs**: Decentralized agents hand off control directly to other agents by invoking specialized routing tools.
- **Debate / Consensus**: Multiple agent instances with distinct system prompts evaluate the same query, critique each other's outputs, and run consensus voting loops before responding.

---

## 2. The Explicit Handoff Pattern: Swarm Trace

The Handoff pattern (seen in modern agent frameworks like Swarm) is a lightweight alternative to centralized supervisor architectures. Instead of a central router parsing all text history, specialized agents return a pointer to the next agent when their criteria are met.

```
                          [User Query]
                                │
                                ▼
                       ┌─────────────────┐
                       │  Support Agent  │
                       └────────┬────────┘
                                │ (invokes tool)
                                ▼
                 ┌─────────────────────────────┐
                 │  Handoff to Banking Agent   │
                 └──────────────┬──────────────┘
                                │ (swaps pointer)
                                ▼
                       ┌─────────────────┐
                       │  Banking Agent  │
                       └─────────────────┘
```

### Trace of Swarm Handoff State Transition
Let's walk through an account query: "I want to dispute a $50 transaction on my savings card."

- **Step 1: Ingestion & Routing**
  - Active Agent: `SupportAgent`
  - Context History $H_0$: `[User: "I want to dispute a $50 transaction"]`
  - **Thought**: "This query requires banking transaction records. I will call the dispute handoff tool."
  - **Action**: `Call dispute_handoff()`
- **Step 2: Tool Execution & Handoff Interception**
  - Inside the orchestrator, the tool `dispute_handoff` is defined as:
    ```python
    def dispute_handoff():
        # Returns the target Agent instance to hand off execution
        return BankingAgent
    ```
  - The tool returns a pointer to `BankingAgent`.
- **Step 3: State Re-routing**
  - The orchestrator intercepts the returned agent object.
  - It swaps the active agent reference: `active_agent = BankingAgent`.
  - The context history is appended with the handoff record:
    $H_1$: `[User: "I want to dispute...", Tool (dispute_handoff): "Handed off to BankingAgent"]`
- **Step 4: Target Agent Execution**
  - The `BankingAgent` is invoked with the updated history $H_1$.
  - **Thought**: "I am now handling a dispute query. I need to list recent transactions."
  - **Action**: `Call list_transactions(user_id=123)`
  - Execution completes within the isolated banking domain.

---

## 3. Communication Patterns: Shared State vs. Message Passing

- **Shared State Channels**:
  Agents share access to a single global state object (e.g. a centralized transaction graph). Node transitions update fields directly.
  - *Risk*: Concurrent write race conditions.
- **Message Passing**:
  Agents communicate strictly by passing messages over channels. Each agent maintains its own private memory context, reducing token overhead.
  - *Risk*: Message routing complexity scales with agent counts.

---

## 4. System Failure Modes: Deadlocks & Infinite Loop Chattering

- **Agentic Deadlock**:
  Agent A is blocked waiting for database schemas from Agent B, while Agent B is waiting for credentials from Agent A. Neither agent can proceed.
- **Infinite Loop Chattering**:
  Two agents continuously swap redundant validation responses without reaching termination:
  `Agent A: "Is this correct?"` $\rightarrow$ `Agent B: "Please double check."` $\rightarrow$ `Agent A: "Here is the same output..."`
  - *Mitigation*: Implement loop detectors that calculate token similarity distributions across steps and terminate if the sequence becomes repetitive.

---

## 5. Human-in-the-Loop (HITL) Architectures

For operations that modify critical data (e.g., executing financial transfers or deleting resources), graphs pause execution using approval gates:

1. **State Serialized Interruption**:
   The workflow runner saves the agent's current state variables ($S_t$) to a persistent store (e.g., Postgres) and suspends the execution thread.
2. **Event Wakeup Hook**:
   The system exposes an API endpoint (e.g., `/approve?thread_id=xyz`). When a human clicks approve, the orchestrator deserializes $S_t$, updates the validation flag, and restarts the execution worker from the exact interrupted node.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Multi-agent coordination reduces the error rates of long prompts by dividing complex tasks into smaller, specialized agents with distinct system instructions and tools.
- **Why was it introduced?**
  Introduced to solve the limits of single-agent prompt execution, which struggles to maintain accuracy when managing dozens of tools simultaneously.
- **What are its limitations?**
  - Latency scales with agent hops.
  - Cost scales as messages are replicated across multiple context windows.
  - Risk of chattering and state synchronization errors.
- **Computational Complexity (Time & Memory)**
  - **Time**: $O(A \cdot T)$ in execution steps, where $A$ is active agents and $T$ is steps per agent.
  - **Memory**: Sharing message history duplicates tokens, scaling as $O(A \cdot L_{\text{history}})$.
- **Component Variable Denotation Legend**
  - $A$: Number of active agents in the system.
  - $T$: Number of execution steps per agent.
  - $L_{\text{history}}$: Message history length in tokens.
- **Production Use Cases**
  - Code refactoring engines where Coder, Compiler, and Reviewer agents run local verification checks.
  - Triaging customer service queries across billing, account recovery, and safety agents.
- **Follow-up questions interviewers ask**
  - *How do you prevent context window overflow in multi-agent chattering loops?* (Enforce a strict step threshold limit ($T_{\text{max}}$) per trajectory run and trace loop repetitions).
  - *Why is a Swarm handoff pattern preferred over a centralized Supervisor?* (A swarm handoff pattern reduces context window overhead by keeping conversations focused on a single specialized agent's history at any point, whereas a supervisor passes full histories to everyone).
