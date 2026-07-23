# Module 01: Agent Fundamentals

Autonomous AI agents represent a shift from static prompt templates to execution architectures that can inspect state, determine tool parameters dynamically, and recover from intermediate failures. This module defines the architectural primitives of agents, maps their lifecycle phases, and provides decision boundaries for choosing them over structured DAG workflows.

---

## 1. Definitions & Boundaries: Workflows vs. Agents vs. Chatbots

In production, engineering systems are classified by their degree of execution path determinism and runtime control loops:

| Dimension | Deterministic Workflow (DAG) | Conversational Chatbot | Autonomous AI Agent |
| :--- | :--- | :--- | :--- |
| **Execution Path** | Fully predefined (Static graph, no runtime branch derivation). | Fixed conversational loops (Input $\rightarrow$ Output). | Dynamic runtime branching (Derives execution steps based on context). |
| **Decision Maker** | Application code / Compiler rules. | LLM token generation (unconstrained text). | LLM-backed Planner (selecting tools and routing paths). |
| **Tool Execution** | Fixed caller nodes in the code. | No tool execution or hardcoded callouts. | Dynamic tool discovery, parameter binding, and execution. |
| **Failure Recovery** | Static try/catch routing. | User-driven prompting for correction. | Autonomous loop reflection and parameter self-correction. |
| **State Scope** | Linear variables / Context objects. | Chat history lists (append-only). | Complex state graphs, working memory, and episodic checkpointers. |

### When is an Agent the WRONG Choice?
Agents are not general-purpose workflow engines. An agent is the wrong architectural choice when:
1. **The task requires $100\%$ predictability**: Finance accounting pipelines or payroll ledger updates cannot tolerate stochastic routing variations.
2. **Low latency is a hard constraint**: If response latency must remain under $100\text{ ms}$, routing through reasoning/planning loops that take multiple LLM turns (averaging $2\text{--}5\text{ seconds}$ per call) is unacceptable.
3. **The cost-per-query budget is extremely narrow**: Running multi-step planning loops increases input token accumulation rapidly, yielding query costs that scale quadratically with steps.

---

## 2. Execution Mechanics: The Observe → Think → Act Loop

The canonical reasoning loop (exemplified by the **ReAct** framework) is a sequential loop that structures how LLMs execute tool-based reasoning:

```
                  ┌──────────────────────────────┐
                  │           Observe            │
                  │   (Collect environment &     │
                  │        tool outputs)         │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │            Think             │
                  │   (Generate reasoning and    │
                  │      next action plan)       │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │             Act              │
                  │   (Format tool parameters    │
                  │      and execute action)     │
                  └──────────────────────────────┘
```

### Hand Trace of ReAct State Transitions
Let's walk through a tiny ReAct trajectory where the goal is to look up an employee's salary and calculate tax. We denote the context state at step $t$ as $S_t$.

- **Step 0: Initialize Context**
  - **User Goal**: "Calculate 20% tax on John's salary."
  - **Available Tools**: `get_salary(employee_name: str)`, `calculate_percentage(val: float, pct: float)`.
  - **State $S_0$**: `{"goal": "Calculate 20% tax on John's salary", "history": []}`

- **Iteration 1: Think & Act**
  - **LLM Output (Thought)**: "I need to get John's salary first to compute the tax. I will call `get_salary` with John as the argument."
  - **LLM Output (Action)**: `Call: get_salary("John")`
  - **State $S_1$**: Appends Thought and Action metadata to history.

- **Iteration 2: Observe & Think**
  - **Tool Output (Observation)**: `$80,000`
  - **LLM Output (Thought)**: "I have John's salary ($80,000). Now I need to calculate 20% of this value. I will call the `calculate_percentage` tool with value 80000 and percentage 0.20."
  - **LLM Output (Action)**: `Call: calculate_percentage(80000, 0.20)`
  - **State $S_2$**: Appends Observation and new Thought/Action.

- **Iteration 3: Observe, Reflect & Respond**
  - **Tool Output (Observation)**: `16000`
  - **LLM Output (Thought)**: "The calculation returned 16000. This is the final tax value. I will now respond to the user."
  - **LLM Output (Response)**: "John's tax is $16,000."
  - **State $S_3$**: Terminated.

---

## 3. The Complete Agent Lifecycle

The production execution pipeline is detailed in the vector-perfect lifecycle diagram below:

<div class="custom-diagram" style="margin: 24px 0; background-color: #f8fafc; padding: 24px; border: 1px solid #cbd5e1; border-radius: 8px; font-family: inherit;">
    <div style="font-weight: bold; color: #0f172a; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; text-align: center; margin-bottom: 20px;">
        End-to-End Autonomous Agent Execution Lifecycle
    </div>
    
    <!-- Top Row: Goal and Plan -->
    <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #64748b;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #64748b;">01. Goal</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">User Objective</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #2563eb;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #2563eb;">02. Plan</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Decompose Tasks</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #10b981;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #10b981;">03. Reason</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Determine Turn</div>
        </div>
    </div>
    
    <!-- Connector row -->
    <div style="display: flex; justify-content: flex-end; padding-right: 60px; margin-bottom: 20px;">
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(90deg);">➔</div>
    </div>

    <!-- Middle Row: Memory, Tools, Reflect -->
    <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #f59e0b;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #f59e0b;">06. Reflect</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Evaluate Step</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(180deg);">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #ef4444;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #ef4444;">05. Call Tools</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Execute API</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(180deg);">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #7c3aed;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #7c3aed;">04. Retrieve Memory</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Context Retrieval</div>
        </div>
    </div>
    
    <!-- Connector row -->
    <div style="display: flex; justify-content: flex-start; padding-left: 60px; margin-bottom: 20px;">
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(90deg);">➔</div>
    </div>

    <!-- Bottom Row: Respond, Store, Terminate -->
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #ec4899;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #ec4899;">07. Respond</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Formulate Output</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #06b6d4;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #06b6d4;">08. Store Memory</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Checkpoint State</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #0f172a;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #0f172a;">09. Terminate</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Return Final Answer</div>
        </div>
    </div>
</div>

---

## 4. Capabilities & Limitations

To design resilient architectures, developers must isolate what an LLM-backed agent can orchestrate from the failure models native to statistical decoders.

### Agent Capabilities:
- **Semantic Routing**: Mapping complex user intents to correct tools based on schema definitions without hardcoded regex patterns.
- **Dynamic Replanning**: Inspecting error traces (e.g. database query timeout) and routing to alternative database replica endpoints dynamically.
- **Contextual Assembly**: Querying distinct data silos (e.g., Jira APIs, Postgres databases, vector files) and stitching them into a unified contextual prompt.

### Agent Limitations & Failure Modes:
- **Error Saturation**: When tool schemas return errors, agents frequently get trapped in infinite retries, draining API token budgets.
- **Parameter Hallucination**: Generating arguments that do not conform to the schema definitions of the tools.
- **Planning Drift**: Losing focus of the original user objective over multi-step trajectory steps, resulting in redundant execution nodes.

---

## 5. Architectural Decision Flow: Workflow vs. Agent

When designing production architectures, apply this standard choice hierarchy:

```
                      Is execution path predictable?
                                   │
                      ┌────────────┴────────────┐
                      ▼ Yes                     ▼ No
              [Structured DAG]         Do we need dynamic tools?
                                                │
                                       ┌────────┴────────┐
                                       ▼ Yes             ▼ No
                                 [AI Agent]     [Simple Chatbot]
```

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Autonomous agents solve the limitation of static DAG workflows by dynamically adjusting execution paths based on intermediate runtime data and error feedback.
- **Why was it introduced?**
  Introduced to allow applications to navigate open-ended tasks (like crawling code repositories or performing complex multi-source research) where the precise sequence of operations cannot be mapped beforehand.
- **What are its limitations?**
  Higher costs, increased latency, susceptibility to infinite chattering loops, planning drift, and stochastic execution behaviors.
- **Computational Complexity (Time & Memory)**
  - **Time**: $O(T \cdot L_{\text{LLM}})$ where $T$ is the number of agent steps/turns and $L_{\text{LLM}}$ is the latency of a single LLM inference turn.
  - **Memory**: $O(T \cdot L_{\text{tokens}})$ as the prompt accumulates historical execution logs over $T$ steps.
- **Component Variable Denotation Legend**
  - $T$: Number of execution steps/turns.
  - $L_{\text{LLM}}$: Single LLM inference turn latency.
  - $L_{\text{tokens}}$: Average token count per execution step.
- **Production Use Cases**
  - Developer agents resolving GitHub issues (cloning, editing, writing tests, creating pull requests).
  - Multi-source financial audits querying disparate external database schemas dynamically.
- **Follow-up questions interviewers ask**
  - *How do you prevent an agent from looping infinitely if a tool consistently fails?* (Implement a hard max-steps counter ($T_{\text{max}}$) and circuit breakers on tool error rates).
  - *How do you limit token usage overhead as step count scales?* (Use sliding window token truncations and recursive summarization of previous trajectories).
