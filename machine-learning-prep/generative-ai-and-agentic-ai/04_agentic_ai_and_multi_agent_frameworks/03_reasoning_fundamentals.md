# Module 03: Reasoning Fundamentals

This module covers the core logical reasoning paradigms that enable foundation models to behave as agents: Chain of Thought (CoT), Self-Consistency, Least-to-Most prompting, and the foundational Reasoning + Acting (ReAct) loop.

> **Notebook Companion**: `03_reasoning_fundamentals.ipynb`

---

## 1. Chain of Thought (CoT) & Self-Consistency

### Chain of Thought (CoT)
- **Concept**: Prompts the LLM to output a sequence of intermediate reasoning steps before generating the final answer (`"Let's think step by step"`).
- **Mechanism**: Shifts the model from single-step pattern matching (high error rate on multi-hop logical tasks) to multi-step execution. This allows the model to allocate more tokens (and compute) to reasoning.

### Self-Consistency
- **Concept**: Generates $K$ independent reasoning paths (CoT) at temperature $T > 0$, then performs a majority vote over the final answers.
- **Why it works**: Eliminates single-path reasoning errors by selecting the consensus result over multiple sample outputs.

```text
Prompt ──► [Path 1 (T=0.7)] ──► Ans: 42  ┐
       ──► [Path 2 (T=0.7)] ──► Ans: 18  ├─► [Majority Vote] ──► Final Ans: 42
       ──► [Path 3 (T=0.7)] ──► Ans: 42  ┘
```

---

## 2. Least-to-Most Prompting

- **Concept**: Solves complex tasks by first decomposing them into simpler sub-tasks, then solving them sequentially (building on top of the solutions from previous sub-tasks).
- **Production Advantage**: Prevents logical degradation in long reasoning chains by focusing model attention on small, manageable contexts.

---

## 3. The ReAct (Reasoning + Acting) Paradigm

Introduced to combine **Reasoning** (CoT) and **Action** (tool execution). ReAct interleaves these steps in a loop:

```text
User Input ──► Thought ──► Action (Tool Call) ──► Observation (Tool Output) ──► Thought ──► Final Answer
```

### Trace Walkthrough of a ReAct Loop:
- **Goal**: Find the age difference between two software engineering frameworks.
  - **Thought 1**: "I need to find the release year of Framework A first. I will use the Search tool."
  - **Action 1**: `Search[Framework A release year]`
  - **Observation 1**: "Framework A was first released in 2013."
  - **Thought 2**: "Now I need to find the release year of Framework B. I will search for it."
  - **Action 2**: `Search[Framework B release year]`
  - **Observation 2**: "Framework B was first released in 2019."
  - **Thought 3**: "Framework A is from 2013, and Framework B is from 2019. The difference is $2019 - 2013 = 6$ years. I have the final answer."
  - **Final Answer**: "The age difference is 6 years."

---

## 4. Comparison of Reasoning Paradigms

| Paradigm | Environmental Interaction | Computation Overhead | Latency | Primary Limitation |
|---|---|---|---|---|
| **Chain of Thought (CoT)** | None (static text) | Low (adds output tokens) | Low | Can hallucinate facts without tools |
| **Self-Consistency** | None (static text) | High (runs $K$ parallel paths) | Low (if run in parallel) | Does not solve facts retrieval |
| **Least-to-Most** | None (static text) | Moderate (chained steps) | Moderate | Requires rigid prompt design |
| **ReAct** | Full (interacts with tools) | Very High (loop token buildup) | High | Subject to infinite execution loops |

---

## 5. Detailed Computational Complexity (Time & Memory)

- **Self-Consistency Voting Time**: $O(K \cdot N_{len})$ generation token cost.
- **ReAct Execution Context Memory**: $O(T^2)$ quadratic context window growth as history scales with every loop turn.
- **Component Denotations**:
  - $K$: Number of parallel sample paths generated for Self-Consistency.
  - $N_{len}$: Average length in tokens of a single reasoning chain.
  - $T$: Number of turns executed in the ReAct loop.

---

## 6. Interview Questions & Production Trade-offs

### What problem does this solve?
LLMs are standard text generators that struggle with multi-hop calculations or information retrieval. Interleaving reasoning (Thought) and actions (Tools) resolves hallucination by grounding observations in external tools.

### Why was it introduced?
To merge the benefits of conversational reasoners (CoT) with task-oriented action engines (tool execution databases).

### What are its limitations?
- **Cascading Failures**: If Observation 1 is incorrect, Thought 2 and all subsequent steps will fail.
- **Token Inflation**: Chaining past prompts and tool results in a loop scales context window size quadratically.

### Production Use Cases:
- Customer billing triage agents verifying purchase history, database entries, and updating client states.
- Automated API route selectors that dynamically fetch parameters and verify routing results.

### Follow-up Questions Interviewers Ask:
1. *Why does ReAct combine reasoning (Thought) and action (Action) rather than just outputting action commands directly?*
   - **Answer**: Direct action outputting (e.g. JSON-only APIs) lacks explanation traces. The intermediate `Thought` acts as a scratchpad where the model aligns context, preventing sudden jumps in logic, stabilizing parameter choices, and allowing developers to debug the agent's inner chain of logic.
2. *How do you build a ReAct parser without external frameworks like LangChain?*
   - **Answer**: Implement a clean loop. Direct the LLM via system prompting to output text containing specific delimiters (e.g., `Thought: <reasoning>`, `Action: <tool_name>[<arguments>]`, `Observation: <results>`). In Python, use regex search patterns to capture the `Action:` block, run the local python function matching that action name, and append `Observation: <tool_output>` to the dialogue history for the next iteration.
