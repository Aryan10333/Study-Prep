# Module 02: Reasoning & Planning Strategies

This module focuses on reasoning and planning paradigms, detailing how models generate intermediate steps, structure thought trees, validate schemas, and dynamically adjust execution routes under deterministic safety boundaries.

---

## 1. Reasoning Paradigms: CoT, ToT, and Reflexion

Reasoning architectures alter the decoding pipeline from flat next-token generation to structured multi-step execution graphs:

```
Chain-of-Thought (CoT):
Input ──► Thought ──► Action ──► Output

Tree-of-Thoughts (ToT):
                  ┌──► Thought A1 (Score: 0.8) ──► Thought A2 (Score: 0.9) [Winner]
Input ──► Planner ┼──► Thought B1 (Score: 0.3) [Pruned]
                  └──► Thought C1 (Score: 0.5) ──► Thought C2 (Score: 0.2) [Pruned]
```

- **Chain-of-Thought (CoT)**: Employs a linear prompt sequence: "Let's think step-by-step". By generating intermediate tokens, the model allocates compute parameters to intermediate states before committing to the final answer.
- **Tree-of-Thoughts (ToT)**: Models planning as search over a tree of intermediate thoughts. It generates multiple thought candidates at each branch, evaluates their probability of task success using a value heuristic, and navigates using Breadth-First Search (BFS) or Depth-First Search (DFS).
- **Reflection / Reflexion**: Introduces a critic loop. The agent acts, gets environment feedback, prompts an evaluator LLM to analyze the trajectory for failure points, and stores the self-reflection in context memory for the next loop.

---

## 2. Tree-of-Thoughts (ToT) Worked Walkthrough

Let's mathematically trace a Tree-of-Thoughts traversal for a mathematical puzzle.

### Mathematical Formulation
Let $s$ be the state of a partial plan. At each step, a candidate thought generator generates a set of next-step thought actions:
$$C(s) = \{a_1, a_2, \dots, a_b\}$$
An evaluator LLM acts as a heuristic function scoring each state:
$$V(s) \in [0, 1.0]$$
The score of a path $P = (s_0, s_1, \dots, s_d)$ is calculated as the product of the branch scores:
$$\text{Score}(P) = \prod_{i=1}^d V(s_i)$$

### Step-by-Step Trajectory Trace
- **State $s_0$ (Root)**: The input puzzle.
- **Step 1: Branch Generation**
  - The model generates two potential first steps:
    - Thought $a_1$: "Group variables by common denominators" (State $s_1$)
    - Thought $a_2$: "Expand expressions using binomial expansion" (State $s_2$)
- **Step 2: Heuristic Evaluation**
  - We prompt the evaluator LLM to score these partial paths:
    - $V(s_1) = 0.80$ (highly viable)
    - $V(s_2) = 0.40$ (moderate chance of success)
- **Step 3: Branch Exploration (Depth 2)**
  - We expand $s_1$ (since $0.80 > 0.40$). Generator outputs two child branches:
    - Thought $a_{1.1}$: "Cancel common denominator variables" (State $s_{1.1}$)
    - Thought $a_{1.2}$: "Multiply denominator terms across equality" (State $s_{1.2}$)
  - Evaluator scores child states:
    - $V(s_{1.1}) = 0.90$
    - $V(s_{1.2}) = 0.30$
- **Step 4: Path Selection Heuristic**
  - Calculate cumulative path scores:
    - $\text{Score}(P_{1.1}) = V(s_1) \times V(s_{1.1}) = 0.80 \times 0.90 = 0.72$
    - $\text{Score}(P_{1.2}) = V(s_1) \times V(s_{1.2}) = 0.80 \times 0.30 = 0.24$
  - The path $(s_0 \rightarrow s_1 \rightarrow s_{1.1})$ is selected as the winner, and the agent proceeds to execution.

---

## 3. Planning Strategies: Decomposition & Replanning

- **Plan-and-Execute**: Decouples macro-planning from execution. A planner LLM generates a complete list of sub-tasks upfront. An executor agent processes each task sequentially. This minimizes planning overhead and latency.
- **Dynamic Replanning**: Used when tasks are stochastic. After each executor step, the system compares the environment output against the plan expectations. If there is a mismatch, the planner is re-invoked to modify the remaining task list.

---

## 4. Structured Output & Schema Validation

Relying on raw LLM strings for routing decisions is a critical production failure mode. To enforce deterministic transitions, systems use structured output frameworks:

- **JSON Schema / Pydantic Validation**:
  We force the LLM to output a JSON object conforming to a schema. If the output fails validation, the system executes an automated retry loop, passing the validation error logs back to the LLM.
- **Grammar-Constrained Decoding (Outlines/Guidance)**:
  Bypasses post-generation validation. It modifies the LLM's token probability logits at the sampling layer. Tokens that do not conform to the regex or schema rules are assigned a probability of zero:
  $$P(\text{token} \mid \text{invalid}) = 0$$
  This guarantees that the model output is $100\%$ valid on the first attempt, saving token costs and reducing latency.

---

## 5. Architectural Design Patterns

```
Router Pattern:
                     ┌──► Tool A
User Query ──► Router┼──► Tool B
                     └──► SFT Model

Supervisor Pattern:
                       ┌──► Researcher Agent
User Query ──► Supervisor ┼──► Coder Agent
                       └──► Reviewer Agent
```

1. **Router**: A lightweight classifier model that directs queries either to specialized tools, specific static code blocks, or a downstream model.
2. **Supervisor**: A central controller agent that delegates tasks to sub-agents, monitors their outputs, and manages execution state transitions.
3. **Reflection Loop**: An agent sends its work to an evaluator agent, receives critique, and iterates until the score meets the target threshold.
4. **Human Approval Gateway**: Pauses execution graph nodes before sensitive operations (e.g. database deletions or payments), resuming only upon manual human validation.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Reasoning and planning strategies solve the limitation of LLMs struggle with complex multi-step reasoning by forcing the model to generate intermediate steps (scratchpad) and validating outputs structurally.
- **Why was it introduced?**
  Introduced to prevent model reasoning drift, enforce type-safety on API calls, and support complex path search and critiquing loops.
- **What are its limitations?**
  - **Latency**: Multiple reasoning steps multiply the time-to-first-token.
  - **Cost**: ToT and reflection loops exponentially increase input/output token usage.
- **Computational Complexity (Time & Memory)**
  - **Time**: ToT with depth $d$ and branching factor $b$ requires $O(b^d)$ LLM evaluation calls if fully explored.
  - **Memory**: Graph states require $O(d)$ path logs.
- **Component Variable Denotation Legend**
  - $b$: Branching factor (number of child nodes per state).
  - $d$: Depth of the search tree.
  - $V(s)$: Heuristic evaluation score of state $s$.
- **Production Use Cases**
  - Automated code generation where SFT output is compiled and tested, feeding compiler errors back to the model in a reflection loop.
  - Legal document validation ensuring strict compliance via schema constraints.
- **Follow-up questions interviewers ask**
  - *How does logit bias constraint compare to post-generation schema validation?* (Logit bias constraint guarantees valid outputs on the first turn and saves token costs, but requires access to model logit probabilities, which is not supported by all cloud API providers).
  - *When is Plan-and-Execute preferred over ReAct?* (Plan-and-Execute is preferred for long, stable workflows with independent sub-tasks to minimize latency. ReAct is preferred when step $N+1$ depends heavily on the result of step $N$).
