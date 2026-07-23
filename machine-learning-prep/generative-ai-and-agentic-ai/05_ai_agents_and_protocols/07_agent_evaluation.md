# Module 07: Agent Evaluation & Benchmarking

Evaluating agents is difficult because their execution paths are dynamic and stochastic. Traditional static Q&A benchmarks fail because they evaluate only the final output. This module details agent evaluation paradigms, key production metrics, benchmarks, and LLM-as-a-Judge validation.

---

## 1. Evaluation Paradigms: Offline Trajectories vs. Online Traces

Production agent evaluation is divided into two phases:

- **Offline Trajectory Benchmarking**:
  Running the agent on a golden dataset of pre-defined tasks. The system logs the entire step-by-step trajectory (tools called, arguments passed, thinking paths) and evaluates intermediate steps against ideal trajectories.
  - *Goal*: Run evaluations before code deployment to catch regressions.
- **Online Live Trace Auditing**:
  Monitoring real-time production traces (using OpenTelemetry or platforms like Langfuse). Real-time telemetry alerts developers if loop steps spike or tool failures rise.
  - *Goal*: Identify live drift, latency bottlenecks, and prompt injection attempts.

---

## 2. Industry Benchmarks: SWE-bench, GAIA, and AgentBench

- **SWE-bench**:
  Evaluates agents on real GitHub issues from open-source repositories. The agent is provided with the repository code and issue description. Task success is validated by running unit tests on the code modified by the agent.
- **GAIA (General AI Assistants)**:
  Exposes complex multi-modal, multi-step assistant tasks (e.g. download a PDF, read page 5, look up financial rates, calculate result). Requires tools like web search, file parsing, and math execution.
- **AgentBench**:
  A systematic benchmark evaluating agents across diverse environments (operating systems, web browsers, databases, and card games).

---

## 3. Core Metrics & Production KPIs: Worked Trace

To measure agent performance, we track task success alongside execution efficiency KPIs.

### Mathematical Formulation of KPIs
1. **Goal Completion Rate (GCR)**:
   $$\text{GCR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Task}_i \text{ Completed successfully})$$
   where $\mathbb{I}$ is the indicator function.
2. **Trajectory Efficiency (TE)**:
   $$\text{TE} = \frac{\text{Steps}_{\text{optimal}}}{\text{Steps}_{\text{actual}}}$$
   measures how close the agent was to the optimal path. If the agent enters redundant loops, the efficiency decays.
3. **Average Tool Call Volume**:
   $$\text{AvgToolCalls} = \frac{1}{N} \sum_{i=1}^N \text{ToolCalls}_i$$
4. **Recovery Success Rate (RSR)**:
   $$\text{RSR} = \frac{\text{Errors Resolved}}{\text{Total Errors Encountered}}$$
   measures the agent's ability to self-correct using reflection loops.

### Hand-Calculation on 3 Test Trajectories
Let's evaluate a file-processing agent on $N=3$ test cases. The optimal path requires $\text{Steps}_{\text{optimal}} = 3$ steps.

| Test Case | Actual Steps | Success? | Tool Calls | Errors Encountered | Errors Resolved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Case 1 | 3 | Yes | 3 | 0 | 0 |
| Case 2 | 5 | Yes | 4 | 2 | 2 (Self-corrected arguments) |
| Case 3 | 6 | No | 4 | 1 | 0 (Infinite loop retry) |

#### Calculations:
- **Goal Completion Rate (GCR)**:
  $$\text{GCR} = \frac{1 + 1 + 0}{3} = 66.67\%$$
- **Trajectory Efficiency (TE) per Case**:
  - Case 1: $\text{TE}_1 = \frac{3}{3} = 1.00$
  - Case 2: $\text{TE}_2 = \frac{3}{5} = 0.60$
  - Case 3: $\text{TE}_3 = \frac{3}{6} = 0.50$
  - **Mean Trajectory Efficiency (MTE)**:
    $$\text{MTE} = \frac{1.00 + 0.60 + 0.50}{3} = 0.70\text{ (or }70\%)$$
- **Average Tool Call Volume**:
  $$\text{AvgToolCalls} = \frac{3 + 4 + 4}{3} = 3.67\text{ calls}$$
- **Recovery Success Rate (RSR)**:
  $$\text{RSR} = \frac{2}{2 + 1} = 66.67\%$$

---

## 4. LLM-as-a-Judge Trajectory Validation

When ground truth tests are unavailable, we evaluate step correctness using a larger model (LLM-as-a-Judge). The evaluator is provided with the goal, tool inputs, and reasoning trajectory:

```
Trajectory Audit Prompt:
Evaluate the agent's intermediate step reasoning:
Goal: [Find John's Manager]
Step 1 Action: call get_salary("John")
Evaluator Verdict: [REJECT] Step 1 called get_salary instead of get_org_chart. Reasoning is invalid for target goal.
```

Using clear scoring rubrics (e.g., rating step precision from $1$ to $5$), we evaluate reasoning steps and tool arguments before code updates are merged.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Agent evaluation measures the accuracy and cost of non-deterministic, multi-step execution graphs.
- **Why was it introduced?**
  Introduced because traditional semantic similarities (BLEU, ROUGE) only measure final text formatting, neglecting intermediate tool errors and infinite loop costs.
- **What are its limitations?**
  - **Cost**: Trajectory evaluation requires running multi-turn loops, which is expensive.
  - **Flakiness**: Stochastic model runs make it difficult to reproduce failures.
- **Computational Complexity (Time & Memory)**
  - **Time**: Evaluator scaling is $O(T \cdot L_{\text{step}})$ where $T$ is the number of trajectory steps.
  - **Memory**: Telemetry servers require storage scaling as $O(N \cdot T \cdot L_{\text{log}})$ bytes.
- **Component Variable Denotation Legend**
  - $N$: Number of test cases evaluated.
  - $\text{Steps}_{\text{optimal}}$: Target steps to solve the task optimally.
  - $\text{Steps}_{\text{actual}}$: Real steps taken by the agent.
- **Production Use Cases**
  - Continuous integration (CI) tests that reject deployment branch merges if agent tool selection accuracy decays below $95\%$.
  - Alerting on production dashboards when average tool call volumes exceed $10$ per request.
- **Follow-up questions interviewers ask**
  - *Why is SWE-bench task success checked by tests rather than LLM verification?* (LLM verification can miss logic bugs or syntax errors. Unit tests verify the modified code behaves correctly under assertions).
  - *How do you evaluate agent safety boundaries dynamically?* (Inject adversarial user queries into offline tests and measure the agent's rate of boundary rejection).
