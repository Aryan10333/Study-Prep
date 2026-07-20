# Top 20 Agentic AI & Multi-Agent Frameworks Interview Questions

This flashcard guide presents 20 high-frequency interview questions asked by top tech companies (Meta, OpenAI, Google, Anthropic, LangChain) covering Agentic AI architectures, ReAct loops, Pydantic tool calling schemas, LangGraph state graphs, CrewAI, AutoGen, and agent memory systems.

---

### Q1: What is the core difference between a passive LLM call and an Agentic AI loop?
- **Answer:** A passive LLM call is single-shot text generation ($X \rightarrow Y$). An Agentic AI system operates in an autonomous loop, interleaving reasoning ("Thoughts") with environmental tool executions ("Actions") and observing feedback ("Observations") to dynamically re-plan until a complex goal is achieved.

---

### Q2: Explain the ReAct (Reasoning + Acting) architecture.
- **Answer:** ReAct structures agent decision-making into a repeating loop:
  1. **Thought:** LLM generates internal reasoning step analyzing task progress.
  2. **Action:** LLM selects and invokes an external tool via JSON Schema arguments.
  3. **Observation:** Tool output data is appended to agent context, allowing the LLM to inspect the result and plan its next turn.

---

### Q3: How do you prevent infinite execution loops in ReAct agents?
- **Answer:** Enforce strict guardrails:
  1. Set a maximum step threshold ($\text{Max\_Steps} \le 8$).
  2. Maintain a tool invocation hash map to detect repeated identical tool calls.
  3. Integrate a Reflexion evaluation node that terminates execution if task progress stalls across 2 consecutive turns.

---

### Q4: How does Pydantic schema validation protect Tool Calling pipelines?
- **Answer:** Pydantic schema validation acts as a security firewall between the LLM and native code execution. It type-checks incoming LLM JSON arguments (e.g. enforcing `int` range bounds), strips unknown parameters, and prevents SQL injection or malformed code execution before function invocation.

---

### Q5: Compare LangGraph, CrewAI, and AutoGen for multi-agent system design.
- **Answer:**
  - **LangGraph:** Best for deterministic, production-grade StateGraph state machines with explicit typed states and cyclic routing.
  - **CrewAI:** Best for role-based hierarchical worker teams automating enterprise workflows.
  - **AutoGen:** Best for autonomous peer-to-peer chat simulations and multi-agent debate networks.

---

### Q6: What is a Supervisor Agent in a Multi-Agent System (MAS)?
- **Answer:** A Supervisor Agent is a centralized router node that receives the global user goal, inspects shared task state, and dynamically routes execution to specialized worker agents (e.g. Researcher, Coder, Reviewer), orchestrating workflow completion.

---

### Q7: Explain Reflexion and how it improves agent problem-solving over time.
- **Answer:** Reflexion adds a self-reflection loop where an evaluator model critiques the agent's failed trajectory, generates a textual feedback summary of *why* the attempt failed, and writes that summary into short-term memory before retrying the task.

---

### Q8: How do you handle tool execution exceptions in production agent systems?
- **Answer:** Catch the exception inside the tool execution wrapper, format the exception traceback into a human-readable string observation (e.g. `"Tool Error: Invalid API Key or Timeout"`), and pass it back into the agent context. This enables the LLM to analyze the error and try a fallback tool.

---

### Q9: What are the three layers of an Agent Memory Architecture?
- **Answer:**
  1. **Short-Term Working Memory:** Sliding message buffer containing recent conversation turns (RAM).
  2. **Episodic Long-Term Memory:** Vector DB index storing semantic vector embeddings of past agent task trajectories.
  3. **Semantic Memory:** Persistent summaries of user entity profiles, preferences, and key facts.

---

### Q10: Write a Python decorator for defining a LangChain tool with Pydantic argument validation.
```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20)

@tool(args_schema=SearchArgs)
def web_search(query: str, max_results: int = 5) -> str:
    """Executes web search for query."""
    return f"Retrieved top {max_results} results for '{query}'"
```

---

### Q11: How do you reduce LLM context token usage in long-running agent chats?
- **Answer:** Implement **Summary Memory Truncation**. Periodically compress older conversation turns into a concise 100-token summary string, retain only the most recent 4 active turns in RAM, and archive full turns in a Vector DB.

---

### Q12: What is the Plan-and-Solve agent paradigm, and when should you prefer it over ReAct?
- **Answer:** Plan-and-Solve asks the LLM to generate a complete high-level sub-goal plan upfront before executing any tools. Prefer Plan-and-Solve for complex multi-step tasks where single-step ReAct agents risk getting sidetracked by initial tool outputs.

---

### Q13: Why is temperature set to 0.0 during Tool Calling inference?
- **Answer:** Temperature $T=0.0$ forces deterministic greedy sampling, guaranteeing that the LLM outputs valid JSON syntax conforming strictly to the requested function signature without creative formatting errors.

---

### Q14: Derive the task success probability of a 2-step ReAct agent given single-shot tool success $p=0.70$ and error recovery rate $r=0.80$.
- **Answer:** 
  $$P(\text{Success}) = 1 - (1 - p)(1 - r) = 1 - (1 - 0.70)(1 - 0.80) = 1 - (0.30)(0.20) = 1 - 0.06 = \mathbf{0.94 \ (94\%)}$$

---

### Q15: What is Human-in-the-Loop (HITL) in Agentic AI pipelines?
- **Answer:** Human-in-the-Loop pauses agent execution at critical high-risk tool call state nodes (such as sending emails, executing database writes, or issuing financial transactions), requiring an explicit human approval or modification before resuming execution.

---

### Q16: How do you prevent peer-to-peer multi-agent chat networks (e.g. AutoGen) from ping-ponging endlessly?
- **Answer:** Set strict `max_consecutive_auto_reply` limits, implement a dedicated Critic agent with explicit `TERMINATE` keyword triggers, and enforce structured state transition graphs using LangGraph.

---

### Q17: What is the difference between synchronous and asynchronous tool calling in Agentic AI?
- **Answer:** Synchronous tool calling blocks agent execution until the tool returns data. Asynchronous tool calling allows the agent to trigger multiple parallel tool calls (e.g. searching 3 distinct databases simultaneously via `asyncio.gather`) and process results as they return.

---

### Q18: What is State Graph persistence in LangGraph?
- **Answer:** State Graph persistence saves every node state checkpoint into a persistent checkpointer database (e.g. SQLite / Redis). This allows agent workflows to be paused, inspected, time-traveled, and resumed after system restarts.

---

### Q19: How do you evaluate Agentic AI performance in production?
- **Answer:**
  1. **Trajectory Efficiency:** Number of steps/tool calls required to reach goal.
  2. **Tool Selection Accuracy:** Percentage of correct tool choices vs invalid calls.
  3. **Task Completion Rate:** Percentage of benchmark tasks successfully solved without human intervention.

---

### Q20: How do you build an agentic web scraper that handles dynamic UI changes?
- **Answer:** Use a ReAct agent equipped with vision capabilities and DOM-parsing tools. If an element selector fails, the agent inspects the screenshot observation, re-locates the UI button visually, and issues a click action on updated coordinates.
