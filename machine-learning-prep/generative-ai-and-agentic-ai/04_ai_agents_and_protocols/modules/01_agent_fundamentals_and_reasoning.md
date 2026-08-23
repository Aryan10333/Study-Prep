# Module 01: Agent Fundamentals & Reasoning Patterns

## 1. Introduction & Intuition

### The Core Bottleneck
A fixed pipeline — call the model once, maybe retrieve something first, generate an answer, done — has a hardcoded control flow: the sequence of steps is decided by the *developer*, at design time, not by the *model*, at run time. That's fine for tasks whose steps are genuinely fixed and known in advance. It breaks down the moment the right sequence of steps actually depends on what happens along the way — the right next action depends on what the *previous* action returned, and that can't be known until the model actually sees it. An agent is what you get when you hand that step-by-step control flow decision to the model itself: instead of "always do A then B then C," the model decides, at each step, whether to act, what to act with, and whether the result means it's done or needs to try something else. That flexibility is genuinely powerful — and genuinely more expensive, slower, and harder to predict than a fixed pipeline, which is exactly why *when* to reach for an agent at all is this module's central question, not just *how* one works.

### High-Level Intuition
A fixed pipeline is a recipe: do these steps, in this order, every time, regardless of how the dish is turning out. An agent is a cook: taste as you go, decide whether it needs more salt, and only move to plating once it actually tastes right — the *sequence* of actions is decided dynamically, in response to what's actually observed, not fixed in advance. This is powerful precisely when you can't write the recipe in advance because the right next step genuinely depends on information you don't have until a previous step returns it. It's needless overhead when the recipe was always going to be the same regardless of what happened along the way — nobody needs a cook improvising a dish that only ever has one correct sequence of steps.

---

## 2. Core Concepts & Mathematical Formulation

This module is architectural throughout — reasoning patterns and the agent-vs-pipeline decision are procedural/comparative concepts, not closed-form calculations, consistent with how `03_advanced_rag` treated its own purely-architectural modules (Query Transformation, GraphRAG). What matters for interview readiness is the *mechanism* and, critically, *when each pattern is actually the right choice* — not a formula to memorize.

### ReAct: Reason + Act Interleaving

#### Intuition & Practical Use
ReAct is the foundational agentic reasoning pattern: interleave explicit reasoning ("Thought") with concrete actions ("Action") and their results ("Observation"), in a loop, until the model decides it has enough information to answer. The model doesn't just silently decide what tool to call — it first reasons in text about *why*, which serves two real purposes: it measurably improves the *quality* of the model's action choices (reasoning before acting reduces impulsive, poorly-justified tool calls), and it makes the agent's behavior *inspectable* — a human debugging a bad run can read the Thought steps and see exactly where the reasoning went wrong, not just guess from the final answer.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 700 320" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="350" y="24" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">ReAct: Reason + Act Interleaving Loop</text>

  <rect x="270" y="55" width="160" height="50" rx="6" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="350" y="85" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="600">Thought</text>

  <rect x="470" y="130" width="160" height="50" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="550" y="160" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">Action (tool call)</text>

  <rect x="270" y="205" width="160" height="50" rx="6" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="350" y="235" text-anchor="middle" font-size="12" fill="#065f46" font-weight="600">Observation</text>

  <rect x="70" y="130" width="160" height="50" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="150" y="152" text-anchor="middle" font-size="11" fill="#854d0e" font-weight="600">Enough info?</text>
  <text x="150" y="167" text-anchor="middle" font-size="9" fill="#854d0e">yes -&gt; final answer</text>

  <g stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow01a)">
    <path d="M420,95 Q470,110 500,128"/>
    <path d="M540,180 Q470,200 420,215"/>
    <path d="M280,225 Q180,190 150,182"/>
    <path d="M230,145 Q250,110 280,90"/>
  </g>
  <defs>
    <marker id="arrow01a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <rect x="70" y="270" width="560" height="35" rx="5" fill="#fef2f2" stroke="#dc2626" stroke-width="1.3"/>
  <text x="350" y="292" text-anchor="middle" font-size="10" fill="#991b1b">No explicit stopping bound here -- Module 09's termination guards are what actually cap this loop in production.</text>
</svg>
</div>

### Chain-of-Thought vs. Agentic Reasoning

#### Intuition & Practical Use
Chain-of-Thought (CoT) is reasoning *within* a single generation — the model writes out intermediate reasoning steps before its final answer, but it's still one call, with no ability to go fetch new information partway through. Agentic reasoning (ReAct and beyond) is reasoning *across* multiple calls, each informed by real, new information (a tool's actual output) the model didn't have when it started. The distinction matters: CoT can only ever reorganize and reason over what the model already knows or was given up front; an agent can reason, then go find out something it didn't know, then reason again with that new fact in hand. CoT reduces reasoning errors on a fixed set of information; agentic reasoning is what's actually needed when the task requires information the model doesn't have until it looks.

### Plan-and-Execute vs. Purely Reactive Agents

#### Intuition & Practical Use
A purely reactive agent (like the ReAct loop above) decides its next single action based only on what it's seen so far — it never commits to a full plan up front, which makes it flexible but means it can wander: re-deciding its approach after every single observation, sometimes losing track of the original goal across many steps. A plan-and-execute agent instead first produces an explicit multi-step plan, *then* executes each step (potentially still reactively within a step), checking back against the plan rather than re-deciding the whole approach from scratch every time. The trade-off is real: an upfront plan gives structure and makes long-horizon tasks less likely to wander, but a plan committed to too early can be wrong in ways a purely reactive agent would have naturally adapted around — the right choice depends on how *predictable* the task's structure is upfront versus how much it depends on information only discoverable along the way.

### When NOT to Use an Agent: A Formal Decision Framework

#### Intuition & Practical Use
This is the single highest-leverage question in this entire topic, and it's worth answering with a structured comparison rather than intuition alone. Four real architectural options exist, in increasing order of flexibility *and* cost/complexity/unpredictability — the right choice is the *least* flexible option that still genuinely satisfies the task, not the most sophisticated one available:

| Architecture | Complexity | Cost | Latency | Reliability | Controllability | Right When |
|---|---|---|---|---|---|---|
| **Single LLM call** | Lowest | Lowest — one call | Lowest — one round trip | Highest — no compounding steps to fail | Highest — fully predictable input/output | The task is answerable from one prompt with no external information or multi-step action needed |
| **Deterministic workflow** | Low-Medium | Low — fixed, known number of calls | Predictable — fixed number of steps | High — each step is independently testable | High — the sequence is fixed by the developer, not the model | The steps and their order are genuinely knowable in advance, even if the content varies |
| **Single agent** | Medium-High | Variable, often higher — unbounded step count without a guard | Variable and unpredictable | Lower — errors can compound across self-directed steps | Lower — the model decides the sequence, not the developer | The right sequence of steps genuinely can't be known until runtime, but one coherent line of reasoning suffices |
| **Multi-agent system** | Highest | Highest — cost multiplies across agents | Highest and least predictable | Lowest — new, cross-agent failure modes appear (Module 06) | Lowest — coordination adds another layer of non-determinism | The task genuinely decomposes into specialized sub-problems that benefit from separate, focused agents — not merely because it "sounds complex" |

The real discipline is climbing this table only as far as the task's genuine requirements force you to — a task solvable by a deterministic workflow gains nothing from being built as an agent except more ways to fail, more cost, and less predictability. (`03_advanced_rag` Module 08's Agentic RAG is the retrieval-specific instance of exactly this same table: a single agent reached for specifically when a fixed retrieve-then-generate pass demonstrably isn't good enough, not by default.)

---

## 3. Implementation & Reference Code

Below is a minimal, self-contained ReAct-style loop skeleton — illustrating the Thought → Action → Observation mechanics and the plan-and-execute vs. purely-reactive distinction, not a production framework. Real tool-calling mechanics (schema, parallel execution, idempotency) are Module 02's subject.

```python
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class Step:
    step_type: StepType
    content: str


@dataclass
class ReactTrace:
    """Full Thought/Action/Observation history for one agent run -- the inspectable
    record ReAct's whole value proposition depends on."""
    steps: list[Step] = field(default_factory=list)

    def log(self, step_type: StepType, content: str) -> None:
        self.steps.append(Step(step_type, content))


def react_loop(query: str, reason_fn, act_fn, is_sufficient_fn, max_steps: int = 5) -> tuple[str, ReactTrace]:
    """A minimal ReAct loop: Thought -> Action -> Observation, repeated until
    is_sufficient_fn says the model has enough information, or max_steps is hit.
    max_steps is a real, hard-coded termination guard -- never left implicit."""
    trace = ReactTrace()
    context = query

    for _ in range(max_steps):
        thought = reason_fn(context)
        trace.log(StepType.THOUGHT, thought)

        if is_sufficient_fn(thought):
            trace.log(StepType.FINAL_ANSWER, thought)
            return thought, trace

        action = act_fn(thought)
        trace.log(StepType.ACTION, action)

        observation = f"[real tool execution result for: {action}]"
        trace.log(StepType.OBSERVATION, observation)
        context = f"{context}\nThought: {thought}\nAction: {action}\nObservation: {observation}"

    trace.log(StepType.FINAL_ANSWER, "[terminated: max_steps reached without sufficient information]")
    return "[terminated: max_steps reached]", trace


def plan_and_execute(query: str, plan_fn, execute_step_fn) -> tuple[list[str], ReactTrace]:
    """Plan-and-execute: commit to a full plan upfront, then execute each step,
    contrasted against react_loop's step-by-step re-deciding above."""
    trace = ReactTrace()
    plan = plan_fn(query)
    trace.log(StepType.THOUGHT, f"Committed plan: {plan}")

    results = []
    for step in plan:
        result = execute_step_fn(step)
        trace.log(StepType.ACTION, step)
        trace.log(StepType.OBSERVATION, result)
        results.append(result)
    return results, trace


if __name__ == "__main__":
    # Simulate a ReAct run that needs exactly 2 real tool calls before it has enough information
    call_count = {"n": 0}

    def fake_reason(context: str) -> str:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return f"I need more information (step {call_count['n']})"
        return "I now have enough information to answer: the answer is 42"

    def fake_act(thought: str) -> str:
        return f"lookup_tool(query='step {call_count['n']}')"

    def fake_is_sufficient(thought: str) -> bool:
        return "enough information to answer" in thought

    answer, trace = react_loop("What is the answer?", fake_reason, fake_act, fake_is_sufficient, max_steps=5)
    print(f"Final answer: {answer}")
    print(f"Trace length: {len(trace.steps)} steps")
    for step in trace.steps:
        print(f"  [{step.step_type.value}] {step.content}")

    assert "42" in answer
    assert len(trace.steps) == 8  # 2 full (thought,action,observation) cycles = 6, + final (thought,final_answer) pair = 8
    print("\nReAct loop verified: converged in 2 tool-call cycles, trace fully inspectable.")

    # Verify the termination guard: a reasoner that never becomes sufficient still stops at max_steps
    def never_sufficient(context: str) -> str:
        return "still thinking..."

    _, exhausted_trace = react_loop("Unanswerable", never_sufficient, fake_act, lambda t: False, max_steps=3)
    action_steps = [s for s in exhausted_trace.steps if s.step_type == StepType.ACTION]
    assert len(action_steps) == 3, "Loop must stop at max_steps, not loop forever"
    print(f"Termination guard verified: stopped after {len(action_steps)} action steps, not indefinitely")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Handling tasks whose correct sequence of steps genuinely depends on information only available at runtime, which a fixed, developer-authored pipeline structurally cannot adapt to.
* **Why Introduced over Legacy Approaches:** A fixed pipeline or a single CoT-reasoning call can only ever reorganize information already available up front; an agentic loop can go fetch genuinely new information mid-task and incorporate it into its next decision, which neither a fixed pipeline nor single-call CoT can do.
* **Key Failure Modes & Limitations:** Unbounded reasoning cycles without an explicit stopping criterion (the `max_steps` guard in the reference code exists specifically to prevent this); non-determinism making the same query potentially take a different path on different runs; and reasoning-quality-dependent action choices — a flawed Thought step directly produces a flawed Action.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Cost scales with the number of Thought/Action/Observation cycles actually taken, bounded by an explicit step limit — worst case is `max_steps` × one cycle's cost, not unbounded, provided a termination guard is actually implemented (the same discipline `03_advanced_rag` Module 08's self-correction loop required).
* **Space/Memory Footprint:** The full trace (all Thoughts/Actions/Observations) typically needs to stay in context for the next step's reasoning, so context size grows with trajectory length — a real, compounding cost distinct from a fixed pipeline's constant per-call context size.
* **Primary Bottleneck Type:** Latency-bound, specifically on the *variable* number of sequential LLM calls per task — the defining operational difference from a fixed pipeline's constant, predictable latency.
* **Variable Legend:** `max_steps` = the termination guard's bound; no additional closed-form formula variables, per this module's prose/procedural scope.

### 3. Production & Scalability
* **Deployment Considerations:** Always implement an explicit step/latency/cost bound before deploying any agentic loop — an unbounded loop is a real production incident waiting to happen on a genuinely unanswerable or malformed query (Module 09 covers the production-hardening version of this guard in depth); log the full Thought/Action/Observation trace for every run, since it's the primary tool for debugging a bad agent decision after the fact.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why not just always use an agent, since it's strictly more flexible than a fixed pipeline?
        *   *A:* Flexibility isn't free — an agent's cost, latency, and reliability are all worse than a fixed pipeline's for a task the pipeline could already handle; the decision framework's whole point is climbing to more flexible (and more expensive/unpredictable) architectures only as far as the task's genuine requirements force you to.
    2.  *Q:* How would you decide between a purely reactive agent and a plan-and-execute agent for a given task?
        *   *A:* If the task's structure is genuinely knowable upfront (even if the specific content varies), plan-and-execute reduces wandering and keeps long-horizon tasks on track; if the right next step genuinely can't be known until you see the previous step's result, a purely reactive loop adapts more naturally than a plan committed to too early.
