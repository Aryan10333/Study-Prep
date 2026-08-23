# Module 07: Agent Frameworks Landscape

## 1. Introduction & Intuition

### The Core Bottleneck
Every concept in this topic so far — reasoning loops, tool calling, MCP, context/state/memory, durable orchestration, multi-agent coordination — can be built from scratch. Frameworks exist because building all of it correctly, every time, for every new project, is real, repeated engineering effort most teams shouldn't pay for from zero. But a framework is never a neutral, free abstraction: it makes real opinionated choices about how control flow, state, and observability work, and those choices become *your* system's choices the moment you adopt it. The interview-relevant skill isn't memorizing each framework's current API surface — that changes constantly — it's being able to compare them along the dimensions that actually matter for a production decision, and know when the right choice is to build the custom loop instead.

### High-Level Intuition
Choosing an agent framework is like choosing a prefabricated building system instead of building from raw materials. It's genuinely faster to get something solid up quickly, and a good prefab system has already solved problems you'd otherwise hit yourself. But you're also now living inside its assumptions about where walls can go — and if your actual requirements don't fit those assumptions, you're either fighting the system or living with a compromise you didn't choose. The right question is never "which prefab system is best," it's "do my requirements fit any prefab system's assumptions well enough to be worth the speed, or do I need to build custom."

---

## 2. Core Concepts & Mathematical Formulation

This module is a comparative survey, kept deliberately conceptual rather than API-tutorial — framework APIs change fast enough that memorizing today's syntax has a short shelf life, while the underlying architectural trade-offs are stable and interview-relevant regardless of which specific framework version is current.

### Six Dimensions for Comparing Any Agent Framework

#### Intuition & Practical Use
Rather than a feature-by-feature tour of each framework, every framework in this space can be compared along six stable dimensions — the actual questions worth asking before adopting one:

| Dimension | What It Actually Asks |
|---|---|
| **Architecture** | Is control flow graph-based, conversational (message-passing between agents), role-based (agents defined by persona/responsibility), or provider-native (built into a specific model provider's own API)? |
| **Control** | How much explicit control flow does the framework give you (you define every transition) vs. how much does it decide for you (an implicit loop you configure but don't fully see)? |
| **State** | How does the framework represent and persist state — is durable checkpointing (Module 05) built in, bolted on, or absent entirely? |
| **Observability** | How easy is it to trace a run step by step after the fact — does the framework expose a full execution trace, or does debugging mean adding your own logging inside its abstractions? |
| **Extensibility** | How easily do custom tools, custom agents, or custom logic plug into the framework's own abstractions, versus requiring you to work around them? |
| **Lock-In** | How much of your code becomes framework-specific and hard to port elsewhere if you later need to switch — is your core logic expressed in the framework's own types/abstractions, or kept mostly independent of it? |

### The Major Frameworks, Compared Along These Dimensions

#### Intuition & Practical Use
*   **LangGraph** — graph-based architecture (Module 05's explicit nodes/edges/state model is directly what it implements); high explicit control (you define the graph); durable state/checkpointing built in as a first-class feature; strong observability via full graph execution traces; high extensibility (custom nodes are just functions); real lock-in risk since your control flow is expressed directly in its graph abstraction.
*   **AutoGen** — conversational multi-agent architecture (agents coordinate by exchanging messages, closer to this topic's peer-to-peer pattern than orchestrator-worker by default); lower explicit control (the conversation pattern drives a lot of the flow implicitly); state/checkpointing support varies by version and is less central to its core model than LangGraph's; observability requires tracing the message exchange itself; extensibility is strong for adding new conversational agents; moderate lock-in, concentrated in its conversation-orchestration abstractions.
*   **CrewAI** — role-based multi-agent architecture (agents defined by an explicit persona/role/goal, closer to Module 06's orchestrator-worker pattern by convention); moderate explicit control (roles and tasks are defined explicitly, but execution flow within a role is more implicit); state handling is comparatively lighter-weight than LangGraph's; observability tooling is improving but was historically thinner than graph-based alternatives; extensibility is strong for adding new roles/tasks; lock-in is concentrated in its role/task/crew abstractions.
*   **OpenAI's Agents SDK / Assistants API** — provider-native architecture (built directly into a specific model provider's own platform); control and state are largely managed by the provider's own infrastructure rather than your own code, trading control/portability for reduced operational burden; observability is whatever the provider's own tooling exposes; extensibility is bounded by what the provider's API surface supports; the highest lock-in of the four, since the architecture is inseparable from that specific provider's platform.

### When to Build a Custom Agent Loop vs. Adopt a Framework

#### Intuition & Practical Use
Build custom when the task's requirements are narrow and well-understood enough that a framework's general-purpose abstractions add more overhead (learning curve, working around its opinions, lock-in) than they save — Module 01's own reference ReAct loop is a real, complete example of how little code a well-scoped custom loop actually needs. Adopt a framework when the task genuinely needs several of the capabilities a mature framework has already solved well — durable checkpointing, multi-agent coordination, rich observability — and rebuilding all of that from scratch would cost more engineering time than the framework's opinions cost in flexibility. This is, again, the same complexity-ladder discipline from Module 01: adopt the framework's added machinery only as far as the task's genuine requirements justify it, evaluated against these same six dimensions rather than "which one is popular."

---

## 3. Implementation & Reference Code

Framework-specific code is intentionally out of scope here — this module is about comparison criteria, not API tutorials, and framework APIs shift fast enough that concrete code examples would go stale quickly. Below instead is a small, structural comparison helper: a data-driven way to score candidate frameworks against a specific project's actual requirements along the six dimensions above, illustrating the *decision process*, not any one framework's syntax.

```python
from dataclasses import dataclass, field


@dataclass
class FrameworkProfile:
    name: str
    # Scored 1 (low) - 5 (high) along each dimension, illustrative ratings for comparison purposes
    control: int
    durable_state: int
    observability: int
    extensibility: int
    lock_in: int  # higher = MORE lock-in (worse, for a project wanting portability)


@dataclass
class ProjectRequirements:
    needs_durable_state: bool
    needs_multi_agent: bool
    portability_matters: bool


def score_framework(profile: FrameworkProfile, reqs: ProjectRequirements) -> float:
    """Illustrative scoring: rewards dimensions the project actually needs,
    penalizes lock-in specifically when portability matters to this project."""
    score = profile.control + profile.observability + profile.extensibility
    if reqs.needs_durable_state:
        score += profile.durable_state
    if reqs.portability_matters:
        score -= profile.lock_in
    return score


if __name__ == "__main__":
    langgraph = FrameworkProfile("LangGraph", control=5, durable_state=5, observability=5, extensibility=4, lock_in=4)
    provider_native = FrameworkProfile("Provider-Native SDK", control=2, durable_state=3, observability=3, extensibility=2, lock_in=5)

    # A project that genuinely needs durable state and cares about portability
    reqs = ProjectRequirements(needs_durable_state=True, needs_multi_agent=False, portability_matters=True)

    langgraph_score = score_framework(langgraph, reqs)
    provider_score = score_framework(provider_native, reqs)
    print(f"LangGraph score for this project's requirements: {langgraph_score}")
    print(f"Provider-native SDK score for this project's requirements: {provider_score}")
    assert langgraph_score > provider_score, "The framework matching this project's real requirements should score higher"

    # The SAME frameworks, for a project that does NOT care about portability and doesn't need durable state
    reqs_different = ProjectRequirements(needs_durable_state=False, needs_multi_agent=False, portability_matters=False)
    langgraph_score2 = score_framework(langgraph, reqs_different)
    provider_score2 = score_framework(provider_native, reqs_different)
    print(f"\nSame frameworks, different requirements:")
    print(f"LangGraph score: {langgraph_score2}")
    print(f"Provider-native SDK score: {provider_score2}")
    print("\nVerified: the 'right' framework depends entirely on the specific project's requirements, not a fixed ranking.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Avoiding re-solving already-solved problems (durable state, multi-agent coordination, observability tooling) from scratch on every new agent project, at the cost of adopting a framework's specific opinions about how those problems should be solved.
* **Why Introduced over Legacy Approaches:** Hand-rolled agent loops before frameworks matured meant every team independently reinvented checkpointing, tool-calling glue, and tracing — often incompletely or inconsistently; mature frameworks concentrate that engineering effort into a shared, battle-tested layer.
* **Key Failure Modes & Limitations:** Adopting a framework whose opinions don't actually fit the task's real requirements creates friction that outweighs the time saved; heavy lock-in makes migrating away from a framework choice expensive if requirements change substantially later; thin observability in some frameworks pushes debugging effort back onto the team anyway, eroding part of the framework's promised value.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not a framework-level concern — the underlying LLM calls and tool executions cost the same regardless of which orchestration layer wraps them; framework choice affects engineering velocity and observability, not model-level compute cost.
* **Space/Memory Footprint:** Varies by framework's state-management approach — a framework with first-class durable checkpointing (Module 05) has a real, explicit storage footprint for that state; a thinner framework may leave state management (and its footprint) entirely up to the implementing team.
* **Primary Bottleneck Type:** Not a runtime bottleneck — the real cost here is engineering/organizational: learning curve, lock-in risk, and the gap between a framework's opinions and the project's actual needs.
* **Variable Legend:** No closed-form formula variables, per this module's prose/comparative scope; the reference code's dimension scores are illustrative ratings for comparison purposes, not measured benchmarks.

### 3. Production & Scalability
* **Deployment Considerations:** Re-evaluate the six-dimension comparison against the *project's* actual requirements, not a generic "which framework is best" ranking — the same framework can be the clear right choice for one project and clear overkill (or a poor fit) for another, exactly as the reference code's scoring example demonstrates with the same two frameworks scoring oppositely under different requirements.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Your team has already built significant custom logic on a hand-rolled agent loop. When would switching to a framework actually be worth the migration cost?
        *   *A:* When the custom loop is now solving problems (durable state at scale, multi-agent coordination, production-grade observability) that a mature framework already solves well, and the ongoing cost of maintaining that custom machinery exceeds the one-time migration cost plus the framework's lock-in risk — not simply because a framework exists.
    2.  *Q:* How would you evaluate lock-in risk concretely before committing to a framework?
        *   *A:* Look at how much of your core business logic would need to be expressed directly in the framework's own types/abstractions (its graph nodes, its role definitions) versus how much can stay as plain, portable code that the framework merely orchestrates — the more logic lives inside framework-specific abstractions, the higher the real cost of migrating away later.
