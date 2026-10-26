# Module 05: A2A Protocol & Agent Interoperability

> **Builds on:** `04_ai_agents_and_protocols` Module 03 (MCP architecture, primitives, discovery, auth basics) and Module 06 (multi-agent systems *inside* one application). This module covers the protocol for agents that talk to agents **owned by someone else**: another team, another company, another trust domain. Spec details follow A2A v1.0 (the latest released version at the time of writing); check the official specification at a2a-protocol.org before relying on field names.

## 1. Introduction & Intuition

### The Core Bottleneck
MCP standardized how an agent connects to **tools and data**: the agent stays in charge, calls a function, gets a result. But many enterprise integrations aren't function calls. A travel company's booking agent, a bank's claims agent and an insurer's underwriting agent each have their own reasoning, tools, data and policies, and their owners will not expose those internals as a list of functions. What they will expose is: "here is what my agent can do; send it a task; it will work on it, maybe ask you a question, and eventually return a result."

Before a standard, every such integration was a custom API with its own task model, polling scheme, streaming format and auth flow. **A2A (Agent2Agent)** standardizes that layer: how an agent discovers another agent, sends it work, follows a long-running task, answers its questions, and receives results, while each agent stays *opaque* to the other.

### High-Level Intuition
MCP is like giving your assistant a set of power tools: they pick them up and use them directly. A2A is like hiring a contractor: you read their business card (what they do, how to contact them, what credentials they need), hand them a job, and they do it their own way with their own tools. They may call back with a question ("which tile did you want?"), they send progress updates, and they hand over finished work. You never see inside their toolbox, and you don't need to.

---

## 2. Core Concepts & Mathematical Formulation

This module is about a protocol; there is no formula.

### Core Objects

| Object | What it is | Key fields / notes |
|---|---|---|
| **Agent Card** | A JSON document describing an agent: identity, skills, endpoints, auth requirements | Published at `https://{domain}/.well-known/agent-card.json`; can be signed (JWS) |
| **Task** | The unit of work, with an id and a state that moves through a lifecycle | Returned by the server; used to poll, subscribe, resume or cancel |
| **Message** | One communication turn from a user/client or the agent | Has a role, one or more parts, a `messageId`, optional `contextId` and `taskId` |
| **Artifact** | An output the agent produces for a task (a document, a booking record, a file) | Made of parts; the spec says results should go in artifacts, not messages |
| **Part** | The smallest content unit | Text, a file (inline or by reference), or structured data |
| **Context** (`contextId`) | Groups related tasks and messages into one conversation | Lets a follow-up task refer to earlier ones |

**Agent Card fields** (v1.0, abridged). Required: `name`, `description`, `supportedInterfaces` (each with `url`, `protocolBinding`, `protocolVersion`), `version`, `capabilities` (e.g. `streaming`, `pushNotifications`, `extendedAgentCard`), `defaultInputModes`, `defaultOutputModes`, `skills` (each with `id`, `name`, `description`, `tags`). Optional: `provider`, `documentationUrl`, `securitySchemes`, `securityRequirements`, `signatures`, `iconUrl`.

**Bindings:** the same operations are defined over JSON-RPC, gRPC and HTTP+JSON/REST. Push notifications to a client's webhook use plain HTTP with JSON payloads.

**Core operations:** `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, the push-notification-config operations (create, get, list, delete), and `GetExtendedAgentCard` (a fuller card for authenticated clients).

### The Task Lifecycle
A task's state is one of: `SUBMITTED`, `WORKING`, `INPUT_REQUIRED`, `AUTH_REQUIRED`, `COMPLETED`, `FAILED`, `CANCELED`, `REJECTED` (plus `UNSPECIFIED` as an enum default).

* **Terminal:** `COMPLETED`, `FAILED`, `CANCELED`, `REJECTED`. Nothing happens after these; a follow-up is a new task (it can share the `contextId` and reference the old task).
* **Interrupted:** `INPUT_REQUIRED` (the agent needs information from the client) and `AUTH_REQUIRED` (the agent needs the client to authenticate or authorize something). The task keeps its id; the client sends a message with that `taskId` to resume it.
* **`REJECTED`** differs from `FAILED`: the agent declined the task (out of scope, policy) rather than trying and failing.

How the client follows a task depends on how long it runs:

| Pattern | Mechanism | Use when |
|---|---|---|
| Request/response | `SendMessage` returns a completed task or a message | Seconds-long tasks |
| Polling | `GetTask` on an interval | Simple clients; tolerant of delay |
| Streaming | `SendStreamingMessage` / `SubscribeToTask` (server-sent status and artifact updates) | Interactive UIs; minutes-long tasks with progress |
| Push notifications | The server calls the client's webhook on state changes | Hours-long tasks; client may be offline |

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 820 470" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <defs>
    <marker id="m05a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#475569"/></marker>
  </defs>
  <text x="410" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">MCP vs A2A, and the A2A Task Lifecycle</text>
  <rect x="20" y="44" width="370" height="170" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="205" y="66" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e3a8a">MCP: agent → tools and data</text>
  <rect x="40" y="90" width="120" height="50" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="100" y="120" text-anchor="middle" font-size="12" fill="#1e3a8a">Your agent</text>
  <rect x="250" y="80" width="120" height="30" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="310" y="100" text-anchor="middle" font-size="11" fill="#1e3a8a">tool: query_db</text>
  <rect x="250" y="120" width="120" height="30" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="310" y="140" text-anchor="middle" font-size="11" fill="#1e3a8a">resource: docs</text>
  <line x1="160" y1="108" x2="245" y2="96" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="160" y1="122" x2="245" y2="134" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <text x="205" y="180" text-anchor="middle" font-size="11" fill="#334155">your agent reasons; tools just execute</text>
  <text x="205" y="198" text-anchor="middle" font-size="11" fill="#334155">and return results</text>
  <rect x="430" y="44" width="370" height="170" rx="8" fill="#ffffff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="615" y="66" text-anchor="middle" font-size="13" font-weight="bold" fill="#4c1d95">A2A: agent ↔ opaque agent</text>
  <rect x="450" y="95" width="120" height="50" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="510" y="125" text-anchor="middle" font-size="12" fill="#4c1d95">Client agent</text>
  <rect x="660" y="95" width="120" height="50" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="720" y="118" text-anchor="middle" font-size="12" fill="#4c1d95">Remote agent</text>
  <text x="720" y="134" text-anchor="middle" font-size="10" fill="#4c1d95">own tools, memory</text>
  <line x1="615" y1="80" x2="615" y2="160" stroke="#dc2626" stroke-width="2" stroke-dasharray="5,4"/>
  <text x="615" y="176" text-anchor="middle" font-size="10" fill="#991b1b">trust boundary</text>
  <line x1="570" y1="112" x2="655" y2="112" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="655" y1="130" x2="575" y2="130" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <text x="615" y="198" text-anchor="middle" font-size="11" fill="#334155">tasks, messages, artifacts; internals stay hidden</text>
  <text x="410" y="238" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Task states (simplified)</text>
  <rect x="30" y="300" width="110" height="40" rx="6" fill="#f1f5f9" stroke="#64748b"/>
  <text x="85" y="325" text-anchor="middle" font-size="12" fill="#0f172a">SUBMITTED</text>
  <rect x="220" y="300" width="110" height="40" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="275" y="325" text-anchor="middle" font-size="12" fill="#1e3a8a">WORKING</text>
  <rect x="400" y="248" width="150" height="34" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="475" y="270" text-anchor="middle" font-size="12" fill="#78350f">INPUT_REQUIRED</text>
  <rect x="400" y="398" width="150" height="34" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="475" y="420" text-anchor="middle" font-size="12" fill="#78350f">AUTH_REQUIRED</text>
  <rect x="640" y="262" width="150" height="34" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="715" y="284" text-anchor="middle" font-size="12" fill="#14532d">COMPLETED</text>
  <rect x="640" y="306" width="150" height="34" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="715" y="328" text-anchor="middle" font-size="12" fill="#7f1d1d">FAILED</text>
  <rect x="640" y="350" width="150" height="34" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="715" y="372" text-anchor="middle" font-size="12" fill="#7f1d1d">CANCELED</text>
  <rect x="30" y="400" width="110" height="34" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="85" y="422" text-anchor="middle" font-size="12" fill="#7f1d1d">REJECTED</text>
  <line x1="140" y1="320" x2="215" y2="320" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="85" y1="340" x2="85" y2="395" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="330" y1="306" x2="395" y2="272" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="410" y1="282" x2="335" y2="312" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#m05a)"/>
  <line x1="330" y1="336" x2="395" y2="410" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="330" y1="312" x2="635" y2="280" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="330" y1="320" x2="635" y2="323" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <line x1="330" y1="328" x2="635" y2="366" stroke="#475569" stroke-width="1.5" marker-end="url(#m05a)"/>
  <text x="410" y="458" text-anchor="middle" font-size="11" fill="#334155">Dashed: client answers, same task id resumes. Amber states are interrupted; green/red are terminal.</text>
</svg>
</div>

*Simplified: the spec also allows interrupted tasks to be canceled or to fail, and allows a submitted task to fail or be canceled before work starts.*

### Security Model
* **Auth is declared, not invented.** The Agent Card lists `securitySchemes` (for example OAuth 2.0 / OpenID Connect, API keys, HTTP bearer, mutual TLS) and which are required. Credentials travel at the transport layer (HTTP headers), never inside message content.
* **In-task authorization.** If the remote agent needs extra authorization mid-task (say, the user must consent to sharing a document), it moves the task to `AUTH_REQUIRED` rather than failing.
* **Signed Agent Cards.** Cards may be signed with JWS over a canonicalized (JCS) form of the card; clients should verify a signature before trusting a card's claims, since a spoofed card could redirect your agent's tasks and data to an attacker.
* **Least disclosure.** Opaqueness cuts both ways: the client shouldn't send more context than the task needs (no full conversation history "just in case"), because the remote agent is outside your trust boundary. Module 06 covers the threats.

### MCP vs A2A
| | MCP | A2A |
|---|---|---|
| **Connects** | Agent → tools, resources, prompts | Agent → another agent |
| **Who reasons** | Your agent; the server executes | Both; the remote agent plans and acts on its own |
| **Unit of work** | A tool call (usually short, synchronous) | A task (can run long; interrupts; streaming; push) |
| **Visibility** | Tool schemas are fully visible to your agent | Remote internals are hidden; only the Agent Card and skills are visible |
| **Typical boundary** | Inside your app, or a vendor's tool server | Across teams or organizations |
| **Governance** | Open spec, governed by a Linux Foundation body | Open spec, governed by a Linux Foundation body |

They are **complementary**: a remote A2A agent very often uses MCP internally to reach its own tools.

### When Not to Use A2A
* A deterministic operation with a fixed input/output schema (a tax calculation, a price lookup) is a **plain API or an MCP tool**. Wrapping it as an agent adds latency, cost and nondeterminism for no benefit.
* Sub-agents inside your own application share your trust domain and code. Your orchestration framework (topic `04` Module 06) is simpler than a network protocol between them.
* A2A earns its keep when the other side is **independently owned**, needs to keep its internals private, and does work that involves judgment, multiple steps or human interaction.

### Worked Example 1: A Full Task Trace
A company's travel-assistant agent (client) delegates to a partner airline's booking agent (remote).

1. **Discovery.** The client fetches `https://airline.example/.well-known/agent-card.json`, verifies the card's signature, sees a `book_flight` skill, a JSON-RPC endpoint, streaming support, and an OAuth 2.0 requirement. It obtains a token for that agent.
2. **Send.** `SendStreamingMessage` with a text part: "Book BOM→BLR on 3 Nov, morning, economy, for employee E-1042," plus a structured-data part with the traveler's ID. The task enters `SUBMITTED`, then `WORKING`; the stream sends status updates.
3. **Interrupt.** The airline agent finds two morning flights at different prices and policy tiers, and moves the task to `INPUT_REQUIRED` with a message: "07:05 for ₹6,200 or 09:40 for ₹4,900?"
4. **Resume.** The client's agent checks its travel policy (cheapest within a 3-hour window) and sends a message with the same `taskId`: "09:40." The task returns to `WORKING`.
5. **Complete.** The task moves to `COMPLETED` with an artifact: a structured booking record (PNR, fare, times). The client stores the artifact, not the conversation.
6. **Cancellation branch.** If the employee cancels the trip during step 3, the client calls `CancelTask`; the task moves to `CANCELED` (terminal) and the airline agent releases any hold.

### Worked Example 2: MCP Tool, A2A Agent, or Plain HTTP?
| Scenario | Choice | Reason |
|---|---|---|
| Query your own orders database | **MCP tool** | Your data, deterministic, your agent should reason over the result |
| Delegate a booking to a partner airline's agent | **A2A** | Independently owned, multi-step, may need clarification, internals private |
| Get today's FX rate from a pricing service | **Plain HTTP / MCP tool** | Deterministic, fixed schema; an agent adds nothing |
| Hand a claim to an insurer's claims agent that reviews documents and may ask for more | **A2A** | Long-running, interrupts for input, judgment involved, cross-organization |

---

## 3. Implementation & Reference Code

No SDK needed: an Agent Card checked against the v1.0 required fields, and a task object that enforces a simplified lifecycle (legal transitions, terminal states, resume by task id, cancellation).

```python
import uuid

REQUIRED_CARD_FIELDS = {"name", "description", "supportedInterfaces", "version",
                        "capabilities", "defaultInputModes", "defaultOutputModes", "skills"}

agent_card = {
    "name": "Airline Booking Agent",
    "description": "Searches and books flights for corporate travelers.",
    "supportedInterfaces": [{"url": "https://airline.example/a2a",
                             "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    "version": "2.3.0",
    "capabilities": {"streaming": True, "pushNotifications": True},
    "securitySchemes": {"oauth": {"type": "oauth2"}},     # abridged
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [{"id": "book_flight", "name": "Book a flight",
                "description": "Find and book a flight within a travel policy.",
                "tags": ["travel", "booking"]}],
}

def validate_card(card):
    missing = REQUIRED_CARD_FIELDS - card.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    for iface in card["supportedInterfaces"]:
        assert {"url", "protocolBinding", "protocolVersion"} <= iface.keys()
        assert iface["url"].startswith("https://"), "agents should be served over HTTPS"
    for skill in card["skills"]:
        assert {"id", "name", "description", "tags"} <= skill.keys()
    return True

assert validate_card(agent_card)

TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED"}
INTERRUPTED = {"INPUT_REQUIRED", "AUTH_REQUIRED"}
LEGAL = {   # simplified lifecycle
    "SUBMITTED":      {"WORKING", "REJECTED", "FAILED", "CANCELED"},
    "WORKING":        {"INPUT_REQUIRED", "AUTH_REQUIRED", "COMPLETED", "FAILED", "CANCELED"},
    "INPUT_REQUIRED": {"WORKING", "FAILED", "CANCELED"},
    "AUTH_REQUIRED":  {"WORKING", "FAILED", "CANCELED"},
}

class Task:
    def __init__(self, context_id=None):
        self.id = str(uuid.UUID(int=0x1042))                  # fixed id for a deterministic demo
        self.context_id = context_id or "ctx-travel-1"
        self.state, self.history, self.artifacts = "SUBMITTED", ["SUBMITTED"], []

    def transition(self, new_state):
        if self.state in TERMINAL:
            raise RuntimeError(f"task {self.id} is terminal ({self.state}); start a new task")
        if new_state not in LEGAL[self.state]:
            raise ValueError(f"illegal transition {self.state} -> {new_state}")
        self.state = new_state
        self.history.append(new_state)

    def resume_with(self, task_id, message):
        """Client answers an interrupt by sending a message with the same task id."""
        assert task_id == self.id, "resume must target the interrupted task"
        assert self.state in INTERRUPTED, "only interrupted tasks can be resumed"
        self.transition("WORKING")

# Happy path with one clarification (Worked Example 1)
t = Task()
t.transition("WORKING")
t.transition("INPUT_REQUIRED")                 # "07:05 or 09:40?"
t.resume_with(t.id, "09:40")
t.transition("COMPLETED")
t.artifacts.append({"parts": [{"data": {"pnr": "X7K2QF", "fare_inr": 4900}}]})
assert t.history == ["SUBMITTED", "WORKING", "INPUT_REQUIRED", "WORKING", "COMPLETED"]

# Terminal states are final
try:
    t.transition("WORKING")
    raise AssertionError("should not reach here")
except RuntimeError as e:
    print("blocked:", e)

# Cancellation while waiting for input
c = Task()
c.transition("WORKING"); c.transition("INPUT_REQUIRED"); c.transition("CANCELED")
assert c.state in TERMINAL

# Illegal jump straight from SUBMITTED to COMPLETED
try:
    Task().transition("COMPLETED")
    raise AssertionError("should not reach here")
except ValueError as e:
    print("blocked:", e)
print("all lifecycle checks passed")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** A standard way for independently owned agents to discover each other, exchange long-running tasks with interrupts and streaming, and return results, without exposing their internals.
* **Why Introduced over Legacy Approaches:** Custom point-to-point APIs reinvent task state, polling, streaming and auth for every integration; exposing a partner's agent as MCP tools would force them to reveal internals and give up control of their own reasoning.
* **Key Failure Modes & Limitations:** Trusting an unsigned or spoofed Agent Card; tasks with no timeout or cancellation path, leaving resources held; oversharing context with a remote agent outside your trust boundary; using A2A for deterministic operations that should be APIs; version skew between clients on different spec versions.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not compute-bound on the protocol side; end-to-end latency is dominated by the remote agent's own model calls, and can be minutes to hours for interrupted tasks.
* **Space/Memory Footprint:** Both sides persist task state for the task's lifetime (including while interrupted); push-notification configs and artifacts need retention policies.
* **Primary Bottleneck Type:** Latency and reliability of a remote dependency you don't control, plus human response time for interrupted tasks.
* **Variable Legend:** No closed-form variables; key quantities are task duration, interrupt count, and timeout/cancellation deadlines.

### 3. Production & Scalability
* **Deployment Considerations:** Verify Agent Card signatures and pin the expected issuer; use short-lived, narrowly scoped OAuth tokens per remote agent; set a deadline per delegated task and cancel on expiry; prefer push notifications over polling for long tasks; store artifacts (the results) as the system of record, not the chat; send only the minimum context a task needs; trace each delegated task end to end with the remote task id.
* **Common Interviewer Follow-Up Questions:**
    1.  *Q:* What's the difference between MCP and A2A, and can they be used together?
        *   *A:* MCP connects an agent to tools and data; your agent does the reasoning and the server executes. A2A connects an agent to another agent that does its own reasoning, keeps its own tools and memory private, and works on tasks that can run long and pause for input. They're complementary: a remote A2A agent typically uses MCP internally for its own tools.
    2.  *Q:* A partner's agent sometimes takes hours and occasionally never finishes. How do you design the client side?
        *   *A:* Use push notifications (or streaming while a user is watching) instead of tight polling; persist the remote task id and state durably; set a deadline and call `CancelTask` when it passes; treat `INPUT_REQUIRED` and `AUTH_REQUIRED` as first-class states that route to the right person or policy; make the downstream action idempotent on the task id so a duplicate completion doesn't double-book.
    3.  *Q:* Why would you not expose an internal pricing service over A2A?
        *   *A:* It's deterministic with a fixed schema. An agent wrapper adds latency, cost and nondeterminism, and makes testing harder. Expose it as a plain API or an MCP tool; reserve A2A for independently owned, multi-step, judgment-heavy work.
