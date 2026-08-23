# Module 09: Production Agent Systems, Safety & Security

## 1. Introduction & Intuition

### The Core Bottleneck
A bad output from a single-turn LLM call is bad text — annoying, sometimes costly, but rarely irreversible on its own. A bad decision from an agent with real tool access is a bad *action* — a sent email, a deleted record, a payment charged — and once it's executed, it's executed. This is the qualitative shift production agent systems introduce: the blast radius of a mistake is no longer bounded by "the user reads bad text and (hopefully) doesn't act on it," it's bounded by whatever the agent's tools are actually capable of doing in the real world. Every mechanism in this module exists to keep that blast radius bounded and intentional, not to make the agent smarter.

### High-Level Intuition
Giving an intern read-only access to a shared document is low-risk even if they make a mistake reading it. Giving that same intern the keys to the production database, unsupervised, on day one, is a fundamentally different risk — not because the intern got worse, but because the *consequences of a mistake* changed entirely. Production agent safety is the discipline of scoping what an agent can actually do to match how much you actually trust its judgment for that specific action, the same way you wouldn't hand an intern unsupervised production database access before they'd earned that trust — and building in a way to catch and undo mistakes when they happen anyway, because they will.

---

## 2. Core Concepts & Mathematical Formulation

This module stays architectural/procedural throughout — production hardening and security policy are structural concerns, consistent with how `03_advanced_rag` Module 09 treated its own production-hardening content.

### Guardrails & Permission Boundaries

#### Intuition & Practical Use
Not every action an agent decides on should execute immediately and unsupervised. A guardrail policy classifies actions into at least two tiers: fully autonomous (the agent executes without waiting) and approval-gated (the agent proposes the action and waits for explicit confirmation before it executes) — the production policy layer built directly on Module 02's confirmation-gate mechanics and Module 05's human-in-the-loop interrupt capability. The classification itself is the real design decision: a read-only lookup is a reasonable candidate for full autonomy; an irreversible, high-consequence action (deleting data, sending a payment, publishing something publicly) is a reasonable candidate for a mandatory approval gate, regardless of how confident the agent's own reasoning appears.

### Rate Limiting & Cost/Step Budgets

#### Intuition & Practical Use
The production enforcement of Module 02's per-task cost model and Module 01's step-count termination guard: a hard ceiling on how many steps, how much cost, or how much real-world action an agent can take within a bounded window, regardless of what its own reasoning decides. This exists specifically to bound the agent's *own* potential for runaway behavior — a genuinely malformed or adversarial input that causes the agent to loop or escalate shouldn't be able to consume unbounded cost or take unbounded real-world action just because no single step's reasoning looked obviously wrong in isolation.

### Indirect Prompt Injection

#### Intuition & Practical Use
Direct prompt injection is a malicious instruction from the user themselves, trying to manipulate the model directly. **Indirect** prompt injection is different and, for a tool-using agent, often more dangerous: malicious instructions arrive not from the user, but embedded in content the agent processes as *data* and ends up treating as *instructions* — because the agent has no structural way to distinguish "text I'm supposed to read" from "text I'm supposed to obey" once both are sitting in the same context window. Four real sources this shows up from in practice:
*   **Tool outputs** — a fetched webpage or API response crafted to contain hidden instructions.
*   **Retrieved content** — a poisoned document surfaced by RAG-based retrieval (a risk `03_advanced_rag` covers on the retrieval side; this module covers the same underlying risk from the agent-tool-execution side, not a duplicate treatment).
*   **Files** — an uploaded or read document with hidden instructions embedded in its text.
*   **External APIs** — a third-party response deliberately crafted to hijack the agent's next action.

The common thread across all four: the agent's context window doesn't inherently distinguish trusted instructions from untrusted data once they're both just text in the same prompt — which is exactly why the mitigations below don't rely on the model "just knowing better."

### Layered Mitigation: Isolation, Validation, Least Privilege, Approval Gates, Auditing

#### Intuition & Practical Use
No single mitigation is a complete defense against indirect prompt injection — production security here is a layered set of controls, each catching what the others might miss:
*   **Isolation** — sandboxing tool/content execution so untrusted content can't directly act, only be read (the subject of the next subsection in depth).
*   **Validation** — checking or sanitizing untrusted content before it enters context, reducing the odds a hidden instruction ever reaches the model in an interpretable form.
*   **Least-privilege tool access** — scoping each agent/tool to the minimum permissions it genuinely needs, so *even a successful* injection has a small blast radius, because the compromised agent still can't do anything beyond its narrow, already-limited grant.
*   **Approval gates** — requiring human confirmation before a sensitive action executes, the same Module 02 confirmation-gate mechanic now applied specifically as a security control against an agent that's been manipulated into proposing a harmful action.
*   **Auditing** — recording every action taken, so a successful injection that does slip through is at minimum detectable and traceable after the fact, not silently invisible.

None of these mitigations depend on training the model to be better at recognizing injected instructions — they depend on structurally limiting what damage is possible even when the model gets fooled, which is a fundamentally more robust posture than trying to make the model injection-proof.

### Authorization Boundaries & Sandboxing

#### Intuition & Practical Use
Authorization boundaries are the structural mechanism least-privilege access is actually built on: per-agent, per-tool, and per-action permission scoping, enforced at the point of execution, not just documented as a policy nobody checks. Sandboxing takes this further — isolating the *environment* a tool executes in (a restricted filesystem, no network access, a disposable container) so that even a tool call the agent was never supposed to make, executing with more permission than intended due to a bug or a successful injection, is still contained by the environment it's running in, not just by the permission check that was supposed to prevent it in the first place. Defense in depth: the permission check is supposed to prevent the bad action; the sandbox is what limits the damage if the permission check somehow fails.

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 780 260" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="390" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Permission Boundary: Autonomous vs. Approval-Gated Actions</text>

  <rect x="30" y="55" width="330" height="150" rx="8" fill="#ecfdf5" stroke="#059669" stroke-width="1.6"/>
  <text x="195" y="78" text-anchor="middle" font-size="11.5" fill="#065f46" font-weight="700">Fully Autonomous</text>
  <g font-size="9" fill="#065f46">
    <text x="195" y="102" text-anchor="middle">Read-only lookups</text>
    <text x="195" y="120" text-anchor="middle">Low-consequence, reversible actions</text>
    <text x="195" y="138" text-anchor="middle">Within rate-limit / cost budget</text>
  </g>
  <text x="195" y="175" text-anchor="middle" font-size="8.5" fill="#065f46" font-weight="600">Executes immediately, no wait</text>

  <rect x="420" y="55" width="330" height="150" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="1.6"/>
  <text x="585" y="78" text-anchor="middle" font-size="11.5" fill="#991b1b" font-weight="700">Approval-Gated</text>
  <g font-size="9" fill="#991b1b">
    <text x="585" y="102" text-anchor="middle">Irreversible actions (delete, publish)</text>
    <text x="585" y="120" text-anchor="middle">High-consequence (payments, external comms)</text>
    <text x="585" y="138" text-anchor="middle">Anything touching an authorization boundary</text>
  </g>
  <text x="585" y="175" text-anchor="middle" font-size="8.5" fill="#991b1b" font-weight="600">Proposed, then WAITS for human confirmation</text>

  <text x="390" y="230" text-anchor="middle" font-size="9" fill="#64748b">The classification itself is the real design decision -- not how confident the agent's own reasoning looks.</text>
</svg>
</div>

### Deployment Patterns: Synchronous vs. Long-Running Async Agents

#### Intuition & Practical Use
A synchronous request/response agent runs within one HTTP request's lifetime — the caller waits, gets a result, done; simple to reason about, but bounded by whatever timeout the request path can tolerate. A long-running async/background agent (Module 05's durable-execution machinery is exactly what makes this viable) can run for far longer, checkpointing its progress and notifying the caller (or waiting for approval) asynchronously — necessary for genuinely long tasks, but it also means the agent's actions are happening in the background, unsupervised in real time, for however long the task runs, which is exactly why guardrails and approval gates matter *more*, not less, for this deployment pattern.

---

## 3. Implementation & Reference Code

Below is a minimal guardrail policy engine — classifying proposed actions into autonomous vs. approval-gated tiers, enforcing a rate/cost budget, and logging every decision to an audit trail, illustrating the layered-mitigation mechanics above.

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ActionTier(Enum):
    AUTONOMOUS = "autonomous"
    APPROVAL_GATED = "approval_gated"
    BLOCKED = "blocked"  # exceeds this agent's authorization boundary entirely


@dataclass
class ProposedAction:
    tool_name: str
    is_reversible: bool
    estimated_cost: float


@dataclass
class AuditEntry:
    timestamp: str
    tool_name: str
    tier: ActionTier
    executed: bool


@dataclass
class GuardrailPolicy:
    """Classifies proposed actions and enforces a hard budget -- the production
    policy layer built on Module 02's confirmation-gate and Module 01's step-guard ideas."""
    authorized_tools: set[str]
    max_total_cost: float
    audit_log: list[AuditEntry] = field(default_factory=list)
    spent: float = 0.0

    def classify(self, action: ProposedAction) -> ActionTier:
        if action.tool_name not in self.authorized_tools:
            return ActionTier.BLOCKED  # least-privilege: never authorized for this agent at all
        if not action.is_reversible:
            return ActionTier.APPROVAL_GATED  # irreversible -> always gated, regardless of "confidence"
        return ActionTier.AUTONOMOUS

    def try_execute(self, action: ProposedAction, execute_fn, approve_fn=None) -> str:
        tier = self.classify(action)

        if tier == ActionTier.BLOCKED:
            self._log(action, tier, executed=False)
            raise PermissionError(f"'{action.tool_name}' is outside this agent's authorization boundary")

        if self.spent + action.estimated_cost > self.max_total_cost:
            self._log(action, tier, executed=False)
            raise RuntimeError(f"Budget exceeded: ${self.spent + action.estimated_cost:.4f} > ${self.max_total_cost:.4f}")

        if tier == ActionTier.APPROVAL_GATED:
            if approve_fn is None or not approve_fn(action):
                self._log(action, tier, executed=False)
                return "[action proposed, awaiting/denied approval -- NOT executed]"

        result = execute_fn(action)
        self.spent += action.estimated_cost
        self._log(action, tier, executed=True)
        return result

    def _log(self, action: ProposedAction, tier: ActionTier, executed: bool) -> None:
        self.audit_log.append(AuditEntry(datetime.now().isoformat(), action.tool_name, tier, executed))


if __name__ == "__main__":
    policy = GuardrailPolicy(authorized_tools={"search_docs", "send_email"}, max_total_cost=0.05)

    # Autonomous, reversible, authorized -> executes immediately
    lookup = ProposedAction(tool_name="search_docs", is_reversible=True, estimated_cost=0.01)
    result = policy.try_execute(lookup, execute_fn=lambda a: "search results...")
    print(f"search_docs result: {result}")
    assert policy.audit_log[-1].tier == ActionTier.AUTONOMOUS and policy.audit_log[-1].executed

    # Irreversible, authorized -> approval-gated, denied
    send_email = ProposedAction(tool_name="send_email", is_reversible=False, estimated_cost=0.01)
    denied = policy.try_execute(send_email, execute_fn=lambda a: "email sent", approve_fn=lambda a: False)
    print(f"send_email (denied): {denied}")
    assert policy.audit_log[-1].tier == ActionTier.APPROVAL_GATED and not policy.audit_log[-1].executed

    # Irreversible, authorized -> approval-gated, approved this time
    approved = policy.try_execute(send_email, execute_fn=lambda a: "email sent", approve_fn=lambda a: True)
    print(f"send_email (approved): {approved}")
    assert policy.audit_log[-1].executed

    # Outside authorization boundary entirely -> blocked, never even reaches approval
    delete_action = ProposedAction(tool_name="delete_database", is_reversible=False, estimated_cost=0.0)
    try:
        policy.try_execute(delete_action, execute_fn=lambda a: "deleted")
        raise AssertionError("Should have been blocked")
    except PermissionError as e:
        print(f"\nBlocked (outside authorization boundary): {e}")
        assert policy.audit_log[-1].tier == ActionTier.BLOCKED

    print(f"\nFull audit trail ({len(policy.audit_log)} entries):")
    for entry in policy.audit_log:
        print(f"  [{entry.tier.value}] {entry.tool_name} -> executed={entry.executed}")
    print("\nGuardrail policy verified: least-privilege blocking, approval gating, and audit logging all functioning correctly.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Bounding the real-world blast radius of an agent's mistakes — a category of risk that doesn't exist for a text-only LLM call, since an agent with tool access can take irreversible real actions, not just generate bad text.
* **Why Introduced over Legacy Approaches:** Trusting the model's own reasoning as the sole safety mechanism has no structural defense against a confidently-wrong decision or a successful indirect prompt injection; guardrails, budgets, and layered security controls provide a defense that holds even when the model's own judgment fails.
* **Key Failure Modes & Limitations:** Misclassifying an irreversible action as autonomous removes the one safety check that would have caught a bad decision before it executed; relying on a single mitigation (e.g., only least-privilege, with no auditing) leaves a real gap if that one control fails; rate/cost budgets that are too loose don't actually bound worst-case behavior.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not compute-bound — guardrail classification and budget checks are cheap, constant-time operations relative to the LLM call or tool execution they gate.
* **Space/Memory Footprint:** Audit logs accumulate proportionally to total agent action volume across the system's lifetime — a real, monotonically-growing storage cost requiring its own retention policy, the security-specific analog of Module 04's unbounded-memory-growth concern.
* **Primary Bottleneck Type:** Not a runtime bottleneck — the real cost is engineering discipline: correctly classifying every action's risk tier, and correctly scoping every tool's authorized permission set, neither of which a system can verify automatically without deliberate, ongoing review.
* **Variable Legend:** No closed-form formula variables, per this module's prose/procedural scope.

### 3. Production & Scalability
* **Deployment Considerations:** Default to the most restrictive tier (approval-gated, or even blocked) for any newly-added tool or action type until it's been deliberately reviewed and reclassified, rather than defaulting to autonomous and hoping nothing goes wrong; treat audit logs as a first-class production artifact from day one, since they're what makes a successful injection or a bad decision detectable and traceable after the fact rather than silently invisible.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* How would you defend against indirect prompt injection from a document your RAG pipeline retrieves?
        *   *A:* Layer the defenses rather than relying on one: validate/sanitize retrieved content before it enters context, ensure the agent's tools operating on that content carry least-privilege access so a successful injection has limited reach, gate any consequential action behind human approval, and audit-log everything so a slip-through is at least detectable — the same layered approach this module applies generally, with the retrieval-specific detection/filtering itself covered on `03_advanced_rag`'s side.
    2.  *Q:* Why is a long-running async agent a bigger safety concern than a synchronous one, even with identical tools and permissions?
        *   *A:* A synchronous agent's actions happen within one bounded request the caller is actively waiting on; a long-running async agent's actions happen in the background, unsupervised in real time, for however long the task runs — the same guardrails matter more, not less, because there's a longer real-world window for something to go wrong before anyone's actively watching.
