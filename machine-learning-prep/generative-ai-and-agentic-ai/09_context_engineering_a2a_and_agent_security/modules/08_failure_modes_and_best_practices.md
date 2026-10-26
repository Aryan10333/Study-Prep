# Module 08: Common Failure Modes & Best Practices

> **Builds on:** Modules 01–07 of this topic. General agent failure modes (looping, tool misuse, bad planning) are covered in `04_ai_agents_and_protocols` Modules 08–09, and evaluation and observability in `07_llm_evaluation_observability_and_guardrails`. This module collects the failures specific to context, interop and the tool supply chain, and how to catch each one early.

## 1. Introduction & Intuition

### The Core Bottleneck
Almost none of the failures in this topic throw an error. A busted prompt cache still returns correct answers, just at 10x the cost. A compaction that drops a constraint still produces confident output, just wrong. A rug-pulled tool still works, while also leaking data. An A2A task with no deadline never fails; it simply never finishes. These are **silent failures**, and the only defense is to decide in advance what signal would reveal each one, and to watch that signal.

### High-Level Intuition
A car with a slow oil leak drives perfectly until the engine seizes. The fix is not better driving; it's a warning light wired to the oil pressure. Each failure below comes with its warning light: the metric or check that turns a silent failure into a visible one.

---

## 2. Core Concepts & Mathematical Formulation

This module is a reference of failures and checks; there is no formula. Each row names the symptom you would actually see, the signal that detects it, the root cause and the fix.

### Context Failures

| Failure | Symptom | Detection signal | Root cause | Fix |
|---|---|---|---|---|
| **Over-stuffing** | Quality drops late in long tasks | Window length per step vs $L_{\text{usable}}$ (Module 02) | Everything retrieved or returned stays in history | Truncate tool output; compress below $L_{\text{usable}}$; isolate exploration |
| **Cache busting** | Cost and time-to-first-token jump after a release | Cached-token ratio per call drops toward 0 | Timestamp/ID near the top; unsorted tools; in-place edits | Stable-first layout; deterministic serialization; lint prompts in CI (Section 3) |
| **Lossy compaction** | Agent re-does work or violates an earlier constraint after a compaction | Eval cases that span a compaction; constraint-violation checks | Summary dropped a decision or rule | Structured notes as source of truth; keep constraints in the static prefix |
| **Stale instruction / notes files** | Agent confidently uses an old build command or path | Instruction file older than the code it describes; failing first commands | Files never reviewed | Version with the code; review in PRs; date entries |
| **Memory poisoning** | A wrong "fact" keeps reappearing across sessions | Memory writes traced to their source; contradiction checks | Untrusted content written to long-term memory | Only write memory from trusted sources; record provenance; expire entries |
| **Lossy sub-agent hand-offs** | Orchestrator misses something a sub-agent found | Trace sub-agent outputs vs orchestrator's final answer | Free-form summaries | Structured return format (findings, evidence, open questions) |
| **Tool-definition bloat** | High static cost; model picks the wrong tool | Tokens of tool definitions per step; tool-selection accuracy | Many servers connected with full schemas | Progressive disclosure; fewer, better-described tools |
| **Uncontrolled fan-out** | Token bill several times higher than expected | Tokens per task by sub-agent; sub-agent count per task | No cap on parallel sub-agents | Cap fan-out and per-sub-agent budget (Module 04) |

### Interop Failures (A2A)

| Failure | Symptom | Detection signal | Root cause | Fix |
|---|---|---|---|---|
| **No deadline / no cancel** | Tasks stuck forever; partner holds resources | Age of non-terminal tasks | No timeout policy | Per-task deadline; `CancelTask` on expiry |
| **Unhandled interrupts** | Tasks sit in `INPUT_REQUIRED` / `AUTH_REQUIRED` | Count and age of interrupted tasks | Client treats only completed/failed | Route interrupts to a policy or a person; alert on age |
| **Assumed trust** | Tasks and data sent to a spoofed agent | Unsigned or unverified Agent Cards in use | Card fetched and used without verification | Verify card signatures; pin expected issuers and domains |
| **Oversharing** | Partner receives full histories and internal data | Payload size and fields per delegated task | Context forwarded "just in case" | Send the minimum each task needs |
| **Duplicate side effects** | Double bookings after retries | Duplicate artifacts per task id | Non-idempotent handling of completion | Idempotency on the remote task id |

### Supply-Chain and Security Failures

| Failure | Symptom | Detection signal | Root cause | Fix |
|---|---|---|---|---|
| **Unpinned definitions** | Behavior changes with no deploy | Definition hash changes in logs | No pinning | Hash-pin; quarantine on change (Module 07) |
| **Over-broad scopes** | One leaked token exposes everything | Scopes granted vs used per tool | Broad scopes requested up front | Minimal scopes with step-up |
| **Secrets in context or logs** | Keys appear in traces or model output | Secret scanners on prompts, traces and logs | Keys in system prompt, env vars, workspace files | Vault + injection at call time; redaction |
| **Approval fatigue** | Users approve everything instantly | Median time-to-approve; prompts per session | Every action prompts | Risk tiers; prompt only for irreversible or external actions |
| **Unaudited third-party skills** | Unknown code running with agent privileges | Inventory diff vs registry | One-click installs outside the registry | Registry-only loading; provenance review |
| **Lethal combination in one session** | Data exfiltration through a legitimate channel | Sessions that combine private data + untrusted content + outbound tool | No session-level policy | Separate them, or gate the outbound step |
| **Gateway fails open** | Unchecked calls during an outage | Calls without gateway decision IDs | Fail-open default | Fail closed for anything above read-only |

### Pre-Launch Checklist for a Production Agent

**Context**
- [ ] Per-segment token breakdown logged for every step.
- [ ] $L_{\text{usable}}$ measured for the main task types; compaction/isolation triggers set below it.
- [ ] Stable prefix first and byte-identical; cached-token ratio monitored with an alert.
- [ ] Tool output truncated or summarized at the harness; size limits per tool.
- [ ] Instruction file reviewed and versioned; notes file structured.

**Loop and harness**
- [ ] Step cap, cost cap and no-progress detection on the orchestrator and on every sub-agent.
- [ ] Checkpoints include pending approvals; resume tested after a simulated crash.
- [ ] Sub-agent return format defined; fan-out capped.

**Interop (if using A2A)**
- [ ] Agent Card signatures verified; expected issuers pinned.
- [ ] Deadline and cancellation for every delegated task; interrupted states routed.
- [ ] Minimum-necessary context per task; idempotent handling of results.

**Tool supply chain**
- [ ] Every server and skill in the registry with source, version, approver, scopes.
- [ ] Definition hashes pinned and checked on every load and update.
- [ ] Tools namespaced; definitions scanned; cross-tool references flagged.
- [ ] Audience-bound, short-lived, minimal-scope tokens; no passthrough.
- [ ] Secrets only in a vault; prompts, traces and logs scanned for secrets.
- [ ] Local servers sandboxed; egress allowlists per tool; SSRF protections on.
- [ ] Approval tiers defined; prompts per session tracked.
- [ ] Red-team run with a poisoned test server passed.

---

## 3. Implementation & Reference Code

Two checks that can run in CI: a prompt linter that catches common cache-busters before they ship, and an agent-config audit against part of the checklist above.

```python
import re

CACHE_BUSTERS = [
    (r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "timestamp"),
    (r"\b(request|trace|session)[_-]?id\s*[:=]", "per-request id"),
    (r"\{\{\s*(now|time|date|uuid|random)\s*\}\}", "dynamic template variable"),
]

def lint_prompt(stable_prefix: str, prefix_tool_names: list, head_chars: int = 4000):
    """Flag likely cache-busters in the part of the prompt meant to be cached."""
    issues = []
    head = stable_prefix[:head_chars]
    for pat, label in CACHE_BUSTERS:
        m = re.search(pat, head, flags=re.IGNORECASE)
        if m:
            issues.append(f"{label} in stable prefix at char {m.start()}: {m.group(0)!r}")
    if prefix_tool_names != sorted(prefix_tool_names):
        issues.append("tool definitions are not sorted; order may change between requests")
    return issues

good = "You are a support agent for Acme.\nRules: be concise."
bad = "You are a support agent.\nCurrent time: 2026-10-26 14:03:17\nrequest_id=8f3a\n{{ now }}"
assert lint_prompt(good, ["lookup_order", "refund"]) == []
bad_issues = lint_prompt(bad, ["refund", "lookup_order"])
assert len(bad_issues) == 4
for i in bad_issues:
    print("cache-buster:", i)

def audit_agent_config(cfg: dict):
    """Check an agent's config against the parts of the pre-launch checklist
    that are machine-checkable. Returns a list of failed checks."""
    failed = []
    if not cfg.get("max_steps") or not cfg.get("max_cost_usd"):
        failed.append("missing step or cost cap")
    for sub in cfg.get("sub_agents", []):
        if not sub.get("max_steps"):
            failed.append(f"sub-agent {sub['name']} has no step cap")
    if cfg.get("max_parallel_sub_agents", 0) > 8:
        failed.append("fan-out above 8 parallel sub-agents")
    if cfg.get("compact_at_tokens", 10**9) > cfg.get("usable_context_tokens", 0):
        failed.append("compaction trigger is above the measured usable context")
    for srv in cfg.get("mcp_servers", []):
        name = srv["name"]
        if not srv.get("pinned_hashes"):
            failed.append(f"{name}: tool definitions not pinned")
        if srv.get("version", "latest") == "latest":
            failed.append(f"{name}: version not pinned")
        if any(s.endswith(":*") or s in {"*", "admin", "all"} for s in srv.get("scopes", [])):
            failed.append(f"{name}: wildcard or admin scope")
        if srv.get("runs") == "local" and not srv.get("sandboxed"):
            failed.append(f"{name}: local server not sandboxed")
        if srv.get("egress") in (None, ["*"]):
            failed.append(f"{name}: no egress allowlist")
    if cfg.get("secrets_in_prompt"):
        failed.append("secrets present in the prompt")
    if cfg.get("gateway_failure_mode") != "closed":
        failed.append("gateway does not fail closed")
    return failed

cfg = {
    "max_steps": 60, "max_cost_usd": 5.0,
    "usable_context_tokens": 32_000, "compact_at_tokens": 28_000,
    "max_parallel_sub_agents": 4,
    "sub_agents": [{"name": "explorer", "max_steps": 15}, {"name": "tester"}],
    "mcp_servers": [
        {"name": "github", "version": "3.4.1", "pinned_hashes": {"create_issue": "ab12"},
         "scopes": ["issues:write"], "runs": "remote", "egress": ["api.github.com"]},
        {"name": "web-tools", "version": "latest", "pinned_hashes": {},
         "scopes": ["files:*"], "runs": "local", "sandboxed": False, "egress": ["*"]},
    ],
    "secrets_in_prompt": False, "gateway_failure_mode": "closed",
}
failed = audit_agent_config(cfg)
for f in failed:
    print("FAIL:", f)
assert "sub-agent tester has no step cap" in failed
assert sum(f.startswith("web-tools") for f in failed) == 5      # every supply-chain check fails
assert not any(f.startswith("github") for f in failed)
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Turns the silent failures of context, interop and the tool supply chain into named, detectable conditions with a signal and a fix.
* **Why Introduced over Legacy Approaches:** Traditional monitoring watches errors and latency. These failures raise neither: they show up as higher cost, worse answers, stuck tasks or quiet data leaks, so each needs its own purpose-built signal.
* **Key Failure Modes & Limitations:** Checklists that are ticked once and never re-run; signals collected but never alerted on; static config audits that miss runtime behavior (a server that's pinned but whose responses carry injected content).

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** All checks here are cheap (regex, config inspection, counters); the cost is engineering time to wire signals and alerts.
* **Space/Memory Footprint:** Per-step token breakdowns, definition hashes and audit logs add modest storage proportional to traffic.
* **Primary Bottleneck Type:** Observability coverage: a failure you have no signal for is a failure you'll find from a customer or an incident.
* **Variable Legend:** $L_{\text{usable}}$ from Module 02; other quantities are counters and ratios (cached-token ratio, prompts per session, task age).

### 3. Production & Scalability
* **Deployment Considerations:** Run the prompt linter and config audit in CI so regressions block merges. Put the cached-token ratio, window length vs $L_{\text{usable}}$, non-terminal task age, definition-hash changes and prompts-per-session on the same dashboard as cost and quality. Re-run the full checklist on every model change, new MCP server or harness release.
* **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Name three silent failures in a production agent and how you'd detect each.
        *   *A:* Cache busting (no errors, cost jumps): monitor the cached-token ratio per call and alert on drops. Lossy compaction (agent forgets a constraint): eval cases that span a compaction, plus constraint-violation checks on outputs. A rug-pulled tool (works normally while leaking): hash-pin tool definitions and alert on any change. Others: stuck A2A tasks (alert on non-terminal task age) and approval fatigue (track time-to-approve).
    2.  *Q:* What would you put on a pre-launch checklist for an agent that uses third-party MCP servers?
        *   *A:* Registry-only loading with pinned versions and definition hashes; namespaced tools and scanned definitions; audience-bound, short-lived, minimal-scope tokens with no passthrough; secrets in a vault, never in context or logs; sandboxed local servers with per-tool egress allowlists; approval tiers; step and cost caps everywhere; and a red-team run with a deliberately poisoned server.
    3.  *Q:* Your agent's quality is fine in testing but degrades for real users on long sessions. Where do you look?
        *   *A:* Window length per step in production traces versus the $L_{\text{usable}}$ you measured (test sessions are usually shorter); whether compaction is triggering and what it drops; whether real tool outputs are much larger than test fixtures; stale instruction files; and any memory written from untrusted content. Then add long-session cases to the eval suite.
