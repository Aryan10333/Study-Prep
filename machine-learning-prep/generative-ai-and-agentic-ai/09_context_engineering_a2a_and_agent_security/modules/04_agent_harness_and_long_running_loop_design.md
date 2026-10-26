# Module 04: Agent Harness & Long-Running Loop Design

> **Builds on:** `04_ai_agents_and_protocols` Module 01 (the agent loop), Module 04 (memory types and summarization triggers), Module 05 (graph orchestration, durable execution, checkpoints, human-in-the-loop interrupts) and Module 06 (multi-agent architectures). This module does not re-explain those. It covers how they are assembled into a **harness** that keeps an agent reliable over hours of work and across context resets.

## 1. Introduction & Intuition

### The Core Bottleneck
The same model can be a mediocre agent or an excellent one depending on what surrounds it. In 2026 the industry's attention moved from "which model" and "which agent framework" to the **harness**: the loop, the tools and how they are loaded, the permission layer, what persists between steps and sessions, and where a human checks in. Coding agents such as Claude Code and Codex made this concrete, because their quality on long tasks comes as much from harness design (instruction files, on-demand tool loading, sub-agents, compaction, checkpoints) as from the model.

The hard case is the **long-running agent**: a task that takes hundreds of steps, outgrows any context window (Modules 01–02) and may span several sessions. Without deliberate design, it forgets earlier decisions after a compaction, repeats work after a restart, and drifts from the original goal.

### High-Level Intuition
A long engineering project survives people going home every night because of what lives *outside* anyone's head: a project README, a ticket with the current status, a lab notebook of what was tried, and commits that can be rolled back. A good agent harness gives the agent the same things: an instruction file it reads at the start, notes it writes as it goes, checkpoints it can resume from, helpers it can delegate to, and a manager (the human) who reviews at sensible points instead of watching every keystroke.

---

## 2. Core Concepts & Mathematical Formulation

This module is architectural. Its worked examples use the token formula from Module 01; there is no new formula. Memory-summarization trigger math is owned by topic `04` Module 04.

### Anatomy of a Harness

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 820 400" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <defs>
    <marker id="m04a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#475569"/></marker>
  </defs>
  <text x="410" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">Agent Harness: Inner Loop, Supporting Services, Human Outer Loop</text>
  <rect x="20" y="50" width="780" height="60" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="410" y="76" text-anchor="middle" font-size="13" font-weight="bold" fill="#713f12">Human outer loop</text>
  <text x="410" y="96" text-anchor="middle" font-size="11" fill="#713f12">sets the goal, edits instruction files, approves risky actions, reviews at checkpoints</text>
  <rect x="290" y="150" width="240" height="120" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2"/>
  <text x="410" y="176" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Inner loop</text>
  <text x="410" y="200" text-anchor="middle" font-size="11" fill="#334155">1. context manager builds window</text>
  <text x="410" y="218" text-anchor="middle" font-size="11" fill="#334155">2. model picks an action</text>
  <text x="410" y="236" text-anchor="middle" font-size="11" fill="#334155">3. permission check, then run tool</text>
  <text x="410" y="254" text-anchor="middle" font-size="11" fill="#334155">4. record result, check stop rules</text>
  <rect x="20" y="140" width="210" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="125" y="163" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e3a8a">Tool registry</text>
  <text x="125" y="182" text-anchor="middle" font-size="11" fill="#1e3a8a">short index, details on demand</text>
  <rect x="20" y="224" width="210" height="56" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="125" y="247" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e3a8a">Permission layer</text>
  <text x="125" y="266" text-anchor="middle" font-size="11" fill="#1e3a8a">allow / ask / deny per action</text>
  <rect x="590" y="140" width="210" height="56" rx="8" fill="#e0e7ff" stroke="#4f46e5"/>
  <text x="695" y="163" text-anchor="middle" font-size="12" font-weight="bold" fill="#312e81">Context manager</text>
  <text x="695" y="182" text-anchor="middle" font-size="11" fill="#312e81">select, compress, isolate</text>
  <rect x="590" y="224" width="210" height="56" rx="8" fill="#e0e7ff" stroke="#4f46e5"/>
  <text x="695" y="247" text-anchor="middle" font-size="12" font-weight="bold" fill="#312e81">Sub-agents</text>
  <text x="695" y="266" text-anchor="middle" font-size="11" fill="#312e81">clean windows, return summaries</text>
  <rect x="150" y="320" width="520" height="60" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="410" y="344" text-anchor="middle" font-size="12" font-weight="bold" fill="#14532d">Durable state (outside the window)</text>
  <text x="410" y="364" text-anchor="middle" font-size="11" fill="#14532d">instruction file · agent notes file · checkpoints · task list · audit log</text>
  <line x1="230" y1="168" x2="285" y2="190" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
  <line x1="230" y1="252" x2="285" y2="235" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
  <line x1="585" y1="168" x2="535" y2="190" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
  <line x1="535" y1="235" x2="585" y2="252" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
  <line x1="410" y1="273" x2="410" y2="315" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
  <line x1="410" y1="113" x2="410" y2="145" stroke="#475569" stroke-width="1.5" marker-end="url(#m04a)"/>
</svg>
</div>

| Component | Job | Design choice that matters most |
|---|---|---|
| **Inner loop** | Build context → model acts → check permission → run tool → record | Stop conditions: step cap, cost cap, "done" check, no-progress detection |
| **Tool registry** | What the model can call | Load full schemas only when needed (progressive disclosure) |
| **Permission layer** | Decide allow / ask a human / deny per action | Tiered by reversibility and blast radius (topic `04` Module 09) |
| **Context manager** | Apply write / select / compress / isolate each step | Trigger compression below $L_{\text{usable}}$ and the cost break-even (Modules 02–03) |
| **Sub-agents** | Run a sub-task in a clean window | What they return (a short, structured summary) |
| **Durable state** | Everything that must survive a reset | What is written, by whom, and when it is read back |
| **Human outer loop** | Goal-setting, approvals, review | Where checkpoints for review sit, so the human isn't prompted every step |

### Instruction Files and Agent-Written Notes
Two kinds of durable, plain-text context have become standard:

* **Instruction files** (an `AGENTS.md` or `CLAUDE.md`-style file in the repo or workspace): written by humans, read at the start of every session. They hold what a new team member would need: how to build and test, conventions, things not to touch, where key code lives. They are version-controlled, reviewable and editable, unlike instructions buried in a hard-coded system prompt. Keep them short and specific: every line is paid on every step (Module 01).
* **Agent-written notes** (a progress or notes file, a task list): written by the agent as it works: what it has done, what it learned, what is left, decisions and why. After a compaction or a fresh session, reading the notes restores the essentials cheaply, instead of relying on a lossy summary of the raw history.

Both are the *write* operation from Module 01. Their failure mode is staleness: an instruction file that describes last quarter's build command, or notes the agent never updates, confidently mislead every later step. Treat them like code: reviewed, dated, and pruned.

### Progressive Disclosure and Just-in-Time Context
Instead of loading everything up front, load a cheap *index* and fetch detail on demand:

* **Tools:** list each tool by name and one-line purpose; load the full schema when the model wants to use it, or group rarely used tools behind a single search/lookup tool.
* **Skills:** packaged instructions (for example a folder with a `SKILL.md`) where only the name and description sit in context, and the body and any scripts are read when the skill is relevant.
* **Data:** keep file paths, query handles and IDs in context, not file contents; read the content when needed, and only the relevant part.

The trade-off: every on-demand fetch is an extra step, and if the index description is poor, the model never fetches the thing it needed. Indexes need the same care as tool descriptions.

### Sub-Agents as Context Isolation
Topic `04` Module 06 covers multi-agent architectures. Here the question is narrower: when does giving a sub-task its own clean window help, and what does it cost?

* **Benefit:** each sub-agent works in a small, focused window, below the usable limit, with no distractors from other sub-tasks. The orchestrator receives only a condensed result.
* **Cost:** total tokens usually go *up*. Each sub-agent re-reads its own static context, and fan-out encourages broader exploration. Anthropic has reported that its multi-agent research system used roughly 15x the tokens of a normal chat interaction, in exchange for better results on broad research tasks.
* **Loss:** whatever the sub-agent learned but did not put in its summary is gone. The summary format is a design decision, not an afterthought.

### Long-Horizon Patterns
For tasks longer than one window or one session:

1. **Compact with notes, not just a summary.** When compacting, also make sure the notes file holds decisions and open items; the summary can then be short.
2. **Checkpoint and resume.** Persist the loop's state (step, task list, notes pointer, pending approvals) so a crash or pause resumes instead of restarting. Use topic `04` Module 05's durable execution for production.
3. **One sub-goal at a time.** Keep a task list; work one item to completion, test it, record it, then move on. This limits how much a single compaction can lose.
4. **Human outer loop.** The human sets the goal and reviews at milestones (a finished sub-goal, a risky action, a failed test streak) rather than approving every step. That keeps oversight meaningful without approval fatigue (Module 07).
5. **No-progress detection.** Stop or escalate when the agent repeats the same action, the same error recurs N times, or the cost cap is near.

### Worked Example 1 (No New Formula): One Agent vs Sub-Agents
A research-and-fix task, costed with Module 01's formula $T\,C_s + g\,T(T+1)/2$. Usable context from Module 02: 32k.

* **Single agent:** $T = 30$, $C_s = 6{,}000$, $g = 2{,}000$. Total $= 180{,}000 + 2{,}000 \times 465 = 1{,}110{,}000$ tokens. Final window $= 6{,}000 + 60{,}000 = 66{,}000$, about twice the usable limit, so late steps run in degraded territory.
* **Orchestrator + 3 sub-agents:** orchestrator $T = 6$, $C_s = 4{,}000$, $g = 1{,}500$ (each sub-agent returns a ~1,000-token summary) gives $24{,}000 + 1{,}500 \times 21 = 55{,}500$. Each sub-agent $T = 14$, $C_s = 5{,}000$, $g = 1{,}800$ gives $70{,}000 + 1{,}800 \times 105 = 259{,}000$. Total $= 55{,}500 + 3 \times 259{,}000 = 832{,}500$ tokens. Largest window $= 5{,}000 + 25{,}200 = 30{,}200$, under the limit.
* **Fan out to 8 sub-agents** (broader exploration): orchestrator $T = 10$ gives $40{,}000 + 1{,}500 \times 55 = 122{,}500$. Total $= 122{,}500 + 8 \times 259{,}000 = 2{,}194{,}500$ tokens, about **2x** the single agent.

Isolation reliably keeps every window under the usable limit. Whether total tokens go down (3 sub-agents) or up (8 sub-agents) depends on how much the design fans out. Fan out when breadth improves the answer, not by default.

### Worked Example 2: Surviving Three Context Resets
A migration task spans three sessions. What each durable artifact preserves:

| After reset… | Instruction file | Notes file | Checkpoint | Without any of them |
|---|---|---|---|---|
| **Reset 1** (compaction) | Build and test commands, "don't edit generated files" | "Migrated 14/40 endpoints; auth endpoints need the new token format" | Current step and task list | Re-discovers the build command; may re-migrate finished endpoints |
| **Reset 2** (crash) | Same | Same, plus "endpoint 22 fails a flaky test; skip and revisit" | Pending approval for a DB migration is preserved | Re-requests the approval; loses the flaky-test finding |
| **Reset 3** (new day) | Same | Full progress log | Resumes at endpoint 31 | Starts over |

### Worked Example 3: Progressive Disclosure for Tools
40 tools with full schemas cost about 8,000 tokens per step. A 600-token index (name + one-line purpose) with on-demand schema loading saves $7{,}400$ tokens per step, or $148{,}000$ input tokens over a 20-step task, before caching. The cost is one extra step each time a not-yet-loaded tool is first used.

---

## 3. Implementation & Reference Code

A minimal, deterministic harness with a scripted mock model: a tool index with on-demand loading, a permission check, an agent-written notes file, checkpoint/resume and stop conditions. It shows the structure; production harnesses add durable execution (topic `04` Module 05) and real model calls.

```python
import json, os, tempfile

TOOLS = {   # full schemas live outside the context; only the index is shown to the model
    "read_file":  {"desc": "Read a file",           "risk": "low",  "schema_tokens": 220},
    "run_tests":  {"desc": "Run the test suite",    "risk": "low",  "schema_tokens": 180},
    "write_file": {"desc": "Write a file",          "risk": "medium", "schema_tokens": 260},
    "deploy":     {"desc": "Deploy to production",  "risk": "high", "schema_tokens": 300},
    "query_db":   {"desc": "Run a read-only SQL query", "risk": "low", "schema_tokens": 340},
    "search_web": {"desc": "Search the web",        "risk": "low",  "schema_tokens": 210},
    "send_email": {"desc": "Send an email",         "risk": "high", "schema_tokens": 280},
    "open_ticket": {"desc": "Open a support ticket", "risk": "medium", "schema_tokens": 250},
}
INDEX_TOKENS = 15 * len(TOOLS)            # ~15 tokens per index line

class Harness:
    def __init__(self, workdir, max_steps=10, approve=lambda action: False):
        self.notes_path = os.path.join(workdir, "NOTES.md")
        self.ckpt_path = os.path.join(workdir, "checkpoint.json")
        self.max_steps, self.approve = max_steps, approve
        self.loaded = set()                          # tool schemas loaded so far
        self.state = {"step": 0, "done": [], "last_actions": []}

    # --- durable state -------------------------------------------------
    def write_note(self, text):
        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write(f"- step {self.state['step']}: {text}\n")

    def checkpoint(self):
        with open(self.ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"state": self.state, "loaded": sorted(self.loaded)}, f)

    def resume(self):
        with open(self.ckpt_path, encoding="utf-8") as f:
            data = json.load(f)
        self.state, self.loaded = data["state"], set(data["loaded"])

    # --- context and permissions ---------------------------------------
    def tool_tokens(self):
        return INDEX_TOKENS + sum(TOOLS[t]["schema_tokens"] for t in self.loaded)

    def permitted(self, tool):
        risk = TOOLS[tool]["risk"]
        if risk == "low":
            return True
        return self.approve(tool)                    # medium/high risk: ask the human

    # --- inner loop ----------------------------------------------------
    def run(self, script):
        """script: list of tool names a mock model would choose, in order."""
        while self.state["step"] < self.max_steps and self.state["step"] < len(script):
            tool = script[self.state["step"]]
            self.state["step"] += 1
            recent = self.state["last_actions"][-2:]
            if recent == [tool, tool]:               # no-progress detection
                self.write_note(f"stopped: repeated {tool} 3x")
                return "stuck"
            self.state["last_actions"].append(tool)
            self.loaded.add(tool)                    # just-in-time schema load
            if not self.permitted(tool):
                self.write_note(f"{tool} needs approval; paused")
                self.checkpoint()
                return "paused_for_approval"
            self.state["done"].append(tool)
            self.write_note(f"ran {tool}")
            self.checkpoint()
        return "finished"


workdir = tempfile.mkdtemp()
script = ["read_file", "run_tests", "write_file", "run_tests", "deploy"]

h = Harness(workdir, approve=lambda t: t == "write_file")   # human approves writes, not deploys
assert h.run(script) == "paused_for_approval"                # stops at deploy
assert h.state["done"] == ["read_file", "run_tests", "write_file", "run_tests"]
# Only the 4 tools actually used were loaded; the other 4 stayed as one index line each
all_schemas = sum(t["schema_tokens"] for t in TOOLS.values())
assert len(h.loaded) == 4 and h.tool_tokens() < all_schemas
print("tool tokens with disclosure:", h.tool_tokens(), "vs all schemas up front:", all_schemas)

# Simulate a crash and a new process that resumes from the checkpoint
h2 = Harness(workdir, approve=lambda t: True)                # deploy now approved
h2.resume()
assert h2.state["step"] == 5 and h2.state["done"][-1] == "run_tests"
h2.state["step"] -= 1                                       # retry the paused step
assert h2.run(script) == "finished" and h2.state["done"][-1] == "deploy"

# No-progress detection
h3 = Harness(tempfile.mkdtemp())
assert h3.run(["run_tests"] * 5) == "stuck"

print(open(h2.notes_path, encoding="utf-8").read())
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Keeps agent quality stable over long tasks and across resets by deciding what lives in the window, what lives outside it, and where humans check in.
* **Why Introduced over Legacy Approaches:** "One big prompt plus a loop" works for short tasks but forgets decisions after compaction, restarts from scratch after a crash, and either asks the human too often or not at all. Harness design addresses each of these explicitly.
* **Key Failure Modes & Limitations:** Stale instruction or notes files; indexes so vague the model never loads the tool it needs; sub-agent summaries that drop the key finding; uncontrolled fan-out multiplying cost; checkpoints that don't capture pending approvals; no stop condition, so a stuck agent burns budget.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Per-step cost follows Module 01. Isolation changes the shape: each sub-agent's cost is quadratic in *its own* step count, which is why splitting a long loop into short ones can cut total tokens, while wider fan-out multiplies them.
* **Space/Memory Footprint:** Durable state (notes, checkpoints, audit logs) grows with task length and needs pruning or rotation; a notes file that is never pruned becomes its own context bloat.
* **Primary Bottleneck Type:** Quality over long horizons (drift, forgetting) and total token cost from fan-out, rather than per-call latency.
* **Variable Legend:** $T$, $C_s$, $g$ as in Module 01, applied separately to the orchestrator and to each sub-agent; $L_{\text{usable}}$ from Module 02.

### 3. Production & Scalability
* **Deployment Considerations:** Version instruction files with the code they describe. Make the notes format structured (done / decisions / open items) so it survives compaction. Put cost and step caps on both the orchestrator and each sub-agent. Trace sub-agent hand-offs so a lost finding can be found. Use durable execution for anything that runs longer than a request.
* **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Your coding agent works well for 20 minutes and then starts repeating itself and undoing earlier fixes. What's going on and what do you change?
        *   *A:* The window has grown past its usable length, and a compaction probably dropped earlier decisions. Add a structured notes file the agent updates after each sub-goal (done, decisions, open items), compact earlier with the notes as the source of truth, work one task-list item at a time, and add no-progress detection (same action or same error repeated) that stops or escalates.
    2.  *Q:* When would you use sub-agents, and when would you avoid them?
        *   *A:* Use them when sub-tasks are separable and benefit from a clean, focused window (exploring a large codebase, parallel research), and when the result can be summarized compactly. Avoid them for tightly coupled work where the hand-off would lose context, or when cost matters more than breadth: fan-out usually increases total tokens, sometimes several-fold.
    3.  *Q:* What goes in an `AGENTS.md`-style instruction file, and what doesn't?
        *   *A:* What a new teammate needs and can't easily infer: build and test commands, conventions, boundaries (files not to edit, actions that need approval), and where things live. Not long documentation (link it instead), secrets, or anything that changes per task. Keep it short, since every line is paid on every step, and review it like code so it doesn't go stale.
