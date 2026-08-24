# Module 07: Reliability Engineering — Redundancy, Fallbacks & Multi-Provider Architecture

## 1. Introduction & Intuition

### The Core Bottleneck
A GenAI system's real, dominant external dependency — a third-party model provider API, or a real GPU fleet under real contention — is inherently less reliable than a typical stateless microservice's dependencies. Designing for that dependency's real failure (a provider outage, a rate-limit exhaustion, a degraded-latency incident) is not optional hardening; it's a real, structural requirement this module owns, distinct from simply hoping the dependency stays healthy.

### High-Level Intuition
A single point of failure in a phone network is mitigated by real, redundant routing — if one line is down, the call finds another real path, possibly at slightly lower quality, rather than simply failing. Reliability engineering for a GenAI system applies the identical real logic: circuit breakers stop hammering a failing dependency, fallback chains find a real alternate path (even at a real, accepted capability cost), and retry/backoff policies make sure the recovery attempt itself doesn't become a second real outage.

---

## 2. Core Concepts & Mathematical Formulation

### Exponential Backoff With Real Jitter — the Production-Critical Term

#### Purpose & High-level Intuition
$$\text{Delay}_n = \min(\text{Delay}_{\text{base}} \times 2^n + \text{Jitter}_n, \text{Delay}_{\text{max}})$$

Where $\text{Jitter}_n$ is real random noise (e.g., uniform over $[0, \text{Delay}_{\text{base}} \times 2^n]$ — "full jitter"). **Stated explicitly as the term that matters most in production**: pure exponential backoff without jitter is real but insufficient alone — many real clients that failed at the same moment (e.g., all hitting a provider's rate limit simultaneously) would otherwise retry in real lockstep at $\text{Delay}_0, \text{Delay}_1, \text{Delay}_2, \ldots$ together, re-synchronizing into a real thundering-herd retry burst against the already-degraded dependency. Real jitter breaks that synchronization by spreading real retry attempts across time.

### Retry Eligibility — a Required Real Taxonomy Before Any Retry Logic

#### Intuition & Practical Use
Not every real failure should be retried — treating all errors as uniformly retryable is itself a real reliability bug. Four real categories: **transient errors** (a real, likely-temporary fault — retry); **rate limits** (retry, but real backoff should honor a provider's stated `Retry-After` header where available, rather than guessing); **timeouts** (retry only if the request is real, confirmed idempotent — a timeout leaves the real completion state unknown, and blindly retrying a non-idempotent request risks a real duplicated side effect); **non-retryable errors** (a real client-side/validation fault — e.g., a malformed request — retrying only adds real load with zero chance of success).

---

### Worked Example: Real Backoff-With-Jitter vs. a Real Naive-Immediate-Retry Storm

Real stated assumptions: 3 real retry attempts, $\text{Delay}_{\text{base}} = 200\text{ms}$, $\text{Delay}_{\text{max}} = 4000\text{ms}$, and (for the jittered case) a real, fixed jitter draw of $[50, 150, 300]\text{ms}$ per attempt for reproducibility.

*   **Real backoff-with-jitter, total delay across 3 attempts:**
    $$\min(200{+}50, 4000) + \min(400{+}150, 4000) + \min(800{+}300, 4000) = 250 + 550 + 1100 = 1900\text{ms}$$

*   **Real naive-immediate-retry (0ms delay every attempt), 1,000 real clients failing simultaneously:** all 1,000 real clients re-hit the dependency at $t=0$ instantly (3 times each, back-to-back) — a real, synchronized load spike of up to 3,000 real requests landing on the already-degraded dependency within milliseconds, with zero real time for it to recover.

*   **Real interpretation.** The jittered backoff's real total delay (1,900ms) is a per-client number that spreads real retry attempts across a real time window; the naive-immediate case's real danger isn't its total delay (which is technically zero) — it's that **1,000 real clients converge on the identical retry instant**, which is exactly the real thundering-herd failure mode jitter exists to prevent.

### Worked Example (No Formula): Retry Eligibility Applied to 4 Real Scenarios

| Scenario | Real category | Retry decision |
|---|---|---|
| Provider returns a real `503 Service Unavailable` | Transient error | **Retry** (with backoff+jitter) |
| Provider returns `429 Too Many Requests` with `Retry-After: 30` | Rate limit | **Retry**, honoring the real stated `Retry-After: 30s`, not a generic backoff guess |
| Request times out after 10s, but it's a real read-only query (no side effect) | Timeout, confirmed idempotent | **Retry** |
| Request times out after 10s, and it's a real "create ticket" tool call | Timeout, NOT confirmed idempotent | **Do NOT blindly retry** — check for a real duplicate first (e.g., via a real idempotency key), or surface the real ambiguous state rather than risk a duplicate ticket |
| Provider returns a real `400 Bad Request` (malformed payload) | Non-retryable | **Do not retry** — the real request will fail identically every time |

### Worked Example (No Formula): Primary vs. Fallback Capability Trade-off

A real user prompt requiring a 100K-token real document to be summarized. The real primary model supports a real 128K-token context window; the real fallback model (invoked after the primary's real circuit breaker opens) supports only a real 32K-token context window. The fallback **cannot** process the same real request unmodified — the real system must either real-chunk the document for the fallback (a real, different processing path, not a transparent substitution) or explicitly degrade the real response (e.g., summarize only the first 32K tokens, clearly flagged as partial). This is the real, stated capability trade-off accepted for availability — treating the fallback as a drop-in equivalent would silently produce a real, incomplete result.

---

## 3. Implementation & Reference Code

```python
import random
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    TIMEOUT_IDEMPOTENT = "timeout_idempotent"
    TIMEOUT_NON_IDEMPOTENT = "timeout_non_idempotent"
    NON_RETRYABLE = "non_retryable"


def is_retry_eligible(category: ErrorCategory) -> bool:
    """Real retry-eligibility taxonomy -- only timeout_non_idempotent and
    non_retryable are real, correct 'do not retry' cases."""
    return category not in (ErrorCategory.TIMEOUT_NON_IDEMPOTENT, ErrorCategory.NON_RETRYABLE)


def backoff_delay_ms(attempt: int, base_ms: float, max_ms: float, jitter_ms: float) -> float:
    return min(base_ms * (2 ** attempt) + jitter_ms, max_ms)


if __name__ == "__main__":
    random.seed(42)  # deterministic for reproducibility

    # Real jittered backoff: 3 attempts with a fixed real jitter draw for reproducibility
    fixed_jitters = [50, 150, 300]
    base_ms, max_ms = 200, 4000
    delays = [backoff_delay_ms(n, base_ms, max_ms, fixed_jitters[n]) for n in range(3)]
    total_delay = sum(delays)
    print(f"Real per-attempt jittered delays (ms): {delays}")
    print(f"Real total delay across 3 attempts: {total_delay}ms")
    assert delays == [250, 550, 1100]
    assert total_delay == 1900

    # Real retry-eligibility taxonomy applied to the module's own 4 scenarios (5 rows, one non-retryable each way)
    scenarios = {
        "503 transient": ErrorCategory.TRANSIENT,
        "429 rate limit": ErrorCategory.RATE_LIMIT,
        "timeout, idempotent read": ErrorCategory.TIMEOUT_IDEMPOTENT,
        "timeout, non-idempotent create-ticket call": ErrorCategory.TIMEOUT_NON_IDEMPOTENT,
        "400 malformed request": ErrorCategory.NON_RETRYABLE,
    }
    print()
    for name, category in scenarios.items():
        decision = "RETRY" if is_retry_eligible(category) else "DO NOT RETRY"
        print(f"{name}: {decision}")

    assert is_retry_eligible(ErrorCategory.TRANSIENT) is True
    assert is_retry_eligible(ErrorCategory.RATE_LIMIT) is True
    assert is_retry_eligible(ErrorCategory.TIMEOUT_IDEMPOTENT) is True
    assert is_retry_eligible(ErrorCategory.TIMEOUT_NON_IDEMPOTENT) is False
    assert is_retry_eligible(ErrorCategory.NON_RETRYABLE) is False

    print("\nVerified: real jittered backoff totals 1900ms across 3 attempts (a per-client number, not")
    print("a synchronization risk), and the retry-eligibility taxonomy correctly withholds retry only")
    print("for the two real categories (non-idempotent timeout, non-retryable) where retrying is unsafe")
    print("or pointless -- confirming the module's own stated taxonomy is not just descriptive but decidable.")
```

### Circuit Breaker + Fallback Chain, Visualized

<div style="max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 1000 420" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Segoe UI', sans-serif;">
  <defs>
    <marker id="a7" markerWidth="8" markerHeight="8" refX="6" refY="2.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,5 L7,2.5 z" fill="#475569" />
    </marker>
  </defs>
  <text x="500" y="24" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">Circuit Breaker State Machine + Provider Fallback Chain</text>

  <!-- circuit breaker state machine -->
  <circle cx="140" cy="110" r="55" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="140" y="105" text-anchor="middle" font-size="12" font-weight="700" fill="#065f46">CLOSED</text>
  <text x="140" y="121" text-anchor="middle" font-size="9.5" fill="#065f46">requests flow</text>

  <circle cx="380" cy="110" r="55" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="380" y="105" text-anchor="middle" font-size="12" font-weight="700" fill="#991b1b">OPEN</text>
  <text x="380" y="121" text-anchor="middle" font-size="9.5" fill="#991b1b">fail fast, no calls</text>

  <circle cx="260" cy="230" r="55" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="260" y="220" text-anchor="middle" font-size="11.5" font-weight="700" fill="#92400e">HALF-</text>
  <text x="260" y="236" text-anchor="middle" font-size="11.5" font-weight="700" fill="#92400e">OPEN</text>
  <text x="260" y="252" text-anchor="middle" font-size="9" fill="#92400e">probe traffic</text>

  <path d="M195,105 L325,105" fill="none" stroke="#475569" stroke-width="1.5" marker-end="url(#a7)"/>
  <text x="260" y="95" text-anchor="middle" font-size="9.5" fill="#475569">error rate exceeds threshold</text>

  <path d="M350,155 L290,195" fill="none" stroke="#475569" stroke-width="1.5" marker-end="url(#a7)"/>
  <text x="365" y="185" text-anchor="middle" font-size="9" fill="#475569">cooldown elapses</text>

  <path d="M225,195 L165,155" fill="none" stroke="#475569" stroke-width="1.5" marker-end="url(#a7)"/>
  <text x="150" y="185" text-anchor="middle" font-size="9" fill="#475569">probe succeeds</text>

  <path d="M300,255 C340,290 370,190 385,165" fill="none" stroke="#475569" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#a7)"/>
  <text x="400" y="260" text-anchor="middle" font-size="9" fill="#475569">probe fails</text>

  <!-- fallback chain -->
  <rect x="580" y="55" width="150" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="655" y="80" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">Primary Model</text>
  <text x="655" y="98" text-anchor="middle" font-size="9.5" fill="#1e3a8a">128K context</text>
  <line x1="655" y1="115" x2="655" y2="145" stroke="#dc2626" stroke-width="1.5" marker-end="url(#a7)"/>
  <text x="700" y="132" text-anchor="middle" font-size="9" fill="#991b1b">circuit OPEN</text>

  <rect x="580" y="150" width="150" height="60" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="655" y="175" text-anchor="middle" font-size="11.5" font-weight="700" fill="#92400e">Secondary Model</text>
  <text x="655" y="193" text-anchor="middle" font-size="9.5" fill="#92400e">32K context (real capability drop)</text>
  <line x1="655" y1="210" x2="655" y2="240" stroke="#dc2626" stroke-width="1.5" marker-end="url(#a7)"/>
  <text x="700" y="227" text-anchor="middle" font-size="9" fill="#991b1b">also unavailable</text>

  <rect x="580" y="245" width="150" height="60" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="655" y="270" text-anchor="middle" font-size="11.5" font-weight="700" fill="#991b1b">Cached / Canned</text>
  <text x="655" y="288" text-anchor="middle" font-size="9.5" fill="#991b1b">last-resort fallback</text>

  <rect x="770" y="55" width="210" height="250" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="875" y="80" text-anchor="middle" font-size="11.5" font-weight="700" fill="#5b21b6">Real Trade-off, Stated</text>
  <text x="875" y="104" text-anchor="middle" font-size="9.5" fill="#5b21b6">Each fallback tier accepts</text>
  <text x="875" y="120" text-anchor="middle" font-size="9.5" fill="#5b21b6">a real capability loss</text>
  <text x="875" y="136" text-anchor="middle" font-size="9.5" fill="#5b21b6">(context window, quality,</text>
  <text x="875" y="152" text-anchor="middle" font-size="9.5" fill="#5b21b6">safety behavior) in</text>
  <text x="875" y="168" text-anchor="middle" font-size="9.5" fill="#5b21b6">exchange for availability --</text>
  <text x="875" y="184" text-anchor="middle" font-size="9.5" fill="#5b21b6">never a transparent,</text>
  <text x="875" y="200" text-anchor="middle" font-size="9.5" fill="#5b21b6">equivalent substitute.</text>
  <text x="875" y="224" text-anchor="middle" font-size="9.5" fill="#5b21b6">Retry/backoff+jitter applies</text>
  <text x="875" y="240" text-anchor="middle" font-size="9.5" fill="#5b21b6">independently at each tier,</text>
  <text x="875" y="256" text-anchor="middle" font-size="9.5" fill="#5b21b6">gated by its own real</text>
  <text x="875" y="272" text-anchor="middle" font-size="9.5" fill="#5b21b6">retry-eligibility check.</text>
</svg>
</div>

*   **Diagram Interpretation:** The real 3-state circuit breaker (closed/open/half-open) on the left, gating whether a real request is even attempted, and a real 3-tier fallback chain on the right with each tier's real capability drop stated explicitly rather than implied as a transparent substitution.

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Bounding a GenAI system's real exposure to an inherently less reliable dependency (provider API, GPU fleet) through circuit breaking, jittered retries, and a real, capability-aware fallback chain.
* **Why Introduced over Legacy Approaches:** A stateless-microservice-style "just retry on failure" approach ignores two real GenAI-specific risks this module addresses directly: retry storms from synchronized client retries, and a fallback provider's real, different capability profile silently producing a degraded or wrong result.
* **Key Failure Modes & Limitations:** Naive immediate retry without jitter causing a real thundering-herd amplification of an already-degraded dependency; blindly retrying a non-idempotent, timed-out tool call and risking a real duplicated side effect; treating a fallback model as a drop-in equivalent and silently truncating or degrading a real response without flagging it.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not applicable — this module's math is real retry-timing arithmetic, not model compute.
* **Space/Memory Footprint:** Not applicable at this module's level; real fallback-chain storage (cached/canned responses) is typically small and bounded.
* **Primary Bottleneck Type:** A real availability/reliability bottleneck — the risk is a synchronized retry storm or an unsafely-retried side effect, not a computational one.
* **Variable Legend:** $\text{Delay}_{\text{base}}$/$\text{Delay}_{\text{max}}$ = real backoff bounds, $\text{Jitter}_n$ = real per-attempt random noise, the 4 real retry-eligibility categories (transient, rate limit, timeout-idempotent, timeout-non-idempotent, non-retryable).

### 3. Production & Scalability
* **Deployment Considerations:** Real production systems typically track circuit-breaker state and fallback-tier usage as first-class real observability signals (`07_llm_evaluation_observability_and_guardrails`'s own tracing content, referenced) — a real spike in fallback-tier traffic is itself an actionable alert, not just an invisible internal failover.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why does the circuit breaker need a real HALF-OPEN state instead of just flipping back to CLOSED after a cooldown?
        *   *A:* Flipping directly back to CLOSED risks immediately re-exposing full real traffic to a dependency that might still be degraded — HALF-OPEN sends a real, small amount of probe traffic first, only fully re-opening if that probe genuinely succeeds, bounding the real risk of a premature full recovery.
    2.  *Q:* Your fallback model handles the request but produces a visibly lower-quality response — what do you do?
        *   *A:* Surface that real degradation explicitly (a real "reduced-context" or "degraded-mode" flag in the response/metadata) rather than presenting it identically to a normal response — the real, stated capability trade-off from this module's own worked example should be visible downstream, not hidden.
