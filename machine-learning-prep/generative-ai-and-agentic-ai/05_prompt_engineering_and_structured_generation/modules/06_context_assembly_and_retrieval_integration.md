# Module 06: Context Assembly & Prompt-Level Retrieval Integration

## 1. Introduction & Intuition

### The Core Bottleneck
A RAG system's retriever (`03_advanced_rag`) can return the perfect set of chunks, and an agent's tools (`04_ai_agents_and_protocols`) can return perfectly correct results — and the final prompt can still perform badly, because *how those results get assembled into the actual prompt text* is a separate, real engineering problem with its own failure modes. Where retrieved/tool content gets placed, how much of the token budget each segment of the prompt is allowed to consume, and what gets cut when the budget is exceeded are all decisions made at the prompt-construction layer, downstream of retrieval succeeding — and getting them wrong can waste a retrieval system's correct results just as thoroughly as a bad retrieval would have.

### High-Level Intuition
Retrieval handing back the right documents is like a research assistant handing you the right five source books for a report. Context assembly is what you do next — deciding which of those books' pages actually make it onto your desk, in what order, and what you do when your desk is too small to hold everything. A perfectly-chosen set of books, dumped onto a desk in a way that buries the most important page in the middle of a disorganized pile, still produces a worse report than the same books, arranged deliberately.

---

## 2. Core Concepts & Mathematical Formulation

### Prompt Template Structure for RAG/Tool Contexts

#### Intuition & Practical Use
A production prompt assembling retrieved or tool content is not one undifferentiated block of text — it's a template with distinct, deliberately-ordered segments: system instructions, few-shot examples (if any), retrieved/tool content, and conversation history, each with clear delimiters marking where one segment ends and the next begins (Module 01's delimiter design pattern, applied here specifically to multi-segment assembly). Treating this as an explicit template, not string concatenation improvised per call, is what makes the budget-allocation and truncation decisions below tractable and testable rather than ad hoc.

### Context Ordering & Position Bias ("Lost in the Middle")

#### Intuition & Practical Use
Where a piece of content sits within the prompt measurably affects how well a model uses it — content near the beginning or end of context tends to be attended to more reliably than content buried in the middle of a long prompt, an effect commonly called "lost in the middle" (the same phenomenon `03_advanced_rag` Module 01 covers from the retrieval-ranking side; this module covers it from the prompt-*construction* side specifically: given a fixed set of retrieved chunks already selected, where do you *place* them in the assembled prompt). The practical implication: the single most relevant retrieved chunk shouldn't be buried in the middle of ten chunks just because that's the order retrieval happened to return them in — deliberate placement (e.g., most relevant chunks near the start or end) is a real, low-cost lever independent of retrieval quality itself.

### Context Budget Allocation Across Segments

#### Intuition & Practical Use
A model's context window is a hard, finite resource shared across every segment in the template — system instructions, few-shot examples, retrieved/tool content, and conversation history are all competing for the same fixed token budget, and an assembly strategy needs an explicit policy for how that budget gets divided, not an implicit "whatever's left over" approach. A sane default policy: reserve a fixed, small allocation for system instructions (typically stable and short), a fixed allocation for few-shot examples (if used at all — Module 01 covers the cost/benefit of including them), and split the *remaining* budget between retrieved content and conversation history based on which the specific task weighs more heavily — a single-turn RAG query might allocate almost the entire remaining budget to retrieved content, while a long multi-turn conversation needs a more even split.

### Prompt-Level Deduplication & Truncation Strategies

#### Intuition & Practical Use
Two concrete, practical failure modes worth guarding against explicitly: **duplicate content** — a retriever or tool occasionally returning overlapping or near-identical chunks, silently wasting real budget on redundant text the model gains nothing from seeing twice; and **naive truncation** — cutting a segment's content at a fixed character/token limit without regard to structure, which can truncate mid-sentence or mid-JSON-object, producing genuinely malformed or confusing input. A real assembly pipeline should deduplicate near-identical retrieved chunks before budget allocation (cheap relative to the waste of not doing so), and truncate at meaningful boundaries (chunk/sentence boundaries, not arbitrary character counts) when a segment must be cut down to fit its allocation.

---

### Hand Calculation: Prompt Segment Budget Allocation
An 8,000-token context window, with a fixed 400-token system-instruction allocation, a fixed 600-token few-shot allocation (3 examples), and the remaining budget split 70/30 between retrieved content and conversation history for a RAG-heavy, single-turn-weighted task — with a 300-token reserve held back for the model's own output.

*   **Step 1: Compute the remaining budget after fixed allocations and output reserve.**
    $$\text{remaining} = 8{,}000 - 400 - 600 - 300 = 6{,}700 \text{ tokens}$$

*   **Step 2: Split the remaining budget 70/30 between retrieved content and conversation history.**
    $$\text{retrieved\_budget} = 0.7 \times 6{,}700 = 4{,}690 \text{ tokens}, \qquad \text{conversation\_budget} = 0.3 \times 6{,}700 = 2{,}010 \text{ tokens}$$

*   **Step 3: A real retrieval batch returns 6 chunks averaging 900 tokens each (5,400 tokens total) — over the 4,690-token retrieved budget by 710 tokens.**
    Trim order (deliberate policy, not arbitrary): drop the *lowest-ranked* whole chunk first (the retriever's own relevance ranking already orders these), rather than truncating every chunk proportionally — dropping the single 900-token lowest-ranked chunk brings the total to 4,500 tokens, now under budget, while every remaining chunk stays intact and un-truncated (avoiding the naive-truncation failure mode above).

The concrete lesson: when the retrieved-content budget is exceeded, the deliberate choice is *which whole chunks to drop* (informed by the retriever's own ranking), not *where to cut across all chunks* — dropping the weakest whole unit preserves the structural integrity of everything that remains, which naive proportional truncation would not.

---

## 3. Implementation & Reference Code

Below is a self-contained implementation of the segment-budget-allocation hand calculation, plus a whole-chunk trimming strategy (drop lowest-ranked chunks first, never truncate mid-chunk) matching the worked example above.

```python
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    rank: int  # 1 = most relevant, per the retriever's own ranking
    token_count: int
    text: str


@dataclass
class SegmentBudget:
    context_window: int
    system_tokens: int
    few_shot_tokens: int
    output_reserve: int
    retrieved_fraction: float  # of the REMAINING budget after fixed allocations + reserve

    def remaining_after_fixed(self) -> int:
        return self.context_window - self.system_tokens - self.few_shot_tokens - self.output_reserve

    def retrieved_budget(self) -> int:
        return int(self.remaining_after_fixed() * self.retrieved_fraction)

    def conversation_budget(self) -> int:
        return self.remaining_after_fixed() - self.retrieved_budget()


def fit_chunks_to_budget(chunks: list[RetrievedChunk], budget: int) -> list[RetrievedChunk]:
    """Drops the LOWEST-ranked whole chunks first until the total fits the budget.
    Never truncates a chunk's own text -- every chunk that survives stays fully intact."""
    kept = sorted(chunks, key=lambda c: c.rank)  # best-ranked first
    total = sum(c.token_count for c in kept)
    while total > budget and kept:
        dropped = kept.pop()  # drop the current lowest-ranked (worst) chunk
        total -= dropped.token_count
    return kept


if __name__ == "__main__":
    # Hand calc verification
    budget = SegmentBudget(
        context_window=8000, system_tokens=400, few_shot_tokens=600,
        output_reserve=300, retrieved_fraction=0.7,
    )
    remaining = budget.remaining_after_fixed()
    retrieved_budget = budget.retrieved_budget()
    conversation_budget = budget.conversation_budget()
    print(f"Remaining after fixed allocations + reserve: {remaining}")
    print(f"Retrieved budget (70%): {retrieved_budget}")
    print(f"Conversation budget (30%): {conversation_budget}")
    assert remaining == 6700
    assert retrieved_budget == 4690
    assert conversation_budget == 2010
    print("\nHand calc verified: 6,700 remaining, split into 4,690 / 2,010.")

    # Chunk-fitting verification: 6 chunks, 900 tokens each, ranks 1-6
    chunks = [RetrievedChunk(rank=i, token_count=900, text=f"chunk_{i}") for i in range(1, 7)]
    total_before = sum(c.token_count for c in chunks)
    print(f"\nTotal chunk tokens before fitting: {total_before} (budget: {retrieved_budget})")
    assert total_before == 5400

    fitted = fit_chunks_to_budget(chunks, retrieved_budget)
    total_after = sum(c.token_count for c in fitted)
    print(f"Chunks kept: {[c.rank for c in fitted]}, total tokens: {total_after}")
    assert total_after <= retrieved_budget
    assert total_after == 4500  # dropped exactly 1 chunk (rank 6, lowest-ranked)
    assert len(fitted) == 5
    assert 6 not in [c.rank for c in fitted], "The lowest-ranked chunk (rank 6) must be the one dropped"
    assert 1 in [c.rank for c in fitted], "The highest-ranked chunk must never be dropped while a lower-ranked one remains"
    print("\nChunk-fitting verified: exactly the lowest-ranked whole chunk was dropped; all surviving chunks remain fully intact (no mid-chunk truncation).")
```

---

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 760 280" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="380" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Prompt Segment Stacking &amp; Position Bias</text>

  <rect x="230" y="45" width="300" height="36" rx="5" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="380" y="68" text-anchor="middle" font-size="10.5" fill="#065f46" font-weight="600">System Instructions (400 tok)</text>

  <rect x="230" y="86" width="300" height="36" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="380" y="109" text-anchor="middle" font-size="10.5" fill="#1e3a8a" font-weight="600">Few-Shot Examples (600 tok)</text>

  <rect x="230" y="127" width="300" height="70" rx="5" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>
  <text x="380" y="147" text-anchor="middle" font-size="10.5" fill="#854d0e" font-weight="700">Retrieved Content (up to 4,690 tok)</text>
  <text x="380" y="163" text-anchor="middle" font-size="8.5" fill="#854d0e">Most relevant chunk placed near an EDGE</text>
  <text x="380" y="176" text-anchor="middle" font-size="8.5" fill="#854d0e">of this segment, not buried in the middle</text>
  <text x="380" y="189" text-anchor="middle" font-size="8.5" fill="#854d0e">("lost in the middle" mitigation)</text>

  <rect x="230" y="205" width="300" height="36" rx="5" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="380" y="228" text-anchor="middle" font-size="10.5" fill="#5b21b6" font-weight="600">Conversation History (up to 2,010 tok)</text>

  <text x="600" y="65" font-size="9" fill="#64748b">High attention</text>
  <text x="600" y="163" font-size="9" fill="#991b1b" font-weight="600">Attention dips here --</text>
  <text x="600" y="176" font-size="9" fill="#991b1b" font-weight="600">place carefully</text>
  <text x="600" y="223" font-size="9" fill="#64748b">High attention</text>

  <rect x="30" y="250" width="700" height="20" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1"/>
  <text x="380" y="264" text-anchor="middle" font-size="8.5" fill="#991b1b">8,000-token window: 400 + 600 + up to 4,690 + up to 2,010 + 300 (output reserve) = 8,000</text>
</svg>
</div>

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Turning a correct retrieval/tool result into an actually well-used prompt — placement, ordering, and budget allocation are separate engineering decisions downstream of retrieval succeeding, and getting them wrong wastes a correct retrieval result just as thoroughly as a bad one would have.
* **Why Introduced over Legacy Approaches:** Naively concatenating retrieved chunks in whatever order retrieval returned them, with no explicit budget policy, ignores both the real "lost in the middle" position effect and the real risk of exceeding the context window with no deliberate trimming strategy.
* **Key Failure Modes & Limitations:** Burying the most relevant retrieved content in the middle of a long context; truncating mid-chunk/mid-sentence when budget is exceeded, producing malformed input; silently including duplicate or near-duplicate retrieved chunks, wasting real budget.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** Not itself compute-heavy — budget allocation and chunk-fitting are cheap, local operations; the real cost impact is indirect, through how much of the (expensive) context window ends up filled with genuinely useful vs. wasted (duplicate/misplaced) content.
* **Space/Memory Footprint:** Directly determines how much of the finite context window (and its KV-cache memory cost) each segment consumes — a fixed-size resource this module's whole job is allocating well.
* **Primary Bottleneck Type:** Context-budget-bound — every token allocated to one segment is a token unavailable to another, making the allocation policy itself the primary lever, not a secondary detail.
* **Variable Legend:** Segment allocations (system/few-shot/retrieved/conversation tokens) summing to the total context window minus the output reserve.

### 3. Production & Scalability
* **Deployment Considerations:** Make budget-allocation policy explicit and testable (fixed allocations for stable segments, a deliberate split for variable ones) rather than implicit; deduplicate retrieved content before allocation, not after; when trimming to fit a budget, drop whole, lowest-ranked units rather than truncating across all units — trust the retriever's own ranking (`03_advanced_rag`) as the trimming signal.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Retrieval returns 10 relevant chunks but your budget only fits 6 — how do you decide what to keep?
        *   *A:* Keep the top-ranked whole chunks up to the budget and drop the rest entirely, rather than truncating every chunk proportionally — a full chunk conveys complete information; a proportionally-truncated one risks conveying a broken fragment of several, which is worse than fewer complete ones.
    2.  *Q:* Where would you place the single most important piece of retrieved content in a long prompt?
        *   *A:* Near the beginning or end of the retrieved-content segment specifically, not in the middle — exploiting the same position-attention effect the "lost in the middle" phenomenon describes, as a deliberate, low-cost placement decision independent of retrieval quality itself.
