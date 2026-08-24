# Module 03: PagedAttention & Memory-Efficient Serving

## 1. Introduction & Intuition

### The Core Bottleneck
Module 02 established that KV cache memory can be real and substantial. But *how* that memory gets allocated matters just as much as how much of it exists: a naive serving system that reserves a fixed, worst-case-length block of memory for every sequence — because it can't know in advance how long a generation will actually run — wastes real, often enormous amounts of that memory on space that's reserved but never used. PagedAttention exists to fix that allocation problem specifically. It's worth stating precisely what it does *not* do: **PagedAttention does not compress the KV cache or shrink its real per-token size** — that's Module 02's and Module 05's territory (fewer KV heads, lower precision). PagedAttention instead improves how the *existing* real memory footprint gets allocated, utilized, and shared.

### High-Level Intuition
This is the same problem operating-system virtual memory solved for programs decades ago, applied here to KV caches: instead of reserving one long, contiguous block of memory sized for the worst case a sequence *might* reach, allocate memory in small, fixed-size blocks (pages) on demand, only as a sequence actually grows. A sequence that ends up short only ever claims a few real blocks; the memory it didn't use was never reserved in the first place, so other sequences can borrow it instead of it sitting empty and locked to one request.

---

## 2. Core Concepts & Mathematical Formulation

### Contiguous (Naive) KV Cache Allocation and Its Waste

#### Intuition & Practical Use
Without a paging scheme, a serving system that wants to guarantee it never runs out of room for a growing sequence has to pre-reserve a contiguous memory region sized for the *maximum* sequence length the system supports — for every sequence, regardless of how long that particular sequence actually turns out to be. Most real sequences finish well short of the maximum, so most of that pre-reserved region sits real, allocated, and completely unused for the sequence's entire lifetime — genuine, wasted memory that directly limits how many concurrent sequences can fit in a fixed GPU memory budget.

### Block-Based (Paged) KV Cache Allocation

#### Intuition & Practical Use
PagedAttention instead divides the KV cache into small, fixed-size blocks (e.g., 16 tokens per block) drawn from a shared pool. A sequence claims new blocks only as it actually grows, one block at a time — never reserving the full worst-case length up front. The only real waste left is *internal fragmentation*: the unused remainder of a sequence's *last* block, bounded above by (block size − 1) tokens per sequence, a tiny, fixed cost regardless of how long the maximum supported sequence length is. This is also what enables real, efficient *memory sharing* — e.g., beam search candidates or shared prompt prefixes can point at the same underlying blocks instead of duplicating them, a real capability contiguous allocation cannot offer since each sequence's memory is a single private, non-shareable region.

---

### Worked Example (No Formula): Fragmentation Waste, Contiguous vs. Paged
Five real, variable-length sequences (token counts): $\{120,\ 340,\ 890,\ 45,\ 600\}$, actually used tokens $= 1{,}995$. Maximum supported sequence length: $2{,}048$ tokens. Paged block size: $16$ tokens.

*   **Step 1: Contiguous allocation.** Every sequence reserves the full $2{,}048$-token maximum, regardless of its real length.
    $$\text{Total reserved} = 2{,}048 \times 5 = 10{,}240 \text{ token-slots}, \quad \text{Waste} = 10{,}240 - 1{,}995 = 8{,}245 \text{ slots} \approx 80.5\% \text{ wasted}$$

*   **Step 2: Paged allocation.** Each sequence claims $\lceil L / 16 \rceil$ blocks (16 tokens each) — only the last block per sequence carries any real waste.

    | Sequence length | Blocks claimed | Slots reserved | Waste (slots) |
    |---|---|---|---|
    | 120 | 8 | 128 | 8 |
    | 340 | 22 | 352 | 12 |
    | 890 | 56 | 896 | 6 |
    | 45 | 3 | 48 | 3 |
    | 600 | 38 | 608 | 8 |

    $$\text{Total reserved} = 2{,}032 \text{ slots}, \quad \text{Waste} = 2{,}032 - 1{,}995 = 37 \text{ slots} \approx 1.8\% \text{ wasted}$$

*   **Step 3: Real comparison.** Contiguous allocation wastes $\approx 80.5\%$ of its reserved memory on this exact workload; paged allocation wastes $\approx 1.8\%$ — a real $\approx 44\times$ reduction in wasted memory, with no change whatsoever to the real per-token KV cache cost established in Module 02. This is precisely "improves allocation and utilization" as distinct from "compresses the cache" — the same 1,995 real tokens' worth of KV data is stored either way; only how much *extra*, unused memory is reserved alongside it changes.

<div style="text-align:center">

<svg viewBox="0 0 900 430" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;height:auto;font-family:sans-serif">
  <style>
    .lbl{font-size:13px;fill:#333}
    .hdr{font-size:15px;font-weight:bold;fill:#222}
    .used{fill:#4c78a8}
    .waste{fill:#e0e0e0;stroke:#bbb;stroke-width:1}
    .wasteHatch{fill:#f4b183}
  </style>

  <text x="225" y="24" text-anchor="middle" class="hdr">Contiguous Allocation (reserve max length = 2048)</text>
  <g font-size="12" fill="#333">
    <!-- 5 rows, bar width scaled: 2048 tokens = 420px -->
    <text x="10" y="55" class="lbl">len=120</text>
    <rect x="70" y="42" width="420" height="18" class="waste"/>
    <rect x="70" y="42" width="24.6" height="18" class="used"/>

    <text x="10" y="85" class="lbl">len=340</text>
    <rect x="70" y="72" width="420" height="18" class="waste"/>
    <rect x="70" y="72" width="69.7" height="18" class="used"/>

    <text x="10" y="115" class="lbl">len=890</text>
    <rect x="70" y="102" width="420" height="18" class="waste"/>
    <rect x="70" y="102" width="182.5" height="18" class="used"/>

    <text x="10" y="145" class="lbl">len=45</text>
    <rect x="70" y="132" width="420" height="18" class="waste"/>
    <rect x="70" y="132" width="9.2" height="18" class="used"/>

    <text x="10" y="175" class="lbl">len=600</text>
    <rect x="70" y="162" width="420" height="18" class="waste"/>
    <rect x="70" y="162" width="123.0" height="18" class="used"/>
  </g>
  <text x="70" y="200" class="lbl" fill="#4c78a8">■ used tokens</text>
  <text x="200" y="200" class="lbl" fill="#999">■ reserved-but-unused (waste)</text>
  <text x="225" y="222" text-anchor="middle" class="hdr" fill="#b05a3a">Total waste: 8,245 / 10,240 slots ≈ 80.5%</text>

  <line x1="0" y1="245" x2="900" y2="245" stroke="#ccc" stroke-width="1"/>

  <text x="675" y="270" text-anchor="middle" class="hdr">Paged (Block-Based) Allocation (block = 16 tokens)</text>
  <g font-size="12" fill="#333">
    <!-- widths scaled at same px/token factor as above (420/2048 ~ 0.205 px/token), grouped per sequence, waste shown only in last block -->
    <text x="480" y="300" class="lbl">len=120</text>
    <rect x="540" y="288" width="24.6" height="18" class="used"/>
    <rect x="564.6" y="288" width="1.6" height="18" class="wasteHatch"/>

    <text x="480" y="325" class="lbl">len=340</text>
    <rect x="540" y="313" width="69.7" height="18" class="used"/>
    <rect x="609.7" y="313" width="2.5" height="18" class="wasteHatch"/>

    <text x="480" y="350" class="lbl">len=890</text>
    <rect x="540" y="338" width="182.5" height="18" class="used"/>
    <rect x="722.5" y="338" width="1.2" height="18" class="wasteHatch"/>

    <text x="480" y="375" class="lbl">len=45</text>
    <rect x="540" y="363" width="9.2" height="18" class="used"/>
    <rect x="549.2" y="363" width="0.6" height="18" class="wasteHatch"/>

    <text x="480" y="400" class="lbl">len=600</text>
    <rect x="540" y="388" width="123.0" height="18" class="used"/>
    <rect x="663.0" y="388" width="1.6" height="18" class="wasteHatch"/>
  </g>
  <text x="675" y="420" text-anchor="middle" class="hdr" fill="#2e7d32">Total waste: 37 / 2,032 slots ≈ 1.8% (~44x less)</text>
</svg>

</div>

*   **Diagram Interpretation:** Both panels plot the same five real sequences at the same relative token scale. Contiguous allocation (top) reserves the full maximum length per sequence, leaving most of each bar as real, unused reserved memory. Paged allocation (bottom) reserves only whole blocks as needed, so waste is confined to a thin sliver at the end of each sequence's last block — the real, direct picture behind the $\approx 44\times$ waste reduction computed above.

---

## 3. Implementation & Reference Code

```python
import math


def contiguous_allocation_waste(lengths: list[int], max_len: int) -> dict:
    reserved = max_len * len(lengths)
    used = sum(lengths)
    waste = reserved - used
    return {"reserved": reserved, "used": used, "waste": waste, "waste_pct": waste / reserved * 100}


def paged_allocation_waste(lengths: list[int], block_size: int) -> dict:
    reserved = 0
    used = sum(lengths)
    for length in lengths:
        blocks = math.ceil(length / block_size)
        reserved += blocks * block_size
    waste = reserved - used
    return {"reserved": reserved, "used": used, "waste": waste, "waste_pct": waste / reserved * 100}


if __name__ == "__main__":
    lengths = [120, 340, 890, 45, 600]
    MAX_LEN = 2048
    BLOCK_SIZE = 16

    contiguous = contiguous_allocation_waste(lengths, MAX_LEN)
    paged = paged_allocation_waste(lengths, BLOCK_SIZE)

    print(f"Contiguous: reserved={contiguous['reserved']}, used={contiguous['used']}, "
          f"waste={contiguous['waste']} ({contiguous['waste_pct']:.2f}%)")
    print(f"Paged:      reserved={paged['reserved']}, used={paged['used']}, "
          f"waste={paged['waste']} ({paged['waste_pct']:.2f}%)")

    assert contiguous["used"] == paged["used"] == sum(lengths), "Real stored token count is identical either way -- PagedAttention doesn't compress"
    assert paged["waste_pct"] < contiguous["waste_pct"], "Paged allocation must waste strictly less than contiguous on this workload"
    reduction_factor = contiguous["waste_pct"] / paged["waste_pct"]
    print(f"Waste reduction factor: {reduction_factor:.1f}x")
    assert reduction_factor > 20, "Real reduction factor should be large for this illustrative max_len/block_size ratio"

    # Per-sequence internal fragmentation bound: each sequence wastes at most (block_size - 1) slots
    for length in lengths:
        blocks = math.ceil(length / BLOCK_SIZE)
        per_seq_waste = blocks * BLOCK_SIZE - length
        assert per_seq_waste < BLOCK_SIZE, "Internal fragmentation per sequence is bounded by block_size - 1, regardless of max_len"

    print("\nVerified: identical real stored-token count both ways (no compression); paged allocation's waste is bounded")
    print("per-sequence by block_size, independent of the maximum supported sequence length.")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** Real memory wasted by pre-reserving worst-case-length contiguous KV cache regions per sequence, which directly limits how many concurrent sequences fit in a fixed GPU memory budget.
* **Why Introduced over Legacy Approaches:** Naive contiguous allocation forces a real trade-off between over-provisioning (waste, as shown above) and under-provisioning (risk of running out of room mid-generation and having to reject or evict a real, in-flight sequence); paging removes that trade-off by allocating on demand.
* **Key Failure Modes & Limitations:** Describing PagedAttention as "compressing" or "shrinking" the KV cache is a common, real misconception — it changes *allocation*, not the real per-token memory cost from Module 02's formula; a system still needs enough *total* real GPU memory for the KV data actually being stored, paging only eliminates the *extra*, unused reservation on top of that.

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** No change to attention's real FLOPs — this is a memory-management technique, not a compute optimization (contrast with Module 04's FlashAttention, which targets real HBM IO during the attention computation itself).
* **Space/Memory Footprint:** Real waste bounded by (block size − 1) tokens per sequence, independent of the maximum supported sequence length — versus contiguous allocation's waste, which grows directly with how far below the maximum a real sequence's actual length falls.
* **Primary Bottleneck Type:** Memory-utilization-bound — the real constraint paging relieves is "how many concurrent sequences' KV caches fit in the GPU's fixed memory budget," which directly caps real effective batch size and therefore real throughput.
* **Variable Legend:** $L$ = a sequence's real token length, block size = fixed page size in tokens (a real serving-engine configuration parameter, commonly small, e.g. 16).

### 3. Production & Scalability
* **Deployment Considerations:** Smaller block sizes reduce real internal fragmentation further but add real per-block bookkeeping overhead; production engines (vLLM) tune this trade-off empirically rather than picking the theoretical minimum block size.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Does PagedAttention reduce the total GPU memory a model needs to serve requests?
        *   *A:* Not directly — it reduces *wasted* memory from over-reservation, letting more real concurrent sequences fit in the same fixed budget; the real per-token KV cache cost itself (Module 02's formula) is unchanged. Actually shrinking that per-token cost is Module 02's GQA/MQA and Module 05's quantization.
    2.  *Q:* How does block-based allocation enable KV-cache sharing across sequences (e.g., beam search)?
        *   *A:* Because each block is an independently addressable unit in a shared pool, multiple sequences that share an identical prefix can have their sequence-specific block tables point at the same real underlying blocks instead of duplicating that data — a real capability a single contiguous per-sequence allocation has no way to offer.
