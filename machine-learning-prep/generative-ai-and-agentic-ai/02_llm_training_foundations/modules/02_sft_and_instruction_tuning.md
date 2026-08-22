# Module 02: Supervised Fine-Tuning (SFT) & Instruction Tuning

## 1. Introduction & Intuition

### The Core Bottleneck
A raw pretrained model is a next-token predictor trained on raw web text — it completes "The capital of France is" fluently, but has no learned notion of "answer this instruction helpfully and stop." Ask it a direct question and it's just as likely to continue the prompt with more questions, or ramble into unrelated text, as it is to answer. The bottleneck is behavioral, not architectural: the base model already contains the knowledge needed to answer well, but has never been trained on the specific *format* of "instruction in, helpful response out, then stop."

### High-Level Intuition
Supervised Fine-Tuning (SFT) is showing the model thousands of examples of exactly that format — (instruction, ideal response) pairs — and training it with the same next-token-prediction objective as pretraining, but now over curated conversational data instead of raw web text. Critically, we only want to train the model to *generate* good responses, not to *memorize* the instructions themselves — so the loss is computed only over the response tokens, with the prompt tokens masked out of the loss entirely.

---

## 2. Core Concepts & Mathematical Formulation

### SFT Loss with Prompt-Token Masking

#### Purpose & Intuition
If we computed the standard causal LM loss over the *entire* sequence (prompt + response), the model would waste capacity learning to predict the prompt tokens themselves — which are given as input, not something the model should learn to generate. Masking the loss to only the response tokens focuses every gradient update on what actually matters: producing a good completion given the instruction.

#### Mathematical Formulation
For a tokenized sequence of length $L$ with a binary mask $m_i \in \{0, 1\}$ (1 for response tokens, 0 for prompt tokens):
$$\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_i m_i} \sum_{i=1}^{L} m_i \cdot \log P(w_i \mid w_{<i})$$

The mask ensures the denominator only counts response tokens, so the average loss is not diluted by (and no gradient flows from) the prompt portion.

---

### Hand Calculation: Masked Loss on a Tiny Sequence
Let's compute the masked SFT loss for a toy 6-token sequence: prompt = `["Summarize:", "cats", "sleep"]` (mask = 0), response = `["Cats", "nap", "often"]` (mask = 1).

*   **Step 1: Per-token negative log-probabilities** (hypothetical model outputs)
    Assume the model assigns these $-\log P(w_i \mid w_{<i})$ values to each position:
    | Token | Mask | $-\log P$ |
    |---|---|---|
    | "Summarize:" | 0 | 2.1 |
    | "cats" | 0 | 1.8 |
    | "sleep" | 0 | 3.0 |
    | "Cats" | 1 | 0.9 |
    | "nap" | 1 | 1.4 |
    | "often" | 1 | 0.6 |

*   **Step 2: Zero out the masked (prompt) positions' contribution**
    Only the response row values (0.9, 1.4, 0.6) enter the sum; the prompt rows (2.1, 1.8, 3.0) are excluded entirely, not just down-weighted.

*   **Step 3: Average over response tokens only**
    $$\mathcal{L}_{\text{SFT}} = \frac{0.9 + 1.4 + 0.6}{3} = \frac{2.9}{3} \approx 0.9667$$

Note the denominator is 3 (the number of response tokens), not 6 (the full sequence length) — this is the entire point of masking: the loss magnitude reflects only how well the model predicts the response, unaffected by how "surprising" the prompt tokens were.

---

### Tensor & Shape Tracking
*   **Input token IDs:** `[B, L]`
*   **Loss mask:** `[B, L]` (binary, 1 for response tokens)
*   **Logits:** `[B, L, V]`
*   **Per-token loss (before masking):** `[B, L]`
*   **Masked scalar loss:** `[]` (sum of masked per-token losses, divided by mask sum)

---

### Instruction Dataset Construction & Chat Templates

#### Intuition & Practical Use
Instruction data is formatted with a **chat template** — a fixed set of special tokens/markers (e.g., `<|user|>`, `<|assistant|>`) that structure a conversation into a single flat token sequence the model can learn to parse positionally. Multi-turn conversations extend this pattern, with the loss mask covering *every* assistant turn's tokens (not just the final one), so the model learns to respond well at any point in a multi-turn exchange, not only as the last turn.

#### Data Quality vs. Quantity
A relatively small set (thousands, not millions) of high-quality, diverse instruction examples reliably outperforms a much larger but noisier or repetitive set — SFT is teaching *format and behavior*, not new knowledge, so the marginal value of additional examples drops quickly once the format is well-represented across enough different task types.

### Synthetic Instruction-Data Generation

#### Intuition & Practical Use
Manually writing thousands of high-quality (instruction, response) pairs is expensive. A common alternative is generating them synthetically — prompting a strong existing LLM to produce instruction/response pairs, optionally seeded from a small set of human-written examples (self-instruct-style bootstrapping), then filtering aggressively before using them for SFT.

#### Quality Control Considerations
*   **Diversity:** Synthetic generation left unchecked tends to collapse onto a narrow set of phrasing patterns and topics; explicit diversity sampling (varying instruction types, topics, and difficulty) is required to avoid a homogeneous dataset.
*   **Deduplication:** Near-duplicate synthetic examples (common when sampling repeatedly from the same generator) waste training compute and can bias the model toward over-represented patterns; embedding-based or n-gram-based deduplication is standard.
*   **Contamination:** Synthetic data generated by a model that was itself trained (even indirectly) on benchmark data can leak benchmark-like content into your SFT set, inflating eval scores in a way that doesn't reflect real generalization — decontamination against your eval suite is a required step, not optional hygiene.
*   **Synthetic-Specific Risks:** Models generating synthetic data can propagate their own stylistic quirks, factual errors, or refusal patterns into the student model (a distillation-like effect) — synthetic data should be spot-checked for quality, not assumed correct because it's fluent.

---

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 800 190" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <text x="400" y="24" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">SFT Loss Masking: Prompt Tokens Excluded, Response Tokens Included</text>

  <g transform="translate(40, 55)">
    <!-- token boxes -->
    <rect x="0"   y="0" width="90" height="42" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.3" />
    <text x="45" y="25" text-anchor="middle" font-size="10" fill="#475569">"Summarize:"</text>
    <rect x="98"  y="0" width="90" height="42" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.3" />
    <text x="143" y="25" text-anchor="middle" font-size="10" fill="#475569">"cats"</text>
    <rect x="196" y="0" width="90" height="42" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.3" />
    <text x="241" y="25" text-anchor="middle" font-size="10" fill="#475569">"sleep"</text>

    <rect x="304" y="0" width="90" height="42" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.6" />
    <text x="349" y="25" text-anchor="middle" font-size="10" fill="#065f46" font-weight="bold">"Cats"</text>
    <rect x="402" y="0" width="90" height="42" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.6" />
    <text x="447" y="25" text-anchor="middle" font-size="10" fill="#065f46" font-weight="bold">"nap"</text>
    <rect x="500" y="0" width="90" height="42" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.6" />
    <text x="545" y="25" text-anchor="middle" font-size="10" fill="#065f46" font-weight="bold">"often"</text>

    <!-- mask row -->
    <text x="143" y="66" text-anchor="middle" font-size="10" fill="#94a3b8">mask = 0 (prompt, excluded from loss)</text>
    <text x="447" y="66" text-anchor="middle" font-size="10" fill="#059669" font-weight="bold">mask = 1 (response, included in loss)</text>

    <text x="143" y="-8" text-anchor="middle" font-size="10" fill="#64748b" font-weight="bold">Instruction</text>
    <text x="447" y="-8" text-anchor="middle" font-size="10" fill="#065f46" font-weight="bold">Response</text>
  </g>
</svg>
</div>

---

## 3. Implementation & Reference Code

Below is a self-contained masked cross-entropy loss implementation and a chat-template formatting example.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sft_masked_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Computes SFT cross-entropy loss, averaged only over response (mask=1) tokens.

    logits:    [B, L, V]
    targets:   [B, L]
    loss_mask: [B, L], 1.0 for response tokens, 0.0 for prompt tokens
    """
    B, L, V = logits.shape
    # .reshape() rather than .view(): a common real usage pattern is calling this on
    # a slice (e.g. logits[:, :-1, :] for next-token prediction), which is not
    # contiguous in memory -- .view() throws RuntimeError there, .reshape() copies
    # transparently when needed and works on both fresh and sliced tensors.
    per_token_loss = F.cross_entropy(
        logits.reshape(B * L, V), targets.reshape(B * L), reduction="none"
    ).reshape(B, L)  # [B, L]

    masked_loss = per_token_loss * loss_mask  # [B, L]
    return masked_loss.sum() / loss_mask.sum().clamp(min=1)  # scalar


def format_chat_example(instruction: str, response: str) -> tuple[str, list[int]]:
    """Builds a single flat chat-formatted string and a token-level loss mask marker
    (illustrated at the string level; a real tokenizer would map this to token-level masks)."""
    prompt_part = f"<|user|>\n{instruction}\n<|assistant|>\n"
    response_part = f"{response}<|end|>"
    full_text = prompt_part + response_part
    # 0 for every prompt character's "region", 1 for every response character's "region" (illustrative only)
    mask_marker = [0] * len(prompt_part) + [1] * len(response_part)
    return full_text, mask_marker


if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, V = 2, 6, 1000

    logits = torch.randn(B, L, V)
    targets = torch.randint(0, V, (B, L))
    # First 3 tokens are prompt (masked out), last 3 are response (kept)
    loss_mask = torch.tensor([[0., 0., 0., 1., 1., 1.],
                               [0., 0., 1., 1., 1., 1.]])

    loss = sft_masked_loss(logits, targets, loss_mask)
    print(f"Masked SFT loss: {loss.item():.4f}")

    text, mask = format_chat_example("Summarize: cats sleep", "Cats nap often")
    print(f"\nFormatted example:\n{text}")
    print(f"Prompt region length: {mask.count(0)} chars, Response region length: {mask.count(1)} chars")
```

---

## Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Converting a raw next-token-prediction base model into one that follows the instruction-response format and behaves helpfully when prompted directly.
*   **Why Introduced over Legacy Approaches:** Prompt-engineering a base model (few-shot examples in-context) can partially elicit instruction-following behavior, but SFT bakes the behavior into the weights directly, producing much more reliable zero-shot instruction-following without needing few-shot examples at inference time.
*   **Key Failure Modes & Limitations:** Catastrophic forgetting — aggressive fine-tuning on a narrow instruction distribution can degrade the base model's broader pretraining knowledge and capabilities if the dataset lacks diversity or the learning rate is too high.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Identical to pretraining's forward/backward FLOPs per token — SFT doesn't change the per-token compute cost, only the data distribution and (via masking) which tokens contribute gradient signal.
*   **Space/Memory Footprint:** Full-parameter SFT carries the full training memory footprint discussed in Module 01 (the "16Ψ bytes" wall) unless combined with the parameter-efficient methods in Module 03.
*   **Primary Bottleneck Type:** Data-quality-bound more than compute-bound in practice — the ceiling on SFT quality is usually the diversity and correctness of the instruction dataset, not GPU throughput.
*   **Variable Legend:** $L$ = sequence length, $V$ = vocabulary size, $m_i$ = loss mask at position $i$.

### 3. Production & Scalability
*   **Deployment Considerations:** SFT datasets are typically versioned and evaluated independently of the model itself (dataset ablations), since swapping in a higher-quality instruction set often improves downstream quality more than architectural changes at this stage.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why mask the loss on prompt tokens instead of just training on the full sequence?
        *   *A:* The prompt is given as input at inference time, not generated — training the model to predict it wastes gradient signal on a task it will never need to perform, and can dilute the loss magnitude on the tokens that actually matter (the response).
    2.  *Q:* What's the risk of using purely synthetic data generated by another LLM for SFT?
        *   *A:* Without careful filtering, synthetic data can be low-diversity (collapsing onto the generator model's stylistic patterns), contain factual errors the generator itself has, or leak benchmark-like content if the generator was exposed to eval data — all of which can silently degrade or mis-represent the fine-tuned model's real capabilities.
