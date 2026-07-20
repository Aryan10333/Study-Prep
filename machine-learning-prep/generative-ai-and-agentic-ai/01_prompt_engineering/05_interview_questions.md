# Top 15 Prompt Engineering & In-Context Learning Interview Questions

This flashcard guide presents 15 high-frequency interview questions asked by top tech companies (Meta, Google, OpenAI, Anthropic) covering reasoning prompt architectures, In-Context Learning (ICL) activation shifts, constrained decoding, and prompt compression failure modes.

---

### Q1: Why does Chain-of-Thought (CoT) prompting significantly improve reasoning performance over standard zero-shot prompting?
- **Answer:** Standard greedy decoding forces the model to compute the final answer token $y$ immediately in a single forward pass without additional computation steps. CoT generates intermediate reasoning tokens ($z_1, z_2, \dots, z_k$), which effectively increases the number of transformer forward passes and allows key activation vectors to update in working memory before outputting the final answer.

---

### Q2: What is Tree-of-Thoughts (ToT), and when should you choose ToT over Chain-of-Thought (CoT)?
- **Answer:** ToT frames reasoning as a tree search (using BFS or DFS) where each node represents a candidate thought step and edges represent logical transitions. Evaluator functions score each candidate thought and prune low-scoring branches. Choose ToT over CoT for complex combinatorial problems (e.g. game puzzles, multi-step math planning) where early mistakes in a linear chain cause catastrophic failure.

---

### Q3: Explain how In-Context Learning (ICL) works without updating model parameters.
- **Answer:** ICL operates by shifting intermediate hidden activation vectors within the transformer's attention layers. Demonstration examples inside the prompt act as context keys and values that steer attention heads to aggregate an implicit "task vector" $v_{\text{task}}$. This vector shifts the query's activation path into a task-specific region of representation space without modifying weight matrices.

---

### Q4: What is the "Demonstration Order Sensitivity" problem in In-Context Learning?
- **Answer:** The order of few-shot prompt examples significantly impacts model predictions (causing $20\% - 30\%$ accuracy variance). This occurs because multi-head self-attention exhibits recency bias and positional encoding decay, giving greater weight to examples placed closer to the query token.

---

### Q5: How does constrained decoding with GBNF grammars or Outlines guarantee 100% valid JSON output?
- **Answer:** Constrained decoding operates directly on the output logit vector before softmax sampling. A finite state machine tracks the schema grammar state. At each decoding step, illegal token IDs receive a logit bias overlay of $-\infty$, rendering their sampling probability exactly $0.0$ and ensuring output syntax validity.

---

### Q6: What is the main drawback of applying strict JSON schema constraints from token 0 during decoding?
- **Answer:** Forcing strict JSON schema constraints from token 0 deprives the LLM of free-form Chain-of-Thought planning workspace tokens. This can degrade the model's reasoning accuracy compared to letting the model reason in free text before outputting JSON.

---

### Q7: How does LLMLingua perform perplexity-based prompt compression?
- **Answer:** LLMLingua uses a small, fast auxiliary language model (e.g., Llama-3-8B) to compute the conditional perplexity $PPL(x_i)$ of each prompt token. Tokens with low perplexity ($PPL \approx 1.0$) are predictable connectives and redundant words, whereas high-perplexity tokens are crucial instructions. Pruning low-PPL tokens yields $50\% - 80\%$ context reduction while retaining semantic meaning.

---

### Q8: What is DSPy, and how does it replace manual prompt engineering?
- **Answer:** DSPy is a framework that treats prompts as compiled programs rather than static strings. Developers define declarative Signatures (inputs/outputs) and Modules (CoT, ReAct). A DSPy compiler evaluates candidate instructions and few-shot examples against a metric, optimizing prompt configurations automatically.

---

### Q9: Write a Python function using PyTorch that applies a logit mask to force a model to select only from a specific list of token IDs.
```python
import torch
import torch.nn.functional as F

def mask_logits(logits: torch.Tensor, allowed_ids: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, float('-inf'))
    mask[allowed_ids] = 0.0
    masked_logits = logits + mask
    return F.softmax(masked_logits, dim=-1)
```

---

### Q10: What is Directional Stimulus Prompting?
- **Answer:** Directional Stimulus Prompting uses a small, lightweight policy model to generate task-specific directional hints or keywords that are prepended to the input prompt, guiding the larger frozen LLM toward the desired output.

---

### Q11: Explain Skeleton-of-Thought (SoT) and how it reduces latency in long-form generation.
- **Answer:** SoT reduces latency by first asking the LLM to output a brief skeleton outline of point titles. It then issues parallel API calls to expand each point independently in parallel, reducing overall generation time from $O(N)$ sequential steps to $O(N / k)$.

---

### Q12: Why can providing incorrect labels in few-shot demonstrations fail to steer large LLMs ($> 70\text{B}$)?
- **Answer:** Larger models possess strong pre-trained semantic priors. When few-shot demonstrations conflict with ground-truth facts (e.g., *"Apple -> Fruit", "Banana -> Vehicle"*), strong attention heads override the prompt demonstrations in favor of pre-trained weights, a phenomenon known as label invariance.

---

### Q13: How do you prevent prompt injection attacks in enterprise customer-facing LLM applications?
- **Answer:** Combine input sanitization (stripping system delimiter tokens), dual-LLM architecture (untrusted user input processed by an isolated reader model before reaching the execution model), and output validation (checking outputs against canary tokens and safety classifiers).

---

### Q14: Derive the logit masking probability for a masked token with logit bias $b_i = -\infty$.
- **Answer:** 
  $$p_i = \frac{\exp(z_i + (-\infty))}{\sum_j \exp(z_j + b_j)} = \frac{0}{\sum_j \exp(z_j + b_j)} = \mathbf{0.0}$$
  Thus, the probability of sampling a masked token is identically zero.

---

### Q15: How do you debug a production prompt that intermittently fails with JSON parsing errors under high concurrency?
- **Answer:**
  1. Inspect temperature settings (ensure `temperature=0.0`).
  2. Switch from prompt-based JSON instructions to logit-masked constrained decoding.
  3. Validate that system tokenizers and server models match exactly.
