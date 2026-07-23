# LLM Fine-tuning Fundamentals

Supervised Fine-Tuning (SFT) is the process of adjusting a pre-trained language model's parameters on a curated, instruction-following dataset. This shifts the model from a task-agnostic next-token predictor to a helpful assistant capable of formatting outputs, matching a desired tone, and executing domain-specific reasoning.

---

## 1. Why Fine-Tuning? Domain vs. Task Adaptation

Pre-trained base LLMs (e.g., Llama-3-Base) possess high general knowledge but lack formatting constraints, safety alignment, and specific operational behaviors. SFT bridges this gap by adapting the model along three primary dimensions:

1. **Domain Adaptation**: Injecting domain-specific terminology, nomenclature, and structural reasoning (e.g., medical diagnosis formatting, legal contract drafting).
2. **Behavioral & Style Alignment**: Enforcing a target persona, tone (e.g., professional, empathetic), or conciseness levels.
3. **Structured Outputs**: Training the model to strictly follow data formats (such as JSON schemas, SQL queries, or API calls) without generating conversational filler.

---

## 2. Full Fine-Tuning vs. PEFT vs. RAG vs. Prompt Engineering

When designing a production GenAI system, choosing the correct adaptation technique requires balancing compute budget, latency constraints, and data updates.

| Dimension | Prompt Engineering | Retrieval-Augmented Generation (RAG) | Parameter-Efficient Fine-Tuning (PEFT) | Full Fine-Tuning |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Purpose** | Contextual instruction | Context injection & factual grounding | Format/Style/Domain adaptation | Deep domain & behavior shift |
| **VRAM Training Overhead** | $0$ (Inference only) | $0$ (Inference only) | Low (only adapters require gradients) | Extremely High (Weights, Gradients, AdamW states) |
| **Inference Latency** | High (large context overhead) | High (retrieved context window) | Low (merged or tiny adapter overhead) | Lowest (native weights) |
| **Data Requirements** | None | Raw text database | $10^3 - 10^5$ instruction pairs | $10^5 - 10^7$ text/instruction pairs |
| **Out-of-Distribution Handling** | Poor | High (retrieved dynamic data) | Medium-Low (fixed at training time) | Medium-Low (fixed at training time) |
| **Vulnerability to Hallucinations** | High | Low (grounded in context) | High (un-grounded general lookup) | High (un-grounded general lookup) |

---

## 3. Continual Pre-training vs. Supervised Fine-Tuning (SFT)

A critical architectural decision is distinguishing between **Continual Pre-training (CP)** and **Supervised Fine-Tuning (SFT)**.

### Continual Pre-training (Domain Adaptation)
- **Objective**: Learn new factual knowledge, vocabularies, and language patterns.
- **Loss Function**: Masked or Causal Language Modeling (predict next token over raw unstructured text).
- **Data Style**: Raw, unformatted documents (e.g., medical textbooks, internal wikis).
- **Compute Scale**: High. Typically requires training on billions of tokens using small learning rates.

### Supervised Fine-Tuning (SFT)
- **Objective**: Learn how to behave, respond, and follow task instructions.
- **Loss Function**: Causal Language Modeling calculated exclusively on response tokens (Target-Only Masking).
- **Data Style**: Pairwise Instruction-Response blocks (`{"instruction": "...", "response": "..."}`).
- **Compute Scale**: Low to Medium. Hundreds of millions of tokens over 1–3 epochs.

---

## 4. Catastrophic Forgetting & Mitigation Strategies

When an LLM is fine-tuned on a narrow downstream dataset, it undergoes parameter drift. The optimizer adjusts weight matrices to minimize loss on the new distribution $\mathcal{D}_{\text{SFT}}$, shifting parameters away from the local minima of the pre-training distribution $\mathcal{D}_{\text{pre-train}}$. This results in **Catastrophic Forgetting**—the degradation of general reasoning, common-sense logic, and mathematical abilities.

```
                  W_pretrain (General Minima)
                         \
                          \  (Unregularized Optimizer Steps)
                           \
                            v
                       W_SFT (Forgets general skills; fits downstream task)
```

### Mitigation Strategies

#### 1. Rehearsal / Mixed Fine-Tuning
Mix a portion of the original pre-training dataset (typically 5% to 20% general instruction/conversational datasets like ShareGPT) into the domain training split.
$$\mathcal{D}_{\text{train}} = \alpha \mathcal{D}_{\text{domain}} + (1 - \alpha) \mathcal{D}_{\text{general}}$$

#### 2. Regularization-Based Methods (Elastic Weight Consolidation - EWC)
EWC restricts changes to weights that were critical for pre-training tasks. It calculates the diagonal of the **Fisher Information Matrix** $\mathbf{F}$ to measure parameter importance, adding a quadratic penalty to the loss function:
$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{i,\text{pre-train}})^2$$
Where $F_i$ is the Fisher information score for parameter $i$, and $\lambda$ is the regularizing coefficient.

#### 3. Low-Rank Adaptation (LoRA)
By freezing the base model parameters $W_0$ and only training low-rank adapter matrices, we prevent the base model's pre-trained weights from drifting, completely avoiding catastrophic forgetting of base representations within the frozen blocks.

---

## 5. Data Quality, Diversity, and Deduplication Requirements

In modern post-training, LIMA (Less Is More for Alignment) proved that **data quality and diversity out-perform pure quantity**.

1. **Quality Filtering**: Removing low-quality outputs, incomplete sentences, toxic text, and machine-generated alignment filler.
2. **Diversity**: Ensuring data covers a wide range of semantic intent (e.g., writing, reasoning, coding, classification) rather than repeating identical templates.
3. **Deduplication (MinHash LSH)**: Using MinHash and Locality-Sensitive Hashing (LSH) to identify and discard near-duplicate instructions. Training on repetitive data patterns causes models to overfit, leading to rote memorization and degraded generalization.

---

## 6. End-to-End Fine-tuning Pipeline Architecture

```
[Raw Domain Data] -> [Deduplication & Quality Filter] -> [Tokenization & Chat Templates]
                                                                     |
                                                                     v
[Loss Evaluation] <- [Backward Pass & Optimizer Step] <- [Target-Only Loss Masking]
```

1. **Extraction & Filtering**: Raw domain documents are parsed, deduplicated, and formatted into prompt-response splits.
2. **Tokenization & Formatting**: Data is passed through Jinja2 chat templates to inject roles (`system`, `user`, `assistant`) and tokenized.
3. **Loss Masking**: Prompt tokens are masked with the cross-entropy ignore index `-100`.
4. **Adapter Injection**: Adapter layers (e.g. LoRA) are initialized and attached.
5. **Optimization**: Model updates are executed using FP16/BF16 mixed-precision and gradient accumulation.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Fine-tuning solves the alignment problem—transforming general raw document completion models into structured instruction-following assistants with target styles and domains.
- **Why was it introduced?**
  To bypass the high costs of pre-training from scratch by reusing general representations, and to solve prompt constraints (context window overflow and high token cost) during in-context retrieval.
- **What are its limitations?**
  High compute cost, susceptibility to catastrophic forgetting, inability to naturally retrieve post-training factual updates without retraining, and risk of hallucinations on unseen facts.
- **Computational Complexity (Time & Memory)**
  - **Full Fine-Tuning**: $O(N)$ parameter updates, where training memory scales at $\sim 16 \times \text{parameters}$ under FP16 AdamW.
  - **PEFT (LoRA)**: $O(r \cdot d)$ parameter updates, reducing training state memory footprint significantly (often keeping optimizer states $< 10\%$ of full-tuning size).
- **Component Variable Denotation Legend**
  - $N$: Number of total parameters in the base model.
  - $r$: Rank of the adapter matrices.
  - $d$: Hidden dimension of the transformer layers.
  - $\mathbf{F}_i$: Fisher Information value for index $i$.
  - $\theta$: Current model parameters.
  - $\alpha$: Dataset mixing ratio or EWC regularization weight.
- **Production Use Cases**
  - Developing domain-expert coding assistants formatting strictly in corporate schemas.
  - Fine-tuning translation engines with custom corporate terminologies.
  - Building low-latency classification agents running on consumer-grade edge devices.
- **Follow-up questions interviewers ask**
  - *If a model starts hallucinating facts after SFT, how do you diagnose if it's due to catastrophic forgetting or poor data quality?*
  - *How would you determine the optimal ratio of general-purpose replay data to use in a mixed fine-tuning run?*
