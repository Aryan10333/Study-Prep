# Best Practices & Common Failures

Fine-tuning LLMs involves navigating training instabilities, data contamination, and model behavior drift. Successful production deployments require structured mitigation strategies for these standard failure modes.

---

## 1. Overfitting & Data Leakage Detection

Overfitting in LLMs manifests as rote memorization of SFT training data. The model loses its ability to generalize, producing repetitive outputs or formatting errors when queried outside the exact training distribution.

### Mitigation Strategies
- **Validation Splitting**: Always segregate 5% to 10% of instruction datasets as a validation split. Monitor training vs. validation loss. If validation loss begins to diverge or rise while training loss drops, stop training immediately.
- **De-contamination**: Run similarity analyses (e.g. MinHash LSH) between your fine-tuning split and standard downstream evaluation benchmarks (like MMLU or GSM8k). If test questions leak into training datasets, the evaluations will show artificially high scores that fail to translate to production.
- **Early Stopping**: Set early stopping checkpoints when validation metrics (like rouge scores or structured syntax validation success rates) plateau.

---

## 2. Mitigating Length Bias in Preference Alignment

In preference alignment (DPO, KTO, RLHF), models often discover that generating longer, more verbose responses artificially inflates reward scores or likelihood ratios. This is because longer answers contain more detail and structural indicators, which preference annotators and reward models implicitly favor.

```
                    Raw Preferred Completion (Verbose) -> High Reward
                                           /
                                          / [DPO Updates]
                                         v
                         Model Output becomes overly verbose
```

This behavior, known as **Length Bias**, degrades production throughput and increases token latency.

### Mitigation Strategies
1. **Length Normalization**: Normalize DPO log-likelihoods by token length during training to prevent the model from exploiting the loss function by adding verbose padding:
   $$\text{Normalized Log Likelihood} = \frac{1}{|y|} \log \pi_{\theta}(y|x)$$
2. **Preference Balancing**: Ensure that preference datasets are balanced—meaning dispreferred responses are not systematically shorter than preferred responses.
3. **Reward Penalties**: Introduce explicit negative weight coefficients in reward models or validation judges to penalize verbosity:
   $$\text{Reward}_{\text{adj}} = \text{Reward}_{\text{raw}} - \gamma \cdot \text{Length}(y)$$

---

## 3. Loss Spikes & Instability Mitigation

During SFT or preference optimization, training runs may experience sudden **Loss Spikes** followed by gradient explosion or NaN outputs. This behavior is typically caused by:
- **Extreme gradients** from corrupted or exceptionally long sequences.
- **Softmax saturation** in attention heads under mixed-precision training.
- **High learning rates** combined with lack of warmup steps.

```
Loss
 ^
 |          / \   <- Loss Spike (Gradients Explode / NaN)
 |         /   \
 |________/     \________
 +---------------------------> Steps
```

### Troubleshooting Runbook
1. **Gradient Clipping**: Restrict updates by clipping the global norm of gradients to a maximum value (typically $1.0$):
   $$\mathbf{g}_{\text{clipped}} = \mathbf{g} \cdot \min\left(1, \frac{\delta}{\|\mathbf{g}\|_2}\right)$$
   Where $\delta$ is the maximum norm limit.
2. **Learning Rate Warmup**: Linear increase of learning rate over the first 5% to 10% of total training steps, allowing parameters to adapt to the new distribution without massive parameter shifts.
3. **Precision Selection**: Prefer BF16 over FP16. FP16 has a narrow dynamic range ($6.1 \times 10^{-5}$ to $6.5 \times 10^4$), causing underflow/overflow during attention computation. BF16 matches the dynamic range of FP32, reducing numerical instability.
4. **Data Isolation**: Write validation checks to identify and discard training samples containing empty fields or abnormal lengths.

---

## 4. Hallucination Drift Post-SFT

As SFT forces the model to mimic human assistant behaviors, it also alters the pre-trained factual representations. This causes **Hallucination Drift** (or alignment tax), where the model's factuality degrades across general tasks.

### Mitigation Strategies
- **General Instruction Injection**: Integrate general-purpose SFT conversational data (e.g. 10% ShareGPT) to preserve base reasoning structures.
- **Reference Constraints**: Keep KL penalty terms active or run soft-merging (e.g. SLERP) between the fine-tuned model and the base model to recover factual representations.

---

## 5. Real-World Production Incident Post-Mortem

### Incident: The Evasive Assistant (Refusal Cascade)
*Context*: A company fine-tuned an 8B model to handle customer support questions while avoiding confidential internal queries.

*The Failure*: In the training set, safety annotators marked many general queries containing corporate keywords as "dispreferred" if they mentioned sensitive product areas. After DPO alignment, the model began aggressively refusing *all* queries containing those keywords, even benign requests (e.g. refusing to explain standard billing because it contained the billing system's proprietary name).

*The Diagnosis*: The DPO loss over-corrected parameters, creating a broad refusal attractor basin in weight space.

*The Solution*: The team rebuilt the dataset, splitting safe vs. unsafe uses of target keywords, and mixed in positive examples of the model explaining the safe context. They also lowered DPO $\beta$ to reduce the alignment penalty.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Mitigating training failures prevents waste of expensive compute budgets and ensures that aligned models remain stable and concise in production.
- **Why was it introduced?**
  Standard training configurations designed for pre-training fail during post-training due to narrow dataset shapes and the numerical instability of preference alignment.
- **What are its limitations?**
  Gradient clipping can slow down training convergence if the clip threshold is set too low. General instruction mixing increases the dataset size and training duration.
- **Computational Complexity (Time & Memory)**
  - **Gradient Clipping**: Adds a minor $O(N)$ calculation over the flat gradient vector.
  - **Validation Overhead**: Adds a forward pass over validation data at checkpoints.
- **Component Variable Denotation Legend**
  - $\delta$: Gradient clipping threshold.
  - $|y|$: Length of sequence $y$ in tokens.
  - $\gamma$: Length penalty multiplier.
  - $\pi_{\theta}$: Generative policy probability.
- **Production Use Cases**
  - Fine-tuning translation agents without loss spikes using BF16 mixed precision.
  - Debugging a DPO model that started outputting redundant paragraphs to maximize reward scores.
- **Follow-up questions interviewers ask**
  - *How would you diagnose whether a training crash is due to data corruption or mixed-precision underflow?*
  - *If a validation split shows perfect loss but human evaluators report high hallucination rates, what evaluation mistakes occurred?*
