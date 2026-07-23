# Production Considerations & Serving

Deploying fine-tuned models in enterprise environments requires structured evaluation, infrastructure sizing, and cost-efficient serving patterns.

---

## 1. Decision Framework: Prompting vs. RAG vs. Fine-Tuning

When designing an LLM application, select the architecture based on requirements for factual updates and stylistic control:

```
                          Style / Formatting Constraints?
                                   /           \
                             [No] /             \ [Yes]
                                 /               \
                       Factual Updates?       SFT / PEFT
                        /           \
                  [No] /             \ [Yes]
                      /               \
               Prompt Eng.            RAG
```

- **Prompt Engineering**: Best for quick prototyping, low-complexity tasks, and variable user intent.
- **RAG (Retrieval-Augmented Generation)**: Best for applications requiring real-time external knowledge integration and source auditability.
- **Supervised Fine-Tuning (SFT)**: Best for enforcing strict formatting structures (JSON/code), unique personas, or domain vocabularies.
- **Combined (RAG + SFT)**: The gold standard for enterprise agents—using SFT to train the model on retrieval response styles, and RAG to inject factually correct documents.

---

## 2. VRAM Sizing Calculations for Fine-Tuning

Calculating VRAM footprint requirements is essential for choosing cluster sizing (e.g. single A10G vs. 8x H100s).

### Mathematical Sizing Formulation
The total VRAM required during training is:
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{gradients}} + \text{VRAM}_{\text{optimizer}} + \text{VRAM}_{\text{activations}}$$

Where:
- **Weights VRAM**:
  - FP16/BF16: $2 \times \Phi$ bytes (where $\Phi$ is the base parameter count).
  - 4-bit (NF4): $0.5 \times \Phi$ bytes.
- **Gradients VRAM**:
  - $2 \times \Phi_{\text{trainable}}$ bytes.
- **Optimizer States VRAM** (assuming standard FP32 AdamW):
  - $12 \times \Phi_{\text{trainable}}$ bytes.
- **Activations VRAM** (VRAM needed to store intermediate activations during the forward pass for backpropagation. Without activation checkpointing):
  $$\text{VRAM}_{\text{activations}} \approx b \cdot L \cdot d \cdot n_{\text{layers}} \cdot \left( 10 + \frac{5}{2} \frac{L \cdot n_{\text{heads}}}{d} \right) \text{ bytes}$$
  Where $b$ is the batch size, $L$ is sequence length, $d$ is hidden dimension, $n_{\text{layers}}$ is the number of layers, and $n_{\text{heads}}$ is the attention heads.

---

### Step-by-Step Whiteboard Calculation: Sizing an 8B Model

Let's calculate the VRAM required to train an 8B parameter model ($\Phi = 8 \times 10^9$) under three scenarios, ignoring activation memory:

#### Scenario A: Full SFT (BF16 Precision, FP32 AdamW Optimizer)
- Parameters trainable: $\Phi_{\text{trainable}} = 8 \times 10^9$
1. **Model Weights**: $2 \times 8\text{ B} = 16\text{ GB}$
2. **Gradients**: $2 \times 8\text{ B} = 16\text{ GB}$
3. **Optimizer States**: $12 \times 8\text{ B} = 96\text{ GB}$
4. **Subtotal**: $16 + 16 + 96 = 128\text{ GB}$

*Required Hardware*: Minimum $2\times$ 80GB A100 GPUs (160GB total VRAM) to accommodate weights, gradients, optimizer states, and activation space.

#### Scenario B: LoRA Fine-Tuning ($r = 16$, $1\%$ Trainable Parameters)
- Base weights frozen in BF16: $\Phi = 8 \times 10^9$
- Trainable parameters: $\Phi_{\text{trainable}} = 0.08 \times 10^9$ (80M parameters)
1. **Frozen Base Weights**: $2 \times 8\text{ B} = 16\text{ GB}$
2. **Trainable Weights (BF16)**: $2 \times 0.08\text{ B} = 0.16\text{ GB}$
3. **Gradients**: $2 \times 0.08\text{ B} = 0.16\text{ GB}$
4. **Optimizer States**: $12 \times 0.08\text{ B} = 0.96\text{ GB}$
5. **Subtotal**: $16 + 0.16 + 0.16 + 0.96 = 17.28\text{ GB}$

*Required Hardware*: A single 24GB consumer GPU (e.g. RTX 3090/4090 or A10G) can handle the model and activation memory during training.

#### Scenario C: QLoRA Fine-Tuning (4-bit Base Weights, $1\%$ Trainable Parameters)
- Base weights frozen in NF4: $\Phi = 8 \times 10^9$
- Trainable parameters: $\Phi_{\text{trainable}} = 0.08 \times 10^9$ (80M parameters)
1. **Frozen Base Weights (NF4)**: $0.5 \times 8\text{ B} = 4\text{ GB}$
2. **Trainable Weights (BF16)**: $2 \times 0.08\text{ B} = 0.16\text{ GB}$
3. **Gradients**: $2 \times 0.08\text{ B} = 0.16\text{ GB}$
4. **Optimizer States**: $12 \times 0.08\text{ B} = 0.96\text{ GB}$
5. **Subtotal**: $4 + 0.16 + 0.16 + 0.96 = 5.28\text{ GB}$

*Required Hardware*: Easily runs on lower-tier consumer hardware (16GB VRAM GPUs like the T4 or RTX 4080) with ample space for activations.

---

## 3. Multi-LoRA Serving Infrastructure

In multi-tenant SaaS environments, serving unique models per tenant is costly. **Multi-LoRA Serving** (pioneered by vLLM, LoRAX, and SGLang) allows serving a single base model and dynamically hot-swapping adapters:

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <!-- Step 1: Request -->
    <div style="background-color: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; padding: 8px 16px; border-radius: 6px; font-weight: 600; font-size: 12px; text-align: center; width: fit-content;">Incoming Request for Tenant A</div>
    
    <!-- Arrow -->
    <div style="color: #94a3b8; font-weight: bold; font-size: 14px; margin: -5px 0;">↓</div>

    <!-- Orchestrator -->
    <div style="background-color: #eff6ff; color: #1e40af; border: 2px solid #2563eb; padding: 10px 20px; border-radius: 6px; font-weight: bold; font-size: 13px; text-align: center; width: 70%;">
        vLLM / SGLang / LoRAX Multi-LoRA Orchestrator
    </div>

    <!-- Arrow split -->
    <div style="display: flex; gap: 100px; justify-content: center; width: 100%; color: #94a3b8; font-weight: bold; font-size: 14px; margin: -5px 0;">
        <span>↓</span>
        <span>↓</span>
        <span>↓</span>
    </div>

    <!-- Active Components -->
    <div style="display: flex; gap: 20px; width: 100%; justify-content: center; flex-wrap: wrap;">
        <!-- Base Model -->
        <div style="flex: 1; min-width: 150px; max-width: 180px; background-color: #ffffff; border: 1px solid #e2e8f0; padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; border-top: 3px solid #7c3aed;">
            <div style="color: #5b21b6; font-weight: bold; font-size: 11px;">Base Model (BF16)</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 4px;">[Shared in VRAM]</div>
        </div>
        <!-- Adapter A -->
        <div style="flex: 1; min-width: 150px; max-width: 180px; background-color: #ffffff; border: 1px solid #e2e8f0; padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; border-top: 3px solid #059669;">
            <div style="color: #065f46; font-weight: bold; font-size: 11px;">LoRA Adapter A</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 4px;">[Active / Loaded]</div>
        </div>
        <!-- Adapter B -->
        <div style="flex: 1; min-width: 150px; max-width: 180px; background-color: #ffffff; border: 1px solid #e2e8f0; padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; border-top: 3px solid #94a3b8; opacity: 0.5;">
            <div style="color: #64748b; font-weight: bold; font-size: 11px;">LoRA Adapter B</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 4px;">[Inactive / Cached]</div>
        </div>
    </div>
</div>

1. **Shared Base Model**: The 16-bit base model parameters are kept in VRAM, computing baseline activations once.
2. **Paged Adapter Cache**: Adapter weights ($<100\text{ MB}$) are cached in non-GPU memory or pre-allocated GPU spaces.
3. **Batching & Custom Kernels**: Custom operations (like Triton kernels) gather active request tokens and multiply activations by their corresponding low-rank adapter weights during the forward pass, avoiding VRAM fragmentation.

---

## 4. Automated Evaluation Pipelines

Production readiness requires verification across three evaluation levels:
1. **Deterministic Benches**: Assert syntax correctness, code validity, or extraction match using programmatic unit tests.
2. **Standard LLM Benchmarks**: Evaluate general reasoning metrics (MMLU, GSM8k, MT-Bench) to ensure no catastrophic forgetting.
3. **LLM-as-a-Judge**: Utilize a larger model (e.g. GPT-4o) to compare SFT completions against target templates, assessing relative win rates.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Estimating hardware requirements prevents Out-of-Memory (OOM) failures mid-training, and multi-LoRA serving makes multi-tenant model customization cost-effective.
- **Why was it introduced?**
  Naively serving unique fully fine-tuned models per tenant leads to linear cost scaling, requiring one dedicated GPU cluster per customer.
- **What are its limitations?**
  Multi-LoRA serving introduces a small latency penalty when loading uncached adapters from CPU RAM to GPU memory during high traffic spikes.
- **Computational Complexity (Time & Memory)**
  - **Serving Memory**: base weights VRAM + active adapter weights + KV cache space.
  - **Inference Time Complexity**: $O(L \cdot d)$ base execution + $O(L \cdot r)$ low-rank projection overhead.
- **Component Variable Denotation Legend**
  - $\Phi$: Parameter size of the model.
  - $\Phi_{\text{trainable}}$: Active trainable parameter size.
  - $b$: Training batch size.
  - $L$: Sequence token length.
  - $d$: Hidden dimension size.
- **Production Use Cases**
  - Estimating the GPU budget needed to fine-tune a Llama-3-70B model on internal medical files.
  - Setting up vLLM to serve custom writing assistants for 100 enterprise customers using a single base instance.
- **Follow-up questions interviewers ask**
  - *How does activation checkpointing reduce VRAM footprint at the expense of computational overhead during training?*
  - *Under what latency and throughput conditions would you transition from multi-LoRA serving to merging adapters permanently?*
