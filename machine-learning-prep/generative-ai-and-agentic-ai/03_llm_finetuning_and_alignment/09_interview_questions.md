# 🎯 LLM Fine-Tuning & Alignment: High-Frequency Question Bank

> **Target Role:** AI Engineer / Applied AI Engineer / LLM Engineer (3+ Years Experience)
> **Scope:** Core mechanics, memory math, preference alignment, production deployment, and failure mode scenarios.

---

## 1. Fine-Tuning & SFT Foundations

### 1. When Fine-Tuning is Appropriate
**What is Fine-Tuning, and when would you choose it over RAG, Prompt Engineering, or Continued Pre-Training?**

- **Quick Screen Response**: Fine-tuning adapts a model's style, formatting, and behavioral alignment. Choose it over RAG when you need to enforce strict formatting outputs (like JSON or code), adopt a specific persona, or train on a custom domain vocabulary, and choose it over continued pre-training when the goal is to align instructions rather than inject raw factual knowledge.
- **Key Interview Points**: Format control, behavioral alignment, style adaptation, domain-specific vocabularies.
- **Technical Intuition & Complexity**:
  - **Prompt Engineering**: $O(L)$ context computation at inference, bounded by model context size. High token cost.
  - **RAG**: $O(K \cdot L_c + L_q)$ where $K$ is retrieved documents, enabling live database integration but introducing retrieval latency.
  - **SFT**: Modifies weights permanently to capture target style patterns. Complexity during inference is $O(L_{\text{output}})$, eliminating large prompt overhead.
- **Production & Implementation Details**: Use RAG to inject dynamic, volatile data (e.g. daily news or stock updates). Use SFT to enforce API calling syntax or JSON output matching. Use Continued Pre-training when the base model lacks the vocabulary or linguistic syntax of the target field (e.g. medical terminology or Chinese text).
- **Follow-up Questions**:
  - *If a task requires both real-time information retrieval and strict format compliance, how would you design the hybrid system?*
  - *Under what dataset size does SFT begin to underperform compared to advanced prompt engineering?*
- **Common Mistakes**: Attempting to use SFT to inject fresh, rapidly changing database facts. This leads to hallucination drift because LLMs are not reliable databases.

---

### 2. Mitigating Catastrophic Forgetting
**What is catastrophic forgetting in LLMs, and what architectural or dataset strategies (e.g., replay buffers, elastic weight conservation) do you use to mitigate it?**

- **Quick Screen Response**: Catastrophic forgetting is the loss of pre-trained general skills when a model adapts to a specific task. We mitigate it by mixing general-purpose instruction datasets into the training split, regularizing parameters using Elastic Weight Consolidation (EWC), or utilizing parameter-efficient adapters (like LoRA) that freeze the base model parameters.
- **Key Interview Points**: Parameter drift, mixed fine-tuning, EWC, Fisher Information, frozen base weights.
- **Technical Intuition & Complexity**:
  - Mixed fine-tuning blends domain data with general conversational data (e.g. 10% ShareGPT), maintaining base representations.
  - **EWC (Elastic Weight Consolidation)** adds a quadratic penalty to the loss function based on the Fisher Information diagonal $\mathbf{F}$, which estimates parameter importance:
    $$\mathcal{L}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{i,\text{pre-train}})^2$$
    - *Complexity*: Time complexity is $O(N)$ to compute the diagonal of the Fisher matrix over parameter size $N$.
- **Production & Implementation Details**: In production pipelines, mixed fine-tuning is preferred over EWC because EWC requires calculating second-order derivatives over billions of parameters, which is computationally expensive.
- **Follow-up Questions**:
  - *How does the Fisher Information matrix calculate parameter importance for a downstream classification task?*
  - *Why does training a LoRA adapter prevent catastrophic forgetting in the base model weights?*
- **Common Mistakes**: Believing that lowering the learning rate alone prevents forgetting. An unregularized model trained long enough on a narrow task will always drift.

---

### 3. Instruction Dataset Formatting & Boundaries
**How is an instruction dataset formatted, and how do Chat Templates (e.g., Jinja2) enforce system, user, and assistant boundaries?**

- **Quick Screen Response**: Instruction datasets are formatted in structures like Alpaca (single-turn) or ShareGPT (multi-turn). Tokenizers use Jinja2 templates to convert these structured conversations into raw strings containing special role markers (like `<|im_start|>` and `<|im_end|>`) to enforce clear boundaries between system instructions, user queries, and assistant replies.
- **Key Interview Points**: Jinja2, ChatML, special tokens, roles, role hygiene.
- **Technical Intuition & Complexity**:
  - If a model is trained without explicit role markers, it cannot distinguish between user inputs and system boundaries, making it highly vulnerable to prompt injection attacks.
  - Jinja2 templates are rendered at tokenization time, mapping raw conversational lists to formatted token sequences.
- **Production & Implementation Details**: When training with ChatML markers, you must add these markers as **special tokens** in the tokenizer configuration so they are parsed as single atomic tokens rather than split into sub-words (e.g., treating `<|im_start|>` as ID 128001 rather than three separate sub-words).
- **Follow-up Questions**:
  - *What happens if a user submits ChatML formatting tokens in their query, and how do tokenizers escape them?*
  - *How does `apply_chat_template` differ between Llama-3-Instruct and standard Mistral tokenizers?*
- **Common Mistakes**: Repeating system prompts in every conversation turn during multi-turn training, which wastes VRAM and causes the model to ignore system constraints in long context windows.

---

### 4. Target-Only Loss Masking (`-100` Index)
**What is target-only loss masking in PyTorch (`CrossEntropyLoss(ignore_index=-100)`), and why is it problematic to compute loss over prompt tokens during Supervised Fine-Tuning (SFT)?**

- **Quick Screen Response**: Target-only loss masking replaces the target labels of prompt tokens with `-100`. During training, PyTorch's `CrossEntropyLoss` ignores indices set to `-100`, which ensures the optimizer only updates weights based on the model's response tokens rather than penalizing the model for failing to predict the user prompts.
- **Key Interview Points**: ignore_index=-100, target masking, gradient drift, cross-entropy ignore.
- **Technical Intuition & Complexity**:
  - If we calculate gradients over prompt tokens, the model's weights adapt to predict user queries, leading to model drift and degradation in style alignment.
  - **Mathematical Walkthrough**:
    For sequence: `Prompt (8 tokens) + Response (4 tokens)`
    $$\text{input\_ids} = [t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_{10}, t_{11}, t_{12}]$$
    $$\text{standard\_labels} = [t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_{10}, t_{11}, t_{12}, \text{PAD}]$$
    $$\text{sft\_masked\_labels} = [-100, -100, -100, -100, -100, -100, -100, -100, t_9, t_{10}, t_{11}, t_{12}]$$
- **Production & Implementation Details**: When writing custom PyTorch data collators, identify the tokenizer's prompt end token index, and write a mask replacing all target labels up to that index with `-100`.
- **Follow-up Questions**:
  - *How do you write a custom PyTorch collator that handles target-only loss masking for a multi-turn chat format?*
  - *Does target-only loss masking affect the calculation of activation memory during the forward pass?*
- **Common Mistakes**: Forgetting to mask role markers like `Assistant:` or `<|im_start|>assistant`. These header tokens must be masked out too, leaving only the assistant's generated output active.

---

### 5. Sequence Packing Efficiency
**What is Sequence Packing (e.g., using FlashAttention), and how does it eliminate wasted memory compute on padding tokens across variable-length sequences?**

- **Quick Screen Response**: Sequence packing concatenates multiple short sequences into a single training block (up to the max context size, e.g. 2048) to eliminate padding tokens. To prevent context cross-contamination, we use custom attention masking (e.g., block-diagonal masks in FlashAttention) so tokens only attend to other tokens in their original sequence.
- **Key Interview Points**: Padding elimination, block-diagonal attention mask, FlashAttention, dataset packing.
- **Technical Intuition & Complexity**:
  - Variable-length padding wastes compute, scaling quadratically with pad lengths. Packing achieves $100\%$ token compute efficiency.
  - *Complexity*: Reduces memory scaling from $O(\text{Batch} \cdot L_{\text{max}}^2)$ under standard padding to $O(L_{\text{packed}}^2)$ with zero-padding tokens.
- **Production & Implementation Details**: Use frameworks like trl (Transformer Reinforcement Learning) which provide `ConstantLengthDataset` to automatically package datasets before feeding them to SFTTrainer.
- **Follow-up Questions**:
  - *How does sequence packing modify the position embeddings injected into the self-attention layer?*
  - *How does FlashAttention's block-diagonal attention mask handle sequence boundaries inside packed blocks?*
- **Common Mistakes**: Packing sequences without passing an attention boundary mask, which allows the model to learn relationships across completely independent training samples.

---

## 2. PEFT Mechanics & Math

### 6. LoRA Formulation & Weight Initialization
**Explain Low-Rank Adaptation (LoRA) mathematically ($\Delta W = B \cdot A$). Why is matrix $A$ initialized with Gaussian noise while $B$ is initialized to zero?**

- **Quick Screen Response**: LoRA models weight updates using two low-rank matrices: $\Delta \mathbf{W} = \mathbf{B} \cdot \mathbf{A}$, where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$. Matrix $\mathbf{A}$ is initialized with Gaussian noise to project activations to the low-rank space, while $\mathbf{B}$ is initialized to zero, ensuring $\Delta \mathbf{W} = 0$ at the start of training so that the model behavior is unmodified before training.
- **Key Interview Points**: Low-rank approximation, rank $r$, $\mathbf{B}$ initialized to zero, scaling factor $\frac{\alpha}{r}$.
- **Technical Intuition & Complexity**:
  - Freezing $\mathbf{W}_0$ reduces trainable parameters by $\sim 99\%$.
  - Projection: $\mathbf{h} = \mathbf{W}_0\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$
  - *Complexity*: FLOPs for forward pass: $O(d \cdot k)$ base + $O(r \cdot (d + k))$ adapter.
- **Production & Implementation Details**: Initialize $\mathbf{A}$ with $\mathcal{N}(0, \sigma^2)$ where $\sigma^2 = \frac{1}{r}$, and set $\mathbf{B} = \mathbf{0}$.
- **Follow-up Questions**:
  - *Why would initializing both matrices $\mathbf{A}$ and $\mathbf{B}$ to Gaussian noise degrade early training stability?*
  - *What layers (e.g. query, key, value, projection) are most critical to apply LoRA matrices to?*
- **Common Mistakes**: Initializing $\mathbf{B}$ to a Gaussian distribution. This injects random noise into the pre-trained outputs at step 0, degrading model performance immediately.

---

### 7. Rank ($r$) and Scaling Factor ($\alpha$) Hyperparameters
**What do rank ($r$) and the scaling factor ($\alpha$) control in LoRA, and how do you adjust $\frac{\alpha}{r}$ when scaling model or task complexity?**

- **Quick Screen Response**: Rank $r$ defines the dimension of the low-rank bottleneck, controlling the model's capacity to learn new styles or parameters. The scaling factor $\alpha$ acts as a constant learning rate multiplier for the adapter updates, and we typically keep the ratio $\frac{\alpha}{r}$ constant when adjusting rank $r$ to avoid having to re-tune learning rates.
- **Key Interview Points**: Low-rank bottleneck, learning rate scaling, constant ratio $\frac{\alpha}{r}$.
- **Technical Intuition & Complexity**:
  - Higher $r$ increases capacity but increases VRAM requirements.
  - Scale updates by $\frac{\alpha}{r}$. If you double the rank $r$ to $32$ from $16$, double $\alpha$ to $64$ to maintain a constant update scale ratio of 2.0.
- **Production & Implementation Details**: Rules of thumb: Set $r=8$ or $r=16$ for style alignment, and $r=32$ or $r=64$ for deep domain adaptation. Set $\alpha = 2r$.
- **Follow-up Questions**:
  - *If the ratio $\frac{\alpha}{r}$ is set too high, how does it affect optimizer step convergence?*
  - *How does scaling rank $r$ affect the VRAM footprint of trainable optimizer states?*
- **Common Mistakes**: Treating $\alpha$ as a hyperparameter that must be search-optimized independently. Always bind it to $r$ (e.g., $\alpha = 2r$).

---

### 8. DoRA (Weight-Decomposed LoRA)
**What is DoRA, and how does decoupling weight magnitude $m$ from direction $V$ bring fine-tuning dynamics closer to full fine-tuning performance?**

- **Quick Screen Response**: DoRA decomposes the weight matrix $\mathbf{W}$ into magnitude vectors $\mathbf{m}$ and directional weight matrices $\mathbf{V}$. In full fine-tuning, both update components are modified. DoRA updates the directional component using a LoRA adapter while training the magnitude vector directly, which decouples these updates and improves learning stability.
- **Key Interview Points**: Weight decomposition, magnitude $\mathbf{m}$, direction $\mathbf{V}$, gradient stability.
- **Technical Intuition & Complexity**:
  - Weight formulation:
    $$\mathbf{W} = \mathbf{m} \frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{\|\mathbf{W}_0 + \mathbf{B}\mathbf{A}\|_c}$$
  - Decoupling these updates allows DoRA to match full fine-tuning performance across downstream tasks with no additional inference latency since the weights merge back into standard base matrices.
- **Production & Implementation Details**: DoRA requires $\sim 10\% - 20\%$ more training memory than LoRA due to the column-wise normalization backward pass, but maintains identical inference speeds post-merge.
- **Follow-up Questions**:
  - *How does the column-wise norm operation $\|\cdot\|_c$ calculate gradients during the backward pass?*
  - *Can a DoRA adapter be merged back into a quantized base model (e.g., INT4 or FP4)?*
- **Common Mistakes**: Assuming DoRA introduces extra computation during inference. Like LoRA, DoRA parameters are merged back into base weights for zero-latency deployment.

---

### 9. QLoRA: NF4, Double Quantization, and Paged Optimizers
**Explain QLoRA. How do 4-bit NormalFloat (NF4), Double Quantization, and Paged Optimizers work together to lower VRAM requirements during training?**

- **Quick Screen Response**: QLoRA quantizes the base model weights to 4-bit NormalFloat (NF4), which maps zero-centered Gaussian weights to optimal quantiles. It then uses Double Quantization to compress the quantization scales themselves from 32-bit to 8-bit, and utilizes Paged Optimizers to offload memory spikes to CPU RAM, enabling training of massive models on single-GPU hardware.
- **Key Interview Points**: NF4 quantiles, double quantization, paged optimizers, CUDA memory swapping.
- **Technical Intuition & Complexity**:
  - Quantization mappings are calculated based on normal distribution statistics:
    $$q_j = \text{argmin}_i |w_j - c_i|$$
  - Double Quantization saves $\sim 0.37\text{ GB}$ of VRAM per 7B parameters.
  - NF4 values are dequantized to FP16/BF16 on the fly during the forward pass before activation multiplication.
- **Production & Implementation Details**: Use `bitsandbytes` library configuration in PEFT:
  ```python
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  ```
- **Follow-up Questions**:
  - *How does dynamic dequantization from NF4 to BF16 affect training throughput and epoch training times?*
  - *Why does QLoRA use BF16 as the computation data type if base weights are stored in 4-bit NF4?*
- **Common Mistakes**: Expecting QLoRA to speed up training. Quantizing base weights reduces VRAM footprint, but dequantization operations add computational overhead, slightly slowing training down.

---

### 10. Trade-Off Matrix: Full SFT vs. LoRA vs. DoRA vs. QLoRA
**Walk through the compute, memory overhead, throughput, and accuracy trade-offs between Full SFT, LoRA, DoRA, and QLoRA.**

- **Quick Screen Response**: Full SFT achieves high accuracy but requires massive VRAM. LoRA significantly reduces VRAM at the cost of task capacity. DoRA matches full SFT accuracy by decoupling magnitude/direction updates with a minor training VRAM increase. QLoRA reduces VRAM to the absolute minimum, but introduces dequantization latency overhead.
- **Key Interview Points**: VRAM scaling, throughput overhead, trainable parameter count.
- **Technical Intuition & Complexity**:
  - No fenced code blocks around GFM tables:

| Dimension | Full SFT | LoRA | DoRA | QLoRA |
| :--- | :--- | :--- | :--- | :--- |
| **VRAM Footprint** | Extremely High ($16\Phi$ bytes) | Low ($4\Phi$ + $14\Phi_{\text{trainable}}$) | Low ($4.1\Phi$ + $14\Phi_{\text{trainable}}$) | Lowest ($0.5\Phi$ + $14\Phi_{\text{trainable}}$) |
| **Accuracy** | Baseline | Low-Medium (on complex tasks)| High (matches full SFT) | High (within 1% of LoRA) |
| **Throughput** | High | High | Medium (norm backward steps) | Lowest (dequantization overhead) |
| **Inference Speed** | High | High (post-merge) | High (post-merge) | Low-Medium (if kept unmerged) |

- **Production & Implementation Details**: Choose QLoRA when constrained to a single 24GB GPU. Choose LoRA/DoRA when scaling high-throughput APIs on A100 clusters. Choose Full SFT only when adapting massive multi-modal weights from scratch.
- **Follow-up Questions**:
  - *How does optimizer state VRAM overhead scale when using 8-bit AdamW compared to FP32 AdamW?*
  - *Why does QLoRA require adapter matrices to remain in 16-bit precision?*
- **Common Mistakes**: Storing QLoRA models unmerged in production. Always merge adapter weights back to base parameters to avoid dequantization cost during serving.

---

## 3. Preference Optimization & Alignment

### 11. The Alignment Imperative & HHH Framework
**Why do pre-trained models require post-training alignment, and how is the Helpful, Honest, Harmless (HHH) objective encoded into evaluation and reward datasets?**

- **Quick Screen Response**: Pre-trained base models complete documents rather than answering instructions, and SFT models frequently hallucinate or refuse queries. Alignment uses preference datasets to teach the model target boundaries: helpfulness (following instructions), honesty (calibrating confidence levels to reduce hallucinations), and harmlessness (resisting safety attacks).
- **Key Interview Points**: Helpful Honest Harmless, refusal boundaries, post-training alignment, preference bias.
- **Technical Intuition & Complexity**:
  - Alignment transforms a probability distribution over vocabulary tokens into a utility utility-maximizing policy.
  - Evaluation datasets contain pairwise prompt responses where human annotators rank safety and compliance.
- **Production & Implementation Details**: Balance HHH weights to prevent the "Refusal Cascade" where safety alignment causes the model to refuse safe requests containing sensitive keywords.
- **Follow-up Questions**:
  - *How do you build a robust taxonomy to balance helpfulness and safety constraints in a medical chatbot?*
  - *What alignment metrics quantify if a model has become overly passive or evasive post-RLHF?*
- **Common Mistakes**: Assuming that alignment increases the model's absolute knowledge. Alignment only constrains the base model's pre-trained representations.

---

### 12. Classic RLHF Pipeline Mechanics
**Explain the full RLHF pipeline: how is the Reward Model trained using the Bradley-Terry preference loss, and how does PPO penalize policy drift using KL divergence?**

- **Quick Screen Response**: The RLHF pipeline first trains a Reward Model (RM) using the Bradley-Terry preference loss, predicting preferred responses over dispreferred ones. We then align the SFT generator using PPO, adding a KL-divergence penalty between the active policy and a frozen reference SFT model to prevent the policy from generating gibberish that exploits the reward system.
- **Key Interview Points**: Bradley-Terry loss, actor-critic, reference policy, KL-divergence regularization.
- **Technical Intuition & Complexity**:
  - **Bradley-Terry Reward Loss**:
    $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(\mathbf{r}_{\psi}(x, y_w) - \mathbf{r}_{\psi}(x, y_l)) \right]$$
  - **Regularized Reward Objective**:
    $$R(x, y) = \mathbf{r}_{\psi}(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_{\theta}(y|x) \parallel \pi_{\text{ref}}(y|x))$$
    - *Complexity*: Requires loading 4 copies of the model in VRAM (generator, reference, reward, critic), scaling memory to $\sim 4 \times N$.
- **Production & Implementation Details**: Compute KL divergence at token level:
  $$\text{KL\_token}_t = \log \pi_{\theta}(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t})$$
- **Follow-up Questions**:
  - *How does standard policy gradient update parameters differently than the reward-prediction PPO baseline?*
  - *What optimization steps prevent the KL divergence values from exploding during early PPO steps?*
- **Common Mistakes**: Setting KL coefficient $\beta$ to zero, which allows the model to immediately drift into unreadable text patterns that exploit reward model shortcuts.

---

### 13. Direct Preference Optimization (DPO)
**Derive the intuition behind DPO. How does it re-parameterize the reward function to optimize preference loss directly without training an explicit Reward Model?**

- **Quick Screen Response**: DPO proves that the optimal policy $\pi^*$ can express reward implicitly using the log likelihood ratio of the active policy $\pi_\theta$ and reference SFT model $\pi_{\text{ref}}$. By substituting this representation into the Bradley-Terry preference objective, DPO minimizes a binary cross-entropy loss over log likelihoods directly, completely bypassing the RL actor-critic training loop.
- **Key Interview Points**: Implicit reward model, log-likelihood ratios, single-stage alignment, no critic model.
- **Technical Intuition & Complexity**:
  - **DPO Loss Formula**:
    $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
    - *Complexity*: Reduces memory scaling to $2\times$ model parameters (active + reference).
- **Production & Implementation Details**:
  For prompt $x$, compute forward passes over $y_w$ and $y_l$ under both the active and reference models, collecting token log likelihoods. Calculate DPO loss using the difference between log likelihood ratios.
- **Follow-up Questions**:
  - *How would you derive the mathematical mapping showing how the KL-regularized policy objective leads to the DPO reward representation?*
  - *Why does DPO tend to overfit preference datasets more rapidly than standard RLHF via PPO?*
- **Common Mistakes**: Deleting the reference model weights to save memory. DPO requires the reference model to calculate the baseline likelihood ratios; running without it leads to alignment collapse.

---

### 14. ORPO vs. GRPO
**What are ORPO and GRPO? How does Group Relative Policy Optimization (GRPO) calculate normalized advantages across a sampled group of outputs to eliminate the Critic model in reasoning tasks (e.g., DeepSeek-R1 style alignment)?**

- **Quick Screen Response**: ORPO combines SFT and preference alignment into a single loss function using odds ratios, removing the reference model. GRPO eliminates the PPO critic model during reinforcement learning by sampling a group of outputs for a prompt, and standardizing their rewards within the group to calculate token advantages, significantly saving GPU memory.
- **Key Interview Points**: Odds ratio, group relative advantage, no critic model, DeepSeek-R1 architecture.
- **Technical Intuition & Complexity**:
  - **GRPO Advantage Formula**:
    $$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$
    - *Complexity*: GRPO reduces training memory footprint from PPO's $\sim 4\times$ to $\sim 2\times$ model parameters by eliminating the Value/Critic model, which matches standard SFT+reference models but executes an RL pipeline.
- **Production & Implementation Details**: GRPO is highly popular for training reasoning models (e.g., DeepSeek-R1). We sample $G = 8$ outputs, calculate rewards using compiler execution or answer syntax parsers, and calculate the policy loss weighted by these standardized advantages.
- **Follow-up Questions**:
  - *How does the mathematical formulation of GRPO's group advantages prevent reward hacking in multi-agent reinforcement learning?*
  - *What are the loss function components of ORPO, and how does the odds ratio penalty prevent representation collapse?*
- **Common Mistakes**: Setting GRPO group size $G$ too low (e.g. $G=2$). A small group size provides inaccurate reward averages, leading to unstable advantage gradients.

---

### 15. Neural Reward Models vs. Verifiable Rule-Based Rewards (RLVR)
**Compare Neural Reward Models (RLHF) with Verifiable Rule-Based Rewards (RLVR) for deterministic tasks like math verification and code execution.**

- **Quick Screen Response**: Neural Reward Models evaluate subjective outputs like tone or safety, but are vulnerable to reward hacking. Verifiable Rule-Based Rewards (RLVR) use deterministic environments (like compilers or math calculators) to return binary rewards ($1$ or $0$), ensuring the model cannot exploit the reward metric with conversational formatting or length bias.
- **Key Interview Points**: Reward hacking vulnerability, deterministic verification, verifiable outcomes, math/code validators.
- **Technical Intuition & Complexity**:
  - Neural RMs output scores that drift as the generator shifts parameters, leading to validation leakage.
  - RLVR uses standard Python test suites or math syntax checkers to calculate hard outcomes.
- **Production & Implementation Details**: For math tutors or coding systems, construct Dockerized sandboxes to execute code outputs, returning $+1$ for correct unit test runs and $-1$ for execution failures.
- **Follow-up Questions**:
  - *How do you design a robust sandbox environment to safely run LLM-generated code during RLVR training runs?*
  - *Can a hybrid system combining Neural RMs and RLVR be used to train agents on subjective formatting with hard logic criteria?*
- **Common Mistakes**: Relying on Neural Reward Models to grade programming outputs. Code syntax is deterministic; grading it with a model allows the generator to write convincing-looking comments that get high neural rewards but fail to compile.

---

### 16. Preventing Length Bias in DPO
**What is Length Bias in preference alignment, and what loss normalization strategies prevent DPO models from generating excessively long answers to exploit implicit reward scores?**

- **Quick Screen Response**: Length bias occurs when models discover that generating longer answers increases log-likelihood scores, which artificially inflates DPO implicit rewards. We prevent this by normalizing token log likelihoods by sequence length, using length-controlled preference data, or adding a length penalty coefficient directly to the reward function.
- **Key Interview Points**: Verbosities exploitation, token length normalization, reference ratio normalization.
- **Technical Intuition & Complexity**:
  - DPO loss uses log-likelihood sums. Adding tokens naturally increases the scalar sums, leading to a length bias.
  - **Length-Normalized Likelihood**:
    $$\tilde{\pi}(y|x) = \frac{1}{|y|} \log \pi_{\theta}(y|x)$$
- **Production & Implementation Details**: During DPO data prep, ensure dispreferred response splits are not systematically shorter than preferred responses. In code execution, use the length-normalized version of the DPO loss function to penalize verbosity.
- **Follow-up Questions**:
  - *How does length normalization affect the gradient update steps for shorter sequence samples?*
  - *What are the trade-offs of using LLM-as-a-Judge to evaluate models when length bias is active in the judge model itself?*
- **Common Mistakes**: Assuming that simply instructing the model to "be brief" in SFT prevents DPO length bias. The DPO gradient update will still exploit length patterns to minimize loss.

---

## 4. Production, Multi-LoRA & VRAM Math

### 17. End-to-End VRAM Estimation Formula
**Walk through the exact VRAM calculation formula for fine-tuning an 8B parameter model in FP16/BF16:**
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{gradients}} + \text{VRAM}_{\text{optimizer}} + \text{VRAM}_{\text{activations}}$$
**Calculate the baseline VRAM needed for AdamW SFT vs. LoRA vs. QLoRA.**

- **Quick Screen Response**: Full FP16 AdamW SFT requires $16N$ bytes for training states, which is $128\text{ GB}$ for an 8B model. LoRA reduces this by freezing the base model ($2N$) and only calculating gradient/optimizer states on the adapter parameters ($1\%$), lowering VRAM to $17.28\text{ GB}$. QLoRA quantizes base weights to 4-bit ($0.5N$), reducing VRAM to $5.28\text{ GB}$.
- **Key Interview Points**: Training VRAM breakdown, weights/gradients/optimizer allocation, 8B sizing.
- **Technical Intuition & Complexity**:
  - Parameters: $\Phi = 8 \times 10^9$.
  - State footprints (ignoring activations):
    - **Full SFT (FP16/BF16)**:
      $$\text{VRAM} = \underbrace{2\Phi}_{\text{Weights}} + \underbrace{2\Phi}_{\text{Gradients}} + \underbrace{12\Phi}_{\text{AdamW}} = 16\Phi\text{ bytes} \rightarrow 128\text{ GB}$$
    - **LoRA ($r=16$, $1\%$ trainable)**:
      $$\text{VRAM} = \underbrace{2\Phi}_{\text{Frozen Base}} + \underbrace{2\Phi_{\text{trainable}}}_{\text{Adapter Weights}} + \underbrace{2\Phi_{\text{trainable}}}_{\text{Adapter Gradients}} + \underbrace{12\Phi_{\text{trainable}}}_{\text{Adapter Optimizer}} = 16\text{ GB} + 0.16\text{ GB} + 0.16\text{ GB} + 0.96\text{ GB} = 17.28\text{ GB}$$
    - **QLoRA ($r=16$, 4-bit base, $1\%$ trainable)**:
      $$\text{VRAM} = \underbrace{0.5\Phi}_{\text{Frozen Base NF4}} + \underbrace{2\Phi_{\text{trainable}}}_{\text{Adapter Weights}} + \underbrace{2\Phi_{\text{trainable}}}_{\text{Adapter Gradients}} + \underbrace{12\Phi_{\text{trainable}}}_{\text{Adapter Optimizer}} = 4\text{ GB} + 0.16\text{ GB} + 0.16\text{ GB} + 0.96\text{ GB} = 5.28\text{ GB}$$
- **Production & Implementation Details**: In production, add activation memory and KV cache buffers. Activations typically add $10\text{ GB} - 30\text{ GB}$ depending on batch size $b$ and sequence length $L$.
- **Follow-up Questions**:
  - *How does using the 8-bit AdamW optimizer (from bitsandbytes) modify the optimizer VRAM calculation?*
  - *What is the activation memory footprint reduction when applying Activation Checkpointing?*
- **Common Mistakes**: Failing to allocate memory for gradients and optimizer states when budgeting VRAM, resulting in out-of-memory errors on step 1.

---

### 18. Dynamic Multi-LoRA Serving Infrastructure
**What is Multi-LoRA Serving, and how do engines like vLLM, SGLang, or LoRAX dynamically swap and batch adapter weights in GPU memory without causing CUDA memory fragmentation?**

- **Quick Screen Response**: Multi-LoRA serving hosts a single base model in VRAM and dynamically swops lightweight adapter files ($<100\text{ MB}$) for incoming user requests. It uses pre-allocated memory pools and custom Triton kernels to apply adapter-specific weights batch-wise during the forward pass, avoiding VRAM fragmentation.
- **Key Interview Points**: Dynamic adapter swapping, pre-allocated GPU cache, Triton scatter-gather kernels, LoRAX.
- **Technical Intuition & Complexity**:
  - Instead of loading dedicated model servers for 100 customers, we serve them from a single base model.
  - The request batch gathers active inputs:
    $$\mathbf{h}_i = \mathbf{W}_0\mathbf{x}_i + \mathbf{B}_{\text{tenant}(i)}\mathbf{A}_{\text{tenant}(i)}\mathbf{x}_i$$
  - *Complexity*: Adding adapters adds $O(L \cdot r)$ operations, keeping latency overhead $< 5\%$.
- **Production & Implementation Details**: Pre-allocate an "adapter cache pool" in VRAM. If a request calls for an adapter not present in the GPU cache, it is loaded asynchronously from CPU RAM using PCIe channels.
- **Follow-up Questions**:
  - *How does the Punica kernel optimize batch calculations for multiple adapters of different ranks?*
  - *What metrics determine if a dynamic adapter should be promoted to a dedicated base model instance?*
- **Common Mistakes**: Hard-reloading adapters by reinstantiating the PyTorch model class for every user request, which causes severe latency spikes and CUDA memory errors.

---

### 19. Automated Offline & Online Evaluation Pipelines
**How do you structure an end-to-end evaluation suite before deploying a fine-tuned model (e.g., combining LLM-as-a-Judge, MT-Bench, and deterministic assertion tests)?**

- **Quick Screen Response**: An end-to-end evaluation suite combines deterministic unit tests (to assert schema parsing and output validations), general benchmarks like MT-Bench (to monitor reasoning decay), and LLM-as-a-Judge (using GPT-4o to grade structured outputs and relative win rates against target ground truths).
- **Key Interview Points**: Deterministic assertions, MT-Bench, LLM-as-a-Judge, validation metrics.
- **Technical Intuition & Complexity**:
  - **Deterministic assertions**: Verify JSON keys exist, regex matching, or compiler compatibility.
  - **LLM-as-a-Judge**: Pairwise evaluation comparing candidate outputs to SFT targets, using prompts that randomize order to prevent model position bias.
- **Production & Implementation Details**: Use metrics like **win rate** and **agreement rate** (how often the judge matches human annotators). Run automated offline checks on GitHub Actions before merging new model checkpoints.
- **Follow-up Questions**:
  - *How do you mitigate position bias and verbosity bias when using GPT-4o as a validation judge?*
  - *What downstream metrics measure user satisfaction drift after deploying a model updates to production?*
- **Common Mistakes**: Relying exclusively on general benchmarks (e.g. MMLU). Domain-specific models can score poorly on general trivia but perform exceptionally well on corporate tasks, or vice versa.

---

### 20. Production Incident Post-Mortems & Loss Instability
**What causes loss spikes, gradient explosions, or NaN outputs during fine-tuning, and how do you debug and recover training runs using learning rate warmup, gradient clipping, or precision tuning?**

- **Quick Screen Response**: Loss spikes and NaN outputs are caused by high learning rates, data corruption, or precision overflow in FP16. We debug and stabilize runs by applying gradient clipping (threshold $1.0$), implementing a linear learning rate warmup (first $10\%$ of steps), switching to BF16 precision, and isolating corrupt inputs.
- **Key Interview Points**: FP16 overflow, gradient clipping, learning rate warmup, BF16 dynamic range.
- **Technical Intuition & Complexity**:
  - **FP16 Underflow/Overflow**: FP16 uses a 5-bit exponent and 10-bit mantissa, limiting its range. During softmax calculation, large logits overflow to infinity, producing NaN outputs.
  - **BF16**: Uses an 8-bit exponent (matching FP32), completely eliminating precision overflow issues.
- **Production & Implementation Details**: If a training run crashes:
  1. Inspect the loss log at the crash step.
  2. Implement gradient clipping: `max_grad_norm=1.0`.
  3. Verify data cleaning: remove empty target responses or sequences exceeding context limits.
  4. Ensure learning rate warmup is active (e.g., standard cosine scheduler with warmup).
- **Follow-up Questions**:
  - *How does deep attention weight scaling prevent softmax saturation in deep networks?*
  - *Why do NaN outputs in the forward pass immediately corrupt all downstream parameter updates in AdamW?*
- **Common Mistakes**: Simply resuming training from the last checkpoint without lowering the learning rate or modifying gradient clipping. The model will typically crash again at the exact same dataset step.

---

## 5. High-Frequency System & Scenario Design

### 21. Low-Data Fine-Tuning Strategy (500 Examples)
**You only have 500 high-quality domain examples. Walk me through your complete strategy (data augmentation, SFT vs. Few-Shot RAG, rank/learning rate selection).**

- **Quick Screen Response**: With only 500 samples, SFT is highly prone to overfitting. I would first implement data augmentation using a larger model to generate synthetic variations, evaluate a Few-Shot RAG baseline, and if fine-tuning is required, train a low-rank adapter ($r=8$ or $r=16$) with high regularization and early stopping to prevent memorization.
- **Key Interview Points**: Few-shot RAG baseline, synthetic data augmentation, low-rank bottleneck, high regularization.
- **Technical Intuition & Complexity**:
  - Fine-tuning on 500 samples changes behavior but fails to inject new facts.
  - SFT parameters: Set $r=8$, $\alpha=16$, and implement a high weight decay (e.g., $0.1$) to regularize updates.
- **Production & Implementation Details**: Use a LLM-based augmentation pipeline (e.g., using GPT-4o to rephrase the 500 instructions into 5,000 diverse pairs). Validate SFT checkpoints every 10 steps to stop before overfitting starts.
- **Follow-up Questions**:
  - *How would you measure the semantic diversity of your synthetic dataset variations?*
  - *At what data scale does transitioning from Few-Shot prompting to SFT become economically viable?*
- **Common Mistakes**: Fine-tuning with a high rank ($r=64$) on a tiny dataset, which causes the model to memorize the training data and lose general generalization skills.

---

### 22. Multi-Tenant Enterprise Personalization at Scale
**An enterprise customer needs custom model behavior for 500 distinct business tenants on a restricted budget. Design the complete training, storage, and serving architecture.**

- **Quick Screen Response**: I would design a Multi-LoRA architecture. I would freeze a single base model and train 500 lightweight LoRA adapters ($r=8$, $<50\text{ MB}$ each) for the individual tenants. During serving, I would use vLLM or LoRAX to dynamically swap and batch tenant adapters in memory, avoiding dedicated GPU costs per tenant.
- **Key Interview Points**: Multi-LoRA serving, vLLM adapter batching, S3 storage cache, tenant routing.
- **Technical Intuition & Complexity**:
  - Dedicated serving for 500 tenants: $500 \times \text{Dedicated GPUs} = \text{cost prohibitive}$.
  - Multi-LoRA: $1\times$ GPU node running base model + dynamically cached adapters in VRAM.
  - *Complexity*: Inference memory overhead is limited to base model size + minor adapter cache buffers.
- **Production & Implementation Details**: Store adapter weights in an S3 bucket. Keep active adapters in a local GPU cache pool. Implement tenant-id headers in the API gateway to route requests and trigger dynamic loading.
- **Follow-up Questions**:
  - *How would you configure the cache evictions policy for tenant adapters during low-traffic periods?*
  - *What is the maximum throughput decrease when serving 20 distinct tenant adapters concurrently in a single batch?*
- **Common Mistakes**: Merging all 500 adapters back into base models and running 500 active endpoints. This architecture is financially unfeasible for most startups.

---

### 23. Mitigating Post-SFT Model Degradation
**Your fine-tuned model's loss decreases as expected, but its general reasoning and factual accuracy crater on downstream benchmarks. How do you diagnose and fix this issue?**

- **Quick Screen Response**: This is caused by catastrophic forgetting or data contamination. I would diagnose this by evaluating validation loss trends and checking for duplicate inputs. To fix it, I would inject 10% to 20% general-purpose conversational data into the training split, implement EWC regularization, or use a low-rank adapter.
- **Key Interview Points**: Catastrophic forgetting, dataset contamination, general replay data, weight regularization.
- **Technical Intuition & Complexity**:
  - Downstream accuracy degradation indicates the model's parameters have drifted away from the general pre-trained minima.
  - Solution: Re-train using a lower learning rate, decrease rank $r$, and blend general SFT datasets (like UltraChat or ShareGPT) into the training split.
- **Production & Implementation Details**: Add general benchmarks (e.g. MMLU or GSM8k) to your CI/CD pipeline, halting training runs if general performance drops below a set threshold.
- **Follow-up Questions**:
  - *How do you evaluate if model degradation is caused by learning rate decay scheduling issues?*
  - *What is the impact of mixing general instructions on training throughput and VRAM requirements?*
- **Common Mistakes**: Assuming that a decreasing training loss means the model is training successfully. A model can easily overfit and lose its general reasoning skills while its training loss drops.

---

### 24. Debugging an Underperforming LoRA Adapter
**Your LoRA fine-tuned model performs worse on the target domain than the base pretrained model. What step-by-step diagnostic checklist do you run through?**

- **Quick Screen Response**: I would check:
  1. *Tokenization*: Ensure special chat markers are parsed correctly.
  2. *Loss Masking*: Verify prompt tokens are masked with `-100`.
  3. *Rank & Alpha*: Check if rank $r$ is too small to capture task complexity.
  4. *Target Layers*: Confirm adapters are applied to both attention and projection layers.
  5. *Learning Rate*: Ensure warmup is active.
- **Key Interview Points**: Tokenizer mismatch, SFT loss masking verification, target module configuration, rank capacity.
- **Technical Intuition & Complexity**:
  - Mismatches in special tokens cause model generations to drift.
  - If adapters are only applied to attention projection layers ($W_q, W_v$), the model's learning capacity is highly limited. Applying to MLP/FFN blocks improves representation learning.
- **Production & Implementation Details**: Print tokenized outputs to verify chat templates match base model templates. Verify PEFT target layers configuration:
  ```python
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  ```
- **Follow-up Questions**:
  - *How does applying LoRA adapters to MLP blocks compare to applying them only to attention blocks in terms of parameter growth and final validation loss?*
  - *How would you identify if the SFT training dataset contains formatting noise that conflicts with pre-trained parameters?*
- **Common Mistakes**: Applying LoRA adapters only to query and value projection matrices ($W_q, W_v$). Modern models require adapter layers across FFN/MLP blocks to capture complex domain distributions.

---

### 25. Production Readiness Checklist
**What criteria, guardrails, latency budgets, and A/B test metrics must be satisfied before promoting a fine-tuned LLM from staging to production?**

- **Quick Screen Response**: The model must pass:
  1. *Deterministic assertions* (schema compliance).
  2. *Safety evaluation* (HHH compliance).
  3. *Latency budget checks* (Time-to-First-Token and tokens per second).
  4. *Benchmark validation* (no reasoning decay).
  Once validated, deploy via a canary routing split (e.g. 5%) to monitor business metrics and task completion rates.
- **Key Interview Points**: Latency budget, HHH compliance, canary deployment, Time-to-First-Token (TTFT), tokens per second.
- **Technical Intuition & Complexity**:
  - **TTFT (Time-to-First-Token)**: Scales with prompt size, verifying prompt parsing speed.
  - **Tokens-per-second**: Scales with batch size and output size, verifying model generation throughput.
- **Production & Implementation Details**: Latency target: TTFT $< 200\text{ ms}$, generation speed $> 30\text{ tokens/sec}$. Configure guardrails (like LlamaGuard) to scan inputs and outputs for toxic content before serving.
- **Follow-up Questions**:
  - *How do you write automated canary tests that compare response distribution drift between staging and production models?*
  - *What business metrics (e.g., API usage, session length) best capture model degradation post-deployment?*
- **Common Mistakes**: Deploying a model immediately to $100\%$ of users without a canary canary split, leading to service outages if memory caching configurations fail under traffic spikes.
