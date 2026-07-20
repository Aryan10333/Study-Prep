# Top 20 Fine-Tuning & Model Alignment Interview Questions

This flashcard guide presents 20 high-frequency interview questions asked by top tech companies covering PEFT, LoRA, QLoRA, DPO, RLHF, Knowledge Distillation, and Unsloth optimizations.

---

### Q1: How does LoRA (Low-Rank Adaptation) reduce parameter training memory?
- **Answer:** LoRA freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and introduces low-rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$. Training only $A$ and $B$ reduces trainable parameters by $>99\%$.

---

### Q2: What is the main advantage of DPO over traditional RLHF with PPO?
- **Answer:** DPO mathematically reparameterizes the reward model to express policy loss directly as a closed-form function of chosen vs rejected log probabilities ($\mathcal{L}_{\text{DPO}}$). This eliminates the need to train a separate Reward Model and Value Network, resulting in 3x faster, stable training without RL convergence issues.

---

### Q3: Explain QLoRA (Quantized Low-Rank Adaptation).
- **Answer:** QLoRA quantizes the frozen base model weights into 4-bit NormalFloat (NF4), uses Double Quantization to compress quantization constants, and uses Paged Optimizers to manage memory spikes, allowing fine-tuning of 70B models on a single 48GB GPU.

---

### Q4: What is the rank $r$ and scaling factor $\alpha$ in LoRA?
- **Answer:** Rank $r$ defines the inner dimension of matrices $A$ and $B$. Scaling factor $\alpha$ scales the LoRA update $\Delta W = \frac{\alpha}{r} (B A) x$. Setting $\alpha = 2r$ is a standard practice to keep learning dynamics stable when changing $r$.

---

### Q5: How does Knowledge Distillation transfer reasoning capabilities from a 70B teacher to an 8B student model?
- **Answer:** Knowledge Distillation minimizes the KL divergence between the soft output token probability distribution of the Teacher model and the Student model ($T \ge 2.0$), allowing the student to learn subtle relative token likelihoods instead of hard 0/1 target labels.
