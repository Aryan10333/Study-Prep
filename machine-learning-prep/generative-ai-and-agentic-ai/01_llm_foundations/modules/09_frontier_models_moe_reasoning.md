# Module 09: Architectural Dissection of Frontier Models, MoE, & Deep Thinking Models

## 1. Introduction & Intuition

### The Core Bottleneck
As dense LLMs scale up, the compute cost to train and run them grows. For a dense 100B model, every single token generation requires running the entire 100B parameter network. This creates a major bottleneck: **Inference Compute Footprint**. 

To scale capacity without scaling compute, researchers designed **Mixture of Experts (MoE)**. Instead of using a single large FFN layer, MoE replaces the FFN with multiple parallel **experts** (smaller FFN blocks). A router network selects the top-k experts for each token. Only the parameters of the selected experts are activated. However, MoE creates a memory bottleneck: all experts must be kept in VRAM, which requires a large memory footprint.

Recently, a new bottleneck emerged: **System Limits on Direct Output Generation**. Standard models generate responses in a single forward-pass sweep, which limits their performance on complex mathematical and logical reasoning tasks. To solve this, **Deep Thinking/Reasoning Models** (e.g. OpenAI o1/o3, DeepSeek-R1) shift compute from training to **inference** (Test-Time Compute). By generating long intermediate chains of thoughts, verifying steps, and searching reasoning trees, they resolve complex tasks at the cost of higher generation latency.

### High-Level Intuition
*   **Frontier Configuration Trends**: Modern models (Llama 3, Mistral) have standard configurations: Pre-LN RMSNorm, RoPE positional embeddings, SwiGLU activation functions, Grouped-Query Attention (GQA), and large vocabulary sizes.
*   **Mixture of Experts (MoE)**: Think of a general contractor routing tasks to specialists. Instead of one generalist FFN doing all the work, the router sends math tokens to the mathematician expert, code tokens to the programmer expert, and translation tokens to the linguist expert. The output is a weighted sum of their work.
*   **Deep Thinking Models & Test-Time Compute (TTC)**: Think of solving a hard riddle. Instead of blurting out the first answer that comes to mind (standard LLMs), you pause, write out a list of sub-problems, verify each step, backtrack if you hit a contradiction, and only state the final solution once you are confident. This is implemented via Reinforcement Learning over search trees.

---

### MoE Router Routing & Reasoning Search Trees
Below is an inline SVG demonstrating MoE routing and Reasoning Search paths:

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 800 280" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <!-- MOE ROUTING DIAGRAM -->
  <g transform="translate(10, 20)">
    <rect x="0" y="0" width="370" height="240" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="185" y="25" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Mixture of Experts (Top-2 Routing)</text>
    
    <!-- Input Token -->
    <rect x="30" y="100" width="70" height="35" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5" />
    <text x="65" y="122" text-anchor="middle" font-size="10" font-weight="bold" fill="#334155">Token x</text>
    
    <path d="M 100 117 L 130 117" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-moe)" />
    
    <!-- Router -->
    <rect x="130" y="80" width="70" height="75" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="165" y="115" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">Gating</text>
    <text x="165" y="130" text-anchor="middle" font-size="9" fill="#1d4ed8">Router</text>
    
    <!-- Router Paths -->
    <path d="M 200 100 L 250 65" stroke="#ef4444" stroke-width="1.5" label="0.55" marker-end="url(#arrow-moe)" />
    <path d="M 200 135 L 250 170" stroke="#10b981" stroke-width="1.5" label="0.45" marker-end="url(#arrow-moe)" />
    <text x="220" y="70" font-size="9" fill="#ef4444">w1=0.55</text>
    <text x="220" y="170" font-size="9" fill="#10b981">w2=0.45</text>
    
    <!-- Experts -->
    <rect x="250" y="40" width="90" height="35" rx="4" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" />
    <text x="295" y="62" text-anchor="middle" font-size="10" font-weight="bold" fill="#991b1b">Expert 1 (Active)</text>
    
    <rect x="250" y="100" width="90" height="35" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />
    <text x="295" y="122" text-anchor="middle" font-size="10" fill="#94a3b8">Expert 2 (Idle)</text>
    
    <rect x="250" y="160" width="90" height="35" rx="4" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" />
    <text x="295" y="182" text-anchor="middle" font-size="10" font-weight="bold" fill="#065f46">Expert 3 (Active)</text>
  </g>

  <!-- SEARCH TREE DIAGRAM -->
  <g transform="translate(420, 20)">
    <rect x="0" y="0" width="370" height="240" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="185" y="25" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Reasoning Model Search Tree (TTC)</text>
    
    <!-- Root -->
    <circle cx="185" cy="60" r="12" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="185" y="64" text-anchor="middle" font-size="9" font-weight="bold" fill="#1e3a8a">Start</text>
    
    <!-- Layer 1 -->
    <circle cx="105" cy="120" r="12" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" />
    <text x="105" y="124" text-anchor="middle" font-size="8" fill="#991b1b">Step 1A</text>
    
    <circle cx="265" cy="120" r="12" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" />
    <text x="265" y="124" text-anchor="middle" font-size="8" font-weight="bold" fill="#065f46">Step 1B</text>
    
    <!-- Layer 2 -->
    <circle cx="215" cy="190" r="12" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" />
    <text x="215" y="194" text-anchor="middle" font-size="8" fill="#991b1b">Step 2A</text>
    
    <circle cx="315" cy="190" r="12" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" />
    <text x="315" y="194" text-anchor="middle" font-size="8" font-weight="bold" fill="#065f46">Output</text>
    
    <!-- Paths -->
    <line x1="175" y1="68" x2="115" y2="112" stroke="#ef4444" stroke-width="1.5" />
    <text x="130" y="85" font-size="8" fill="#ef4444">PRM=-0.2</text>
    
    <line x1="195" y1="68" x2="255" y2="112" stroke="#10b981" stroke-width="2" />
    <text x="235" y="85" font-size="8" fill="#10b981">PRM=+0.9</text>
    
    <line x1="255" y1="128" x2="225" y2="182" stroke="#ef4444" stroke-width="1.5" />
    <text x="220" y="155" font-size="8" fill="#ef4444">PRM=-0.5</text>
    
    <!-- Backtrack Arrow -->
    <path d="M 215 178 Q 235 150 255 132" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3" marker-end="url(#arrow-moe)" />
    <text x="255" y="165" font-size="8" fill="#d97706" font-weight="bold">Backtrack</text>
    
    <line x1="275" y1="128" x2="305" y2="182" stroke="#10b981" stroke-width="2" />
    <text x="305" y="155" font-size="8" fill="#10b981">PRM=+0.95</text>
  </g>
  
  <!-- Marker -->
  <defs>
    <marker id="arrow-moe" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### Mathematical Formulations

#### 1. MoE Router Softmax
To select the top-$K$ experts for a token $x$:
$$G(x)_i = \text{Softmax}(\text{KeepTopK}(H(x), K))_i$$
$$\text{where } \text{KeepTopK}(v, K)_i = \begin{cases} v_i & \text{if } v_i \text{ is in top } K \text{ values} \\ -\infty & \text{otherwise} \end{cases}$$

*   **Purpose & High-level Intuition:** The router maps token representation $x$ to a score vector $H(x) = x \cdot W_{\text{gate}}$. To enforce sparsity, we zero out all scores except the top-$K$ experts by setting them to $-\infty$. The softmax converts the remaining scores into routing weights that sum to 1. This routes the token to a subset of experts, keeping compute constant while scaling parameter capacity.

---

### Hand Calculations: Top-2 Router Gating
Let's compute the routing weight vector for a token $x$ mapped to $N_{\text{experts}} = 4$ experts.
Assume the router raw outputs are:
$$H(x) = \begin{pmatrix} 0.5 & 1.2 & 0.1 & 1.0 \end{pmatrix}$$
Let the number of active experts be $K = 2$.

#### Step 1: Apply KeepTopK Filter
Find the top 2 values in $H(x)$:
*   Highest value: $1.2$ at index 1.
*   Second highest value: $1.0$ at index 3.
Zero out all other indices by setting them to $-\infty$:
$$\bar{H}(x) = \begin{pmatrix} -\infty & 1.2 & -\infty & 1.0 \end{pmatrix}$$

#### Step 2: Compute Softmax
$$\begin{aligned}
\text{Sum} &= e^{1.2} + e^{1.0} \\
&\approx 3.3201 + 2.7183 \\
&= 6.0384
\end{aligned}$$
*   **Gating weight for Expert 1**:
    $$\begin{aligned}
    G(x)_1 &= \frac{e^{1.2}}{6.0384} \\
    &= \frac{3.3201}{6.0384} \\
    &\approx 0.5498
    \end{aligned}$$
*   **Gating weight for Expert 3**:
    $$\begin{aligned}
    G(x)_3 &= \frac{e^{1.0}}{6.0384} \\
    &= \frac{2.7183}{6.0384} \\
    &\approx 0.4502
    \end{aligned}$$
*   **Gating weights for Expert 0 and 2**:
    $$G(x)_0 = 0.0, \quad G(x)_2 = 0.0$$

The final routing vector is:
$$G(x) = \begin{pmatrix} 0.0 & 0.5498 & 0.0 & 0.4502 \end{pmatrix}$$

*   **Conclusion:** Token $x$ is routed to Expert 1 (with 55% weight) and Expert 3 (with 45% weight). The parameters of Expert 0 and Expert 2 are not loaded or executed for this token.

---

### Tensor & Shape Tracking
*   **Gating projection matrix ($W_{\text{gate}}$)**: `[d_model, N_experts]`
*   **Expert input (token representation)**: `[B * L, d_model]`
*   **Routed Expert outputs (each)**: `[N_tokens_assigned, d_model]`
*   **Assembled FFN output**: `[B, L, d_model]`

---

## 3. Implementation & Reference Code

Below is a self-contained PyTorch simulation of an MoE routing block routing tokens to parallel expert networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ReLU FFN expert
        return self.w2(F.relu(self.w1(x)))

class SparseMoELayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router projection
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # Collection of experts
        self.experts = nn.ModuleList([Expert(d_model, d_ffn) for _ in range(num_experts)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, d_model]
        B, L, d = x.shape
        flat_x = x.view(-1, d) # [B * L, d_model]
        
        # 1. Compute router scores
        logits = self.router(flat_x) # [B * L, num_experts]
        
        # 2. Get top-k scores and indices
        scores, indices = torch.topk(logits, self.top_k, dim=-1) # [B * L, top_k]
        
        # 3. Softmax over the top-k experts
        weights = F.softmax(scores, dim=-1) # [B * L, top_k]
        
        # 4. Route and execute tokens block-wise
        output = torch.zeros_like(flat_x)
        
        # We loop through experts and gather tokens assigned to each
        for exp_idx in range(self.num_experts):
            # Mask identifying which tokens route to this expert
            token_mask = (indices == exp_idx)
            if not token_mask.any():
                continue
                
            # Get flat indices of assigned tokens
            token_indices, k_indices = torch.where(token_mask)
            
            # Extract inputs and execute expert
            inputs = flat_x[token_indices]
            expert_outputs = self.experts[exp_idx](inputs)
            
            # Weight outputs and accumulate
            weights_extracted = weights[token_indices, k_indices].unsqueeze(-1)
            output[token_indices] += weights_extracted * expert_outputs
            
        return output.view(B, L, d)

# Verification block
if __name__ == "__main__":
    B, L, d = 2, 8, 16
    x = torch.randn(B, L, d)
    
    moe = SparseMoELayer(d_model=d, d_ffn=32, num_experts=4, top_k=2)
    out = moe(x)
    
    print("Input shape:", x.shape)
    print("MoE Output shape:", out.shape)
    assert out.shape == x.shape, "Shape mismatch!"
    print("Sparse MoE Layer successfully executed and verified routing loops!")
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** The quadratic computational cost scaling of scaling dense networks. MoE scales parameter capacity while keeping activation FLOPs constant.
*   **Why Introduced over Legacy Approaches:** MoE enables models to learn specialized tasks (e.g. math, coding, languages) within separate parameters blocks, improving downstream accuracy without increasing inference compute requirements.
*   **Key Failure Modes & Limitations:** Routing collapses: tokens get routed to a subset of experts (e.g. only Expert 1 and 2), causing other experts to remain under-trained. This is mitigated using a **Load Balancing Loss** during training.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Linear projections scale as $O(2 \cdot B \cdot L \cdot d \cdot d_{\text{ffn}} \cdot K)$ where $K$ is active experts (much lower than full $N$-expert evaluations).
*   **Space/Memory Footprint:** Parameter footprint scales as $O(N_{\text{experts}} \cdot 2 \cdot d \cdot d_{\text{ffn}})$. All expert parameters must be loaded into VRAM, increasing memory requirements.
*   **Primary Bottleneck Type:** Memory-routing and inter-GPU communication latency (All-to-All network exchanges).
*   **Variable Legend:** $N_{\text{experts}}$ = Total Expert count, $K$ = Number of active routed experts.

### 3. Production & Scalability
*   **Deployment Considerations:** Serving MoE requires massive VRAM. Model sharding is typically required: experts are split across multiple GPUs using Expert Parallelism.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Detail the difference between Process-Supervised Reward Models (PRMs) and Outcome-Supervised Reward Models (ORMs).
        *   *A:* ORMs evaluate only the final output of a generation (assigning a reward based on whether the final answer is correct). PRMs evaluate every single intermediate step (thought block) of a reasoning trace. PRMs provide a dense feedback signal, which is critical for training deep thinking models because it rewards logical reasoning and penalizes correct answers arrived at via wrong steps.
    2.  *Q:* Explain what Test-Time Compute (TTC) scaling laws represent.
        *   *A:* Traditional scaling laws focus on training compute (increasing parameters and training tokens). TTC scaling laws focus on inference compute: they state that for complex reasoning tasks, the performance of a model scales predictably as we allocate more compute budget at inference time (e.g. by expanding search paths, generating more thoughts, and verifying reasoning chains).
