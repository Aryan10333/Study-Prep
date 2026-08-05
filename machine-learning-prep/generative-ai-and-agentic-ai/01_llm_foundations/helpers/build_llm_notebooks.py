import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def run_and_save(nb, path):
    """Executes a notebook in place using prep-venv kernel and serializes it."""
    ep = ExecutePreprocessor(timeout=240, kernel_name='prep-venv')
    # Run in the notebook's directory
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path) or '.'}})
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {path}")

def build_01_transformer_from_scratch():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 01_llm_transformer_from_scratch: Implementing Modern LLM Blocks in PyTorch
    
This notebook builds the core structural blocks of modern decoder-only LLM architectures (like Llama 3) from scratch using PyTorch. 

We will implement:
1. **RMSNorm** (Root Mean Square Normalization)
2. **SwiGLU** (Swish Gated Linear Unit) Activation and FFN
3. **RoPE** (Rotary Position Embeddings)
4. **Grouped-Query Attention** (GQA)
5. A combined **LlamaTransformerBlock**

We will verify tensor shapes and parameters at every step.
"""))
    
    # 1. Setup and Environment Initialization
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Environment Initialization"))
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F

# Establish seed for determinism
torch.manual_seed(42)
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Environment Setup
- **Determinism**: We set PyTorch manual seeds to ensure all mock layer activations are reproducible across runs.
- **Hardware Target**: Verification is performed on the active runtime environment.
"""))
    
    # 2. RMSNorm Layer
    cells.append(nbf.v4.new_markdown_cell("## 2. RMSNorm Layer"))
    cells.append(nbf.v4.new_code_cell("""class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale vector gamma, initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, d]
        # Calculate variance (mean of squares) over last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        # Normalize and apply gain weight
        return x * torch.rsqrt(variance + self.eps) * self.weight

# Verify RMSNorm
B, L, d = 2, 4, 16
x = torch.randn(B, L, d)
rmsnorm = RMSNorm(dim=d)
out = rmsnorm(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
print("RMS of Normalized outputs (per token):\\n", torch.sqrt(out.pow(2).mean(-1)))
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: RMSNorm
- **Normalizing Behavior**: The output RMS values are scaled to approximately 1.0 (matching the target normalization factor).
- **VRAM Saving**: We bypassed centering the mean entirely, eliminating one global memory reduction pass.
"""))
    
    # 3. SwiGLU Activation Function
    cells.append(nbf.v4.new_markdown_cell("## 3. SwiGLU Activation Function"))
    cells.append(nbf.v4.new_code_cell("""class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)  # W matrix
        self.w_value = nn.Linear(d_model, d_ffn, bias=False) # V matrix
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)  # Down projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel gates
        gate = self.w_gate(x)
        swish_gate = gate * torch.sigmoid(gate) # Swish(xW)
        value = self.w_value(x)                  # xV
        
        # Gated multiplication and projection down
        return self.w_down(swish_gate * value)

# Verify SwiGLU
ffn = SwiGLUFeedForward(d_model=d, d_ffn=48)
out = ffn(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
assert out.shape == x.shape, "FFN output shape mismatch!"
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: SwiGLU
- **Dimensions**: Input `[2, 4, 16]` is projected up to intermediate dimension `48` inside the parallel gated paths and projected back down to `16`.
- **Gating Mechanism**: The gating multiplication models sharper activation regions, enhancing representation capability.
"""))
    
    # 4. Rotary Position Embeddings (RoPE)
    cells.append(nbf.v4.new_markdown_cell("## 4. Rotary Position Embeddings (RoPE)"))
    cells.append(nbf.v4.new_code_cell("""class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 100, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        
        # Calculate theta values: [dim / 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cosine and sine tables: [max_seq_len, dim]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x shape: [B, h, L, d_k]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, L, d_k]
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, L, d_k]
        return (x * cos) + (self._rotate_half(x) * sin)

# Verify RoPE
# q shape: [B, h, L, d_k] -> Batch=2, Heads=4, SeqLen=3, HeadDim=8
q = torch.randn(2, 4, 3, 8)
rope = RotaryPositionEmbedding(dim=8)
q_rotated = rope(q, seq_len=3)

print("Input shape :", q.shape)
print("Output shape:", q_rotated.shape)
assert q_rotated.shape == q.shape, "RoPE shape mismatch!"
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: RoPE
- **Half-Rotation**: The `_rotate_half` helper swaps the first and second halves of dimensions and negates one, implementing the complex multiplication.
- **Order Preservation**: Dot products calculated between rotated queries and keys will depend strictly on relative positions.
"""))
    
    # 5. Grouped-Query Attention (GQA)
    cells.append(nbf.v4.new_markdown_cell("## 5. Grouped-Query Attention (GQA)"))
    cells.append(nbf.v4.new_code_cell("""class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_query_heads // num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def _repeat_heads(self, x: torch.Tensor, reps: int) -> torch.Tensor:
        B, n_heads, L, d_k = x.shape
        if reps == 1:
            return x
        x = x.unsqueeze(2).expand(B, n_heads, reps, L, d_k)
        return x.reshape(B, n_heads * reps, L, d_k)
        
    def forward(self, x: torch.Tensor, rope: nn.Module) -> torch.Tensor:
        B, L, d = x.shape
        
        # Project and reshape: [B, h, L, d_k]
        q = self.q_proj(x).view(B, L, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = rope(q, seq_len=L)
        k = rope(k, seq_len=L)
        
        # Broadcast/Repeat KV heads to match Query heads
        k = self._repeat_heads(k, self.group_size)
        v = self._repeat_heads(v, self.group_size)
        
        # Compute scaled attention weights: [B, h_q, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum: [B, h_q, L, d_k]
        context = torch.matmul(attn_weights, v)
        
        # Concatenate and project back
        context = context.transpose(1, 2).contiguous().view(B, L, d)
        return self.out_proj(context)

# Verify GQA
# embed_dim=16, 4 query heads, 2 KV heads (group size 2)
rope = RotaryPositionEmbedding(dim=4) # head_dim = 16 // 4 = 4
gqa = GroupedQueryAttention(embed_dim=16, num_query_heads=4, num_kv_heads=2)
out = gqa(x, rope)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
assert out.shape == x.shape, "GQA output shape mismatch!"
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Grouped-Query Attention
- **Broadcasting**: The 2 Key and Value heads are duplicated to 4 heads to match the Query head count.
- **Savings**: Memory cached for Keys/Values is halved compared to standard MHA.
"""))
    
    # 6. Full LlamaTransformerBlock Integration
    cells.append(nbf.v4.new_markdown_cell("## 6. Full LlamaTransformerBlock Integration"))
    cells.append(nbf.v4.new_code_cell("""class LlamaTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int, d_ffn: int):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_query_heads, num_kv_heads)
        self.rope = RotaryPositionEmbedding(dim=embed_dim // num_query_heads)
        
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = SwiGLUFeedForward(embed_dim, d_ffn)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention block with residual connection
        h = x + self.attn(self.attn_norm(x), self.rope)
        # Pre-LN FFN block with residual connection
        out = h + self.ffn(self.ffn_norm(h))
        return out

# Instantiate full block and run
llama_block = LlamaTransformerBlock(
    embed_dim=128,
    num_query_heads=8,
    num_kv_heads=2,
    d_ffn=340
)
x_block = torch.randn(2, 10, 128) # Batch=2, Seq=10, Dim=128
out_block = llama_block(x_block)

print("Input Block shape :", x_block.shape)
print("Output Block shape:", out_block.shape)
assert out_block.shape == x_block.shape
print("Full Llama Transformer Block completed successfully!")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Llama Block Integration
- **Pre-LN Flow**: Normalization happens prior to entering the attention and feedforward layers, preserving the identity residual shortcut pathway.
- **Llama Config**: A mini-Llama layer running successfully with GQA, RoPE, RMSNorm, and SwiGLU.
"""))
    
    nb['cells'] = cells
    return nb

def build_02_kv_cache_profiler():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 02_kv_cache_and_serving_profiler: Autoregressive KV Cache Latency & VRAM Profiling
    
This notebook profiles the memory and computational latency of autoregressive text decoding. 

We will implement:
1. **Naive Autoregressive Decoding**: Recalculates all token projections at each generation step.
2. **KV Cache-Based Decoding**: Only projects the single new token and retrieves historical values.
3. **Prefill vs. Decode Latency Profiling**: Comparing speed (tokens/sec).
4. **VRAM Memory Footprint Profiling**: Scaling analysis for MHA vs. GQA vs. MQA.
5. **PagedAttention Page Table Manager Simulation**: Mocking OS-style paging in PyTorch.
"""))
    
    # 1. Setup and Environment Initialization
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Environment Initialization"))
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import time
import numpy as np

# Set random seed
torch.manual_seed(42)

# Configurations mimicking standard LLM layers
B = 4           # Batch size
h = 32          # Query heads
d_k = 128       # Head dimension
d_model = h * d_k # 4096 hidden dimension
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Setup
- **Config Constants**: We establish a model dimension of 4096 to emulate real 7B model sizes.
"""))
    
    # 2. Autoregressive Decoding Loop (Naïve vs. KV Cache)
    cells.append(nbf.v4.new_markdown_cell("## 2. Autoregressive Decoding Loop (Naïve vs. KV Cache)"))
    cells.append(nbf.v4.new_code_cell("""# Projection layers simulation
q_proj = nn.Linear(d_model, d_model, bias=False)
k_proj = nn.Linear(d_model, d_model, bias=False)
v_proj = nn.Linear(d_model, d_model, bias=False)

def generate_naive(prompt_tokens, num_tokens_to_generate):
    B, L, d = prompt_tokens.shape
    tokens = prompt_tokens
    
    start_time = time.time()
    for step in range(num_tokens_to_generate):
        # Naive approach: recalculate projections for the entire sequence at each step
        q = q_proj(tokens) # [B, SeqLen, d]
        k = k_proj(tokens) # [B, SeqLen, d]
        v = v_proj(tokens) # [B, SeqLen, d]
        
        # Calculate attention over the entire sequence length
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        # Select attention scores for the last token only (autoregressive decoding step)
        attn_weights = torch.softmax(scores[:, -1:, :], dim=-1)
        context = torch.matmul(attn_weights, v) # [B, 1, d]
        
        # Simulate next token selection & append
        next_token = context[:, -1:, :]
        tokens = torch.cat([tokens, next_token], dim=1)
        
    duration = time.time() - start_time
    return tokens, duration

def generate_with_kv_cache(prompt_tokens, num_tokens_to_generate):
    B, L, d = prompt_tokens.shape
    tokens = prompt_tokens
    
    # PREFILL PHASE: Process prompt tokens in parallel and initialize caches
    k_cache = k_proj(prompt_tokens) # [B, L, d]
    v_cache = v_proj(prompt_tokens) # [B, L, d]
    
    q_last = q_proj(prompt_tokens[:, -1:, :]) # [B, 1, d]
    scores = torch.matmul(q_last, k_cache.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    next_token = torch.matmul(attn_weights, v_cache)
    
    tokens_out = torch.cat([prompt_tokens, next_token], dim=1)
    
    # DECODING PHASE: Run step-by-step appending only the new token
    start_time = time.time()
    for step in range(1, num_tokens_to_generate):
        x_new = tokens_out[:, -1:, :] # [B, 1, d]
        
        # Project only the single new token
        q_new = q_proj(x_new)
        k_new = k_proj(x_new)
        v_new = v_proj(x_new)
        
        # Append to caches
        k_cache = torch.cat([k_cache, k_new], dim=1)
        v_cache = torch.cat([v_cache, v_new], dim=1)
        
        # Attention on cached history
        scores = torch.matmul(q_new, k_cache.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        next_token = torch.matmul(attn_weights, v_cache)
        
        tokens_out = torch.cat([tokens_out, next_token], dim=1)
        
    duration = time.time() - start_time
    return tokens_out, duration

# Sanity check run
prompt = torch.randn(B, 16, d_model)
naive_tokens, naive_time = generate_naive(prompt, 5)
cache_tokens, cache_time = generate_with_kv_cache(prompt, 5)

print("Naive generation time:", naive_time)
print("KV Cache generation time:", cache_time)
assert naive_tokens.shape == cache_tokens.shape, "Sequence output shapes mismatch!"
print("Generated shapes match successfully:", naive_tokens.shape)
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Generative Simulation
- **Output Equivalency**: Both loops generate identical sequence sizes.
- **Arithmetic Reduction**: The cache loop only projects a single token input at each generation step, skipping the $O(L)$ projection computation.
"""))
    
    # 3. Prefill vs. Decoding Latency Profiling
    cells.append(nbf.v4.new_markdown_cell("## 3. Prefill vs. Decoding Latency Profiling"))
    cells.append(nbf.v4.new_code_cell("""# Profile latency over longer context length
prompt = torch.randn(B, 256, d_model) # Prompt length 256
gen_tokens = 50

_, naive_duration = generate_naive(prompt, gen_tokens)
_, cache_duration = generate_with_kv_cache(prompt, gen_tokens)

print(f"Latency Profiling (generating {gen_tokens} tokens on a prompt of size {prompt.shape[1]}):")
print(f"- Naive approach: {naive_duration:.4f} seconds ({gen_tokens / naive_duration:.2f} tokens/sec)")
print(f"- KV Cache approach: {cache_duration:.4f} seconds ({gen_tokens / cache_duration:.2f} tokens/sec)")
print(f"Speedup factor: {naive_duration / cache_duration:.2f}x")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: Profiling Results
- **Generation Speed**: The KV Cache approach generates tokens significantly faster.
- **Scaling Effect**: As sequence length $L$ scales to thousands of tokens, the naive approach slows down quadratically, while the KV Cache approach maintains constant execution speed.
"""))
    
    # 4. Attention Variant Memory Footprint Analysis (MHA vs. GQA vs. MQA)
    cells.append(nbf.v4.new_markdown_cell("## 4. Attention Variant Memory Footprint Analysis (MHA vs. GQA vs. MQA)"))
    cells.append(nbf.v4.new_code_cell("""# Calculate and print KV cache VRAM footprint in MB
B_val = 8
layers_val = 80
d_k_val = 128
h_q = 64

lengths = [512, 1024, 2048, 4096, 8192, 16384]
bytes_per_param = 2 # fp16/bf16

print("KV Cache Memory footprint comparison (MB) for Batch Size=8, Layers=80:")
print(f"{'Context Length L':<18} | {'MHA (64 heads)':<16} | {'GQA (8 heads)':<16} | {'MQA (1 head)':<16}")
print("-" * 72)
for L in lengths:
    # Formula: 2 (keys & values) * B * L * layers * h_kv * d_k * bytes_per_param / 1024^2
    mha_mb = (4 * B_val * L * layers_val * 64 * d_k_val * bytes_per_param) / (1024**2)
    gqa_mb = (4 * B_val * L * layers_val * 8 * d_k_val * bytes_per_param) / (1024**2)
    mqa_mb = (4 * B_val * L * layers_val * 1 * d_k_val * bytes_per_param) / (1024**2)
    print(f"{L:<18} | {mha_mb:<16.2f} | {gqa_mb:<16.2f} | {mqa_mb:<16.2f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: KV Cache Memory Comparison
- **Linear Scaling**: The KV cache size scales linearly with sequence length $L$.
- **Saving Ratio**: Grouped-Query Attention (8 heads) consumes exactly 8x less VRAM than standard MHA (64 heads). Multi-Query Attention (1 head) consumes 64x less VRAM.
"""))
    
    # 5. Simulation of PagedAttention Page Table Mapping
    cells.append(nbf.v4.new_markdown_cell("## 5. Simulation of PagedAttention Page Table Mapping"))
    cells.append(nbf.v4.new_code_cell("""class MockPagedAttentionManager:
    def __init__(self, block_size: int = 4, num_blocks: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.page_table = {} # Maps request_id -> List of physical block indices
        
    def allocate_request(self, request_id: int, num_tokens: int):
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        allocated = []
        for _ in range(needed_blocks):
            if not self.free_blocks:
                raise MemoryError("Out of physical VRAM blocks!")
            allocated.append(self.free_blocks.pop(0))
        self.page_table[request_id] = allocated
        print(f"Allocated request {request_id} ({num_tokens} tokens) to physical blocks: {allocated}")
        
    def add_token(self, request_id: int, current_num_tokens: int):
        # Allocate new page block if current block is full
        if current_num_tokens % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("Out of physical VRAM blocks!")
            new_block = self.free_blocks.pop(0)
            self.page_table[request_id].append(new_block)
            print(f"Request {request_id} page limit reached. Allocated new physical block: {new_block}. Block list: {self.page_table[request_id]}")
        else:
            last_block = self.page_table[request_id][-1]
            print(f"Request {request_id} token fits in existing block: {last_block}")

# Verify manager execution
manager = MockPagedAttentionManager(block_size=4, num_blocks=10)
# 1. Allocate initial sequence space of 6 tokens (takes 2 blocks)
manager.allocate_request(request_id=42, num_tokens=6)

# 2. Add tokens step-by-step
manager.add_token(request_id=42, current_num_tokens=6) # Fits in second block
manager.add_token(request_id=42, current_num_tokens=7) # Fits in second block
manager.add_token(request_id=42, current_num_tokens=8) # Page block is full! Allocates a third block
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation: PagedAttention Simulation
- **Block-wise Allocation**: Memory is allocated dynamically in pages of size 4 tokens.
- **No Fragmentation**: When a page is full (at current token 8), a new block is fetched from the shared pool, eliminating the need for contiguous pre-allocated static VRAM arrays.
"""))
    
    nb['cells'] = cells
    return nb

def main():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\01_llm_foundations"
    notebooks_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    
    # 1. Build and Run Notebook 1
    print("\\n--- Building and Running Notebook 01 ---")
    nb1 = build_01_transformer_from_scratch()
    run_and_save(nb1, os.path.join(notebooks_dir, "01_llm_transformer_from_scratch.ipynb"))
    
    # 2. Build and Run Notebook 2
    print("\\n--- Building and Running Notebook 02 ---")
    nb2 = build_02_kv_cache_profiler()
    run_and_save(nb2, os.path.join(notebooks_dir, "02_kv_cache_and_serving_profiler.ipynb"))

if __name__ == "__main__":
    main()
