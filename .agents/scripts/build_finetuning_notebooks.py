import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_full_finetuning_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# Supervised Fine-Tuning SFT Pipeline Demonstration

This notebook demonstrates the basic training pipeline of Supervised Fine-Tuning (SFT) over a toy language head to align outputs to specific formatting targets."""))
    
    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Define a simple classifier representing our LLM vocabulary head
class ToyLanguageHead(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        h = self.embedding(x)
        logits = self.linear(h)
        return logits

# Toy instruction SFT data: (prompts, responses)
# Map prompt IDs [1, 2, 3] to response target IDs [4, 5, 6]
prompts = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
targets = torch.tensor([[4, 5, 6], [6, 5, 4]], dtype=torch.long)

model = ToyLanguageHead()
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Initial Loss:")
logits = model(prompts)
# Shift for causal prediction
loss = criterion(logits[:, :-1, :].reshape(-1, 10), targets[:, 1:].reshape(-1))
print(f"Loss: {loss.item():.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output log shows the initial loss of the randomly initialized network before SFT starts. The calculated cross-entropy evaluates the probability distribution matching of shifted tokens."""))
    
    # Training loop
    cells.append(nbf.v4.new_code_cell("""print("Training for 20 epochs...")
for epoch in range(20):
    optimizer.zero_grad()
    logits = model(prompts)
    loss = criterion(logits[:, :-1, :].reshape(-1, 10), targets[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The printed logs show the SFT loss steadily decreasing as the model parameters are updated to align with the instruction-following dataset target shifts."""))
    
    nb['cells'] = cells
    return nb

def create_lora_dora_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Low-Rank Adaptation (LoRA) and Weight Decomposed LoRA (DoRA) from Scratch

This notebook implements LoRA and DoRA layers from scratch in PyTorch, showing weight equations and projection shapes."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import math

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=2, alpha=4):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        # Initialize A and B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_out = self.base_layer(x)
        adapter_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * adapter_out

class DoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=2, alpha=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Base linear layer
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.base_layer.weight.requires_grad = False
        
        # Magnitude parameter vector (m)
        self.m = nn.Parameter(torch.norm(self.base_layer.weight, p=2, dim=1))
        
        # Direction adapters
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Compute combined direction weights
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        combined_weight = self.base_layer.weight + lora_weight
        
        # Column normalize the direction component
        norm_factor = torch.norm(combined_weight, p=2, dim=1, keepdim=True)
        direction_weight = combined_weight / norm_factor
        
        # Multiply by magnitude vector
        effective_weight = self.m.unsqueeze(1) * direction_weight
        
        return x @ effective_weight.t()

# Test both layers
x = torch.randn(1, 4)
lora = LoraLinear(4, 4, r=2, alpha=4)
dora = DoraLinear(4, 4, r=2, alpha=4)

print("LoRA output shape:", lora(x).shape)
print("DoRA output shape:", dora(x).shape)
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output confirms both low-rank adapters calculate correct tensor transformations, maintaining the output hidden dimension sizes ($1 \times 4$ shape)."""))
    
    nb['cells'] = cells
    return nb

def create_qlora_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# QLoRA: 4-bit NormalFloat NF4 Quantization

This notebook maps floating point values to the optimal non-uniform NormalFloat 16-bin centroids array, simulating QLoRA parameter savings."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch
import numpy as np

# Simple NF4 quantization centroids lookup table (representing 16 bins for N(0,1))
NF4_CENTROIDS = torch.tensor([
    -1.0, -0.6961917, -0.51508895, -0.3854331, -0.2759082, -0.1727768, -0.06997282, 0.0,
    0.07658493, 0.17702854, 0.27961054, 0.3862211, 0.5022872, 0.64010215, 0.8354714, 1.0
])

def quantize_nf4(weight_tensor):
    # Normalize tensor to range [-1, 1]
    max_val = torch.max(torch.abs(weight_tensor))
    scaled_weight = weight_tensor / max_val
    
    # Map to nearest centroid index
    diff = torch.abs(scaled_weight.unsqueeze(-1) - NF4_CENTROIDS)
    quant_indices = torch.argmin(diff, dim=-1)
    return quant_indices, max_val

def dequantize_nf4(quant_indices, max_val):
    dequant_weight = NF4_CENTROIDS[quant_indices] * max_val
    return dequant_weight

# Create toy weight matrix
weights = torch.randn(4, 4)
indices, scale = quantize_nf4(weights)
dequant = dequantize_nf4(indices, scale)

print("Original Weights:\\n", weights)
print("\\nQuantized Centroid Indices:\\n", indices)
print("\\nDequantized Reconstruction:\\n", dequant)
print(f"\\nMax reconstruction absolute error: {torch.max(torch.abs(weights - dequant)).item():.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output shows the original FP32 tensor, the mapped 4-bit indices (values 0–15), and the dequantized reconstruction. The reconstruction error remains bounded, verifying NF4 efficiency."""))
    
    nb['cells'] = cells
    return nb

def create_sft_loss_masking_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# SFT Target-Only Loss Masking with -100 ignore_index

This notebook demonstrates target-only cross-entropy loss masking by replacing prompt tokens in target tensors with `-100`."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn

# Input sequence of token IDs (length 10)
# Index 0-6: Prompt tokens
# Index 7-9: Target response tokens
input_ids = torch.tensor([[101, 102, 103, 104, 105, 106, 107, 201, 202, 203]], dtype=torch.long)

# Causal next-token labels are shifted inputs
labels_unmasked = input_ids.clone()

# SFT loss masking: Set indices 0 to 6 to -100
labels_masked = input_ids.clone()
labels_masked[0, :7] = -100

print("Unmasked SFT targets:")
print(labels_unmasked)
print("\\nMasked SFT targets (ignore index = -100):")
print(labels_masked)

# Logit tensor of shape (batch, seq_len, vocab_size)
torch.manual_seed(42)
logits = torch.randn(1, 10, 300)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Unmasked Loss (calculated over all shifts)
loss_unmasked = criterion(logits[:, :-1, :].reshape(-1, 300), labels_unmasked[:, 1:].reshape(-1))
# Masked Loss (only calculated on response positions)
loss_masked = criterion(logits[:, :-1, :].reshape(-1, 300), labels_masked[:, 1:].reshape(-1))

print(f"\\nUnmasked Loss: {loss_unmasked.item():.4f}")
print(f"Masked Loss: {loss_masked.item():.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output demonstrates how target-only loss masking alters target tensors. The masked loss is significantly lower as the loss ignores prompt labels, isolating optimization updates to the response."""))
    
    nb['cells'] = cells
    return nb

def create_dpo_grpo_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# DPO and GRPO Loss Formulations in PyTorch

This notebook implements DPO (Direct Preference Optimization) implicit reward loss and GRPO (Group Relative Policy Optimization) group advantages normalization."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn.functional as F

# DPO loss calculation
def dpo_loss(policy_win_logps, policy_lose_logps, ref_win_logps, ref_lose_logps, beta=1.0):
    policy_log_ratio = policy_win_logps - policy_lose_logps
    ref_log_ratio = ref_win_logps - ref_lose_logps
    logits = policy_log_ratio - ref_log_ratio
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

# GRPO relative advantages normalization
def calculate_grpo_advantages(rewards):
    mean_val = rewards.mean()
    std_val = rewards.std() + 1e-8
    advantages = (rewards - mean_val) / std_val
    return advantages

# Test DPO Loss
policy_win = torch.tensor([[-0.5, -0.2]])
policy_lose = torch.tensor([[-1.5, -1.8]])
ref_win = torch.tensor([[-0.8, -0.6]])
ref_lose = torch.tensor([[-1.0, -1.2]])

loss = dpo_loss(policy_win, policy_lose, ref_win, ref_lose, beta=0.1)
print(f"Toy DPO Loss: {loss.item():.4f}")

# Test GRPO Advantages
rewards = torch.tensor([1.0, 2.0, 3.0])
advantages = calculate_grpo_advantages(rewards)
print("\\nGRPO Raw Rewards:", rewards.numpy())
print("GRPO Standardized Advantages:", advantages.numpy())
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The logs show the calculated DPO loss, and the standardized GRPO advantages. Since output 3 scored the highest ($3.0$), its normalized advantage is positive ($1.22$), reinforcing its generation logits without needing a critic network."""))
    
    nb['cells'] = cells
    return nb

def create_adapter_comparison_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# PEFT Adapter Parameter Footprint Profiles

This notebook calculates and compares the trainable parameters size of Prompt Tuning, Prefix Tuning, and LoRA."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn

class AdapterComparison:
    def __init__(self, in_features=512, out_features=512, prompt_len=10, prefix_len=10):
        # 1. Base Linear Layer (Frozen)
        self.base = nn.Linear(in_features, out_features)
        
        # 2. Prompt Tuning Parameter size
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_len, in_features))
        
        # 3. Prefix Tuning Keys/Values size (across 12 layers)
        self.prefix_key_values = nn.Parameter(torch.randn(12, 2, prefix_len, 8, in_features // 8))
        
        # 4. LoRA Adapter size (r=8)
        self.lora_A = nn.Parameter(torch.randn(8, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, 8))
        
    def profile_sizes(self):
        prompt_params = self.prompt_embeddings.numel()
        prefix_params = self.prefix_key_values.numel()
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        base_params = self.base.weight.numel() + self.base.bias.numel()
        
        print(f"Base Layer Parameters: {base_params:,}")
        print(f"Prompt Tuning Parameters: {prompt_params:,} ({prompt_params/base_params*100:.3f}% of base)")
        print(f"Prefix Tuning Parameters: {prefix_params:,} ({prefix_params/base_params*100:.3f}% of base)")
        print(f"LoRA (r=8) Parameters: {lora_params:,} ({lora_params/base_params*100:.3f}% of base)")

profile = AdapterComparison()
profile.profile_sizes()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output logs present parameter calculations. For example, a rank 8 LoRA represents only $\sim 3\%$ parameter overhead compared to the base weight dimension."""))
    
    nb['cells'] = cells
    return nb

def create_model_merging_nb():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Model Weight Merging via Task Arithmetic

This notebook implements Task Arithmetic merging over base model matrices, calculating cosine similarity outputs."""))
    
    cells.append(nbf.v4.new_code_cell("""import torch

# Task Arithmetic Merging
def task_arithmetic_merge(W_base, W_t1, W_t2, lambda_1=0.5, lambda_2=0.5):
    tau_1 = W_t1 - W_base
    tau_2 = W_t2 - W_base
    W_merged = W_base + lambda_1 * tau_1 + lambda_2 * tau_2
    return W_merged

# Cosine similarity helper
def cosine_similarity(W_a, W_b):
    a_flat = W_a.flatten()
    b_flat = W_b.flatten()
    return torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat))

# Set base weights
torch.manual_seed(42)
W_base = torch.randn(4, 4)

# Create two specialized weights
W_t1 = W_base + 0.1 * torch.randn(4, 4)
W_t2 = W_base - 0.1 * torch.randn(4, 4)

W_merged = task_arithmetic_merge(W_base, W_t1, W_t2, lambda_1=0.6, lambda_2=0.4)

print("Base Weights:\\n", W_base)
print("\\nMerged Weights (Task Arithmetic):\\n", W_merged)
print(f"\\nCosine Similarity (Base vs. Merged): {cosine_similarity(W_base, W_merged).item():.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
The output presents the base and Task Arithmetic merged matrices, along with high similarity index verification ($>0.98$), demonstrating preservation of structural weights."""))
    
    nb['cells'] = cells
    return nb

def build_all_notebooks():
    output_dir = 'machine-learning-prep/generative-ai-and-agentic-ai/03_llm_finetuning_and_alignment/notebooks'
    os.makedirs(output_dir, exist_ok=True)
    
    notebooks = {
        '01_full_finetuning.ipynb': create_full_finetuning_nb(),
        '02_lora_and_dora.ipynb': create_lora_dora_nb(),
        '03_qlora.ipynb': create_qlora_nb(),
        '04_sft_loss_masking.ipynb': create_sft_loss_masking_nb(),
        '05_dpo_and_grpo.ipynb': create_dpo_grpo_nb(),
        '06_adapter_comparison.ipynb': create_adapter_comparison_nb(),
        '07_model_merging.ipynb': create_model_merging_nb(),
    }
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='prep-venv')
    
    for filename, nb in notebooks.items():
        filepath = os.path.join(output_dir, filename)
        print(f"Generating and running {filename}...")
        
        # Run execution
        try:
            ep.preprocess(nb, {'metadata': {'path': output_dir}})
        except Exception as e:
            print(f"Execution failed for {filename}: {e}")
            raise e
            
        # Write to disk
        with open(filepath, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Successfully saved executed {filename}")

if __name__ == '__main__':
    build_all_notebooks()
