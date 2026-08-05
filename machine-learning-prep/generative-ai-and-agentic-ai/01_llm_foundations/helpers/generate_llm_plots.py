import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def setup_premium_style():
    """Sets standard premium chart styling conforming to repository visualization guides."""
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-",
        "xtick.color": "#475569",
        "ytick.color": "#475569"
    })

def generate_sinusoidal_encoding(output_dir):
    """Generates Sinusoidal Positional Encoding Heatmap (Module 02)."""
    pos = 100
    d = 128
    
    pe = np.zeros((pos, d))
    for p in range(pos):
        for i in range(d // 2):
            denom = np.power(10000.0, (2 * i) / d)
            pe[p, 2 * i] = np.sin(p / denom)
            pe[p, 2 * i + 1] = np.cos(p / denom)
            
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(pe, cmap="coolwarm", center=0, ax=ax, 
                cbar_kws={"label": "Embedding Value", "pad": 0.02})
    
    ax.set_xlabel("Embedding Dimension Channel (d)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Sequence Position (pos)", fontsize=10, fontweight="semibold")
    ax.set_title("Sinusoidal Positional Encoding Matrix Heatmap", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sinusoidal_positional_encoding.png"), dpi=300)
    plt.close()

def generate_chinchilla_scaling(output_dir):
    """Generates Chinchilla Optimal Scaling Contours (Module 08)."""
    # Create parameter and token grids (in log space)
    params = np.logspace(8, 11, 200) # 100M to 100B
    tokens = np.logspace(9, 13, 200)  # 1B to 10T
    
    P, T = np.meshgrid(params, tokens)
    # Compute in FLOPs: C = 6 * N * D
    C = 6 * P * T
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    # Plot contours of constant compute
    # Using log10(C) for smooth contour intervals
    contours = ax.contour(P, T, np.log10(C), levels=[20, 21, 22, 23, 24, 25], 
                          colors='#64748b', linewidths=1.0, linestyles='--')
    ax.clabel(contours, inline=True, fmt=lambda x: f"$10^{{ {int(x)} }}$ FLOPs", fontsize=8)
    
    # Highlight Chinchilla optimal line (D ~ 20 * N)
    optimal_tokens = 20 * params
    ax.plot(params, optimal_tokens, color="#2563eb", linewidth=2.5, label="Chinchilla Optimal ($D \\approx 20N$)")
    
    # Highlight Kaplan (under-training) region (D ~ 1.5 * N)
    kaplan_tokens = 1.5 * params
    ax.plot(params, kaplan_tokens, color="#dc2626", linewidth=1.5, linestyle=":", label="Kaplan et al. (Under-trained)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters ($N$)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Training Tokens ($D$)", fontsize=10, fontweight="semibold")
    ax.set_title("Compute Scaling Frontier: Kaplan vs. Chinchilla", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chinchilla_scaling.png"), dpi=300)
    plt.close()

def generate_attention_patterns(output_dir):
    """Generates Full Self-Attention vs. Sliding Window Attention masking (Module 05)."""
    L = 16
    window = 4
    
    # Full Causal Mask
    full_mask = np.tril(np.ones((L, L)))
    
    # Sliding Window Causal Mask
    sw_mask = np.zeros((L, L))
    for r in range(L):
        for c in range(L):
            if c <= r and (r - c) < window:
                sw_mask[r, c] = 1.0
                
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    # Custom color map for active vs masked
    cmap = sns.color_palette(["#f1f5f9", "#2563eb"], as_cmap=True)
    
    sns.heatmap(full_mask, cmap=cmap, cbar=False, linewidths=0.5, linecolor="#cbd5e1", ax=axes[0])
    axes[0].set_title("Standard Causal Self-Attention", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Key Position", fontsize=9)
    axes[0].set_ylabel("Query Position", fontsize=9)
    
    sns.heatmap(sw_mask, cmap=cmap, cbar=False, linewidths=0.5, linecolor="#cbd5e1", ax=axes[1])
    axes[1].set_title(f"Sliding Window Attention (w={window})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Key Position", fontsize=9)
    axes[1].set_ylabel("Query Position", fontsize=9)
    
    plt.suptitle("Attention Matrix Masking Grids (Blue = Active, Gray = Masked)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_patterns.png"), dpi=300)
    plt.close()

def generate_kv_cache_memory(output_dir):
    """Generates VRAM consumption scaling of KV cache for MHA, GQA, MQA (Module 05 & 06)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    # Parameters
    B = 4          # Batch size
    H = 32         # Query heads
    d_head = 128   # Head dimension
    n_layers = 32  # Number of layers
    bytes_per_param = 2 # fp16/bf16
    
    context_lengths = np.linspace(0, 32768, 500)
    
    # KV cache size formula: 2 * 2 * B * L * n_layers * n_heads_kv * d_head * bytes_per_param
    # Factors for n_heads_kv:
    # MHA: 32 heads
    # GQA: 8 heads (group size 4)
    # MQA: 1 head
    
    mha_size_gb = (4 * B * context_lengths * n_layers * 32 * d_head * bytes_per_param) / (1024**3)
    gqa_size_gb = (4 * B * context_lengths * n_layers * 8 * d_head * bytes_per_param) / (1024**3)
    mqa_size_gb = (4 * B * context_lengths * n_layers * 1 * d_head * bytes_per_param) / (1024**3)
    
    ax.plot(context_lengths, mha_size_gb, color="#dc2626", label="Multi-Head Attention (MHA) [32 KV Heads]", linewidth=2)
    ax.plot(context_lengths, gqa_size_gb, color="#2563eb", label="Grouped-Query Attention (GQA) [8 KV Heads]", linewidth=2)
    ax.plot(context_lengths, mqa_size_gb, color="#059669", label="Multi-Query Attention (MQA) [1 KV Head]", linewidth=2)
    
    ax.set_xlabel("Context Length (Sequence Length L)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("KV Cache VRAM Footprint (GB)", fontsize=10, fontweight="semibold")
    ax.set_title("KV Cache VRAM Consumption Scaling (Batch Size B=4, Layers=32)", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.set_xlim(0, 32768)
    ax.set_ylim(0, max(mha_size_gb) * 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kv_cache_memory.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    setup_premium_style()
    
    # Resolve absolute path for output plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating plots and saving to: {os.path.abspath(output_dir)}")
    generate_sinusoidal_encoding(output_dir)
    generate_chinchilla_scaling(output_dir)
    generate_attention_patterns(output_dir)
    generate_kv_cache_memory(output_dir)
    print("All plots generated successfully!")
