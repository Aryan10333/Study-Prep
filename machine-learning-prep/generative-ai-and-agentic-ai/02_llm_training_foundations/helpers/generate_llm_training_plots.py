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


def generate_memory_breakdown(output_dir):
    """Module 01: Stacked bar comparing DDP baseline vs. ZeRO stages memory per GPU for a 7B model."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    stages = ['DDP\n(no ZeRO)', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3']
    params_gb = np.array([2, 2, 2, 2 / 8])
    grads_gb = np.array([2, 2, 2 / 8, 2 / 8])
    optim_gb = np.array([12, 12 / 8, 12 / 8, 12 / 8])

    x = np.arange(len(stages))
    ax.bar(x, params_gb, label='Params (fp16, 2$\\Psi$ bytes)', color='#3b82f6')
    ax.bar(x, grads_gb, bottom=params_gb, label='Gradients (fp16, 2$\\Psi$ bytes)', color='#10b981')
    ax.bar(x, optim_gb, bottom=params_gb + grads_gb, label='Optimizer States (fp32, 12$\\Psi$ bytes)', color='#f59e0b')

    totals = params_gb + grads_gb + optim_gb
    for i, total in enumerate(totals):
        ax.text(i, total + 2, f'{total * 7:.1f} GB', ha='center', fontsize=10, fontweight='bold', color='#0f172a')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10.5)
    ax.set_ylabel('Memory per GPU (bytes per parameter, $\\Psi$)', fontsize=10.5, fontweight='semibold')
    ax.set_title('Per-GPU Memory vs. ZeRO Partitioning Stage (N=8 GPUs, 7B Model)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)
    ax.set_ylim(0, 20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_zero_memory_breakdown.png'), dpi=300)
    plt.close()


def generate_lora_vram_comparison(output_dir):
    """Module 03: Bar chart comparing VRAM footprint of Full FT vs. LoRA vs. QLoRA for a 7B model."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    methods = ['Full Fine-Tuning\n(bf16 + Adam)', 'LoRA\n(bf16 base + fp32 adapters)', 'QLoRA\n(NF4 base + fp32 adapters)']
    # Full FT: 16 bytes/param x 7B = 112GB. LoRA: base frozen bf16 (2 bytes/param, 14GB) + tiny trainable adapter optimizer state (~0.2GB for r=8 on attn+FFN). QLoRA: base 4-bit (0.5 bytes/param, 3.5GB) + same tiny adapter state.
    vram_gb = [112, 14.2, 3.7]
    colors = ['#ef4444', '#3b82f6', '#10b981']

    bars = ax.bar(methods, vram_gb, color=colors, width=0.55)
    for bar, val in zip(bars, vram_gb):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f'{val:.1f} GB', ha='center', fontsize=11, fontweight='bold', color='#0f172a')

    ax.set_ylabel('Training VRAM Footprint (GB)', fontsize=10.5, fontweight='semibold')
    ax.set_title('VRAM Footprint: Full Fine-Tuning vs. LoRA vs. QLoRA (7B Model)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.set_ylim(0, 130)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_lora_vram_comparison.png'), dpi=300)
    plt.close()


def generate_kl_reward_tradeoff(output_dir):
    """Module 04: Illustrative reward vs. KL-divergence trade-off curve during RLHF/PPO training."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    kl = np.linspace(0, 10, 200)
    reward = 10 * (1 - np.exp(-kl / 3)) - 0.15 * kl  # rises then trades off as policy drifts too far
    ax.plot(kl, reward, color='#3b82f6', linewidth=2.5)

    optimal_idx = np.argmax(reward)
    ax.scatter([kl[optimal_idx]], [reward[optimal_idx]], color='#dc2626', zorder=5, s=60)
    ax.annotate('KL penalty $\\beta$ tuned here', xy=(kl[optimal_idx], reward[optimal_idx]),
                xytext=(kl[optimal_idx] + 1.5, reward[optimal_idx] - 1.5),
                fontsize=10, color='#991b1b', fontweight='semibold',
                arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.3))

    ax.set_xlabel('KL Divergence from Reference Policy $D_{KL}(\\pi_\\theta \\| \\pi_{ref})$', fontsize=10.5, fontweight='semibold')
    ax.set_ylabel('Mean Reward Model Score (illustrative)', fontsize=10.5, fontweight='semibold')
    ax.set_title('Reward vs. KL-Divergence Trade-off During PPO Training (Illustrative)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.set_xlim(0, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_kl_reward_tradeoff.png'), dpi=300)
    plt.close()


def generate_lr_schedule(output_dir):
    """Module 07: Linear warmup + cosine decay learning rate schedule."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    total_steps = 1000
    warmup_steps = 100
    peak_lr = 3e-4
    steps = np.arange(0, total_steps)

    lr = np.where(
        steps < warmup_steps,
        peak_lr * (steps / warmup_steps),
        0.5 * peak_lr * (1 + np.cos(np.pi * (steps - warmup_steps) / (total_steps - warmup_steps)))
    )

    ax.plot(steps, lr, color='#3b82f6', linewidth=2.5)
    ax.axvline(warmup_steps, color='#94a3b8', linestyle='--', linewidth=1.2)
    ax.text(warmup_steps + 15, peak_lr * 1.03, 'Warmup ends\n(step 100)', fontsize=9, color='#475569')

    ax.set_xlabel('Training Step', fontsize=10.5, fontweight='semibold')
    ax.set_ylabel('Learning Rate', fontsize=10.5, fontweight='semibold')
    ax.set_title('Linear Warmup + Cosine Decay Learning Rate Schedule', fontsize=12, fontweight='bold', color='#0f172a')
    ax.set_ylim(0, peak_lr * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_lr_schedule.png'), dpi=300)
    plt.close()


def generate_reward_hacking(output_dir):
    """Module 08: Illustrative divergence between proxy reward score and true response quality."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    steps = np.linspace(0, 100, 200)
    proxy_reward = 2 + 6 * (1 - np.exp(-steps / 25))
    true_quality = 2 + 6 * (1 - np.exp(-steps / 25)) - np.maximum(0, (steps - 55) * 0.055)

    ax.plot(steps, proxy_reward, color='#3b82f6', linewidth=2.5, label='Proxy Reward Model Score')
    ax.plot(steps, true_quality, color='#dc2626', linewidth=2.5, linestyle='--', label='True Response Quality (human-judged)')
    ax.axvline(55, color='#94a3b8', linestyle=':', linewidth=1.2)
    ax.text(57, 1.5, 'Divergence begins\n(reward hacking onset)', fontsize=9, color='#991b1b')

    ax.set_xlabel('PPO Training Step (illustrative)', fontsize=10.5, fontweight='semibold')
    ax.set_ylabel('Score (illustrative scale)', fontsize=10.5, fontweight='semibold')
    ax.set_title('Reward Hacking: Proxy Reward vs. True Quality Divergence (Illustrative)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_reward_hacking.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(output_dir, exist_ok=True)
    setup_premium_style()

    generate_memory_breakdown(output_dir)
    generate_lora_vram_comparison(output_dir)
    generate_kl_reward_tradeoff(output_dir)
    generate_lr_schedule(output_dir)
    generate_reward_hacking(output_dir)
    print("All plots generated.")
