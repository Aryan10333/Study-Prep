import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def set_style():
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })


def majority_vote_probability(p: float, k: int) -> float:
    """threshold = k//2 + 1 (STRICT majority, correct for both odd and even k) --
    NOT math.ceil(k/2), which would wrongly count an exact tie as "majority
    correct" for even k. See Module 02's text for the real bug this fixes."""
    threshold = k // 2 + 1
    total = 0.0
    for i in range(threshold, k + 1):
        total += math.comb(k, i) * (p ** i) * ((1 - p) ** (k - i))
    return total


def plot_02_self_consistency_accuracy_vs_k(output_dir: str) -> str:
    """REAL, computed curve directly from Module 02's own binomial formula at p=0.6.
    Restricted to ODD k only, matching real self-consistency practice -- even k
    is provably worse than k-1 under the correct strict-majority threshold
    (an extra sample can only create an unresolvable tie, never lower the bar)."""
    p = 0.6
    ks = [1, 3, 5, 7, 9]
    accuracies = [majority_vote_probability(p, k) for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(ks, accuracies, marker='o', markersize=6, linewidth=2.0, color='#3b82f6', label=f'Majority-vote accuracy (p={p})')
    ax.axhline(y=p, color='#94a3b8', linestyle='--', linewidth=1.3, label=f'Single-sample baseline (p={p})')
    ax.fill_between(ks, accuracies, alpha=0.08, color='#3b82f6')

    for k, acc in zip(ks, accuracies):
        ax.annotate(f'{acc:.3f}', xy=(k, acc), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8, color='#1e3a8a')

    ax.set_xlabel('k (number of independent samples, ODD only -- see caption)', fontsize=9.5, fontweight='semibold', color='#334155')
    ax.set_ylabel('P(majority correct)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Self-Consistency Majority-Vote Accuracy vs. k\n(REAL, computed from the module\'s own formula; odd k only)', fontsize=11, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xticks(ks)
    ax.set_ylim(0.5, 0.85)
    ax.legend(fontsize=8.5, loc='lower right')
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, '02_self_consistency_accuracy_vs_k.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_03_expected_retries_vs_validity_probability(output_dir: str) -> str:
    """REAL, computed curve directly from Module 03's own E[attempts] = 1/p formula."""
    ps = [0.1 * i for i in range(1, 11)]
    expected_attempts = [1.0 / p for p in ps]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(ps, expected_attempts, marker='o', markersize=6, linewidth=2.0, color='#7c3aed')
    ax.fill_between(ps, expected_attempts, alpha=0.08, color='#7c3aed')

    for p, e in zip(ps, expected_attempts):
        if p in (0.5, 0.85, 1.0):
            ax.annotate(f'p={p:.2f}\nE={e:.2f}', xy=(p, e), xytext=(6, 10), textcoords='offset points',
                        fontsize=8, color='#5b21b6')

    ax.set_xlabel('p (per-attempt schema-validity probability)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('E[attempts] = 1/p', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Expected Retry Attempts vs. Validity Probability\n(REAL, computed from the module\'s own formula)', fontsize=11.5, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 11)
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, '03_expected_retries_vs_validity_probability.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_04_conceptual_latency_overhead_vs_sequence_length(output_dir: str) -> str:
    """CONCEPTUAL / ILLUSTRATIVE ONLY -- no notebook in this topic measures real
    constrained-decoding latency overhead. Invented, illustrative curves showing
    only the qualitative shape (overhead grows with sequence length; CFG costs
    more than FSM), per Module 04's explicit framing."""
    seq_lengths = [16, 32, 64, 128, 256, 512]
    fsm_overhead_pct = [1.5, 1.8, 2.2, 2.6, 3.0, 3.4]  # invented, illustrative
    cfg_overhead_pct = [3.0, 4.2, 5.8, 7.9, 10.5, 13.6]  # invented, illustrative

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(seq_lengths, fsm_overhead_pct, marker='o', markersize=6, linewidth=2.0, color='#059669', label='FSM/regex constraint (cheaper)')
    ax.plot(seq_lengths, cfg_overhead_pct, marker='s', markersize=6, linewidth=2.0, color='#dc2626', label='CFG constraint (more expensive)')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('Illustrative Latency Overhead (%)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Conceptual Per-Token Latency Overhead vs. Sequence Length\n(ILLUSTRATIVE -- NOT a Measured Production Curve)', fontsize=11, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(s) for s in seq_lengths])
    ax.legend(fontsize=8.5, loc='upper left')
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, '04_conceptual_latency_overhead_vs_sequence_length.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_09_prompt_caching_savings_vs_hit_rate(output_dir: str) -> str:
    """REAL, computed curve directly from Module 09's own cost-model formula, at
    the module's own stated ILLUSTRATIVE pricing (not a specific provider's real rate)."""
    price_full = 2.50 / 1_000_000
    price_cached = 0.25 / 1_000_000
    price_out = 10.0 / 1_000_000
    t_cached, t_uncached, t_out = 1000, 200, 150

    cost_no_cache = (t_cached + t_uncached) * price_full + t_out * price_out
    cost_cached = t_cached * price_cached + t_uncached * price_full + t_out * price_out

    hit_rates = [0.1 * i for i in range(0, 11)]
    savings_pct = []
    for h in hit_rates:
        blended_cost = h * cost_cached + (1 - h) * cost_no_cache
        savings = (1 - blended_cost / cost_no_cache) * 100
        savings_pct.append(savings)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(hit_rates, savings_pct, marker='o', markersize=6, linewidth=2.0, color='#0284c7')
    ax.fill_between(hit_rates, savings_pct, alpha=0.08, color='#0284c7')

    ax.set_xlabel('Cache Hit Rate (fraction of calls hitting a warm cache)', fontsize=9.5, fontweight='semibold', color='#334155')
    ax.set_ylabel('Cost Savings vs. No Caching (%)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Prompt-Caching Cost Savings vs. Cache-Hit-Rate\n(REAL, computed from the module\'s own formula; illustrative pricing)', fontsize=10.5, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, max(savings_pct) * 1.15)
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, '09_prompt_caching_savings_vs_hit_rate.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    set_style()

    p1 = plot_02_self_consistency_accuracy_vs_k(PLOTS_DIR)
    print(f"Generated: {p1}")

    p2 = plot_03_expected_retries_vs_validity_probability(PLOTS_DIR)
    print(f"Generated: {p2}")

    p3 = plot_04_conceptual_latency_overhead_vs_sequence_length(PLOTS_DIR)
    print(f"Generated: {p3}")

    p4 = plot_09_prompt_caching_savings_vs_hit_rate(PLOTS_DIR)
    print(f"Generated: {p4}")


if __name__ == "__main__":
    main()
