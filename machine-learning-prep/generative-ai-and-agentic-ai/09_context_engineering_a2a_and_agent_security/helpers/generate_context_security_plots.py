"""Plots for topic 09 (context engineering, A2A & agent security).

Every number here comes from a module's own hand calculation or is an
explicitly illustrative table stated in that module. Run with the repo venv:
    .venv\\Scripts\\python.exe helpers\\generate_context_security_plots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

# Shared loop parameters (Module 01 / Module 03 hand calculations)
C_S, G = 5_000, 1_500          # static context tokens, history tokens per step
R, W, O = 0.1, 1.25, 5.0       # cache-read, cache-write, output price multipliers (illustrative)
K, S = 8, 2_000                # compact every K steps into an S-token summary


def _set_style():
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })


# ---------- cost models (mirrors Module 03 reference code) ----------
def cost_no_cache(T):
    return sum(C_S + G * t for t in range(1, T + 1))


def cost_keep_all(T):
    c = 0.0
    for t in range(1, T + 1):
        c += W * (C_S + G) if t == 1 else R * (C_S + G * (t - 1)) + W * G
    return c


def cost_compact(T, k=K, s=S):
    c, hist, prev = 0.0, 0, None
    for t in range(1, T + 1):
        hist += G
        window = C_S + hist
        c += W * window if prev is None else R * prev + W * (window - prev)
        prev = window
        if t % k == 0 and t < T:
            c += R * prev + O * s      # summarization call: cached read + output
            hist, prev = s, C_S        # summary replaces history; only C_S stays cached
    return c


def plot_01_cumulative_tokens(output_dir):
    steps = np.arange(1, 41)
    base = np.cumsum(C_S + G * steps)
    trimmed = np.cumsum(1_500 + G * steps)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(steps, base / 1e3, color="#dc2626", lw=2.2, label="Full tool schemas ($C_s$ = 5,000)")
    ax.plot(steps, trimmed / 1e3, color="#2563eb", lw=2.2, label="Tool index only ($C_s$ = 1,500)")
    ax.scatter([20], [415], color="#dc2626", zorder=5)
    ax.annotate("415k at step 20", xy=(20, 415), xytext=(8, 620),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=9, color="#0f172a")
    ax.scatter([20], [345], color="#2563eb", zorder=5)
    ax.annotate("345k at step 20", xy=(20, 345), xytext=(23, 180),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=9, color="#0f172a")
    ax.set_xlabel("Agent step $T$", fontsize=10, color="#334155")
    ax.set_ylabel("Cumulative input tokens (thousands)", fontsize=10, color="#334155")
    ax.set_title("Agent-loop input tokens grow quadratically (g = 1,500 tokens/step)",
                 fontsize=11, fontweight="bold", color="#0f172a", pad=12)
    ax.set_xlim(0, 44)
    ax.set_ylim(0, base[-1] / 1e3 * 1.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    path = os.path.join(output_dir, "01_cumulative_tokens_vs_step.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("saved", path)


def plot_02_context_rot(output_dir):
    lengths_k = np.array([4, 8, 16, 32, 64, 128])
    single = np.array([0.99, 0.99, 0.98, 0.98, 0.97, 0.95])
    multihop = np.array([0.91, 0.90, 0.89, 0.87, 0.79, 0.66])
    delta = 0.05
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    x = np.log2(lengths_k)
    ax.plot(x, single, "o-", color="#2563eb", lw=2.2, label="Single-needle lookup")
    ax.plot(x, multihop, "o-", color="#dc2626", lw=2.2, label="Multi-hop reasoning")
    ax.axhline(single[0] - delta, color="#2563eb", ls="--", lw=1, alpha=0.7)
    ax.axhline(multihop[0] - delta, color="#dc2626", ls="--", lw=1, alpha=0.7)
    ax.annotate("$L_{usable}$ = 128k", xy=(np.log2(128), 0.95), xytext=(np.log2(40), 1.02),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=9)
    ax.annotate("$L_{usable}$ = 32k", xy=(np.log2(32), 0.87), xytext=(np.log2(6), 0.76),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}k" for v in lengths_k])
    ax.set_xlabel("Context length (tokens)", fontsize=10, color="#334155")
    ax.set_ylabel("Task accuracy", fontsize=10, color="#334155")
    ax.set_title("Illustrative: usable context depends on the task ($\\delta$ = 0.05)",
                 fontsize=11, fontweight="bold", color="#0f172a", pad=12)
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    ax.set_ylim(0.6, 1.06)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    path = os.path.join(output_dir, "02_context_rot_usable_limit.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("saved", path)


def plot_03_cache_policies(output_dir):
    steps = np.arange(1, 61)
    nc = np.array([cost_no_cache(t) for t in steps])
    ka = np.array([cost_keep_all(t) for t in steps])
    cp = np.array([cost_compact(t) for t in steps])
    be = next(t for t in steps if cost_compact(t) < cost_keep_all(t))
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(steps, nc / 1e3, color="#94a3b8", lw=2, label="No caching")
    ax.plot(steps, ka / 1e3, color="#2563eb", lw=2.2, label="Keep full history, cached")
    ax.plot(steps, cp / 1e3, color="#059669", lw=2.2, label="Compact every 8 steps, cached")
    ax.axvline(be, color="#475569", ls="--", lw=1)
    ax.text(be + 1, 330, f"break-even: step {be}", fontsize=9, color="#0f172a")
    ax.set_xlabel("Agent step $T$", fontsize=10, color="#334155")
    ax.set_ylabel("Cumulative cost (thousand base-token units)", fontsize=10, color="#334155")
    ax.set_title("Caching vs compaction (illustrative multipliers r=0.1, w=1.25)",
                 fontsize=11, fontweight="bold", color="#0f172a", pad=12)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 1200)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    path = os.path.join(output_dir, "03_cache_policy_cost_vs_step.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("saved", path, "break-even", be)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _set_style()
    assert cost_no_cache(20) == 415_000
    assert round(cost_keep_all(20)) == 81_750
    assert round(cost_compact(20)) == 93_150
    plot_01_cumulative_tokens(OUTPUT_DIR)
    plot_02_context_rot(OUTPUT_DIR)
    plot_03_cache_policies(OUTPUT_DIR)
