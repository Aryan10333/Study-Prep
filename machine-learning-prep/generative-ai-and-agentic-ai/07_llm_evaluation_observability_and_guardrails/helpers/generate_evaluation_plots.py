"""Generates the 4 plots referenced across Topic 07's Track 1 modules.

All 4 are computed directly from each module's own verified hand-calc data/formula
(real, not illustrative) -- no illustrative-only plot in this topic.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1 (Module 02): N-gram overlap vs. constructed/annotated correctness
# ---------------------------------------------------------------------------
def plot_02_ngram_overlap_vs_correctness():
    # Real data from Module 02's own Example 2 hand calc
    labels = ["wrong_fluent\n(BLEU-1=0.8571)", "correct_rephrased\n(BLEU-1=0.7143)"]
    bleu_scores = [6 / 7, 5 / 7]
    correctness = [0, 1]  # constructed/annotated: 0=stipulated wrong, 1=stipulated correct
    colors = ["#c96", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, bleu_scores, color=colors, width=0.5)
    for bar, c in zip(bars, correctness):
        label = "constructed-correct" if c == 1 else "constructed-WRONG"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                ha="center", fontsize=9)

    ax.set_ylabel("BLEU-1 Score (real, computed)")
    ax.set_title("N-gram Overlap vs. Constructed/Annotated Correctness", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "02_ngram_overlap_vs_correctness.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 (Module 04): Cohen's kappa vs. raw observed agreement
# ---------------------------------------------------------------------------
def plot_04_kappa_vs_raw_agreement():
    p_o = 0.75  # fixed at the module's own real observed agreement
    p_e_values = [x / 100 for x in range(0, 71, 2)]  # 0.00 to 0.70
    kappa_values = [(p_o - p_e) / (1 - p_e) for p_e in p_e_values]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(p_e_values, kappa_values, color="#4c78a8", linewidth=2)
    ax.axhline(p_o, color="#888", linestyle="--", linewidth=1, label=f"Raw agreement (p_o={p_o})")
    ax.scatter([0.60], [(p_o - 0.60) / (1 - 0.60)], color="#c96", zorder=5, s=70,
               label="Module's own example (p_e=0.60, κ=0.375)")

    ax.set_xlabel("Chance Agreement Rate (p_e)")
    ax.set_ylabel("Cohen's Kappa (κ)")
    ax.set_title(f"Cohen's Kappa vs. Chance Agreement, at Fixed Raw Agreement (p_o={p_o})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "04_kappa_vs_raw_agreement.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 (Module 06): Self-consistency agreement vs. constructed correctness
# ---------------------------------------------------------------------------
def plot_06_self_consistency_vs_correctness():
    # Real data from Module 06's own three constructed scenarios
    scenario_names = ["Scenario A\n(agreement=0.80)", "Scenario B\n(agreement=0.90)", "Scenario C\n(agreement=0.40)"]
    agreements = [0.80, 0.90, 0.40]
    correctness_labels = ["constructed-WRONG", "constructed-correct", "constructed-WRONG"]
    colors = ["#c96", "#2e7d32", "#c96"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(scenario_names, agreements, color=colors, width=0.5)
    for bar, label in zip(bars, correctness_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, label,
                ha="center", fontsize=9)

    ax.set_ylabel("Self-Consistency Agreement Rate (real, computed)")
    ax.set_title("Self-Consistency Agreement vs. Constructed/Annotated Correctness", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "06_self_consistency_vs_correctness.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 (Module 08): Guardrail latency, sequential vs. parallel
# ---------------------------------------------------------------------------
def plot_08_guardrail_latency():
    # Real, computed directly from the module's own latency formulas, extending the
    # 3-check worked example to a swept range of check counts using the same
    # illustrative per-check latency values, cycled.
    base_latencies = [40, 60, 25]
    n_checks_range = list(range(1, 8))

    sequential = []
    parallel = []
    for n in n_checks_range:
        latencies = [base_latencies[i % len(base_latencies)] for i in range(n)]
        sequential.append(sum(latencies))
        parallel.append(max(latencies))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(n_checks_range, sequential, marker="o", color="#c96", label="Sequential (sum)")
    ax.plot(n_checks_range, parallel, marker="o", color="#2e7d32", label="Parallel (max, under independence assumption)")
    ax.scatter([3], [125], color="#000", zorder=5, marker="x", s=80, label="Module's own 3-check example")
    ax.scatter([3], [60], color="#000", zorder=5, marker="x", s=80)

    ax.set_xlabel("Number of Guardrail Checks")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Guardrail Stack Latency: Sequential vs. Parallel (real, computed)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "08_guardrail_latency_sequential_vs_parallel.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_02_ngram_overlap_vs_correctness()
    plot_04_kappa_vs_raw_agreement()
    plot_06_self_consistency_vs_correctness()
    plot_08_guardrail_latency()
    print("\nAll 4 plots generated.")
