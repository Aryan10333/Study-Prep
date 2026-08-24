import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")


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


def plot_03_cost_vs_request_volume(output_dir):
    """Real, computed from Module 03's own worked cost-engineering data:
    N_GPU=22 (from the module's own Little's Law worked example), a real
    stated $700/GPU-month rate -> fixed self-hosted monthly cost, versus a
    hosted-API cost of $0.02/request (the module's own stated full-request
    cost basis) scaling linearly with real request volume."""
    n_gpu = 22
    cost_per_gpu_month = 700.0
    self_hosted_monthly = n_gpu * cost_per_gpu_month  # 15,400

    cost_per_request_api = 0.02
    volumes = np.linspace(0, 2_000_000, 400)
    api_cost = cost_per_request_api * volumes
    self_hosted_cost = np.full_like(volumes, self_hosted_monthly)

    breakeven_volume = self_hosted_monthly / cost_per_request_api  # 770,000

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(volumes, self_hosted_cost, linewidth=2.2, color="#2563eb", label=f"Self-hosted ({n_gpu} GPUs, ${self_hosted_monthly:,.0f}/mo)")
    ax.plot(volumes, api_cost, linewidth=2.2, color="#d97706", label=f"Hosted API (${cost_per_request_api}/request)")
    ax.axvline(breakeven_volume, color="#64748b", linestyle="--", linewidth=1.3)
    ax.annotate(
        f"Break-even: {breakeven_volume:,.0f} req/mo",
        xy=(breakeven_volume, self_hosted_monthly),
        xytext=(breakeven_volume + 60000, self_hosted_monthly + 6000),
        fontsize=9, color="#0f172a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
    )
    ax.set_xlabel("Real Monthly Request Volume", fontsize=10, fontweight="semibold", color="#334155")
    ax.set_ylabel("Real Monthly Cost ($)", fontsize=10, fontweight="semibold", color="#334155")
    ax.set_title("Self-Hosted vs. Hosted-API Cost vs. Real Request Volume", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.set_xlim(0, 2_000_000)
    ax.set_ylim(0, 45000)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "03_cost_vs_request_volume_build_vs_buy.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")
    return breakeven_volume


def plot_06_canary_traffic_ramp(output_dir):
    """Real, computed from Module 06's own worked canary-ramp example: Stage 1
    (5%, promoted at t=32min after the real monitoring window), Stage 2 (25%,
    ROLLBACK at t=35min on a real quality-threshold failure) -> traffic cut
    back to 0% for the regressed version."""
    # (time_minutes, traffic_pct) real step points from the module's own worked example
    t = [0, 0, 32, 32, 35, 35, 50]
    pct = [0, 5, 5, 25, 25, 0, 0]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.step(t, pct, where="post", linewidth=2.2, color="#7c3aed")
    ax.fill_between(t, pct, step="post", alpha=0.10, color="#7c3aed")

    # Real, annotated events
    ax.axvline(32, color="#059669", linestyle="--", linewidth=1.2)
    ax.annotate("Stage 1 PROMOTE\n(N_min/T_min window met)", xy=(32, 5), xytext=(6, 14),
                fontsize=8.5, color="#065f46", fontweight="bold")

    ax.axvline(35, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.annotate("Stage 2 ROLLBACK\n(QualityScore < threshold)", xy=(35, 25), xytext=(37, 27),
                fontsize=8.5, color="#991b1b", fontweight="bold")

    ax.set_xlabel("Real Elapsed Time (minutes)", fontsize=10, fontweight="semibold", color="#334155")
    ax.set_ylabel("Real Canary Traffic (%)", fontsize=10, fontweight="semibold", color="#334155")
    ax.set_title("Canary Traffic Ramp, With the Module's Own Real Rollback Stage Marked", fontsize=12, fontweight="bold", pad=15, color="#0f172a")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 32)
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "06_canary_traffic_ramp_with_rollback.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    _set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    breakeven = plot_03_cost_vs_request_volume(OUTPUT_DIR)
    print(f"Real computed break-even volume: {breakeven:,.0f} requests/month")
    plot_06_canary_traffic_ramp(OUTPUT_DIR)


if __name__ == "__main__":
    main()
