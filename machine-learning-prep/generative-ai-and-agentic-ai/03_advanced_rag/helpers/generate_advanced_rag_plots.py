import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def generate_cost_crossover(out_path):
    """Module 01: Long Context vs. RAG cost crossover, matching the module's hand calculation."""
    corpus_tokens = 50_000
    context_tokens = 2_000
    price_token = 2.50 / 1_000_000
    price_embed = 0.02 / 1_000_000
    embed_once = corpus_tokens * price_embed

    n_queries = list(range(0, 101))
    cost_lc = [n * corpus_tokens * price_token for n in n_queries]
    cost_rag = [embed_once + n * context_tokens * price_token for n in n_queries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_queries, cost_lc, label="Long Context (re-process full corpus/query)", color="#dc2626", linewidth=2)
    ax.plot(n_queries, cost_rag, label="RAG (one-time index + small context/query)", color="#059669", linewidth=2)
    ax.axvline(0.0083, color="#94a3b8", linestyle=":", linewidth=1.3, label="Break-even (~0.008 queries)")

    ax.set_xlabel("Number of Queries")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Long Context vs. RAG: Total Cost vs. Query Volume\n(50K-token corpus, 2K-token retrieved context)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.5, max(cost_lc) * 1.1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_embedding_dim_tradeoff(out_path):
    """Module 03: Embedding dimensionality vs. storage / query latency trade-off (illustrative)."""
    dims = [64, 128, 256, 384, 512, 768, 1024, 1536]
    n_vectors = 1_000_000
    storage_gb = [(d * 4 * n_vectors) / 1e9 for d in dims]
    # Illustrative latency model: roughly linear in dimensionality for a fixed corpus size
    latency_ms = [2.0 + 0.018 * d for d in dims]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2563eb"
    ax1.set_xlabel("Embedding Dimensionality (d)")
    ax1.set_ylabel("Index Storage (GB) @ 1M vectors, fp32", color=color1)
    ax1.plot(dims, storage_gb, marker="o", color=color1, linewidth=2, label="Storage (GB)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xlim(0, 1650)
    ax1.set_ylim(0, max(storage_gb) * 1.2)

    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("Illustrative Query Latency (ms)", color=color2)
    ax2.plot(dims, latency_ms, marker="s", color=color2, linewidth=2, linestyle="--", label="Latency (ms)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(latency_ms) * 1.2)

    ax1.set_title("Embedding Dimensionality vs. Storage and Query Latency\n(illustrative trade-off, not a specific product's benchmark)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_recall_vs_latency(out_path):
    """Module 04: Recall vs. latency curve across ANN configurations (illustrative efSearch/nprobe sweep)."""
    # Illustrative diminishing-returns curve: recall approaches 1.0 as latency (search breadth) increases
    ef_values = [10, 20, 50, 100, 200, 400, 800]
    recall = [1 - math.exp(-ef / 120) for ef in ef_values]
    latency_ms = [1.0 + 0.045 * ef for ef in ef_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(latency_ms, recall, marker="o", color="#7c3aed", linewidth=2)
    for ef, lat, rec in zip(ef_values, latency_ms, recall):
        ax.annotate(f"efSearch={ef}", (lat, rec), textcoords="offset points", xytext=(6, -8), fontsize=8, color="#5b21b6")

    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall vs. Latency Across ANN Search Breadth\n(illustrative HNSW efSearch sweep -- diminishing returns at high recall)")
    ax.set_xlim(-5, max(latency_ms) * 1.15)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_pq_compression_bar(out_path):
    """Module 04: PQ compression ratio bar chart, matching the module's hand calculation (d=768, m=96, k=256)."""
    labels = ["Raw fp32\n(768 dims x 4 bytes)", "PQ-Compressed\n(96 subvectors x 1 byte)"]
    values = [3072, 96]
    colors = ["#dc2626", "#059669"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 60, f"{val} bytes", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Bytes per Vector")
    ax.set_title("IVF-PQ Compression: 32x Storage Reduction\n(d=768, m=96 subvectors, k=256 centroids)")
    ax.set_ylim(0, 3072 * 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_retrieval_metrics_bar(out_path):
    """Module 09: Illustrative Recall@5 / MRR / NDCG@5 bar chart, matching the module's hand calculation."""
    labels = ["Recall@5", "MRR", "NDCG@5"]
    values = [2 / 3, 0.5, 0.693]
    colors = ["#2563eb", "#7c3aed", "#059669"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics for One Toy Query\n(relevant={D2,D5,D9}, retrieved=[D7,D2,D5,D3,D8])")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    generate_cost_crossover(os.path.join(PLOTS_DIR, "01_cost_crossover.png"))
    generate_embedding_dim_tradeoff(os.path.join(PLOTS_DIR, "03_embedding_dim_tradeoff.png"))
    generate_recall_vs_latency(os.path.join(PLOTS_DIR, "04_recall_vs_latency.png"))
    generate_pq_compression_bar(os.path.join(PLOTS_DIR, "04_pq_compression.png"))
    generate_retrieval_metrics_bar(os.path.join(PLOTS_DIR, "09_retrieval_metrics.png"))
