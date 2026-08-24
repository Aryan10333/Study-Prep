"""Generates the 4 plots referenced across Topic 06's Track 1 modules.

Plots 1, 2, and 4 are computed directly from each module's own verified formula/reference
code (real, not illustrative). Plot 3 is explicitly labeled illustrative -- no real notebook
measurement exists yet at Track 1 time (Track 2's notebooks may add a real measured version).
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1 (Module 01): Roofline -- decode vs. prefill arithmetic intensity
# ---------------------------------------------------------------------------
def plot_01_roofline():
    peak_flops = 312e12
    peak_bandwidth = 2039e9
    ridge = peak_flops / peak_bandwidth  # 153.0 FLOPs/byte, matches Module 01's verified value

    i_values = [10 ** (x / 20) for x in range(0, 100)]  # 1 to ~1e5, log-spaced
    attainable = [min(peak_flops, i * peak_bandwidth) for i in i_values]

    i_decode, i_prefill, i_batched = 1.0, 512.0, 256.0
    perf_decode = min(peak_flops, i_decode * peak_bandwidth)
    perf_prefill = min(peak_flops, i_prefill * peak_bandwidth)
    perf_batched = min(peak_flops, i_batched * peak_bandwidth)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(i_values, attainable, color="#4c78a8", linewidth=2, label="Roofline (attainable FLOPs/s)")
    ax.axvline(ridge, color="#888", linestyle="--", linewidth=1, label=f"Ridge point (I={ridge:.1f})")

    ax.scatter([i_decode], [perf_decode], color="#c96", zorder=5, s=70, label=f"Decode (I={i_decode:.1f}, memory-bw-bound)")
    ax.scatter([i_prefill], [perf_prefill], color="#2e7d32", zorder=5, s=70, label=f"Prefill, 512 tok (I={i_prefill:.0f}, compute-bound)")
    ax.scatter([i_batched], [perf_batched], color="#8856a7", zorder=5, marker="^", s=70,
               label=f"256 concurrent decode (I={i_batched:.0f}, shifted compute-bound)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity I = FLOPs / Byte")
    ax.set_ylabel("Attainable Performance (FLOPs/s)")
    ax.set_title("Roofline Model: Decode vs. Prefill (real, computed from Module 01's formula)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "01_roofline_decode_vs_prefill.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 (Module 02): KV cache memory vs. sequence length, MHA vs. GQA
# ---------------------------------------------------------------------------
def plot_02_kv_cache_memory():
    GiB = 1024 ** 3

    def kv_cache_gb(n_layers, n_kv_heads, d_head, batch_size, seq_len, bytes_per_elem=2):
        return 2 * batch_size * seq_len * n_layers * n_kv_heads * d_head * bytes_per_elem / GiB

    batch_size = 32
    seq_lens = list(range(0, 8193, 256))
    seq_lens[0] = 1
    mha_mem = [kv_cache_gb(32, 32, 128, batch_size, L) for L in seq_lens]
    gqa_mem = [kv_cache_gb(32, 8, 128, batch_size, L) for L in seq_lens]
    weight_mem_gb = 7_000_000_000 * 2 / GiB

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(seq_lens, mha_mem, color="#c96", linewidth=2, label="MHA (32 KV heads)")
    ax.plot(seq_lens, gqa_mem, color="#2e7d32", linewidth=2, label="GQA (8 KV heads)")
    ax.axhline(weight_mem_gb, color="#4c78a8", linestyle="--", linewidth=1.5, label=f"Model weight memory ({weight_mem_gb:.2f} GB)")

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("KV Cache Memory (GB)")
    ax.set_title(f"KV Cache Memory vs. Sequence Length (B={batch_size}, real, computed)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "02_kv_cache_memory_vs_sequence_length.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 (Module 06): Illustrative throughput/latency vs. batch size
# ---------------------------------------------------------------------------
def plot_03_batching_illustrative():
    batch_sizes = list(range(1, 33))
    # Illustrative conceptual shapes only -- no real notebook measurement exists yet.
    static_throughput = [b / (1 + 0.15 * math.log(b + 1)) for b in batch_sizes]
    continuous_throughput = [b / (1 + 0.02 * math.log(b + 1)) for b in batch_sizes]
    static_latency = [10 + 0.6 * b for b in batch_sizes]
    continuous_latency = [10 + 0.25 * b for b in batch_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(batch_sizes, static_throughput, color="#c96", label="Static batching")
    axes[0].plot(batch_sizes, continuous_throughput, color="#2e7d32", label="Continuous batching")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Relative Throughput (illustrative)")
    axes[0].set_title("Throughput vs. Batch Size (illustrative)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch_sizes, static_latency, color="#c96", label="Static batching")
    axes[1].plot(batch_sizes, continuous_latency, color="#2e7d32", label="Continuous batching")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Relative Latency (illustrative)")
    axes[1].set_title("Latency vs. Batch Size (illustrative)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Illustrative Only -- No Real Notebook Measurement Yet at Track 1 Time", fontsize=10, color="#a55")
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "06_batching_throughput_latency_illustrative.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 (Module 07): Expected speedup vs. acceptance rate
# ---------------------------------------------------------------------------
def plot_04_speculative_speedup():
    def expected_accepted_tokens(alpha, k):
        if alpha >= 0.999999:
            return float(k + 1)
        return (1 - alpha ** (k + 1)) / (1 - alpha)

    def expected_speedup(alpha, k, draft_cost_ratio):
        accepted = expected_accepted_tokens(alpha, k)
        cost_round = k * draft_cost_ratio + 1.0
        return accepted / cost_round

    k, draft_cost_ratio = 4, 0.2
    alphas = [x / 200 for x in range(1, 200)]
    speedups = [expected_speedup(a, k, draft_cost_ratio) for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(alphas, speedups, color="#4c78a8", linewidth=2)
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1, label="Speedup = 1.0x (break-even)")
    ax.scatter([0.8], [expected_speedup(0.8, k, draft_cost_ratio)], color="#2e7d32", zorder=5, s=60, label="α=0.8 (module's Step 1-2 example, ≈1.87x)")
    ax.scatter([0.3], [expected_speedup(0.3, k, draft_cost_ratio)], color="#c96", zorder=5, s=60, label="α=0.3 (module's low-acceptance case, ≈0.79x)")

    ax.set_xlabel("Acceptance Rate α")
    ax.set_ylabel("Expected Speedup vs. Standard Autoregressive Decoding")
    ax.set_title(f"Expected Speedup vs. Acceptance Rate (k={k}, draft cost={draft_cost_ratio}x, real)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "07_speculative_decoding_speedup_vs_acceptance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_01_roofline()
    plot_02_kv_cache_memory()
    plot_03_batching_illustrative()
    plot_04_speculative_speedup()
    print("\nAll 4 plots generated.")
