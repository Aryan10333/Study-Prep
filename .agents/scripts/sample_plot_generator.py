import os
import numpy as np
import matplotlib
# Enforce headless Agg backend for execution in script environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_premium_plots(output_dir):
    """Generates standard premium charts conforming to repository visualization guides."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Set Premium Visual Styles
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-",
        "xtick.color": "#475569",
        "ytick.color": "#475569"
    })
    
    # --- Plot A: Dual Bar Chart (e.g. Metric comparison) ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    labels = ['Dense Retrieval\n(Vector Cosine)', 'Sparse Retrieval\n(BM25 Score)']
    latencies = [15.4, 3.2] # Mock latency measurements in ms
    
    bars = ax.bar(labels, latencies, color=['#3b82f6', '#0f172a'], width=0.4, edgecolor='#1e293b', linewidth=0.8)
    
    # Add metrics values text on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0f172a')
                    
    ax.set_ylabel('Inference Latency (ms)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Retrieval Pipeline Latency Bounds', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_ylim(0, 18.0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    bar_chart_path = os.path.join(output_dir, 'sample_latency_bar_chart.png')
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    print(f"Generated sample bar chart: {bar_chart_path}")

    # --- Plot B: Metric Scaling Line Plot ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    scaling_x = [1, 2, 4, 8] # Number of parallel execution workers
    throughput_y = [120, 230, 410, 680] # Inferences per second
    
    ax.plot(scaling_x, throughput_y, marker='o', markersize=6, linewidth=2.0, color='#8b5cf6', label='Throughput')
    ax.fill_between(scaling_x, throughput_y, alpha=0.08, color='#8b5cf6')
    
    for x, y in zip(scaling_x, throughput_y):
        ax.annotate(f'{y} inf/s', xy=(x, y), xytext=(8, -3), textcoords='offset points', fontsize=8, color='#334155')
        
    ax.set_xlabel('Parallel Worker Count', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_ylabel('Throughput (Inferences / Sec)', fontsize=10, fontweight='semibold', color='#334155')
    ax.set_title('Inference Scaling Characteristics', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xticks(scaling_x)
    ax.set_ylim(0, 750)
    sns.despine()
    plt.tight_layout()
    
    line_chart_path = os.path.join(output_dir, 'sample_scaling_line_plot.png')
    plt.savefig(line_chart_path, dpi=300)
    plt.close()
    print(f"Generated sample line plot: {line_chart_path}")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as temp_output:
        generate_premium_plots(temp_output)
