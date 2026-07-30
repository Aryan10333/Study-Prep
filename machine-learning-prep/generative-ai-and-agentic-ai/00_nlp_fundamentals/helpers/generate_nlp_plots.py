import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_plots():
    output_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals\plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set premium aesthetic configurations
    sns.set_theme(style="whitegrid", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#cbd5e1",
        "grid.color": "#f1f5f9",
        "grid.linestyle": "-"
    })
    
    # 1. Stemming vs Lemmatization Latency Comparison
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    methods = ['Porter Stemmer\n(Heuristic Slicing)', 'WordNet Lemmatizer\n(Grammatical Lookup)']
    latencies = [1.2, 4.8]  # mock latencies in milliseconds
    bars = ax.bar(methods, latencies, color=['#3b82f6', '#8b5cf6'], width=0.45, edgecolor='#1e293b', linewidth=0.8)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e293b')
                    
    ax.set_ylabel('Inference Latency (ms / 1000 tokens)', fontsize=10, fontweight='semibold')
    ax.set_title('Lexical Reduction Pipeline Latency', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 6.0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stem_vs_lemma_latency.png'), dpi=300)
    plt.close()
    
    # 2. Cosine Similarity Heatmap of TF-IDF vectors
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    doc_labels = ['Doc 1: Cat/Feline', 'Doc 2: Feline/Rug', 'Doc 3: Dog/Garden']
    similarity_matrix = np.array([
        [1.00, 0.34, 0.00],
        [0.34, 1.00, 0.05],
        [0.00, 0.05, 1.00]
    ])
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=doc_labels, yticklabels=doc_labels, cbar=True, 
                vmin=0, vmax=1, linewidths=1.0, linecolor='#e2e8f0',
                annot_kws={"size": 11, "weight": "bold", "color": "#0f172a"}, ax=ax)
    ax.set_xticklabels(doc_labels, rotation=15, ha='right')
    ax.set_yticklabels(doc_labels, rotation=0)
    ax.set_title('Sparse Document Cosine Similarity Matrix', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tfidf_similarity_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Perplexity vs. N-gram Context Size
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    n_gram_order = [1, 2, 3, 4]
    perplexity = [180, 45, 18, 14]
    ax.plot(n_gram_order, perplexity, marker='o', markersize=8, linewidth=2.5, color='#4f46e5', label='Perplexity')
    ax.fill_between(n_gram_order, perplexity, alpha=0.08, color='#4f46e5')
    
    # Annotation labels
    for x, y in zip(n_gram_order, perplexity):
        ax.annotate(f'PPL: {y}', xy=(x, y), xytext=(8, 4), textcoords='offset points', fontsize=8, fontweight='bold', color='#4f46e5')
        
    ax.set_xlabel('N-gram Context Order (N)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Perplexity (Lower is Better)', fontsize=10, fontweight='semibold')
    ax.set_title('Model Perplexity vs. Context Order', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(n_gram_order)
    ax.set_xticklabels(['Unigram', 'Bigram', 'Trigram', '4-gram'])
    ax.set_ylim(0, 210)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perplexity_decay.png'), dpi=300)
    plt.close()
    
    # 4. Word Embedding Analogies 2D projection scatter plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    words = ['king', 'man', 'queen', 'woman']
    coords = np.array([
        [1.8, 1.2],
        [0.5, 1.2],
        [1.8, 2.5],
        [0.5, 2.5]
    ])
    ax.scatter(coords[:, 0], coords[:, 1], color='#ef4444', s=120, edgecolors='#1e293b', zorder=5)
    for word, (x, y) in zip(words, coords):
        ax.text(x + 0.08, y - 0.05, word, fontsize=11, fontweight='bold', color='#1e293b')
    
    # Draw vector shift lines with improved aesthetics
    ax.annotate("", xy=(1.8, 1.2), xytext=(0.5, 1.2),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2, ls="-", shrinkA=8, shrinkB=8))
    ax.annotate("", xy=(1.8, 2.5), xytext=(0.5, 2.5),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2, ls="-", shrinkA=8, shrinkB=8))
    ax.annotate("", xy=(0.5, 2.5), xytext=(0.5, 1.2),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=2, ls="-", shrinkA=8, shrinkB=8))
    ax.annotate("", xy=(1.8, 2.5), xytext=(1.8, 1.2),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=2, ls="-", shrinkA=8, shrinkB=8))
                
    # Labels for vectors
    ax.text(1.15, 1.35, 'Gender Vector', color='#2563eb', ha='center', fontsize=9, fontweight='semibold')
    ax.text(0.12, 1.85, 'Royalty\nVector', color='#059669', ha='center', fontsize=9, fontweight='semibold')
    
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0.5, 3.2)
    ax.set_xlabel('Embedding Dimension 1 (Semantic Latent Space)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Embedding Dimension 2 (Semantic Latent Space)', fontsize=10, fontweight='semibold')
    ax.set_title('Word Embedding Analogy Space Projection', fontsize=12, fontweight='bold', pad=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_analogy_projection.png'), dpi=300)
    plt.close()
    
    # 5. RNN vs. LSTM Gradient Norm Decay across sequence length
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    seq_steps = np.arange(1, 31)
    rnn_grad_norms = np.exp(-0.32 * seq_steps)
    lstm_grad_norms = np.ones_like(seq_steps) * 0.93 + np.random.normal(0, 0.015, size=30)
    ax.plot(seq_steps, rnn_grad_norms, label='Standard RNN (Multiplicative Decay)', color='#dc2626', linewidth=2.2)
    ax.plot(seq_steps, lstm_grad_norms, label='LSTM CEC (Additive Propagation)', color='#059669', linewidth=2.2)
    ax.set_xlabel('Backpropagation Steps (Time Gap)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Gradient Norm Ratio', fontsize=10, fontweight='semibold')
    ax.set_title('Gradient Flow Decay Comparison', fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_flow_comparison.png'), dpi=300)
    plt.close()
    
    # 6. Attention Weight Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    words_q = ['The', 'feline', 'sat']
    words_k = ['The', 'cat', 'sat']
    att_weights = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.02, 0.03, 0.95]
    ])
    sns.heatmap(att_weights, annot=True, fmt=".2f", cmap='Purples', 
                xticklabels=words_k, yticklabels=words_q, cbar=True, 
                vmin=0, vmax=1, linewidths=1.0, linecolor='#e2e8f0',
                annot_kws={"size": 11, "weight": "bold", "color": "#0f172a"}, ax=ax)
    ax.set_title('Self-Attention Alignment Weight Matrix', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_matrix_heatmap.png'), dpi=300)
    plt.close()
    
    # 7. Data Drift distribution divergence check
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.linspace(-3, 8, 200)
    train_dist = np.exp(-0.5 * (x - 0)**2) / np.sqrt(2 * np.pi)
    prod_dist = np.exp(-0.5 * (x - 3.2)**2) / np.sqrt(2 * np.pi)
    ax.fill_between(x, train_dist, color='#2563eb', alpha=0.15, label='Training Distribution P(X)')
    ax.fill_between(x, prod_dist, color='#f59e0b', alpha=0.15, label='Production Distribution P(X)')
    ax.plot(x, train_dist, color='#2563eb', linewidth=2, label='_nolegend_')
    ax.plot(x, prod_dist, color='#f59e0b', linewidth=2, label='_nolegend_')
    ax.set_xlabel('Lexical Feature Space Projection', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Probability Density', fontsize=10, fontweight='semibold')
    ax.set_title('Data Drift (Covariate Shift) Divergence', fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_drift_distributions.png'), dpi=300)
    plt.close()
    
    # 8. BLEU Brevity Penalty Decay
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ratio = np.linspace(0.05, 1.5, 200)
    bp = np.where(ratio > 1.0, 1.0, np.exp(1.0 - 1.0/ratio))
    ax.plot(ratio, bp, color='#ef4444', linewidth=2.5, label='Brevity Penalty (BP)')
    ax.fill_between(ratio, bp, alpha=0.08, color='#ef4444')
    ax.axvline(1.0, color='#64748b', linestyle='--', linewidth=1, label='c = r (Reference Length)')
    
    ax.set_xlabel('Length Ratio (c / r)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Penalty Value (BP)', fontsize=10, fontweight='semibold')
    ax.set_title('BLEU Brevity Penalty vs. Length Ratio', fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    ax.set_ylim(-0.05, 1.05)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bleu_brevity_penalty.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_all_plots()
    print("All NLP Fundamentals plots generated successfully!")
