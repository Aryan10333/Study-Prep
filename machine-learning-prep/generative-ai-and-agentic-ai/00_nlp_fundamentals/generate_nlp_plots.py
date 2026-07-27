import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_plots():
    output_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals\plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Stemming vs Lemmatization Latency Comparison
    plt.figure(figsize=(6, 4))
    methods = ['Porter Stemmer', 'WordNet Lemmatizer']
    latencies = [1.2, 4.8]  # mock latencies in milliseconds
    plt.bar(methods, latencies, color=['#2563eb', '#10b981'], width=0.5)
    plt.ylabel('Inference Latency (ms / 1000 tokens)')
    plt.title('Lexical Reduction Pipeline Latency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stem_vs_lemma_latency.png'), dpi=150)
    plt.close()
    
    # 2. Cosine Similarity Heatmap of TF-IDF vectors
    plt.figure(figsize=(6, 5))
    doc_labels = ['Doc 1: Cat/Feline', 'Doc 2: Feline/Rug', 'Doc 3: Dog/Garden']
    similarity_matrix = np.array([
        [1.00, 0.34, 0.00],
        [0.34, 1.00, 0.05],
        [0.00, 0.05, 1.00]
    ])
    sns.heatmap(similarity_matrix, annot=True, cmap='Blues', xticklabels=doc_labels, yticklabels=doc_labels, cbar=True, vmin=0, vmax=1)
    plt.title('Sparse Document Vector Cosine Similarity Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tfidf_similarity_heatmap.png'), dpi=150)
    plt.close()
    
    # 3. Perplexity vs. N-gram Context Size
    plt.figure(figsize=(6, 4))
    n_gram_order = [1, 2, 3, 4]
    perplexity = [180, 45, 18, 14]
    plt.plot(n_gram_order, perplexity, marker='o', linewidth=2.5, color='#7c3aed')
    plt.xlabel('N-gram Context Order (N)')
    plt.ylabel('Perplexity (Test Set)')
    plt.title('Model Perplexity vs. Context Order')
    plt.xticks(n_gram_order, ['Unigram', 'Bigram', 'Trigram', '4-gram'])
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perplexity_decay.png'), dpi=150)
    plt.close()
    
    # 4. Word Embedding Analogies 2D projection scatter plot
    plt.figure(figsize=(6, 5))
    words = ['king', 'man', 'queen', 'woman']
    coords = np.array([
        [1.8, 1.2],
        [0.5, 1.2],
        [1.8, 2.5],
        [0.5, 2.5]
    ])
    plt.scatter(coords[:, 0], coords[:, 1], color='#ef4444', s=100)
    for word, (x, y) in zip(words, coords):
        plt.text(x + 0.08, y - 0.05, word, fontsize=12, fontweight='bold')
    
    # Draw vector shift lines
    plt.arrow(0.5, 1.2, 1.3, 0.0, head_width=0.05, head_length=0.08, fc='#3b82f6', ec='#3b82f6', length_includes_head=True, label='Gender Offset')
    plt.arrow(0.5, 2.5, 1.3, 0.0, head_width=0.05, head_length=0.08, fc='#3b82f6', ec='#3b82f6', length_includes_head=True)
    plt.arrow(0.5, 1.2, 0.0, 1.3, head_width=0.05, head_length=0.08, fc='#10b981', ec='#10b981', length_includes_head=True, label='Royalty Offset')
    plt.arrow(1.8, 1.2, 0.0, 1.3, head_width=0.05, head_length=0.08, fc='#10b981', ec='#10b981', length_includes_head=True)
    
    plt.xlim(0, 2.5)
    plt.ylim(0.5, 3.2)
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Word Embedding Analogy Space Projection')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_analogy_projection.png'), dpi=150)
    plt.close()
    
    # 5. RNN vs. LSTM Gradient Norm Decay across sequence length
    plt.figure(figsize=(6, 4))
    seq_steps = np.arange(1, 31)
    rnn_grad_norms = np.exp(-0.35 * seq_steps)
    lstm_grad_norms = np.ones_like(seq_steps) * 0.95 + np.random.normal(0, 0.02, size=30)
    plt.plot(seq_steps, rnn_grad_norms, label='Standard RNN (Multiplicative Decay)', color='#ef4444', linewidth=2)
    plt.plot(seq_steps, lstm_grad_norms, label='LSTM CEC (Additive Propagation)', color='#10b981', linewidth=2)
    plt.xlabel('Backpropagation Steps (Time Gap)')
    plt.ylabel('Gradient Norm Ratio')
    plt.title('Gradient Flow Decay comparison')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_flow_comparison.png'), dpi=150)
    plt.close()
    
    # 6. Attention Weight Matrix Heatmap
    plt.figure(figsize=(6, 5))
    words_q = ['The', 'feline', 'sat']
    words_k = ['The', 'cat', 'sat']
    att_weights = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.02, 0.03, 0.95]
    ])
    sns.heatmap(att_weights, annot=True, cmap='Purples', xticklabels=words_k, yticklabels=words_q, cbar=True, vmin=0, vmax=1)
    plt.title('Self-Attention Alignment Weight Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_matrix_heatmap.png'), dpi=150)
    plt.close()
    
    # 7. Data Drift distribution divergence check
    plt.figure(figsize=(6, 4))
    x = np.linspace(-3, 8, 200)
    train_dist = np.exp(-0.5 * (x - 0)**2) / np.sqrt(2 * np.pi)
    prod_dist = np.exp(-0.5 * (x - 3)**2) / np.sqrt(2 * np.pi)
    plt.fill_between(x, train_dist, color='#2563eb', alpha=0.3, label='Training Distribution P(X)')
    plt.fill_between(x, prod_dist, color='#f59e0b', alpha=0.3, label='Production Distribution P(X)')
    plt.plot(x, train_dist, color='#2563eb', linewidth=2)
    plt.plot(x, prod_dist, color='#f59e0b', linewidth=2)
    plt.xlabel('Lexical Feature Space Projection')
    plt.ylabel('Probability Density')
    plt.title('Data Drift (Covariate Shift) Divergence')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_drift_distributions.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    generate_all_plots()
    print("All NLP Fundamentals plots generated successfully!")
