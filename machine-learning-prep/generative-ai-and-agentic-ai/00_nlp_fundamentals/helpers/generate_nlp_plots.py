import os
import numpy as np
import textwrap
import matplotlib
# Enforce headless Agg backend for execution in script environments
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

def generate_zipfs_law(output_dir):
    """Generates Zipf's Law rank-frequency curve (Module 04)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Simulate rank and frequency
    ranks = np.arange(1, 1000)
    frequencies = 10000 / (ranks ** 1.0)
    
    # Linear scale plot
    axes[0].plot(ranks, frequencies, color='#2563eb', linewidth=2)
    axes[0].set_xlabel('Word Rank (r)', fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('Word Frequency (f)', fontsize=10, fontweight='semibold')
    axes[0].set_title('Linear Rank vs. Frequency Scale', fontsize=11, fontweight='bold', color='#0f172a')
    
    # Log-Log scale plot
    axes[1].loglog(ranks, frequencies, color='#7c3aed', linewidth=2, label='Empirical Slope = -1.0')
    axes[1].set_xlabel('Log Word Rank (r)', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('Log Word Frequency (f)', fontsize=10, fontweight='semibold')
    axes[1].set_title('Log-Log Rank vs. Frequency Scale', fontsize=11, fontweight='bold', color='#0f172a')
    axes[1].legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
    
    fig.suptitle("Zipf's Law word frequency distribution", fontsize=13, fontweight='bold', y=0.98, color='#0f172a')
    sns.despine(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_zipfs_law.png'), dpi=300)
    plt.close()

def generate_bm25_saturation(output_dir):
    """Generates BM25 saturation vs TF-IDF scaling curves (Module 04)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    tf = np.linspace(0, 15, 200)
    
    # Calculations
    tf_linear = tf
    tf_log = np.log(1 + tf)
    
    k1_1 = 1.2
    k1_2 = 2.0
    
    bm25_k1_1 = (tf * (k1_1 + 1)) / (tf + k1_1)
    bm25_k1_2 = (tf * (k1_2 + 1)) / (tf + k1_2)
    
    ax.plot(tf, tf_linear, color='#dc2626', linestyle='--', label='Linear TF (Raw Count)', linewidth=1.5)
    ax.plot(tf, tf_log, color='#059669', linestyle='-.', label='Log-Scaled TF log(1 + tf)', linewidth=1.5)
    ax.plot(tf, bm25_k1_1, color='#2563eb', label=f'BM25 Term contribution (k1 = {k1_1})', linewidth=2)
    ax.plot(tf, bm25_k1_2, color='#0284c7', label=f'BM25 Term contribution (k1 = {k1_2})', linewidth=2)
    
    ax.set_xlabel('Term Frequency (f) in Document', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Term Score Contribution', fontsize=10, fontweight='semibold')
    ax.set_title('Score Saturation Curves: TF-IDF vs. Okapi BM25', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, 15)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_tfidf_vs_bm25_saturation.png'), dpi=300)
    plt.close()

def generate_gradient_flow(output_dir):
    """Generates simulated gradient norm decay over recurrent steps (Module 09)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    steps = np.arange(1, 40)
    
    # Vanilla RNN (exponential decay)
    rnn_grads = np.exp(-0.25 * steps)
    # LSTM (stable flow via linear carry)
    lstm_grads = 0.8 / (1 + 0.01 * steps)
    
    ax.plot(steps, rnn_grads, color='#dc2626', marker='x', markersize=4, label='Vanilla RNN cell (Vanishing Gradients)', linewidth=1.5)
    ax.plot(steps, lstm_grads, color='#2563eb', marker='o', markersize=4, label='LSTM cell state (Stable gradient flow)', linewidth=2)
    
    ax.set_xlabel('Backpropagation Recurrent Steps (Time steps back)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Backpropagated Gradient Norm', fontsize=10, fontweight='semibold')
    ax.set_title('Gradient Signal Stability Over Long Sequences', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_rnn_gradient_flow.png'), dpi=300)
    plt.close()

def generate_drift_psi(output_dir):
    """Generates Expected vs Actual distribution histograms for data drift (Module 10)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    bins = ['Tech', 'News', 'Finance', 'Slang/Emojis', 'Spam Ads']
    expected = [0.35, 0.25, 0.20, 0.05, 0.15]
    actual = [0.20, 0.15, 0.12, 0.38, 0.15] # Heavy shift toward Emojis/Slang
    
    x = np.arange(len(bins))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, expected, width, label='Expected Baseline Distribution', color='#0f172a', edgecolor='#1e293b')
    rects2 = ax.bar(x + width/2, actual, width, label='Actual Serving Stream (Drifted)', color='#dc2626', edgecolor='#991b1b')
    
    ax.set_xlabel('Vocabulary Classification Category Bins', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Probability Frequency Proportion', fontsize=10, fontweight='semibold')
    ax.set_title('Covariate Text Distribution Shift (PSI Validation)', fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_data_drift_psi.png'), dpi=300)
    plt.close()

def draw_linguistic_levels(output_dir):
    """Draws levels of linguistic analysis schematic (Module 01)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.axis('off')
    
    levels = [
        ("PRAGMATICS", "Contextual & situational meanings of text segments.", "#1e3a8a", "#eff6ff"),
        ("SEMANTICS", "Literal meanings of words, word combinations, and phrase structures.", "#1d4ed8", "#eff6ff"),
        ("SYNTAX", "Structural and grammatical rules governing sentence compositions.", "#2563eb", "#eff6ff"),
        ("MORPHOLOGY", "Morphemes, prefix/suffix root structures, and word boundaries.", "#3b82f6", "#eff6ff"),
        ("PHONOLOGY", "Phonemes, sound constructs, and characters.", "#60a5fa", "#eff6ff")
    ]
    
    y = 0.8
    for name, desc, border, bg in levels:
        # Wrap description to 35 characters per line to prevent cutting off text
        wrapped_desc = textwrap.fill(desc, width=35)
        
        # Draw box (shifted left to 0.12)
        ax.text(0.12, y, name, fontsize=11, fontweight='bold', color=border,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=bg, edgecolor=border, lw=1.5),
                ha='center', va='center')
        ax.text(0.26, y, wrapped_desc, fontsize=9.5, color='#334155', ha='left', va='center')
        
        if y > 0.2:
            # Draw arrow pointing up (shifted left to 0.12)
            ax.annotate('', xy=(0.12, y + 0.05), xytext=(0.12, y - 0.05),
                        arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
        y -= 0.15
        
    ax.set_title("Levels of Linguistic Analysis Hierarchy", fontsize=12, fontweight='bold', pad=10, color='#0f172a', loc='center')
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0.1, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_linguistic_levels.png'), dpi=300)
    plt.close()

def draw_subword_tree(output_dir):
    """Draws a subword tree parsing hierarchy (Module 03)."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.axis('off')
    
    # Root word
    ax.text(0.5, 0.8, '"unfriendliness"', fontsize=12, fontweight='bold', color='#0f172a',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#f1f5f9', edgecolor='#94a3b8', lw=1),
            ha='center', va='center')
            
    # Subword splits (Level 1)
    ax.text(0.25, 0.5, '"un-"\n(Prefix)', fontsize=10, fontweight='semibold', color='#2563eb',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#eff6ff', edgecolor='#3b82f6', lw=1.2),
            ha='center', va='center')
            
    ax.text(0.75, 0.5, '"friendliness"\n(Base Compound)', fontsize=10, fontweight='semibold', color='#475569',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1),
            ha='center', va='center')
            
    # Subword splits (Level 2)
    ax.text(0.6, 0.2, '"friend"\n(Root Morpheme)', fontsize=9.5, fontweight='semibold', color='#059669',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#ecfdf5', edgecolor='#10b981', lw=1.2),
            ha='center', va='center')
            
    ax.text(0.9, 0.2, '"-ly" + "-ness"\n(Suffixes)', fontsize=9.5, fontweight='semibold', color='#7c3aed',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#f5f3ff', edgecolor='#8b5cf6', lw=1.2),
            ha='center', va='center')
            
    # Connective lines
    ax.plot([0.5, 0.25], [0.73, 0.58], color='#64748b', lw=1.5)
    ax.plot([0.5, 0.75], [0.73, 0.58], color='#64748b', lw=1.5)
    ax.plot([0.75, 0.6], [0.42, 0.28], color='#94a3b8', lw=1.2)
    ax.plot([0.75, 0.9], [0.42, 0.28], color='#94a3b8', lw=1.2)
    
    ax.set_title("Subword Vocabulary Tree Fragmentation Example", fontsize=11, fontweight='bold', pad=10, color='#0f172a', loc='center')
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_subword_tree.png'), dpi=300)
    plt.close()

def draw_word2vec_projection(output_dir):
    """Draws Word2Vec CBOW vs. Skip-gram projection layouts (Module 05)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    for ax in axes:
        ax.axis('off')
        
    # --- Skip-gram (Target -> Contexts) ---
    axes[0].text(0.2, 0.5, 'Input Context\nWords\n[w_t-1, w_t+1]', fontsize=9.5, color='#475569', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#f8fafc', edgecolor='#cbd5e1'))
    axes[0].text(0.5, 0.5, 'Projection Layer\n(Sum/Mean Embedding)\n[h]', fontsize=9.5, fontweight='semibold', color='#2563eb', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#eff6ff', edgecolor='#3b82f6', lw=1.5))
    axes[0].text(0.8, 0.5, 'Output Target\nWord\n[w_t]', fontsize=9.5, color='#0f172a', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#f1f5f9', edgecolor='#94a3b8'))
                 
    axes[0].annotate('', xy=(0.35, 0.5), xytext=(0.28, 0.5), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.5))
    axes[0].annotate('', xy=(0.65, 0.5), xytext=(0.58, 0.5), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.5))
    axes[0].set_title("CBOW Model Architecture", fontsize=11, fontweight='bold', pad=10, color='#0f172a')
    
    # --- CBOW (Contexts -> Target) ---
    axes[1].text(0.2, 0.5, 'Input Target\nWord\n[w_t]', fontsize=9.5, color='#0f172a', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#f1f5f9', edgecolor='#94a3b8'))
    axes[1].text(0.5, 0.5, 'Projection Layer\n(Embedding Lookup)\n[h]', fontsize=9.5, fontweight='semibold', color='#2563eb', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#eff6ff', edgecolor='#3b82f6', lw=1.5))
    axes[1].text(0.8, 0.5, 'Output Context\nWords\n[w_t-1, w_t+1]', fontsize=9.5, color='#475569', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#f8fafc', edgecolor='#cbd5e1'))
                 
    axes[1].annotate('', xy=(0.35, 0.5), xytext=(0.28, 0.5), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.5))
    axes[1].annotate('', xy=(0.65, 0.5), xytext=(0.58, 0.5), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.5))
    axes[1].set_title("Skip-Gram Model Architecture", fontsize=11, fontweight='bold', pad=10, color='#0f172a')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_word2vec_projection.png'), dpi=300)
    plt.close()

def draw_fasttext_ngrams(output_dir):
    """Draws FastText n-gram mean pooling projection (Module 06)."""
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
    ax.axis('off')
    
    ax.text(0.15, 0.8, 'Word String\n"<where>"', fontsize=10, fontweight='bold', color='#0f172a', ha='center', va='center')
    
    ngrams = ['"<wh"', '"whe"', '"her"', '"ere"', '"re>"', '"where" (whole)']
    y = 0.8
    for ng in ngrams:
        ax.text(0.45, y, ng, fontsize=9, color='#475569', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#f8fafc', edgecolor='#cbd5e1'))
        
        ax.annotate('', xy=(0.35, y), xytext=(0.23, 0.8), arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1))
        ax.annotate('', xy=(0.62, 0.5), xytext=(0.52, y), arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1))
        y -= 0.12
        
    ax.text(0.8, 0.5, 'EmbeddingBag\nMean Aggregation\nVector [h]', fontsize=10, fontweight='bold', color='#059669', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#ecfdf5', edgecolor='#10b981', lw=1.5))
            
    ax.set_title("FastText Subword Embedding Aggregation Flow", fontsize=11, fontweight='bold', pad=10, color='#0f172a')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.1, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_fasttext_ngrams.png'), dpi=300)
    plt.close()

def draw_hmm_lattice(output_dir):
    """Draws Hidden Markov Model transition state lattice (Module 07)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    ax.axis('off')
    
    # States
    states_t0 = ['Noun', 'Verb', 'Adj']
    states_t1 = ['Noun', 'Verb', 'Adj']
    
    y0 = [0.7, 0.5, 0.3]
    y1 = [0.7, 0.5, 0.3]
    
    # Draw State Boxes at t-1
    for s, y in zip(states_t0, y0):
        ax.text(0.2, y, f'{s} (t-1)', fontsize=9, fontweight='semibold', color='#0f172a', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='#f1f5f9', edgecolor='#64748b'))
                
    # Draw State Boxes at t
    for s, y in zip(states_t1, y1):
        ax.text(0.8, y, f'{s} (t)', fontsize=9, fontweight='semibold', color='#2563eb', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='#eff6ff', edgecolor='#3b82f6', lw=1.2))
                
    # Draw transitions
    for y_prev in y0:
        for y_curr in y1:
            ax.annotate('', xy=(0.72, y_curr), xytext=(0.28, y_prev),
                        arrowprops=dict(arrowstyle="->", color='#94a3b8', alpha=0.6, lw=1))
                        
    # Annotate Transition Probability A_ij and Emission Probability B_jk
    ax.text(0.5, 0.65, 'Transition Matrix A\nP(State_t | State_t-1)', fontsize=8, color='#475569', ha='center', va='center', fontweight='semibold')
    
    ax.set_title("HMM Hidden State Transition Lattice Diagram", fontsize=11, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.2, 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_hmm_lattice.png'), dpi=300)
    plt.close()

def draw_lstm_cell(output_dir):
    """Draws detailed LSTM gating structure schematic (Module 09)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.axis('off')
    
    # Background box representing cell boundary
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#64748b', lw=2, linestyle='-'))
    
    # State paths
    ax.annotate('Cell State C_t-1', xy=(0.1, 0.75), xytext=(-0.05, 0.75), arrowprops=dict(arrowstyle="<-", color='#8b5cf6', lw=1.5))
    ax.annotate('Cell State C_t', xy=(1.05, 0.75), xytext=(0.9, 0.75), arrowprops=dict(arrowstyle="->", color='#8b5cf6', lw=2))
    ax.plot([0.1, 0.9], [0.75, 0.75], color='#8b5cf6', lw=2) # State rail
    
    # Inputs
    ax.text(0.25, -0.05, 'Input x_t', fontsize=9, fontweight='semibold', color='#0f172a', ha='center')
    ax.text(0.55, -0.05, 'Hidden State h_t-1', fontsize=9, fontweight='semibold', color='#475569', ha='center')
    
    # Draw Gating operations
    ax.text(0.3, 0.4, 'Forget Gate\nf_t = \u03c3(...)', fontsize=8.5, fontweight='bold', color='#dc2626', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='#fef2f2', edgecolor='#ef4444'))
            
    ax.text(0.5, 0.4, 'Input Gate\ni_t = \u03c3(...)\nCandidate Cell\nC~ = tanh(...)', fontsize=8, fontweight='bold', color='#059669', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='#ecfdf5', edgecolor='#10b981'))
            
    ax.text(0.75, 0.4, 'Output Gate\no_t = \u03c3(...)', fontsize=8.5, fontweight='bold', color='#0284c7', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='#eff6ff', edgecolor='#0284c7'))
            
    # Connect gates to cell state line
    ax.annotate('', xy=(0.3, 0.75), xytext=(0.3, 0.48), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.48), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    ax.annotate('', xy=(0.75, 0.75), xytext=(0.75, 0.48), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    
    ax.annotate('Hidden State h_t', xy=(1.05, 0.25), xytext=(0.9, 0.25), arrowprops=dict(arrowstyle="->", color='#2563eb', lw=2))
    ax.plot([0.75, 0.9], [0.25, 0.25], color='#2563eb', lw=1.5)
    ax.annotate('', xy=(0.75, 0.25), xytext=(0.75, 0.32), arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.2))
    
    ax.set_title("LSTM Gated Recurrent Unit Architecture & State Flow", fontsize=11, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_lstm_cell.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    output_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals\plots"
    os.makedirs(output_dir, exist_ok=True)
    setup_premium_style()
    
    # Generate premium charts
    generate_zipfs_law(output_dir)
    generate_bm25_saturation(output_dir)
    generate_gradient_flow(output_dir)
    generate_drift_psi(output_dir)
    
    # Draw premium schematics
    draw_linguistic_levels(output_dir)
    draw_subword_tree(output_dir)
    draw_word2vec_projection(output_dir)
    draw_fasttext_ngrams(output_dir)
    draw_hmm_lattice(output_dir)
    draw_lstm_cell(output_dir)
    print("All premium NLP plots and schematics successfully created!")
