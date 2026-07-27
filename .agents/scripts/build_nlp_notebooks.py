import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def run_and_save(nb, path):
    ep = ExecutePreprocessor(timeout=240, kernel_name='prep-venv')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path) or '.'}})
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {path}")

def build_01_text_preprocessing():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 01_text_preprocessing: Cleaning, Normalization, Stemming, and Subword Simulation

This notebook implements classical text preprocessing steps (Porter stemming and WordNet lemmatization) using NLTK, and simulates a basic Byte-Pair Encoding (BPE) subword merge loop.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 1. Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

raw_text = "The cats were studying studying studies in Seattle's libraries! https://example.com"

# 2. Basic regex cleaning
cleaned_text = re.sub(r"https?://\S+", "", raw_text)
cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text).lower()
print("Cleaned Text:", cleaned_text)

# 3. Tokenize
tokens = word_tokenize(cleaned_text)
print("Tokens:", tokens)

# 4. Stemming vs Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed = [stemmer.stem(t) for t in tokens]
lemmatized = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

print("\nLexical Reduction Comparison:")
print(f"{'Original':<12} | {'Stemmed':<12} | {'Lemmatized':<12}")
print("-" * 42)
for o, s, l in zip(tokens, stemmed, lemmatized):
    print(f"{o:<12} | {s:<12} | {l:<12}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Byte-Pair Encoding (BPE) Simulation
Let's simulate a basic bottom-up BPE tokenizer training merge loop on a tiny vocabulary.
"""))

    cells.append(nbf.v4.new_code_cell(r"""from collections import Counter, defaultdict

# Tiny BPE training corpus
corpus = {
    "l o w _": 5,
    "l o w e r _": 2,
    "n e w e s t _": 6
}

def get_stats(corpus):
    pairs = defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, corpus):
    new_corpus = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in corpus:
        w_new = p.sub(''.join(pair), word)
        new_corpus[w_new] = corpus[word]
    return new_corpus

# Run 5 BPE merge iterations
vocab = set("l o w e r n s t _".split())
print("Initial Vocab:", sorted(vocab))

for i in range(5):
    pairs = get_stats(corpus)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    corpus = merge_vocab(best_pair, corpus)
    merged_token = ''.join(best_pair)
    vocab.add(merged_token)
    print(f"\nIteration {i+1}: Merging {best_pair} (frequency={pairs[best_pair]})")
    print("Updated Corpus State:", corpus)

print("\nFinal BPE Vocab:", sorted(vocab))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- **Lexical Reduction**: Stemming cuts off suffixes heuristically (e.g. `"studying"` $\rightarrow$ `"studi"`), whereas lemmatization resolves tokens to morphological base forms using grammatical tagging dictionary lookups (e.g. `"studies"` $\rightarrow$ `"study"`).
- **BPE merges**: The simulator finds adjacent character pairs (e.g., `(e, s)` then `(es, t)`) and groups them into single, multi-character subwords.
"""))
    
    nb['cells'] = cells
    return nb

def build_02_bag_of_words_tfidf():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 02_bag_of_words_tfidf: Vector Representations from Scratch

This notebook builds Bag of Words (BoW) and Term Frequency - Inverse Document Frequency (TF-IDF) representation matrices from scratch using NumPy, and compares results against Scikit-Learn's estimators.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Tiny Corpus
corpus = [
    "cat feline mat",
    "feline rug mat"
]

# 1. Map vocabulary
vocab = sorted(list(set(" ".join(corpus).split())))
word_to_idx = {w: i for i, w in enumerate(vocab)}
print("Vocabulary mapping:", word_to_idx)

# 2. Bag of Words (BoW) from scratch
bow_matrix = np.zeros((len(corpus), len(vocab)))
for doc_idx, doc in enumerate(corpus):
    for word in doc.split():
        if word in word_to_idx:
            bow_matrix[doc_idx, word_to_idx[word]] += 1

print("\nBag of Words Matrix (from scratch):\n", bow_matrix)

# 3. Smooth TF-IDF from scratch
# Smooth IDF formulation: log((1 + N) / (1 + DF)) + 1
N = len(corpus)
df = np.sum(bow_matrix > 0, axis=0)
idf = np.log((1 + N) / (1 + df)) + 1

# Calculate TF-IDF
tfidf_matrix = bow_matrix * idf

# L2 normalization to match Scikit-Learn standard
norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
tfidf_norm = tfidf_matrix / norms

print("\nTF-IDF Matrix (from scratch, normalized):\n", tfidf_norm)

# 4. Compare with Scikit-Learn
vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True)
sklearn_tfidf = vectorizer.fit_transform(corpus).toarray()
print("\nScikit-Learn TF-IDF Matrix:\n", sklearn_tfidf)

# Check assertion
assert np.allclose(tfidf_norm, sklearn_tfidf, atol=1e-5)
print("\nSUCCESS: Custom TF-IDF matrix matches Scikit-Learn output exactly!")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The hand-coded matrix implementation calculates raw term frequencies, applies the smooth IDF equation, normalizes vectors by their $L_2$ Euclidean norms, and matches the output of Scikit-Learn's `TfidfVectorizer` exactly.
"""))
    
    nb['cells'] = cells
    return nb

def build_03_word2vec():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 03_word2vec: Word Representation Models using Gensim

This notebook trains Continuous Bag-of-Words (CBOW) and Skip-gram Word2Vec embedding models on a custom text corpus using Gensim, and inspects vector representations.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""from gensim.models import Word2Vec

# Custom token corpus
sentences = [
    ["the", "feline", "sat", "on", "the", "mat"],
    ["the", "cat", "sat", "on", "the", "mat"],
    ["a", "feline", "rested", "on", "the", "rug"],
    ["the", "cat", "rested", "on", "the", "rug"],
    ["dogs", "run", "in", "the", "park"],
    ["cats", "sleep", "on", "the", "mat"]
]

# 1. Train Continuous Bag-of-Words (CBOW) model
cbow_model = Word2Vec(sentences=sentences, vector_size=20, window=2, min_count=1, sg=0, epochs=100)

# 2. Train Skip-gram model
sg_model = Word2Vec(sentences=sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100)

# 3. Retrieve representations
print("=== CBOW Embedding for 'feline' ===")
print(cbow_model.wv["feline"])

print("\n=== Similarity lookup ('cat' vs 'feline') ===")
cbow_sim = cbow_model.wv.similarity("cat", "feline")
sg_sim = sg_model.wv.similarity("cat", "feline")
print(f"CBOW Cosine Similarity: {cbow_sim:.4f}")
print(f"Skip-gram Cosine Similarity: {sg_sim:.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The model projects semantic correlations into the dense embedding vectors.
- Words sharing similar context profiles (like `"cat"` and `"feline"`) yield high cosine similarity values, while non-related terms exhibit low scores.
"""))
    
    nb['cells'] = cells
    return nb

def build_04_glove_fasttext():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 04_glove_fasttext: Character N-Grams and Out-of-Vocabulary Resolution

This notebook trains a FastText model using Gensim to show how character n-grams resolve Out-of-Vocabulary (OOV) lookup failures that cause static Word2Vec to crash.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""from gensim.models import FastText
from gensim.models import Word2Vec

# Tiny training corpus
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["feline", "sat", "on", "the", "rug"],
    ["dogs", "run", "in", "the", "garden"]
]

# 1. Train standard Word2Vec (vocabulary is fixed to training tokens)
w2v = Word2Vec(sentences, vector_size=10, window=2, min_count=1, epochs=10)

# 2. Train FastText (stores character n-grams)
ft = FastText(sentences, vector_size=10, window=2, min_count=1, min_n=3, max_n=6, epochs=10)

# 3. Attempt OOV word retrieval (e.g. 'cats' - not in training vocabulary)
print("Vocabulary keys in training data:", list(w2v.wv.key_to_index.keys()))

try:
    vector = w2v.wv["cats"]
except KeyError as e:
    print("\n[Word2Vec Error]: Word 'cats' is out of vocabulary!")

# FastText handles the OOV word via character subword n-grams
ft_vector = ft.wv["cats"]
print("\nFastText Vector for OOV word 'cats':\n", ft_vector)

# Similar words lookup
print("\nFastText similarity 'cat' vs 'cats':", ft.wv.similarity("cat", "cats"))
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- Standard Word2Vec raises a `KeyError` when queried with the Out-of-Vocabulary word `"cats"`.
- FastText handles this by decomposing `"cats"` into character n-grams (e.g., `cat`, `ats`) and summing their vectors to generate an embedding.
"""))
    
    nb['cells'] = cells
    return nb

def build_05_ngram_language_models():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 05_ngram_language_models: N-gram Estimations and Perplexity Calculations

This notebook builds a Bigram Language Model from scratch, implements Laplace smoothing, and computes Perplexity metrics over sample text sequences.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import math
from collections import Counter, defaultdict

# 1. Corpus
corpus = "the cat sat on the mat the cat rested on the rug".split()
vocab = list(set(corpus))
vocab_size = len(vocab)

# 2. Count Unigrams and Bigrams
unigrams = Counter(corpus)
bigrams = Counter(zip(corpus[:-1], corpus[1:]))

# 3. Probability estimation with Laplace (add-1) smoothing
def get_bigram_prob(w1, w2):
    count_bigram = bigrams[(w1, w2)]
    count_unigram = unigrams[w1]
    # Smooth calculation
    return (count_bigram + 1) / (count_unigram + vocab_size)

print("Vocabulary:", vocab)
print("\nSmoothed transition probabilities:")
print("P(cat | the) =", get_bigram_prob("the", "cat"))
print("P(rug | the) =", get_bigram_prob("the", "rug")) # non-zero despite not appearing

# 4. Calculate Perplexity on a test sequence
test_sequence = ["the", "cat", "rested", "on", "the", "mat"]

def compute_perplexity(seq):
    log_prob_sum = 0.0
    # Bigram sequence modeling
    for i in range(1, len(seq)):
        w1, w2 = seq[i-1], seq[i]
        prob = get_bigram_prob(w1, w2)
        log_prob_sum += math.log(prob)
        
    avg_log_prob = log_prob_sum / (len(seq) - 1)
    perplexity = math.exp(-avg_log_prob)
    return perplexity

ppl = compute_perplexity(test_sequence)
print(f"\nPerplexity of sequence {test_sequence}: {ppl:.4f}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The bigram language model calculates transitions.
- Laplace smoothing prevents zero probabilities for unseen transitions (e.g. `"the rug"`), allowing the model to compute a valid, finite perplexity value.
"""))
    
    nb['cells'] = cells
    return nb

def build_06_rnn_lstm_gru():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 06_rnn_lstm_gru: Sequence Classifier Comparison in PyTorch

This notebook constructs a simple bidirectional sequence classifier in PyTorch, comparing RNN, LSTM, and GRU architectures.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import torch
import torch.nn as nn

# Model parameters
vocab_size = 100
embedding_dim = 16
hidden_dim = 32
num_classes = 2

# Mock input batch: size=2, sequence_length=5
x = torch.randint(0, vocab_size, (2, 5))

# 1. Define models
class RecurrentClassifier(nn.Module):
    def __init__(self, cell_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            
        # bidirectional outputs are doubled in dimensionality
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        # Take the output of the final time step
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return logits

# 2. Run forward pass
for cell_name in ["RNN", "LSTM", "GRU"]:
    model = RecurrentClassifier(cell_type=cell_name)
    output = model(x)
    print(f"[{cell_name} Classifier] Output Logits Shape: {output.shape}")
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The PyTorch modules define recurrent networks.
- Setting `bidirectional=True` concatenates the forward and backward hidden state vectors, outputting a vector of dimension `hidden_dim * 2` before classification.
"""))
    
    nb['cells'] = cells
    return nb

def build_07_attention():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 07_attention: Query-Key-Value Matrix Calculations from Scratch

This notebook programmatically computes Self-Attention Query-Key-Value dot-product matrix transformations, demonstrating the variance scaling effect of dividing by $\sqrt{d_k}$.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import torch
import torch.nn.functional as F

# Sequence length L=3, dimension d_k=64
torch.manual_seed(42)
L, d_k = 3, 64

# Simulating Query and Key inputs
Q = torch.randn(L, d_k)
K = torch.randn(L, d_k)
V = torch.randn(L, d_k)

# 1. Unscaled Attention
scores_unscaled = torch.matmul(Q, K.T)
weights_unscaled = F.softmax(scores_unscaled, dim=-1)

# 2. Scaled Attention (dividing by sqrt(d_k))
scaling_factor = d_k ** 0.5
scores_scaled = scores_unscaled / scaling_factor
weights_scaled = F.softmax(scores_scaled, dim=-1)

print("=== Raw Similarity Scores ===")
print(scores_unscaled)

print("\n=== Unscaled Softmax Attention Weights (Variance is high, scores saturate) ===")
print(weights_unscaled)
print("Weights variance:", torch.var(weights_unscaled).item())

print("\n=== Scaled Softmax Attention Weights (Normalized variance, sensitive gradients) ===")
print(weights_scaled)
print("Weights variance:", torch.var(weights_scaled).item())
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- Unscaled dot products yield larger values, pushing Softmax inputs into saturating regions where the output weights approach $0$ or $1$, causing vanishing gradients.
- Dividing by $\sqrt{d_k}$ restores unit variance, keeping the weights in a range where gradients can flow during training.
"""))
    
    nb['cells'] = cells
    return nb

def build_08_nlp_pipeline():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 08_nlp_pipeline: End-to-End Production Debugging Loop

This notebook implements an end-to-end NLP classification pipeline, monitors for data/concept drift, detects errors, and applies a diagnostic improvement patch.
"""))
    
    cells.append(nbf.v4.new_code_cell(r"""import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Training data (standard language)
train_data = ["I love this book", "this movie is awesome", "bad experience", "very horrible service"]
train_labels = [1, 1, 0, 0] # 1=positive, 0=negative

# Fit vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data)

# Fit classifier
clf = LogisticRegression()
clf.fit(X_train, train_labels)
print("Spam/Sentiment Classifier trained successfully.")

# 2. Simulate Production Inference with Data Drift (e.g. emojis and slang)
production_inputs = [
    "love it! 😍",
    "horrible service 😡",
    "awesome deal! 🔥",
    "so bad 💀"
]
production_labels = [1, 0, 1, 0]

X_prod = vectorizer.transform(production_inputs)
predictions = clf.predict(X_prod)

# 3. Calculate Performance Metrics
print("\n--- Production Metrics ---")
print(classification_report(production_labels, predictions, target_names=["negative", "positive"]))

# 4. Error Analysis & Model Improvement Loop
print("\n--- Error Analysis Diagnostic ---")
for text, gold, pred in zip(production_inputs, production_labels, predictions):
    if gold != pred:
        print(f"FAIL: Text '{text}' (Gold: {gold}, Predicted: {pred})")
        print("Reason: Out-of-Vocabulary slang / emoji features.")

# Model Improvement: Add training samples containing emojis and retrain
improved_train_data = train_data + ["this is so bad 💀", "awesome product! 🔥"]
improved_labels = train_labels + [0, 1]

# Retrain
vectorizer_imp = TfidfVectorizer()
X_train_imp = vectorizer_imp.fit_transform(improved_train_data)
clf_imp = LogisticRegression()
clf_imp.fit(X_train_imp, improved_labels)

# Re-evaluate
X_prod_imp = vectorizer_imp.transform(production_inputs)
predictions_imp = clf_imp.predict(X_prod_imp)

print("\n--- Improved Post-Patch Metrics ---")
print(classification_report(production_labels, predictions_imp, target_names=["negative", "positive"]))
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""### Output Explanation
- The initial model fails to classify texts containing emojis because it has never seen them during training (Data Drift).
- The debugging loop detects these classification errors, updates the training data with representative examples, and retrains the model to resolve the OOV failures.
"""))
    
    nb['cells'] = cells
    return nb

if __name__ == "__main__":
    output_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals\notebooks"
    os.makedirs(output_dir, exist_ok=True)
    
    builders = [
        ("01_text_preprocessing.ipynb", build_01_text_preprocessing),
        ("02_bag_of_words_tfidf.ipynb", build_02_bag_of_words_tfidf),
        ("03_word2vec.ipynb", build_03_word2vec),
        ("04_glove_fasttext.ipynb", build_04_glove_fasttext),
        ("05_ngram_language_models.ipynb", build_05_ngram_language_models),
        ("06_rnn_lstm_gru.ipynb", build_06_rnn_lstm_gru),
        ("07_attention.ipynb", build_07_attention),
        ("08_nlp_pipeline.ipynb", build_08_nlp_pipeline)
    ]
    
    for filename, builder in builders:
        nb_path = os.path.join(output_dir, filename)
        nb = builder()
        run_and_save(nb, nb_path)
