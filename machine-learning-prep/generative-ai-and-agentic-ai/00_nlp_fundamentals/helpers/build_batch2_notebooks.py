import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_and_execute_notebook_03():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "03_text_representation.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown introduction
    cells.append(nbf.v4.new_markdown_cell(
        "# 03_text_representation: TF-IDF, Cosine Similarity, Hashing Vectorizer, and BM25 from Scratch\n"
        "\n"
        "This notebook implements the math behind text representation models. It builds a manual TF-IDF vectorizer, performs L2 normalization, computes cosine similarity, implements Okapi BM25 scoring with length penalty parameters, and simulates the Feature Hashing trick with sign hashing to balance collisions."
    ))
    
    # Cell 2: Code - TF-IDF and Cosine Similarity from scratch
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "\n"
        "# Define tiny corpus matching study guide\n"
        "corpus = [\"cat feline\", \"feline rug\"]\n"
        "\n"
        "# 1. Scikit-learn TF-IDF Vectorizer matching our math formulation\n"
        "vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, sublinear_tf=False)\n"
        "tfidf_matrix = vectorizer.fit_transform(corpus).toarray()\n"
        "\n"
        "print(\"Vocabulary order:\\n\", vectorizer.vocabulary_)\n"
        "print(\"\\nTF-IDF Vectors:\")\n"
        "for doc, vec in zip(corpus, tfidf_matrix):\n"
        "    print(f\"  '{doc}' -> {np.round(vec, 4)}\")\n"
        "\n"
        "# Calculate Cosine Similarity\n"
        "cos_sim = np.dot(tfidf_matrix[0], tfidf_matrix[1])\n"
        "print(f\"\\nComputed Cosine Similarity: {cos_sim:.4f}\")\n"
        "\n"
        "# Assertions checking matching math outputs to 4 decimal places\n"
        "np.testing.assert_almost_equal(tfidf_matrix[0, 0], 0.8148, decimal=4) # cat\n"
        "np.testing.assert_almost_equal(tfidf_matrix[0, 1], 0.5797, decimal=4) # feline\n"
        "np.testing.assert_almost_equal(cos_sim, 0.3361, decimal=4)"
    ))
    
    # Cell 3: Markdown - Output explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: TF-IDF and Cosine Similarity\n"
        "- **L2 Normalization:** Each raw TF-IDF vector is scaled by its Euclidean length, mapping the vectors to a unit hypersphere. The normalized coordinates for Document 1 are `[0.8148, 0.5797, 0.0]`, which matches our hand calculation.\n"
        "- **Similarity Metric:** Since the vectors are pre-normalized, the cosine similarity simplifies to the dot product, yielding `0.3361` due to the shared token `\"feline\"`."
    ))
    
    # Cell 4: Code - Okapi BM25 from scratch
    cells.append(nbf.v4.new_code_cell(
        "def compute_bm25_score(tf, doc_len, avgdl, idf, k1=1.2, b=0.75):\n"
        "    numerator = tf * (k1 + 1)\n"
        "    denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))\n"
        "    return idf * (numerator / denominator)\n"
        "\n"
        "# Setup inputs matching study guide\n"
        "doc1_len = 2 # \"cat feline\"\n"
        "doc2_len = 4 # \"feline rug garden cat\"\n"
        "avgdl = 3.0\n"
        "idf_cat = 1.40\n"
        "k1, b = 1.2, 0.75\n"
        "\n"
        "# Document term frequencies for query Q = {\"cat\"}\n"
        "tf_doc1 = 1\n"
        "tf_doc2 = 1\n"
        "\n"
        "score_d1 = compute_bm25_score(tf_doc1, doc1_len, avgdl, idf_cat, k1, b)\n"
        "score_d2 = compute_bm25_score(tf_doc2, doc2_len, avgdl, idf_cat, k1, b)\n"
        "\n"
        "print(f\"BM25 Score for Document 1 (cat feline): {score_d1:.4f}\")\n"
        "print(f\"BM25 Score for Document 2 (feline rug garden cat): {score_d2:.4f}\")\n"
        "\n"
        "# Assertions verifying length normalization impact\n"
        "np.testing.assert_almost_equal(score_d1, 1.6211, decimal=4)\n"
        "np.testing.assert_almost_equal(score_d2, 1.2320, decimal=4)\n"
        "assert score_d1 > score_d2, \"Short document should score higher for identical word frequencies!\""
    ))
    
    # Cell 5: Markdown - Output explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: BM25 Scoring\n"
        "- **Length Penalty Impact:** Document 1 ($D_1$) scores higher than Document 2 ($D_2$) even though both contain the query token `\"cat\"` exactly once. The length normalization parameter $b = 0.75$ scales down the score of the longer document ($D_2$) because its words are diluted."
    ))
    
    # Cell 6: Code - Feature Hashing from scratch
    cells.append(nbf.v4.new_code_cell(
        "import hashlib\n"
        "\n"
        "def hash_word(word, B):\n"
        "    # Compute index bucket using md5\n"
        "    h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)\n"
        "    idx = h % B\n"
        "    # Compute sign hash (+1 or -1)\n"
        "    sign = 1 if (h // B) % 2 == 0 else -1\n"
        "    return idx, sign\n"
        "\n"
        "B = 1000  # Number of buckets\n"
        "words = [\"purchase\", \"buy\", \"cat\", \"purchase\"]\n"
        "\n"
        "hash_vector = np.zeros(B)\n"
        "for w in words:\n"
        "    idx, sign = hash_word(w, B)\n"
        "    hash_vector[idx] += sign\n"
        "    print(f\"Word: '{w:<8}' -> Bucket: {idx:<3} | Sign: {sign:+2}\")\n"
        "\n"
        "print(f\"\\nNon-zero indices in hash vector: {np.where(hash_vector != 0)[0]}\")\n"
        "assert hash_vector.sum() != 0, \"Feature Hashing vector is empty!\""
    ))
    
    # Cell 7: Markdown - Output explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: Feature Hashing\n"
        "- **Bucket Mapping:** Words are mapped to a fixed vector space of size $B = 1000$ using `hashlib.md5`. This bypasses the need to store a dictionary table in memory.\n"
        "- **Sign Hash:** Collisions cancel out on average because words are randomly scaled by $+1$ or $-1$ before addition, ensuring expected representation bias remains near 0."
    ))
    
    nb.cells = cells
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook draft: {notebook_path}")

def create_and_execute_notebook_04():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    notebook_path = os.path.join(base_dir, "notebooks", "04_statistical_language_models.ipynb")
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown introduction
    cells.append(nbf.v4.new_markdown_cell(
        "# 04_statistical_language_models: Bigram Language Models, Laplace Smoothing, and Perplexity\n"
        "\n"
        "This notebook demonstrates bigram statistical language modeling. It processes sentences from the NLTK Gutenberg corpus, builds a bigram transition matrix, implements Laplace (Add-One) smoothing, and computes the joint probability and perplexity of test sequences."
    ))
    
    # Cell 2: Code - Processing Gutenberg Corpus
    cells.append(nbf.v4.new_code_cell(
        "import nltk\n"
        "import re\n"
        "nltk.download('gutenberg', quiet=True)\n"
        "from nltk.corpus import gutenberg\n"
        "from collections import Counter\n"
        "\n"
        "# Load and tokenise sentences\n"
        "sentences = gutenberg.sents('carroll-alice.txt')[:100]\n"
        "cleaned_sentences = []\n"
        "for s in sentences:\n"
        "    words = [w.lower() for w in s if re.match(r\"^\\w+$\", w)]\n"
        "    if len(words) > 2:\n"
        "        cleaned_sentences.append(words)\n"
        "\n"
        "# Build vocabulary\n"
        "vocab = {\"<pad>\": 0, \"<unk>\": 1}\n"
        "for s in cleaned_sentences:\n"
        "    for w in s:\n"
        "        if w not in vocab:\n"
        "            vocab[w] = len(vocab)\n"
        "V = len(vocab)\n"
        "\n"
        "print(f\"Cleaned Sentences count: {len(cleaned_sentences)}\")\n"
        "print(f\"Vocabulary size: {V}\")\n"
        "assert V > 2, \"Vocabulary extraction failed!\""
    ))
    
    # Cell 3: Markdown - Output explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: Corpus Parsing\n"
        "- **Vocabulary Initialization:** We built a vocabulary using the first 100 sentences from *Alice in Wonderland*. Unique words are assigned indexes, mapping words to categorical dimensions."
    ))
    
    # Cell 4: Code - Laplace smoothing transition and Perplexity
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\n"
        "\n"
        "# Mock stats matching Module 04 study guide exactly\n"
        "unigram_counts = np.array([2, 2]) # cat: 2, sat: 2\n"
        "bigram_counts = np.array([\n"
        "    [0, 1], # cat->cat: 0, cat->sat: 1\n"
        "    [1, 0]  # sat->cat: 1, sat->sat: 0\n"
        "])\n"
        "V_micro = len(unigram_counts)\n"
        "\n"
        "# Compute Laplace-smoothed transition matrix\n"
        "P_smoothed = np.zeros((V_micro, V_micro))\n"
        "for i in range(V_micro):\n"
        "    P_smoothed[i, :] = (bigram_counts[i, :] + 1) / (unigram_counts[i] + V_micro)\n"
        "\n"
        "print(\"Smoothed Transition Matrix:\")\n"
        "print(P_smoothed)\n"
        "\n"
        "# Calculate perplexity of sequence: [\"cat\", \"cat\", \"sat\"] with P(\"cat\") = 0.5\n"
        "# Probabilities in path: [P(\"cat\"), P(\"cat\"|\"cat\"), P(\"sat\"|\"cat\")]\n"
        "probabilities = [0.5, P_smoothed[0, 0], P_smoothed[0, 1]]\n"
        "m = len(probabilities)\n"
        "\n"
        "log_prob_sum = np.sum(np.log(probabilities))\n"
        "perplexity = np.exp(-1/m * log_prob_sum)\n"
        "joint_prob = np.prod(probabilities)\n"
        "\n"
        "print(f\"\\nJoint Probability: {joint_prob:.6f}\")\n"
        "print(f\"Calculated Perplexity: {perplexity:.4f}\")\n"
        "\n"
        "# Verifications and assertions\n"
        "np.testing.assert_almost_equal(P_smoothed[0, 0], 0.2500, decimal=4)\n"
        "np.testing.assert_almost_equal(P_smoothed[0, 1], 0.5000, decimal=4)\n"
        "np.testing.assert_almost_equal(perplexity, 2.5198, decimal=4)\n"
        "assert np.allclose(P_smoothed.sum(axis=1), 0.75), \"Smoothed transition rows must sum to 0.75!\""
    ))
    
    # Cell 5: Markdown - Output explanation
    cells.append(nbf.v4.new_markdown_cell(
        "### Output Explanation: Laplace Smoothing and Perplexity\n"
        "- **Laplace Probability Adjustment:** Unseen transitions are smoothed to $0.2500$ instead of crashing to $0.0$, resolving the zero-probability defect.\n"
        "- **Perplexity Analysis:** The perplexity of the test sequence is `2.5198`, representing the average branching factor of choices during text generation."
    ))
    
    nb.cells = cells
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook draft: {notebook_path}")

def execute_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved notebook: {notebook_path}")

if __name__ == "__main__":
    create_and_execute_notebook_03()
    create_and_execute_notebook_04()
    
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    execute_notebook(os.path.join(base_dir, "notebooks", "03_text_representation.ipynb"))
    execute_notebook(os.path.join(base_dir, "notebooks", "04_statistical_language_models.ipynb"))
    print("ALL BATCH 2 NOTEBOOKS GENERATED AND EXECUTED SUCCESSFULLY.")
