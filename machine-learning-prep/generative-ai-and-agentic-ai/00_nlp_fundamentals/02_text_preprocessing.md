# Module 02: Text Preprocessing & Subword Tokenization

Preprocessing translates unstructured text strings into normalized token sequences. This module details classical cleaning methods, compares stemming vs. lemmatization, and explains the step-by-step mechanics of modern subword tokenization.

---

## 1. Modern Subword Tokenization Algorithms

Modern language models avoid word-level and character-level limitations by building subword vocabularies. The three primary subword algorithms differ in how they build and prune vocabularies:

### 1. Byte-Pair Encoding (BPE)
BPE (used in GPT models and LLaMA) builds its vocabulary from the bottom up, iteratively merging the most frequent adjacent character pairs.

#### Step-by-Step Merge Example:
Consider a tiny training corpus with token counts:
- `"h u g _"` (appears 10 times)
- `"p u g _"` (appears 5 times)
- `"h u g s _"` (appears 5 times)

1. **Initialize Vocabulary**: Populate vocabulary with unique characters:
   `["h", "u", "g", "p", "s", "_"]`
2. **Find Most Frequent Pair**:
   - `(h, u)` appears $10 + 5 = 15$ times.
   - `(u, g)` appears $10 + 5 + 5 = 20$ times.
   - `(g, _)` appears $10 + 5 = 15$ times.
3. **Merge the Top Pair**:
   - The most frequent pair is `(u, g)`. Merge them into a new token `ug`.
   - Updated corpus tokens: `"h ug _"`, `"p ug _"`, `"h ug s _"`
   - Updated vocabulary: `["h", "u", "g", "p", "s", "_", "ug"]`
4. **Iterate**:
   - In the next iteration, the most frequent pair is `(h, ug)`, appearing 15 times. Merge them into a new token `hug`.
   - Final vocabulary includes: `["ug", "hug"]`

---

### 2. WordPiece
WordPiece (used in BERT) is a bottom-up tokenizer similar to BPE, but instead of merging by raw frequency, it merges pairs that maximize the likelihood of the corpus under a probabilistic unigram model.

#### The Intuitive Scoring Ratio:
To decide which pair to merge, WordPiece evaluates:
$$\text{Score}(A, B) = \frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$$

- *Intuition*: This ratio measures how often $A$ and $B$ appear together compared to their independent occurrences.
- *Example*: The characters `h` and `u` are merged because their combination `hu` occurs far more frequently than their independent probabilities would suggest, whereas unrelated adjacent characters receive low scores.

---

### 3. Unigram Language Modeling
Unigram (used in T5 and SentencePiece) operates in a top-down manner. It starts with a massive vocabulary and iteratively prunes the least useful tokens.

#### Step-by-Step Pruning:
1. **Initialize**: Define a very large vocabulary containing all characters and the most common substrings from the training corpus.
2. **Estimate Probabilities**: Estimate the probability of each token in the vocabulary based on frequency.
3. **Compute Loss**: For each token, calculate how much the overall corpus likelihood would drop if that token were removed.
4. **Prune**: Remove the bottom $10\text{--}20\%$ of tokens that result in the smallest reduction in corpus likelihood.
5. **Iterate**: Repeat steps 2-4 until the vocabulary shrinks to the target budget (e.g. $32,000$ tokens).

---

## 2. Stemming vs. Lemmatization

![Lexical Reduction Latency](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/stem_vs_lemma_latency.png)

> [!NOTE]
> **Plot Explanation & Intuition: Lexical Reduction Latency**
> This chart compares the execution latency of Porter Stemmer vs. WordNet Lemmatizer per 1,000 tokens. 
> - **Porter Stemmer** executes in sub-2ms time because it relies on simple, heuristic suffix-chopping rules (e.g. slicing `"ing"` or `"ies"` based on character lengths) without consulting external databases.
> - **WordNet Lemmatizer** requires a significantly higher latency ($\approx 5$ms) because it relies on dictionary lookups, grammatical validation, and part-of-speech context tags to resolve words to their canonical base form (lemma).
> - **Production Takeaway**: In high-throughput streaming systems (like customer ticket triage), stemming is preferred for speed if morphological precision is not critical. If POS disambiguation is vital (e.g. distinguishing `"saw"` as a noun vs. verb), lemmatization is required despite the $4\times$ latency penalty.

Before vectorization, classical pipelines reduce morphological variations of words to a common base form:

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
  <div style="text-align: center; margin-bottom: 15px;">
    <div style="display: inline-block; background-color: #ffffff; padding: 8px 24px; border: 1px solid #cbd5e1; border-radius: 20px; font-family: monospace; font-size: 15px; font-weight: bold; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">"studies" / "studying"</div>
    <div style="margin: 10px auto 0 auto; width: 2px; height: 16px; border-left: 2px solid #cbd5e1;"></div>
  </div>
  
  <div style="display: flex; justify-content: space-between; gap: 20px;">
    <!-- Stemming Path -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
      <div style="background-color: #3b82f6; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 13px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">Stemming (Porter)</div>
      <div style="color: #64748b; font-size: 11.5px; margin-bottom: 12px;">Heuristic Suffix Truncation</div>
      <div style="font-size: 16px; color: #cbd5e1; margin-bottom: 12px;">&darr;</div>
      <div style="font-family: monospace; font-size: 16px; font-weight: bold; color: #dc2626; margin-bottom: 8px;">"studi"</div>
      <div style="color: #475569; font-size: 11.5px; line-height: 1.4;">Fast, rule-based suffix chopping. Often produces non-dictionary words.</div>
    </div>
    
    <!-- Lemmatization Path -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
      <div style="background-color: #8b5cf6; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 13px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">Lemmatization (WordNet)</div>
      <div style="color: #64748b; font-size: 11.5px; margin-bottom: 12px;">Morphological Lexicon Lookup</div>
      <div style="font-size: 16px; color: #cbd5e1; margin-bottom: 12px;">&darr;</div>
      <div style="font-family: monospace; font-size: 16px; font-weight: bold; color: #16a34a; margin-bottom: 8px;">"study"</div>
      <div style="color: #475569; font-size: 11.5px; line-height: 1.4;">Slower, dictionary-backed check. Always preserves valid root words.</div>
    </div>
  </div>
</div>

- **Stemming (Porter Stemmer)**: A fast, rule-based heuristic that chops off word prefixes and suffixes.
  - *Failure mode*: Over-stemming (collapsing words with different meanings: `"organization"` and `"organs"` both map to `"organ"`) or under-stemming (failing to merge `"alumnus"` and `"alumni"`).
- **Lemmatization (WordNet)**: Uses grammatical tags (POS tagging) and dictionary tables to resolve words to their canonical base forms (lemmas).
  - *Example*: `"better"` maps to `"good"`, `"was"` maps to `"be"`.
  - *Trade-off*: Slower and memory-intensive due to dictionary lookups and POS dependencies.

---

## 3. Classical Text Preprocessing Pipeline Steps

Before applying modeling or vectorization layers, classical NLP pipelines clean, split, and normalize raw inputs. The primary preprocessing steps and techniques include:

### 1. Tokenization (Sentence and Word Level)
Tokenization splits a continuous character stream into semantic blocks:
- **Sentence Tokenization (Segmentation)**: Divides a text block into individual sentences (e.g. using NLTK's `sent_tokenize` or spaCy, which handle punctuation ambiguities like `"Dr. Smith bought an apple."`).
- **Word Tokenization**: Divides sentences into words. Standard word tokenizers handle contraction splittings (e.g. splitting `"don't"` into `["do", "n't"]` or `["dont"]` depending on the grammar template).

### 2. Case Normalization
Case normalization maps all characters to lowercase.
- *Production Trade-offs*: Reduces the vocabulary search space (merging `"Apple"`, `"APPLE"`, and `"apple"` into a single index). However, it degrades performance for Named Entity Recognition (NER), Sentiment Analysis, and POS Tagging because it discards critical structural cues (e.g. distinguishing `"Apple"` the company from `"apple"` the fruit).

### 3. Stopword Removal
Stopwords are high-frequency, low-semantic-value words (e.g. `"is"`, `"the"`, `"at"`, `"which"`).
- *Production Trade-offs*: Essential for sparse retrieval indices (TF-IDF, BM25) and classical classifiers (Naïve Bayes, SVM) to reduce dimensionality and speed up database query searches.
- *Negative Trade-offs*: **Do NOT use stopword removal for sequence models (RNNs, LSTMs, Transformers)**. Removing stopwords destroys syntactic sequence context, positional relationships, and grammatical structure, degrading model outputs.

### 4. Noise Cleaning & Text Normalization
- **HTML/Markdown Stripping**: Removing raw markup tags (e.g. using `BeautifulSoup` or regex patterns like `r"<[^>]*>"`).
- **Contraction Expansion**: Expanding abbreviations and contractions (e.g., mapping `"I've"` to `"I have"`) to align token usage across documents.
- **Emoji Handling**: Stripping emojis or converting them into descriptive word tokens (e.g. mapping `😊` to `"[happy_face]"`) to preserve sentiment in user reviews.

---

## 4. Regular Expressions and Unicode Normalization

Uncleaned raw strings contain encoding anomalies and noise that must be filtered out:

- **Common Regex Cleanups**:
    - URL removal: `re.sub(r"https?://\S+", "", text)`
    - Punctuation removal: `re.sub(r"[^\w\s]", "", text)`
- **Unicode Normalization**: Ensures consistent character representations. For example, the accented character `é` can be represented as:
    - **NFC (Composition)**: Single code point `\u00e9`.
    - **NFD (Decomposition)**: Decomposed into base `e` (`\u0065`) and combining accent character `´` (`\u0301`).

---

> [!IMPORTANT]
> **Production Insight: Unicode Vocabulary Alignment**
> Always apply NFC normalization (e.g. `unicodedata.normalize('NFC', text)`) during both training and inference. If user inputs use NFD encoding (decomposed characters) while the model was trained on NFC, the tokenizer will split the characters differently, mapping them to incorrect token indices and degrading predictions.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Transforms raw, inconsistent character streams into normalized, uniform inputs, reducing vocabulary size and vocabulary sparsity.
- **Why was it introduced?**
  Introduced to prevent out-of-vocabulary (OOV) errors and compress vocabulary matrices for efficient training.
- **What are its limitations?**
  Extreme preprocessing (e.g. discarding casing and punctuation) discards semantic context (such as sentiment cues, code syntax symbols, or structural boundaries).
- **Computational Complexity (Time & Memory)**
    - **BPE Encoding Time**: $O(L \cdot \log |V|)$ where $L$ is sequence length.
    - **Lemmatization Time**: $O(N \cdot D)$ where $N$ is word count and $D$ represents dictionary search depth.
- **Component Variable Denotation Legend**
    - $L$: Sequence token length.
    - $|V|$: Target vocabulary size.
    - $D$: Dictionary lookup size.
- **Production Use Cases**
    - Subword tokenization pipelines in multilingual LLMs.
    - Normalizing raw user queries (e.g. casing and Unicode accents) before search indexing.
- **Follow-up questions interviewers ask**
    - *How does BPE handle a word with unknown characters during inference?* (Characters not seen during training are mapped to character-level bytes or fallback tokens using a byte-fallback vocabulary, preventing `<unk>` errors).
    - *Why should you avoid lowercase normalization for Named Entity Recognition (NER)?* (Named entities like `"Apple"` (company) depend heavily on capitalization to differentiate them from `"apple"` (fruit)).
