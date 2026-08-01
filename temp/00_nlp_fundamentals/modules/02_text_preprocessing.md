# Module 02: Text Preprocessing & Subword Tokenization

Preprocessing translates unstructured text strings into normalized token sequences. This module details classical cleaning methods, compares stemming vs. lemmatization, and explains the step-by-step mechanics of modern subword tokenization.

---

## 1. Morphological Basics

Before analyzing text normalization algorithms, we must understand the structure of human words:
- **Morpheme:** The smallest grammatical unit in a language that carries semantic meaning. Morphemes are the building blocks of words. For example, the word `"unfriendly"` contains three morphemes: `"un-"` (prefix meaning "not"), `"friend"` (noun root), and `"-ly"` (suffix forming an adjective).
- **Prefixes and Suffixes:** Affixes attached to the beginning (prefix) or end (suffix) of a root word to modify its meaning or grammatical class.
- **Inflectional Variations:** Modifications of a word to express different grammatical categories, such as tense, case, voice, aspect, person, number, or gender. For example, `"study"`, `"studies"`, and `"studying"` are inflectional variations of the same root verb.
- **Canonical Base Form (Lemma):** The standard dictionary form of a word, stripped of inflectional affixes. The lemma for `"studies"`, `"studied"`, and `"studying"` is the base form `"study"`.

---

## 2. Modern Subword Tokenization Algorithms

Modern language models avoid word-level and character-level limitations by building subword vocabularies. The three primary subword algorithms differ in how they build and prune vocabularies:

### 1. Byte-Pair Encoding (BPE)
BPE (used in GPT models and LLaMA) builds its vocabulary from the bottom up, iteratively merging the most frequent adjacent character pairs.

#### Step-by-Step Hand Calculation on a Micro-Corpus:
Consider a tiny training corpus with token counts:
- `"h u g _"` (appears 10 times)
- `"p u g _"` (appears 5 times)
- `"h u g s _"` (appears 5 times)

##### 1. Initialize Vocabulary:
Populate the vocabulary with the unique characters present in the corpus:
`Vocabulary = ["h", "u", "g", "p", "s", "_"]` (Vocabulary size $|V| = 6$)

##### 2. Iteration 1 - Find and Merge Most Frequent Pair:
Count all adjacent token pairs in the corpus:
- `(h, u)`: appears in `"h u g _"` (10 times) + `"h u g s _"` (5 times) = $10 + 5 = 15$ times.
- `(u, g)`: appears in `"h u g _"` (10) + `"p u g _"` (5) + `"h u g s _"` (5) = $10 + 5 + 5 = 20$ times.
- `(g, _)`: appears in `"h u g _"` (10) + `"p u g _"` (5) = $10 + 5 = 15$ times.
- `(p, u)`: appears in `"p u g _"` = 5 times.
- `(g, s)`: appears in `"h u g s _"` = 5 times.
- `(s, _)`: appears in `"h u g s _"` = 5 times.

**Selection:** The most frequent pair is `(u, g)` with a count of 20.
- **Merge:** Merge them into a new token `ug`.
- **Updated Corpus:** `"h ug _"` (10), `"p ug _"` (5), `"h ug s _"` (5)
- **Updated Vocabulary:** `["h", "u", "g", "p", "s", "_", "ug"]` (Vocabulary size $|V| = 7$)

##### 3. Iteration 2 - Iterate:
Count adjacent token pairs in the updated corpus:
- `(h, ug)`: appears in `"h ug _"` (10) + `"h ug s _"` (5) = 15 times.
- `(ug, _)`: appears in `"h ug _"` (10) + `"p ug _"` (5) = 15 times.
- `(p, ug)`: appears in `"p ug _"` = 5 times.
- `(ug, s)`: appears in `"h ug s _"` = 5 times.
- `(s, _)`: appears in `"h ug s _"` = 5 times.

**Selection:** There is a tie between `(h, ug)` and `(ug, _)`. We break ties systematically (e.g., character order or first occurrence). Let's merge `(h, ug)`.
- **Merge:** Merge them into a new token `hug`.
- **Updated Corpus:** `"hug _"` (10), `"p ug _"` (5), `"hug s _"` (5)
- **Updated Vocabulary:** `["h", "u", "g", "p", "s", "_", "ug", "hug"]` (Vocabulary size $|V| = 8$)

##### Findings & Interpretation:
BPE starts at the raw character level, ensuring $0\%$ out-of-vocabulary (OOV) errors because all base characters are in the vocabulary. By greedily merging the most frequent pairs, it builds common words (like `"hug"`) into single tokens. Unseen words at inference time (like `"pugs"`) are not mapped to `<unk>`; instead, they decompose cleanly to known subwords `["p", "ug", "s"]`, preserving the semantic roots.

---

### 2. WordPiece
WordPiece (used in BERT) is a bottom-up tokenizer similar to BPE, but instead of merging by raw frequency, it merges pairs that maximize the likelihood of the training corpus under a probabilistic unigram language model.

#### The Intuitive Scoring Ratio:
To decide which pair to merge, WordPiece evaluates:

$$\text{Score}(A, B) = \frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$$

#### Step-by-Step Hand Calculation:
Let's calculate and compare scores for two candidate pairs in a micro-corpus:
- Let the total corpus size be $N = 100$ words.
- **Candidate Pair 1: `(h, u)`**
  - Unigram counts: $\text{Count}(h) = 20$, $\text{Count}(u) = 30$
  - Co-occurrence count: $\text{Count}(h, u) = 15$
  - $$\text{Score}(h, u) = \frac{15}{20 \times 30} = \frac{15}{600} = 0.0250$$
- **Candidate Pair 2: `(p, u)`**
  - Unigram counts: $\text{Count}(p) = 5$, $\text{Count}(u) = 30$
  - Co-occurrence count: $\text{Count}(p, u) = 4$
  - $$\text{Score}(p, u) = \frac{4}{5 \times 30} = \frac{4}{150} \approx 0.0267$$

##### Findings & Interpretation:
Even though the pair `(h, u)` co-occurs more times in absolute terms ($15$ vs $4$), WordPiece selects and merges `(p, u)` first because its score is higher ($0.0267 > 0.0250$). 

The denominator $\text{Count}(A) \times \text{Count}(B)$ represents the expected probability of $A$ and $B$ appearing adjacent by pure chance. The scoring ratio measures the statistical correlation: how much more often do $A$ and $B$ appear together than independent chance would predict? This prevents WordPiece from merging common characters (like stopwords components) just because they are frequent, prioritizing tightly bound morpheme units.

---

### 3. Unigram Language Modeling
Unigram (used in T5 and SentencePiece) operates in a top-down manner. It starts with a massive vocabulary and iteratively prunes the least useful tokens.

#### Step-by-Step Pruning:
1. **Initialize:** Define a very large vocabulary containing all characters and the most common substrings from the training corpus.
2. **Estimate Probabilities:** Estimate the probability of each token in the vocabulary based on frequency.
3. **Compute Loss:** For each token, calculate how much the overall corpus likelihood would drop if that token were removed.
4. **Prune:** Remove the bottom $10\text{--}20\%$ of tokens that result in the smallest reduction in corpus likelihood.
5. **Iterate:** Repeat steps 2-4 until the vocabulary shrinks to the target budget (e.g. $32,000$ tokens).

---

## 3. Stemming vs. Lemmatization

![Lexical Reduction Latency](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/stem_vs_lemma_latency.png)

> [!NOTE]
> **Plot Explanation & Intuition: Lexical Reduction Latency**
> This chart compares the execution latency of Porter Stemmer vs. WordNet Lemmatizer per 1,000 tokens. 
> - **Porter Stemmer** executes in sub-2ms time because it relies on simple, heuristic suffix-chopping rules (e.g. slicing `"ing"` or `"ies"` based on character lengths) without consulting external databases.
> - **WordNet Lemmatizer** requires a significantly higher latency ($\approx 5$ms) because it relies on dictionary lookups, grammatical validation, and part-of-speech context tags to resolve words to their canonical base form (lemma).
> - **Production Takeaway**: In high-throughput streaming systems (like customer ticket triage), stemming is preferred for speed if morphological precision is not critical. If POS disambiguation is vital (e.g. distinguishing `"saw"` as a noun vs. verb), lemmatization is required despite the $4\times$ latency penalty.

Before vectorization, classical pipelines reduce morphological variations of words to a common base form. The algorithms work as follows:

### 1. Stemming (Porter Stemmer)
Stemming is a rule-based heuristic that truncates the ends of words using a sequence of suffix-chopping rules. The Porter Stemmer runs sequentially through five phases:
- **Step 1a Rules:**
  - `SSES` $\rightarrow$ `SS` (e.g., `caresses` $\rightarrow$ `caress`)
  - `IES` $\rightarrow$ `I` (e.g., `ponies` $\rightarrow$ `poni`)
  - `SS` $\rightarrow$ `SS` (e.g., `caress` $\rightarrow$ `caress`)
  - `S` $\rightarrow$ `Ø` (e.g., `cats` $\rightarrow$ `cat`)
- **Step 1b Rules:** Truncates plural verb endings like `-ed` or `-ing` (e.g., `walking` $\rightarrow$ `walk`, `plastered` $\rightarrow$ `plaster`).

*Failure Mode:* Because it is rule-based and does not use a dictionary, it frequently causes **over-stemming** (chopping unrelated words to the same stem: `"organization"` and `"organs"` both map to `"organ"`) or **under-stemming** (failing to link related words: `"alumnus"` and `"alumni"` remain separate).

### 2. Lemmatization (WordNet Lemmatizer)
Lemmatization uses a lexicon (dictionary lookup database) and morphological analysis to resolve words to their canonical base forms. It is dependent on Part-of-Speech (POS) tagging:
- **Without POS Tag:** If you pass the word `"saw"`, the lemmatizer defaults to treating it as a noun, returning `"saw"`.
- **With POS Tag (Verb):** If you pass `"saw"` with a verb POS tag, the lemmatizer looks up the lexicon and resolves it correctly to its base lemma `"see"`.

---

### Python Code Demonstration & Morphological Output Table

The following Python script demonstrates NLTK's `PorterStemmer` vs. `WordNetLemmatizer` on a set of morphological test cases:

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required lexical resources are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["studies", "studying", "saw", "leaves", "better"]

print(f"{'Original Word':<15} | {'Porter Stemmer':<15} | {'WordNet Lemmatizer':<20} | {'Interpretation'}")
print("-" * 75)
for word in words:
    stem = stemmer.stem(word)
    # Lemmatize without POS tag (defaults to Noun)
    lemma_noun = lemmatizer.lemmatize(word, pos='n')
    # Lemmatize with POS tag (verb or adjective)
    if word == "better":
        lemma_pos = lemmatizer.lemmatize(word, pos='a') # adjective
    else:
        lemma_pos = lemmatizer.lemmatize(word, pos='v') # verb
        
    print(f"{word:<15} | {stem:<15} | {lemma_pos:<20} | Stem: suffix-chop; Lemma: base dictionary word")
```

The resulting output matches the following native GFM comparison table:

| Original Word | Porter Stemmer | WordNet Lemmatizer (with POS) | Semantic & Algorithmic Interpretation |
| :--- | :--- | :--- | :--- |
| **"studies"** | `"studi"` | `"study"` | Stemmer chops ending; Lemmatizer resolves root y. |
| **"studying"** | `"studi"` | `"study"` | Both collapse variants, but Stemmer produces non-word `"studi"`. |
| **"saw"** | `"saw"` | `"see"` (as verb) | Lemmatizer uses POS tags to resolve verb `"saw"` $\rightarrow$ `"see"`. |
| **"leaves"** | `"leav"` | `"leave"` (as verb) | Lemmatizer resolves plural verb; Stemmer heuristics chop suffix. |
| **"better"** | `"better"` | `"good"` (as adj) | Lemmatizer maps comparative adjective to canonical base. |

---

## 4. Classical Text Preprocessing Pipeline Steps

Before applying modeling or vectorization layers, classical NLP pipelines clean, split, and normalize raw inputs. The primary preprocessing steps and techniques include:

### 1. Tokenization (Sentence and Word Level)
Tokenization splits a continuous character stream into semantic blocks:
- **Sentence Tokenization (Segmentation):** Divides a text block into individual sentences (e.g. using NLTK's `sent_tokenize` or spaCy, which handle punctuation ambiguities like `"Dr. Smith bought an apple."`).
- **Word Tokenization:** Divides sentences into words. Standard word tokenizers handle contraction splittings (e.g. splitting `"don't"` into `["do", "n't"]` or `["dont"]` depending on the grammar template).

### 2. Case Normalization
Case normalization maps all characters to lowercase.
- *Production Trade-offs:* Reduces the vocabulary search space (merging `"Apple"`, `"APPLE"`, and `"apple"` into a single index). However, it degrades performance for Named Entity Recognition (NER), Sentiment Analysis, and POS Tagging because it discards critical structural cues (e.g. distinguishing `"Apple"` the company from `"apple"` the fruit).

### 3. Stopword Removal
Stopwords are high-frequency, low-semantic-value words (e.g. `"is"`, `"the"`, `"at"`, `"which"`).
- *Production Trade-offs:* Essential for sparse retrieval indices (TF-IDF, BM25) and classical classifiers (Naïve Bayes, SVM) to reduce dimensionality and speed up database query searches.
- *Negative Trade-offs:* **Do NOT use stopword removal for sequence models (RNNs, LSTMs, Transformers)**. Removing stopwords destroys syntactic sequence context, positional relationships, and grammatical structure, degrading model outputs.

### 4. Noise Cleaning & Text Normalization
- **HTML/Markdown Stripping:** Removing raw markup tags (e.g. using `BeautifulSoup` or regex patterns like `r"<[^>]*>"`).
- **Contraction Expansion:** Expanding abbreviations and contractions (e.g., mapping `"I've"` to `"I have"`) to align token usage across documents.
- **Emoji Handling:** Stripping emojis or converting them into descriptive word tokens (e.g. mapping `😊` to `"[happy_face]"`) to preserve sentiment in user reviews.

---

## 5. Regular Expressions and Unicode Normalization

Uncleaned raw strings contain encoding anomalies and noise that must be filtered out:
- **Common Regex Cleanups:**
  - URL removal: `re.sub(r"https?://\S+", "", text)`
  - Punctuation removal: `re.sub(r"[^\w\s]", "", text)`
- **Unicode Normalization:** Ensures consistent character representations. For example, the accented character `é` can be represented as:
  - **NFC (Composition):** Single code point `\u00e9`.
  - **NFD (Decomposition):** Decomposed into base `e` (`\u0065`) and combining accent character `´` (`\u0301`).

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
