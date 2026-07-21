# Module 02: Text Preprocessing & Modern Tokenization Paradigms

This study guide details text cleaning, normalization, Stop Words trade-offs, Stemming vs. Lemmatization, Word vs. Character vs. Subword tokenization, mathematical algorithms for Byte Pair Encoding (BPE), WordPiece, SentencePiece, step-by-step BPE hand-calculations, production Python code, failure modes, and interview flashcards.

> **Notebook Companion**: [02_text_preprocessing_and_tokenization.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/02_text_preprocessing_and_tokenization.ipynb)

---

## 1. Text Preprocessing Taxonomy

Text preprocessing transforms raw, noisy human text into normalized, discrete tokens suitable for numerical feature extraction or neural network embeddings.

```text
Raw Text ──► Noise Removal & Unicode Normalization ──► Tokenization ──► Stopword / Stemming Filtering ──► Normalized Tokens
```

### Preprocessing Operations:
1. **Noise Removal**: Stripping HTML/XML tags, non-printable ASCII codes, and log metadata timestamps.
2. **Case Normalization**: Lowercasing text to eliminate duplicate vocabulary entries (`"Python"` vs `"python"`).
   - *Production Caution*: Preserve casing for Named Entity Recognition (NER) to distinguish `"Apple"` (company) from `"apple"` (fruit).
3. **Stop Words Filtering**: Removing high-frequency words (`"the"`, `"is"`, `"at"`, `"which"`) that carry minimal domain specificity in bag-of-words or TF-IDF models.
   - *Production Caution*: **DO NOT** remove stop words in Sentiment Analysis (e.g., removing *"not"* converts *"not good"* to *"good"*) or Machine Translation.

---

## 2. Stemming vs. Lemmatization

```text
Technique       Mechanism                            Example: "better"  Example: "meeting" (Noun)  Processing Speed  Production Role
--------------------------------------------------------------------------------------------------------------------------------------
Stemming        Heuristic suffix chopping (Regex)     "better"           "meet"                     Ultra-Fast        Legacy search engines
Lemmatization   Morphological & POS dictionary lookup "good"             "meeting"                  Moderate          Precision NLP pipelines
Subword BPE     Frequency-based subword splitting     "bet" + "ter"      "meet" + "ing"             Fast (Lookup)     Modern Transformers/LLMs
```

### 1. Stemming (Porter / Lancaster)
Stemming applies crude, rule-based algorithmic heuristics to slice off word suffixes (e.g., `-ing`, `-ed`, `-es`, `-ly`).
- **Over-stemming Error**: Slicing distinct words to the same incorrect stem (e.g., `"universe"`, `"university"`, `"universal"` $\rightarrow$ `"univers"`).
- **Under-stemming Error**: Failing to reduce morphologically related words to the same root (e.g., `"alumnus"` and `"alumni"` stay separate).

### 2. Lemmatization (WordNet / spaCy)
Lemmatization uses full morphological analysis and Part-of-Speech (POS) context tags to map words to their canonical dictionary root (**lemma**):

$$\text{Lemma}(\text{word}, \text{POS}) = \text{DictionaryLookup}(\text{word}, \text{POS})$$

- `"meeting"` $\xrightarrow{\text{POS = NOUN}}$ `"meeting"`
- `"meeting"` $\xrightarrow{\text{POS = VERB}}$ `"meet"`
- `"better"` $\xrightarrow{\text{POS = ADJ}}$ `"good"`

---

## 3. Tokenization Paradigms: Word vs. Character vs. Subword

```text
Paradigm            Vocabulary Size (|V|)   Sequence Length (T)  Out-Of-Vocabulary (OOV) Rate  Primary Use Case
----------------------------------------------------------------------------------------------------------------------
Word-Level          Large (500,000+)        Short                High (Fails on rare words)    Classical TF-IDF / GloVe
Character-Level     Tiny (100 - 250)        Extremely Long       Zero (100% coverage)          Char-RNN / Spellcheckers
Subword (BPE/Piece) Optimal (30,000-100,000) Balanced             Zero (Splits into sub-units)  Transformers, GPT-4, Llama
```

---

## 4. Subword Tokenization Algorithms & Mathematics

Subword tokenization solves the vocabulary bottleneck by decomposing rare or out-of-vocabulary words into smaller, frequently occurring sub-units (e.g., `"unexplainable"` $\rightarrow$ `["un", "explain", "able"]`).

### 1. Byte Pair Encoding (BPE) (GPT-2, GPT-4, Llama 3)
BPE starts with a base vocabulary of individual characters and iteratively merges the most frequent adjacent pair of symbols in the corpus.

- **Objective Function**: Maximize bigram co-occurrence frequency:

  $$\text{Pair}_{\text{best}} = \arg\max_{(A, B)} \text{Count}(A, B)$$

- **Algorithm Steps**:
  1. Tokenize corpus into individual characters with an end-of-word symbol `</w>`.
  2. Count frequencies of all adjacent symbol pairs $(A, B)$.
  3. Merge the most frequent pair $(A, B) \rightarrow AB$ and add $AB$ to vocabulary.
  4. Repeat until target vocabulary size $|V|_{\text{target}}$ is reached.

### 2. WordPiece (BERT)
WordPiece merges symbol pairs based on likelihood maximization rather than raw frequency count:

$$\text{Score}(A, B) = \frac{\text{Count}(AB)}{\text{Count}(A) \times \text{Count}(B)}$$

Merging $A$ and $B$ prioritizes pairs whose joint occurrence is significantly higher than expected by independent chance.

### 3. SentencePiece / Unigram Language Model (T5, LLaMA)
Instead of starting with small characters and merging upwards, the Unigram model starts with a massive candidate vocabulary and iteratively prunes subwords that minimize the increase in corpus perplexity loss:

$$\mathcal{L}_{\text{unigram}} = -\sum_{i=1}^N \log \sum_{x \in S(W_i)} P(x)$$

---

## 5. Step-by-Step BPE Hand Calculation Example (Andrew Ng Style)

Suppose we have a tiny training corpus with word frequencies:
- `"cost </w>"` : 4 times
- `"costs </w>"` : 2 times
- `"costing </w>"` : 3 times
- `"cozy </w>"` : 1 time

### Initial Base Vocabulary:
$$\text{Base Chars} = \{\text{'c'}, \text{'o'}, \text{'s'}, \text{'t'}, \text{'s'}, \text{'i'}, \text{'n'}, \text{'g'}, \text{'z'}, \text{'y'}, \text{'</w>'}\}$$

### Step 1: Count Adjacent Pairs across Corpus
- Pair `('c', 'o')`: $4 + 2 + 3 + 1 = 10$ occurrences.
- Pair `('o', 's')`: $4 + 2 + 3 = 9$ occurrences.
- Pair `('s', 't')`: $4 + 2 + 3 = 9$ occurrences.
- Pair `('t', '</w>')`: $4$ occurrences.
- Pair `('t', 'i')`: $3$ occurrences.

Highest Frequency Pair: **`('c', 'o')`** with Count = 10.
- **Merge 1**: `('c', 'o')` $\rightarrow$ **`"co"`**
- Corpus updated: `"co s t </w>"`, `"co s t s </w>"`, `"co s t i n g </w>"`, `"co z y </w>"`

### Step 2: Recalculate Pair Frequencies
- Pair `('co', 's')`: $4 + 2 + 3 = 9$ occurrences.
- Pair `('s', 't')`: $4 + 2 + 3 = 9$ occurrences.
- Pair `('co', 'z')`: 1 occurrence.

Highest Frequency Pair: Tie between `('co', 's')` and `('s', 't')` (Count = 9). Select **`('co', 's')`**.
- **Merge 2**: `('co', 's')` $\rightarrow$ **`"cos"`**
- Corpus updated: `"cos t </w>"`, `"cos t s </w>"`, `"cos t i n g </w>"`, `"co z y </w>"`

### Step 3: Recalculate Pair Frequencies
- Pair `('cos', 't')`: $4 + 2 + 3 = 9$ occurrences.
- **Merge 3**: `('cos', 't')` $\rightarrow$ **`"cost"`**
- Corpus updated: `"cost </w>"`, `"cost s </w>"`, `"cost i n g </w>"`, `"co z y </w>"`

**Resulting Learned Subwords**: `["co", "cos", "cost"]` added to vocabulary! When encountering out-of-vocabulary word `"costless"`, BPE tokenizes it into `["cost", "l", "e", "s", "s"]`.

---

## 6. Production Python Code Implementation

```python
import tiktoken
from nltk.stem import PorterStemmer
import spacy

# 1. Stemming vs Lemmatization Demonstration
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

words = ["meeting", "better", "universities", "studies"]

print("=== 1. NLTK Stemming vs spaCy Lemmatization ===")
print(f"{'Word':<15} | {'Porter Stemmer':<15} | {'spaCy Lemmatizer (POS-aware)':<25}")
print("-" * 65)

for word in words:
    stem_val = stemmer.stem(word)
    doc = nlp(word)
    lemma_val = doc[0].lemma_
    print(f"{word:<15} | {stem_val:<15} | {lemma_val:<25}")

# 2. Modern OpenAI BPE Tokenization (tiktoken)
enc = tiktoken.get_encoding("cl100k_base") # Tokenizer used by GPT-4 and text-embedding-3
sample_text = "Unexplainable enterprise microservice architecture failure."

tokens = enc.encode(sample_text)
decoded_subwords = [enc.decode([t]) for t in tokens]

print("\n=== 2. OpenAI tiktoken BPE Subword Split ===")
print("Original Text:   ", repr(sample_text))
print("Token IDs:       ", tokens)
print("Subword Pieces:  ", decoded_subwords)
```

> [!NOTE]
> **Production Tokenization Alert:**
> - `cl100k_base` uses a 100,000 token subword vocabulary. Notice how complex words like `"Unexplainable"` are seamlessly tokenized into subword fragments `["Un", "explain", "able"]`, completely avoiding Out-Of-Vocabulary `<UNK>` fallbacks.

---

## 7. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Stopword Over-filtering**: Stripping stopwords prior to passing text into transformer LLMs ruins grammatical syntax and self-attention positioning.
   - *Fix*: Never strip stopwords for Transformer / LLM models; only strip stopwords for sparse TF-IDF / BM25 search engines.
2. **Tokenizer Vocabulary Mismatch**: Encoding prompt text using `cl100k_base` (GPT-4) and feeding token IDs into a model expecting `p50k_base` (Codex) causes corrupted token inputs.
   - *Fix*: Ensure exact tokenizer model consistency between data preprocessing and model inference pipelines.

---

## 8. Master Interview Flashcards & Questions

#### Q1: Why do modern Large Language Models use Subword Tokenization (BPE/WordPiece) instead of Word-Level or Character-Level Tokenization?
- **Answer:** Word-level tokenization requires huge vocabularies ($|V| > 500k$) and still suffers from Out-Of-Vocabulary (OOV) errors. Character-level tokenization has tiny vocabulary but produces long sequences ($T$), causing quadratic self-attention memory overhead $O(T^2)$. Subword tokenization balances vocabulary size ($\approx 32k - 100k$) and sequence length while guaranteeing 0% OOV errors.

#### Q2: Compare Byte Pair Encoding (BPE) vs. WordPiece merging criteria.
- **Answer:** BPE merges adjacent symbol pairs based purely on raw pair co-occurrence frequency ($\arg\max \text{Count}(A, B)$). WordPiece merges symbol pairs by maximizing the likelihood score $\frac{\text{Count}(AB)}{\text{Count}(A) \times \text{Count}(B)}$, prioritizing pairs that occur together significantly more often than expected by random chance.

#### Q3: What is the primary difference between Stemming and Lemmatization?
- **Answer:** Stemming uses heuristic, rule-based suffix slicing without considering Part-of-Speech (POS) or grammatical context, often producing non-real word stems (`"univers"`). Lemmatization uses morphological dictionary lookup and POS tags to return the true canonical dictionary root (`"good"` for `"better"`).
