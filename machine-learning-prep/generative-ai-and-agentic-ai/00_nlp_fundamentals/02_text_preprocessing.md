# Module 02: Text Preprocessing & Subword Tokenization

Preprocessing translates unstructured text strings into normalized token tokens. This module details classical cleaning methods and explains the step-by-step mathematical algorithms behind modern subword tokenizers (BPE, WordPiece, Unigram).

---

## 1. Modern Subword Tokenization Algorithms

Modern language models avoid word-level and character-level limitations by building subword vocabularies. The three primary subword algorithms differ in how they build and prune vocabularies:

### 1. Byte-Pair Encoding (BPE)
BPE (used in GPT-2, GPT-3, GPT-4, LLaMA, RoBERTa) builds its vocabulary from the bottom up, iteratively merging the most frequent adjacent character pairs.

#### Step-by-Step Algorithm:
1. **Initialize Vocabulary**: Populate vocabulary $|V|$ with all unique characters occurring in the training corpus, plus an end-of-word marker (e.g. `</w>` or `_`).
2. **Decompose Corpus**: Split all words in the training corpus into characters. For example, the corpus `"low low low lower lower newest newest"` is represented as:
   `{'l o w </w>': 3, 'l o w e r </w>': 2, 'n e w e s t </w>': 2}`
3. **Count Pairs**: Scan the corpus to count all adjacent token pairs (e.g. `(l, o)`, `(o, w)`, `(e, s)`).
4. **Merge Most Frequent**: Find the single most frequent adjacent pair. Add it to the vocabulary as a merged token.
   - Example: The pair `(e, s)` appears 2 times in `"newest"`. The pair `(l, o)` appears 5 times. Thus, merge `l` and `o` into `lo`.
   - Update the corpus: `{'lo w </w>': 3, 'lo w e r </w>': 2, 'n e w e s t </w>': 2}`
5. **Iterate**: Repeat steps 3 and 4 until the vocabulary reaches the target size $|V|$ or the maximum frequency drops below a threshold.

---

### 2. WordPiece
WordPiece (used in BERT, DistilBERT, MobileBERT) is a bottom-up tokenizer similar to BPE, but instead of merging by raw frequency, it merges pairs that maximize the likelihood of the corpus under a probabilistic unigram model.

#### Step-by-Step Algorithm:
1. **Initialize Vocabulary**: Populate vocabulary $|V|$ with all individual characters and subwords.
2. **Calculate Merge Scores**: Compute a score for every adjacent pair of tokens $(A, B)$ using:
   $$\text{Score}(A, B) = \frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$$
   *Intuition*: This ratio measures how often $A$ and $B$ appear together compared to their independent occurrences. A high score means $A$ and $B$ are strongly coupled (e.g. `"h"` and `"ug"` in `"hug"`).
3. **Merge**: Add the pair $(A, B)$ with the highest score to the vocabulary.
4. **Iterate**: Repeat steps 2 and 3 until the target vocabulary budget is met.

---

### 3. Unigram Language Modeling
Unigram (used in T5, ALBERT, SentencePiece models) operates in a top-down manner. It starts with a massive vocabulary and iteratively prunes the least useful tokens.

#### Step-by-Step Algorithm:
1. **Initialize Vocabulary**: Define a very large vocabulary $|V_{\text{initial}}|$ consisting of all characters and the most common substrings/words from the training corpus.
2. **Estimate Unigram Model**: Train a unigram language model to estimate probabilities $P(x)$ for all tokens $x \in V$ using Expectation-Maximization (EM). The likelihood of a word $W$ composed of segmentations $\{x_1, \dots, x_k\}$ is:
   $$P(W) = \prod_{i=1}^k P(x_i)$$
3. **Compute Pruning Loss**: For each token $x$ in the current vocabulary, calculate the decrease in corpus log-likelihood if $x$ were removed.
4. **Prune**: Remove the bottom $p\%$ of tokens (usually $10\text{--}20\%$) that result in the smallest reduction in corpus likelihood.
5. **Iterate**: Repeat steps 2-4 until the vocabulary size shrinks to the target budget $|V|$.

---

## 2. Classical Preprocessing: Normalization, Stemming, and Lemmatization

Before vectorization, classical pipelines use deterministic text normalizations:

### Stemming vs. Lemmatization
Understanding the algorithmic trade-offs of these lexical reductions is a common system design question:

```
                  "studies" / "studying"
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Stemming (Porter)             Lemmatization (WordNet)
  (Heuristic truncation)           (Morphological Lookup)
            │                               │
            ▼                               ▼
        "studi"                         "study"
  (Fast, non-dictionary,         (Slower, dictionary-backed,
  results in non-words)           preserves real root word)
```

- **Stemming (Porter Stemmer)**: A fast, rule-based heuristic that chops off word prefixes and suffixes.
  - *Example*: `"studies"`, `"studying"`, `"studied"` all map to `"studi"`.
  - *Failure mode*: Over-stemming (collapsing words with different meanings: `"organization"` and `"organs"` both to `"organ"`) or under-stemming (failing to merge `"alumnus"` and `"alumni"`).
- **Lemmatization (WordNet)**: Uses grammatical tags (POS tagging) and dictionary tables to resolve words to their canonical base forms (lemmas).
  - *Example*: `"better"` maps to `"good"`, `"was"` maps to `"be"`.
  - *Trade-off*: Slower and memory-intensive due to dictionary lookups and POS dependencies.

---

## 3. Regular Expressions and Unicode Normalization

Uncleaned raw strings contain encoding anomalies, boilerplate, and noise that must be filtered out:

- **Regular Expressions (Regex)**: Used to filter out URLs (`https?://\S+`), HTML tags (`<[^>]+>`), or punctuation (`[^\w\s]`).
- **Unicode Normalization**: Ensures consistent character representations. For example, the accented character `é` can be represented as:
  - **NFC (Canonical Decomposition, followed by Canonical Composition)**: Single code point `\u00e9`.
  - **NFD (Canonical Decomposition)**: Decomposed into base `e` (`\u0065`) and combining accent character `´` (`\u0301`).
  - *Production Rule*: Always apply NFC normalization to guarantee consistent token mapping in tokenizers.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Text preprocessing transforms raw, inconsistent character streams into normalized, uniform inputs, reducing vocabulary size and vocabulary sparsity.
- **Why was it introduced?**
  Introduced to solve out-of-vocabulary (OOV) errors, normalize character variations, and compress vocabulary matrices for efficient training.
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
