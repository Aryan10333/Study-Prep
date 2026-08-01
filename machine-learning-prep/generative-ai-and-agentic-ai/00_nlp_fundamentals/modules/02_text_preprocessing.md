---
title: Text Preprocessing & Normalization
category: NLP Foundations
prerequisites: Introduction to NLP & Classical Tasks
---

# Text Preprocessing & Normalization

## 1. Introduction & Intuition

### The Core Bottleneck
Machine learning models process static numerical vectors, not raw strings. To convert text into vectors, we must map words to a fixed vocabulary index. However, raw human text contains capitalization variance, punctuation, typos, and morphologic endings (e.g. `"runs"`, `"running"`, `"ran"`). If we assign a unique vector index to every variation, the vocabulary size ($|V|$) explodes. This increases model parameters (VRAM footprint) and splits statistical data signals, as the model must learn the meaning of `"ran"` and `"running"` independently. The bottleneck is compressing vocabulary variations without losing semantic context.

### High-Level Intuition
Preprocessing is a lossy text compression pipeline. We convert raw text into a standardized, low-entropy canonical form by lowercase folding, stripping punctuation, filtering low-signal words (like `"the"` or `"is"`), and collapsing inflected words down to their root forms.

---

## 2. Core Concepts & Mathematical Formulation

### Stemming vs. Lemmatization
To collapse inflected words down to roots, we use two paradigms:

1.  **Stemming:** A heuristic, rule-based suffix truncation process. It acts on single words without grammatical context or dictionaries.
    *   *The Porter Stemmer:* Uses structural rules (e.g. if word ends in `"-sses"`, map to `"-ss"`; if word ends in `"-ed"`, check syllable count and chop).
2.  **Lemmatization:** A morphological analysis lookup process. It maps inflected words back to their dictionary root (*lemma*) by checking a dictionary (like WordNet) and utilizing the word's Part-of-Speech (POS) context.

#### Intuition & Practical Use
Stemming is a fast, lightweight regex tool used when processing massive corpora quickly (like search engine indexers). Lemmatization is slower but highly accurate, preferred when building grammatical understanding or semantic pipelines (like QA bots).

#### Unicode Normalization: NFC vs. NFD
Unicode characters can have equivalent visual representations but different binary code points:
*   **NFD (Normal Form Decomposition):** Separates characters into base letters and accent marks.
    *   *Example:* `"é"` is decomposed to base letter `"e"` (`\u0065`) and combining acute accent mark `"´"` (`\u0031`).
*   **NFC (Normal Form Composition):** Combines letters and accents into a single code point.
    *   *Example:* `"é"` is represented as a single character (`\u00E9`).

Normalizing all characters to a single form (typically NFC) prevents vocabularies from treating visually identical text as different words.

#### Hand Calculation on a Simple Example
Let's preprocess and compare stemming vs. lemmatization on the sequence:
$$X = \text{"wolves running quickly"}$$

*   **Step 1: Stemming (Porter Algorithm Rules)**
    1.  Word `"wolves"`:
        *   Matches suffix rule: replace `"-ves"` with `"-f"`.
        *   Output stem: `"wolv"` (not a valid dictionary word).
    2.  Word `"running"`:
        *   Matches suffix rule: replace `"-ning"` with `"-n"` (if consonant duplication is sliced).
        *   Output stem: `"run"`.
    3.  Word `"quickly"`:
        *   Matches suffix rule: replace `"-ly"` with `"-li"`.
        *   Output stem: `"quickli"`.
    *   *Stemmed Output:* `["wolv", "run", "quickli"]`

*   **Step 2: Lemmatization (WordNet Morphological Lookup)**
    1.  Word `"wolves"`:
        *   Identify POS: Noun.
        *   WordNet mapping: matches plural form `"wolves"` $\to$ root singular `"wolf"`.
        *   Output lemma: `"wolf"`.
    2.  Word `"running"`:
        *   Identify POS: Verb.
        *   WordNet mapping: matches progressive form `"running"` $\to$ base verb `"run"`.
        *   Output lemma: `"run"`.
    3.  Word `"quickly"`:
        *   Identify POS: Adverb.
        *   WordNet mapping: no morphological root change.
        *   Output lemma: `"quickly"`.
    *   *Lemmatized Output:* `["wolf", "run", "quickly"]`

Notice that stemming produced non-words (`"wolv"`, `"quickli"`), while lemmatization yielded clean dictionary lemmas (`"wolf"`, `"quickly"`).

#### Tensor & Shape Tracking
*   Input text: Raw string characters.
*   Token Array: `[N_tokens]` on CPU during extraction passes.

---

## 3. Implementation & Reference Code

Below is a Python comparison of stemming and lemmatization on a sentence.

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Download resources locally
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def run_preprocessing_comparison():
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    words = ["studies", "studying", "went", "wolves", "running", "was"]
    pos_tags = {
        "studies": wordnet.NOUN,
        "studying": wordnet.VERB,
        "went": wordnet.VERB,
        "wolves": wordnet.NOUN,
        "running": wordnet.VERB,
        "was": wordnet.VERB
    }
    
    print(f"{'Original':<12} | {'Porter Stem':<15} | {'WordNet Lemma':<15}")
    print("-" * 48)
    for w in words:
        stemmed = stemmer.stem(w)
        tag = pos_tags.get(w, wordnet.NOUN)
        lemmed = lemmatizer.lemmatize(w, pos=tag)
        print(f"{w:<12} | {stemmed:<15} | {lemmed:<15}")

if __name__ == "__main__":
    run_preprocessing_comparison()
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Word redundancy mitigation and structural vocabulary compression.
*   **Why Introduced over Legacy Approaches:** Lemmatization is preferred over stemming when downstream semantic classification tasks require valid grammatical roots and word semantics.
*   **Key Failure Modes & Limitations:** Lemmatizers fail when POS-tags are mispredicted. If `"saw"` is tagged as a Noun instead of a Verb, it fails to map to `"see"`.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Stemming runs in $O(L)$ where $L$ is token length. Lemmatization runs in $O(L \log V_D)$ due to lookup tree structures inside WordNet.
*   **Space/Memory Footprint:** Stemming requires zero VRAM. Lemmatization requires keeping the morphology database index loaded in RAM (approx. $10\text{MB}$ to $50\text{MB}$).
*   **Primary Bottleneck Type:** CPU-bound. Heavy string manipulation bottleneck during dataset token tokenization passes.

### 3. Production & Scalability
*   **Deployment Considerations:** For web-scale indexing (e.g. search engines), stemming is integrated directly into the indexing phase. For modern neural nets (like LLMs), character-level subword tokenizers have largely replaced stemming and lemmatization, delegating root mapping to embedding parameter learning.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* Why is lemmatization computationally more expensive than stemming, and when is this trade-off justified?
        *   *A:* Lemmatization requires checking grammatical context and loading a morphology lookup index (database) to resolve roots. The trade-off is justified when semantic accuracy and word validity are critical (e.g., question answering, semantic search), whereas stemming is better for speed-critical, retrieval-focused search indices.
    2.  *Q:* How does Unicode normalization prevent token fragmentation in subword models?
        *   *A:* If Unicode characters like `"é"` exist as NFD (two code points `e` + accent) and NFC (one code point `é`), a subword model will tokenize them as separate tokens. Normalizing to a standard form NFC/NFKC ensures equivalent text is mapped to the same vocabulary indices, preventing vocabulary splits.
