# Module 01: NLP Introduction

Natural Language Processing (NLP) translates unstructured human language into quantitative values. This module details the standard text processing pipeline, maps the historical paradigm shifts from rule-based engines to modern Transformers, and compares tokenization strategies.

---

## 1. The Standard NLP Pipeline

In production systems, processing raw text requires translating unstructured strings into structured numerical matrices.

### Walkthrough of a Sample String: `"Seattle's libraries are awesome! 🌟"`

```
[Raw Input] ──▶ "Seattle's libraries are awesome! 🌟"
                      │
                      ▼
[Preprocessed] ──▶ "seattles libraries are awesome" (lowercased, emoji & punctuation removed)
                      │
                      ▼
[Tokenized]    ──▶ ["seattle", "s", "libraries", "are", "awesome"] (split into tokens)
                      │
                      ▼
[Represented]  ──▶ [42, 107, 856, 12, 93] (indices mapped to dense embedding vectors)
                      │
                      ▼
[Model Feed]   ──▶ [[0.12, -0.4, ...], [0.88, 0.1, ...]] (fed into sequence classifier)
                      │
                      ▼
[Prediction]   ──▶ Sentiment: POSITIVE (Confidence: 98.4%)
```

---

## 2. Tokenization Strategies: Character vs. Word vs. Subword

Text representation splits raw characters into discrete computational units:

| Tokenization Strategy | Vocabulary Size | Out-of-Vocabulary (OOV) Risk | Sequence Length | Key Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Character-Level** | Minimal ($\approx 256$ tokens) | Zero ($0\%$ OOV rate) | Extremely Long | Discards word root semantics; increases attention cost. |
| **Word-Level** | Massive ($>1,000,000$ tokens) | Extremely High | Short | Vocabulary size drains GPU memory; cannot map new words. |
| **Subword-Level** | Balanced ($32,000\text{--}256,000$) | Zero | Balanced | Requires boundary prefix markers (e.g. `##` or ` `). |

### Why Subword Tokenization is Essential for LLMs
To scale models to billions of parameters, vocabulary embedding tables must remain compact. Word-level tokenizers result in massive lookup matrices, draining GPU memory (VRAM). Character-level tokenizers increase sequence length, raising attention computational cost $O(L^2)$ quadratically.

Subword tokenization bridges this gap. By splitting words into common root combinations (e.g., `"unhappy"` $\rightarrow$ `["un", "happy"]`), the vocabulary remains small while unknown words are decomposed without crashing the model.

---

## 3. The Evolutionary Shifts in NLP

The architecture of text systems transitioned through four distinct developmental paradigms:

1. **Rule-Based Systems**: Manual regular expressions and syntax trees (e.g. pattern matchers). Fast and predictable, but fragile when handling typos or grammatical variations.
2. **Statistical Models**: Modeled language probabilistically using frequency counts (e.g. N-grams, Naïve Bayes). Limited to short contexts due to exponential scaling of n-gram count tables.
3. **Deep Learning Sequence Models**: Introduced recurrent loops (RNNs, LSTMs) to maintain continuous hidden state vectors across variable sequence lengths. Suffered from sequential bottleneck delays during training.
4. **Transformers (Modern GenAI)**: Replaced recurrence with Self-Attention query-key-value projections. Enabled parallel training calculations and long-range context mapping.

---

> [!TIP]
> **Production Insight: Latency vs. Vocabulary Constraints**
> When deploying real-time classification models (e.g., streaming chat routing), subword tokenizers (like WordPiece) offer the best balance of speed and coverage. Avoid word-level models in production, as typos will trigger OOV errors, returning useless `<unk>` tokens that degrade classification performance.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Translates unstructured human text into standardized numerical formats for machine learning models.
- **Why was it introduced?**
  Introduced to handle lexical variations and spelling anomalies in human speech.
- **What are its limitations?**
  Preprocessing steps (like lowercase conversions and punctuation removal) can discard semantic context, such as sentiment flags or code syntax indicators.
- **Computational Complexity (Time & Memory)**
  - **Time**: Subword tokenization runs in $O(L)$ linear time, where $L$ is string length.
  - **Memory**: Embedding matrix footprint scales as $O(|V| \cdot d)$ where $|V|$ is vocabulary size and $d$ is embedding dimensionality.
- **Component Variable Denotation Legend**
  - $L$: Input sequence length.
  - $|V|$: Vocabulary size.
  - $d$: Embedding hidden dimension.
- **Production Use Cases**
  - Text categorization for customer service routing.
  - Compressing inputs for multilingual translation models.
- **Follow-up questions interviewers ask**
  - *Why does word-level tokenization fail on specialized text?* (Medical and legal domains contain rare words that are mapped to `<unk>`, causing the model to lose key details).
  - *How does subword vocabulary size impact training memory?* (Larger vocabularies increase the size of the final projection layer, increasing GPU VRAM usage during backpropagation).
