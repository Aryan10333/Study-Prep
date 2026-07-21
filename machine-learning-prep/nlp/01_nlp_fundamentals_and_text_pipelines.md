# Module 01: NLP Fundamentals & Enterprise Text Pipelines

This study guide covers the core principles of Natural Language Processing (NLP), the end-to-end enterprise NLP pipeline, evolution of NLP paradigms, core NLP task taxonomy, engineering challenges, step-by-step vocabulary hand-calculations, clean production Python preprocessing code, failure modes, and interview flashcards.

> **Notebook Companion**: [01_nlp_fundamentals_and_text_pipelines.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/01_nlp_fundamentals_and_text_pipelines.ipynb)

---

## 1. What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) and Computational Linguistics that enables computers to understand, interpret, represent, manipulate, and generate human natural language text and speech.

Unlike tabular or structured data (which consists of fixed numeric schema, continuous features, or categorical codes), natural language text possesses distinct characteristics that present unique engineering challenges:
1. **Unstructured & Non-Euclidean**: Text does not naturally lie in a continuous vector space $\mathbb{R}^d$. It consists of variable-length discrete symbols (words/characters/subwords).
2. **High-Cardinality & Sparse**: Human languages contain hundreds of thousands of distinct words. Naive discrete encoding leads to sparse vector spaces ($|V| \ge 100,000$).
3. **Context-Dependent & Ambiguous**: The semantic meaning of a token depends entirely on surrounding context (e.g., *"bank"* in *"river bank"* vs. *"investment bank"*).
4. **Hierarchical Grammar & Syntax**: Words assemble into phrases, clauses, and sentences governed by complex grammatical dependencies and semantic logic.

```text
Raw Text (Unstructured String) ──► Tokenization & Normalization ──► Numerical Vector Encoding (R^d) ──► Model Inference
```

---

## 2. The End-to-End Enterprise NLP Pipeline

In production enterprise software systems, raw unstructured text must pass through a disciplined, deterministic multi-stage pipeline before model inference or vector store indexing:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION & PARSING                                                            │
│    Extract raw text from PDFs, HTML pages, SQL DBs, OCR streams, Kafka logs.          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. TEXT CLEANING & NORMALIZATION                                                       │
│    Unicode normalization (NFC/NFKD), lowercasing, HTML tag stripping, regex noise.    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. TOKENIZATION & VOCABULARY MAPPING                                                   │
│    Convert raw character stream into token IDs (Word, Subword: BPE/WordPiece).         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. FEATURE REPRESENTATION & EMBEDDING                                                  │
│    Map token IDs to numerical vectors (Sparse: TF-IDF / BM25; Dense: Word2Vec / BERT).│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. MODEL INFERENCE & SEQUENTIAL PROCESSING                                             │
│    Pass vector embeddings into Classifier, Sequence Model (LSTM), or Transformer LLM.  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. POST-PROCESSING & PRODUCTION SERVING                                                │
│    Parse logits/tokens into JSON payload, apply safety guardrails, stream SSE output.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Evolution of NLP Paradigms

```text
Dimension          Rule-Based NLP             Statistical ML NLP         Deep Learning NLP          Modern Transformer LLM
--------------------------------------------------------------------------------------------------------------------------------
Era                1960s – 1990s              1990s – 2010s              2014 – 2019                2019 – Present
Primary Tech       Regex, Context-Free Grammars TF-IDF, Naive Bayes, SVM  RNN, LSTM, GRU, GloVe      BERT, GPT-4, Llama 3
Feature Engineering Hand-crafted lexicons    Bag-of-Words / N-Grams     Dense Static Vectors (Word2Vec) Learned Contextual Self-Attention
Context Radius     Rule clause bounded        Unigram / Bigram window    Sequential hidden state    Global Context (128k+ tokens)
OOV Handling       Fails on missing rules     Mapped to <UNK> token      Character-level / Subword   Subword Tokenization (BPE/tiktoken)
Inference Latency  Ultra-fast (<1ms)          Fast (<5ms)                Moderate (10ms - 50ms)     Heavy (100ms - 1000ms+)
Best Production Use Input validation & regex  High-speed spam filter     Sequence tagging (NER)     Complex QA, RAG, Reasoning
```

---

## 4. Core NLP Task Taxonomy

Enterprise NLP tasks are broadly categorized into **Classification**, **Sequence Tagging**, **Generative Modeling**, and **Information Retrieval**:

1. **Text Classification**: Maps input text $X$ to a discrete class label $y \in \{1, \dots, C\}$.
   - *Examples*: Spam vs Non-Spam, Customer Support Ticket Routing, Intent Recognition.
2. **Named Entity Recognition (NER)**: Assigns token-level sequence tags $y_1, \dots, y_T$ identifying entities (Person, Organization, Location, Date).
   - *Examples*: Extracting drug names from clinical trial notes, extracting invoice line items.
3. **Sentiment Analysis & Opinion Mining**: Evaluates emotional polarity (Positive, Neutral, Negative) or aspect-level sentiment.
4. **Machine Translation & Sequence-to-Sequence**: Maps sequence $X_{1:T}$ in Source Language $A$ to target sequence $Y_{1:S}$ in Language $B$.
5. **Extractive & Abstractive Summarization**: Condenses long-form documents into concise summaries.
6. **Question Answering & Retrieval Augmented Generation (RAG)**: Retrieves relevant knowledge chunks and synthesizes grounded answers.

---

## 5. Fundamental Engineering Challenges in NLP

### 1. Polysemy & Homonymy (Contextual Ambiguity)
- **Polysemy**: A single word has multiple related meanings (e.g., *"apple"* as a fruit vs. *"Apple"* as a technology corporation).
- **Homonymy**: Words share spelling but have unrelated meanings (e.g., *"bark"* of a tree vs. *"bark"* of a dog).

### 2. High-Cardinality & Out-Of-Vocabulary (OOV) Explosion
Language vocabularies follow **Zipf's Law**: the frequency of any word is inversely proportional to its rank in the frequency table:

$$f(k; s, N) = \frac{1/k^s}{\sum_{n=1}^N (1/n^s)}$$

A small fraction of common words account for $>80\%$ of text occurrences, while a massive "long tail" of rare words causes vocabulary explosion. Unseen words at test time cause OOV errors if not handled by subword tokenization.

### 3. Syntactic Ambiguity
- *Example*: *"I saw the man with the telescope."* (Did I use the telescope to see the man, or did the man have a telescope?)

### 4. Long-Range Contextual Dependencies
In long documents, crucial subject-verb or antecedent relationships may be separated by hundreds of tokens, causing Recurrent Neural Networks (RNNs) to suffer from vanishing gradients.

---

## 6. Step-by-Step Hand Calculation Example (Andrew Ng Style)

Suppose we have a tiny enterprise logging dataset of $D = 3$ documents:
- **Doc 1**: `"system error log"`
- **Doc 2**: `"system authentication error"`
- **Doc 3**: `"database log error"`

Let us calculate vocabulary statistics step-by-step:

### 1. Extract Unique Tokens (Vocabulary $V$):
$$\text{Tokens} = \{\text{"system"}, \text{"error"}, \text{"log"}, \text{"authentication"}, \text{"database"}\}$$
$$\text{Vocabulary Size } |V| = \mathbf{5}$$

### 2. Total Token Count ($N_{\text{total}}$):
- Doc 1: 3 tokens
- Doc 2: 3 tokens
- Doc 3: 3 tokens
- $N_{\text{total}} = 3 + 3 + 3 = \mathbf{9\text{ tokens}}$

### 3. Calculate Type-Token Ratio (TTR):
The Type-Token Ratio measures lexical diversity in a corpus:

$$\text{TTR} = \frac{|V|}{N_{\text{total}}} = \frac{5}{9} \approx \mathbf{0.5556}$$

### 4. Word Frequency Distribution Table:

```text
Token             Doc 1 Frequency  Doc 2 Frequency  Doc 3 Frequency  Total Corpus Count  Document Frequency (DF)
------------------------------------------------------------------------------------------------------------------
system            1                1                0                2                   2
error             1                1                1                3                   3
log               1                0                1                2                   2
authentication    0                1                0                1                   1
database          0                0                1                1                   1
```

---

## 7. Production Python Text Preprocessing Implementation

```python
import re
import unicodedata

class EnterpriseTextCleaner:
    """Production-grade text cleaning and normalization pipeline."""
    
    def __init__(self, lower: bool = True, strip_html: bool = True):
        self.lower = lower
        self.strip_html = strip_html
        self.html_regex = re.compile(r'<[^>]+>')
        self.extra_whitespace_regex = re.compile(r'\s+')
        
    def normalize_unicode(self, text: str) -> str:
        """Normalizes unicode characters to NFKD decomposed form."""
        return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        
    def clean(self, text: str) -> str:
        """Executes full deterministic text cleaning pipeline."""
        if not text:
            return ""
            
        # 1. Unicode normalization
        text = self.normalize_unicode(text)
        
        # 2. Strip HTML tags
        if self.strip_html:
            text = self.html_regex.sub(" ", text)
            
        # 3. Lowercasing
        if self.lower:
            text = text.lower()
            
        # 4. Collapse extra whitespace
        text = self.extra_whitespace_regex.sub(" ", text).strip()
        
        return text

# Demonstration execution
cleaner = EnterpriseTextCleaner()
raw_sample = "  <p>ALERT: System <b>Server_01</b> error at 10:45&nbsp;AM! Connection refused. </p>  "
cleaned_sample = cleaner.clean(raw_sample)

print("=== Enterprise Text Preprocessing Output ===")
print("Raw Input:    ", repr(raw_sample))
print("Cleaned Output:", repr(cleaned_sample))
```

> [!NOTE]
> **Production Optimization Alert:**
> - Compiling regular expressions (`re.compile`) at class initialization prevents costly re-compilation overhead during batch processing ($100\text{x}$ speedup on large document streams).

---

## 8. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Regex Catastrophic Backtracking**: Unbounded regex patterns (e.g., `(a+)+`) executed on malicious user input cause CPU spike and denial-of-service (ReDoS).
   - *Remediation*: Avoid nested quantifiers and set explicit execution timeouts on regex engines.
2. **Over-cleaning & Loss of Signal**: Removing punctuation, numbers, or casing can destroy semantic meaning (e.g., removing `"-"` converts `"non-responsive"` to `"non responsive"`, removing uppercase converts `"US"` country to `"us"` pronoun).
   - *Remediation*: Preserve casing and numbers for domain-specific models like NER and Part-of-Speech tagging.
3. **Vocabulary Shift & Out-Of-Vocabulary (OOV)**: Deploying a model trained on general English (e.g., Wikipedia) to clinical medical notes causes high OOV rate and performance collapse.
   - *Remediation*: Use domain-specific tokenization and vocabulary (e.g., BioBERT, ClinicalBERT).

---

## 9. Master Interview Flashcards & Questions

#### Q1: Why is text representation fundamentally more challenging than tabular data representation?
- **Answer:** Tabular data consists of continuous/categorical features in fixed Euclidean dimensions. Text is unstructured, variable-length, discrete, sparse, high-cardinality ($|V| > 100k$), and heavily context-dependent where word meaning changes based on surrounding sequence context.

#### Q2: Explain Zipf's Law and its implications for NLP vocabulary management.
- **Answer:** Zipf's Law states that word frequency is inversely proportional to its frequency rank. A few common words dominate corpus occurrences, while a massive tail of rare words causes vocabulary explosion. Subword tokenization (BPE/WordPiece) addresses this by breaking rare words into common subword units.

#### Q3: Compare Rule-Based NLP vs Statistical ML NLP vs Deep Learning NLP vs LLMs.
- **Answer:** Rule-based uses hand-crafted grammars (fast, rigid). Statistical ML uses BoW/TF-IDF + Naive Bayes/SVM (fast baseline, ignores word order). Deep Learning uses RNNs/LSTMs (captures sequence order, suffers from sequential latency). Transformers/LLMs use self-attention (global context, state-of-the-art accuracy, higher compute cost).
