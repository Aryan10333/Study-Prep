# Module 01: NLP Introduction

Natural Language Processing (NLP) bridges the gap between unstructured human language and quantitative computation. This module details the standard NLP text pipeline, maps the historical paradigm shifts from rule-based filters to modern Transformers, and contrasts tokenization strategies.

---

## 1. The Standard NLP Pipeline

In production systems, processing raw text requires translating unstructured strings into formatted numerical matrices through a sequence of modular processing steps:

<div class="custom-diagram" style="margin: 24px 0; background-color: #f8fafc; padding: 24px; border: 1px solid #cbd5e1; border-radius: 8px; font-family: inherit;">
    <div style="font-weight: bold; color: #0f172a; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; text-align: center; margin-bottom: 20px;">
        Standard Production NLP Pipeline Sequence
    </div>
    
    <!-- Row 1: Raw, Preprocess, Tokenize -->
    <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #64748b;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #64748b;">01. Input</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Raw Text</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #2563eb;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #2563eb;">02. Clean</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Preprocessing</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px;">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #10b981;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #10b981;">03. Segment</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Tokenization</div>
        </div>
    </div>
    
    <!-- Vertical connector -->
    <div style="display: flex; justify-content: flex-end; padding-right: 60px; margin-bottom: 20px;">
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(90deg);">➔</div>
    </div>

    <!-- Row 2: Representation, Model, Prediction -->
    <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #f59e0b;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #f59e0b;">06. Output</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Prediction</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(180deg);">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #ef4444;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #ef4444;">05. Process</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Model Inference</div>
        </div>
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(180deg);">➔</div>
        <div style="flex: 1; min-width: 120px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; text-align: center; border-top: 3px solid #7c3aed;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #7c3aed;">04. Vectorize</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Text Representation</div>
        </div>
    </div>
    
    <!-- Vertical connector -->
    <div style="display: flex; justify-content: flex-start; padding-left: 60px; margin-bottom: 20px;">
        <div style="color: #94a3b8; font-weight: bold; font-size: 16px; transform: rotate(90deg);">➔</div>
    </div>

    <!-- Row 3: Evaluation -->
    <div style="display: flex; justify-content: center; align-items: center;">
        <div style="width: 40%; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; text-align: center; border-top: 3px solid #0f172a;">
            <span style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #0f172a;">07. Evaluate</span>
            <div style="font-size: 12px; font-weight: bold; color: #1e293b; margin-top: 4px;">Model Evaluation</div>
        </div>
    </div>
</div>

---

## 2. Tokenization Strategies: Character vs. Word vs. Subword

Text representation requires splitting raw characters into discrete computational units. Selecting the correct boundary limits token vocabulary size while preserving semantics:

| Tokenization Strategy | Vocabulary Size | Out-of-Vocabulary (OOV) Risk | Computational Overhead | Key Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Character-Level** | Minimal (typically $\approx 100\text{--}250$ tokens covering alphabet/symbols). | Zero ($0\%$ OOV rate, every string is composed of raw characters). | High sequence length (forces models to process long arrays). | Discards semantic meaning of multi-character root words. |
| **Word-Level** | Massive (often $>1,000,000$ tokens to cover entire dictionary). | Extremely High (unseen words like "tokenization" fail to map). | Low sequence length (one token per word). | Vocabulary explosion; unable to share semantic roots. |
| **Subword-Level** | Controlled (typically fixed at $32,000\text{--}256,000$ tokens). | Zero (unknown words are split into characters or byte tokens). | Balanced sequence length. | Requires prefix/suffix tracking character indicators (e.g. `##` or ` `). |

### Why Subword Tokenization Became Necessary for LLMs
To scale models to hundreds of billions of parameters, vocabulary matrices must remain compact. Word-level tokenizers result in massive lookup matrices ($|V| \times d_{\text{model}}$), draining GPU VRAM. Character-level tokenizers increase sequence length, scaling the attention matrix cost $O(L^2)$ quadratically. 

Subword tokenization bridges this gap. By splitting words into common root combinations (e.g., `"unhappy"` $\rightarrow$ `["un", "happy"]`), the vocabulary remains small while unknown words are decomposed without crashing the model.

---

## 3. The Evolutionary Shifts in NLP

The architecture of text systems transitioned through four distinct developmental paradigms:

```
Rule-Based Systems          Statistical Models          Deep Learning Models          Transformers / LLMs
 (Regex & Parsers)      (N-grams & Naive Bayes)          (RNNs & LSTM Cells)            (Self-Attention)
        │                         │                               │                            │
        ▼                         ▼                               ▼                            ▼
Strict String Matches     Markov Probabilities         Sequence State Memory         Parallel Attention
```

1. **Rule-Based Systems**: Reliant on manual regular expressions and grammatical syntax trees. Failed to handle natural language variation or domain shifts.
2. **Statistical Models**: Modeled language probabilistically using frequency counts and Markov assumptions. Limited to short contexts due to exponential scaling of n-gram count tables.
3. **Deep Learning Sequence Models**: Introduced recurrent loops (RNNs, LSTMs) to maintain continuous hidden state vectors across variable sequence lengths. Suffered from sequential bottleneck delays during training.
4. **Transformers (Modern GenAI)**: Replaced recurrence entirely with Self-Attention query-key-value projections. Enabled massively parallel training calculations and long-range context mapping.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Establishing a standard text processing pipeline translates unstructured human language strings into dense numerical formats that machine learning decoders can optimize.
- **Why was it introduced?**
  Introduced to solve the high lexical variation, structural complexity, and unbounded dimensionality of human speech.
- **What are its limitations?**
  Stochastic processing results in text representation drift; cleaning steps can discard semantic metadata (such as negation or emojis).
- **Computational Complexity (Time & Memory)**
  - **Time**: Subword tokenization using pre-compiled trees runs in $O(L)$ time, where $L$ is the string character length.
  - **Memory**: Vocabulary weight matrix footprint scales as $O(|V| \cdot d)$ where $|V|$ is the vocabulary size and $d$ is embedding dimensionality.
- **Component Variable Denotation Legend**
  - $L$: Character length of raw input text.
  - $|V|$: Vocabulary token size.
  - $d$: Hidden dimension size of embedding layer.
- **Production Use Cases**
  - Parsing streaming customer queries for high-volume routing models.
  - Subword vocabulary compression in multilingual translation pipelines.
- **Follow-up questions interviewers ask**
  - *Why does word-level tokenization fail on specialized biomedical text?* (Specialized terminology features high Out-of-Vocabulary occurrences, forcing word-level models to assign the generic `<unk>` token to critical clinical terms).
  - *How does the subword vocabulary limit affect training cost?* (A larger vocabulary $|V|$ increases classification layer parameter count, increasing memory usage during backpropagation calculation steps).
