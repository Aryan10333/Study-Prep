# Transformers: Families & Model Architectures

This guide compares Encoder-only, Decoder-only, and Encoder-Decoder families, mapping their architectural designs to production use cases, walking through step-by-step model parameter counts, and providing a PyTorch script to inspect attention masks.

---

## 1. The Transformer Families

Based on the original design, modern Transformers are divided into three primary architectural families:

```text
Model Family        Example Models   Attention Mask Type                 Primary Production Task
----------------------------------------------------------------------------------------------------------------------
Encoder-only        BERT, RoBERTa    Bidirectional (Full Context)        Search, classification, NER, embedding
Decoder-only        GPT, Llama       Causal (Look-ahead)                 Generative chat, coding, reasoning
Encoder-Decoder     T5, BART         Bidirectional (Enc) / Causal (Dec)  Translation, summarization, rewriting
```

### A. Encoder-Only Models (Autoencoding)
Encoder-only models compute self-attention bidirectionally. Tokens attend to both past and future positions, generating dense semantic embeddings.
- **Production Utility:** Best for extraction and classification tasks (e.g. sentiment classification or Named Entity Recognition).

### B. Decoder-Only Models (Autoregressive)
Decoder-only models enforce causality, preventing tokens from looking at future steps.
- **Production Utility:** Best for generation and interactive reasoning (e.g., text completions, chat loops).

### C. Encoder-Decoder Models (Sequence-to-Sequence)
The Encoder processes input sequences bidirectionally, generating context vectors. The Decoder consumes these states via cross-attention layers while causally generating target outputs.
- **Production Utility:** Best for tasks mapping one sequence length to another (e.g. language translation or document summarization).

---

## 2. Model Families Comparison

| Feature | Encoder-only (BERT) | Decoder-only (GPT/Llama) | Encoder-Decoder (T5) |
| :--- | :--- | :--- | :--- |
| **Masking Style** | Bidirectional (No causal restriction) | Causal (Look-ahead mask) | Bidirectional (Encoder) / Causal (Decoder) |
| **Pre-training Goal** | Masked Language Modeling (MLM) | Causal Language Modeling (CLM) | Span Corruption / Text Reconstruction |
| **Generative Style** | Non-generative (Extraction only) | Autoregressive (token-by-token) | Autoregressive decoding |
| **KV Cache Size** | None (Computed once) | Large (Grows with context) | Medium (Grows with output length) |
| **Best Use Cases** | Sentiment, classification, NER, search | Chat, code generation, summarization | Translation, text-to-text conversion |

---

## 3. Step-by-Step Hand Calculations: Parameter Estimation (Andrew Ng Style)

Let's estimate the total parameter count of a tiny Transformer model to prepare for systems design calculations:
- **Configurations:**
  - Embedding dimension $d_{\text{model}} = 128$
  - Number of Layers $L = 2$
  - Vocabulary Size $V = 1000$
  - FFN Hidden Dimension $d_{\text{ff}} = 4 \times d_{\text{model}} = 512$

---

### Step 1: Compute Token Embedding Parameters
The model maps $V$ tokens to $d_{\text{model}}$ vectors:
$$\text{Params}_{\text{Embedding}} = V \times d_{\text{model}} = 1000 \times 128 = \mathbf{128,000}$$

---

### Step 2: Compute Self-Attention Parameters (Per Layer)
Each attention layer contains four projection matrices ($W_Q, W_K, W_V, W_O$) of shape $d_{\text{model}} \times d_{\text{model}}$ (ignoring biases):
$$\text{Params}_{\text{Attn\_Proj}} = 3 \times \left( d_{\text{model}} \times d_{\text{model}} \right) = 3 \times 128 \times 128 = 49,152$$
$$\text{Params}_{\text{Output\_Proj}} = d_{\text{model}} \times d_{\text{model}} = 128 \times 128 = 16,384$$
$$\text{Params}_{\text{Attn\_Total}} = 49,152 + 16,384 = \mathbf{65,536} \text{ parameters}$$

---

### Step 3: Compute FFN Parameters (Per Layer)
The FFN contains two projection matrices: $W_1$ ($d_{\text{model}} \times d_{\text{ff}}$) and $W_2$ ($d_{\text{ff}} \times d_{\text{model}}$):
$$\text{Params}_{W_1} = d_{\text{model}} \times d_{\text{ff}} = 128 \times 512 = 65,536$$
$$\text{Params}_{W_2} = d_{\text{ff}} \times d_{\text{model}} = 512 \times 128 = 65,536$$
$$\text{Params}_{\text{FFN\_Total}} = 65,536 + 65,536 = \mathbf{131,072} \text{ parameters}$$

---

### Step 4: Sum and Scale across $L=2$ Layers
$$\text{Params}_{\text{Per\_Layer}} = \text{Params}_{\text{Attn\_Total}} + \text{Params}_{\text{FFN\_Total}} = 65,536 + 131,072 = 196,608$$
$$\text{Params}_{\text{Layers\_Total}} = L \times \text{Params}_{\text{Per\_Layer}} = 2 \times 196,608 = \mathbf{393,216}$$
$$\text{Total Parameters} = \text{Params}_{\text{Embedding}} + \text{Params}_{\text{Layers\_Total}} = 128,000 + 393,216 = \mathbf{521,216} \text{ parameters}$$

**Result:** The total parameter count is **$521,216$** (approximately $0.52\text{M}$ parameters).

---

## 4. Production Scenario & Example

### Scenario: Design a Customer Feedback Routing & Reply Pipeline
You are designing an email customer support pipeline. The system must analyze incoming feedback, extract order numbers, evaluate sentiment, and write personalized replies.
- **The Failure Mode:** Using a single large Decoder model (like GPT-4) to perform the entire extraction, classification, and reply generation task is expensive, slow, and prone to formatting errors.
- **The Solution:** We design a multi-stage routing system using specialized models:
  1. **Feedback Classification & Extraction:** We route the incoming text to a fine-tuned **Encoder-only model (BERT)**. Because it reads bidirectionally, it is highly accurate at extracting order entities and classifying sentiment, costing a fraction of decoder execution.
  2. **Personalized Reply Generation:** If the email requires a reply, we pass the extracted metadata to a **Decoder-only model (Llama)** to generate the personalized text.
  This hybrid architecture reduces serving costs by $5\text{x}$ and improves response latency.

---

## 5. PyTorch Attention Mask Inspection

This code prints and verifies causal and bidirectional attention masking tensors:

```python
import torch

def inspect_attention_masks(seq_len):
    # 1. Bidirectional Attention Mask (Encoder style)
    # Allows all positions to attend to all other positions
    encoder_mask = torch.ones(seq_len, seq_len)
    
    # 2. Causal Attention Mask (Decoder style)
    # Upper-triangular matrix filled with -inf to prevent looking ahead
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
    
    print("Encoder Mask (Bidirectional):\n", encoder_mask)
    print("\nDecoder Mask (Causal Logits Adjustment):\n", causal_mask)

# Test sequence length of 4 tokens
inspect_attention_masks(4)
```
