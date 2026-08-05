# Module 04: Tokenization & Subword Processing for LLMs

## 1. Introduction & Intuition

### The Core Bottleneck
Language models cannot directly process raw characters or strings. They map discrete textual tokens to static vector representations. Early models split text by spaces into words. This word-level tokenization has a severe bottleneck: any word not explicitly seen during training (like typos, slang, or new terms) gets mapped to a special Out-of-Vocabulary (`<unk>`) token. This discards all semantic details. 

If we expand the vocabulary to include every possible word, the model's embedding parameters ($V \times d$) grow exponentially, consuming gigabytes of VRAM. Character-level tokenization resolves OOV issues but generates extremely long sequence lengths ($L$), which increases attention complexity quadratically ($O(L^2)$). The bottleneck is finding a tokenization method that balances vocabulary size $V$ and sequence length $L$ while avoiding `<unk>` representations.

### High-Level Intuition
Think of tokenization as finding the optimal subword compression scheme. Instead of assigning a unique vector index to every single word, we break rare or complex words down into smaller, common subword fragments (e.g. `"unfriendliness"` becomes `"un-"`, `"friend"`, and `"-liness"`). 

*   **Byte-Pair Encoding (BPE)**: Starts with character-level tokens and iteratively merges the most frequent adjacent token pairs (bigrams) in the training corpus until the vocabulary reaches a target size.
*   **SentencePiece**: Direct subword tokenization on raw byte streams. It treats spaces as a standard character (represented by a visible blank character `_`), allowing it to build language-independent vocabularies without needing whitespace pre-tokenizers.
*   **Byte-Fallback BPE**: When encountering characters outside the training vocabulary, the tokenizer falls back to their raw UTF-8 byte representation. This completely eliminates Out-of-Vocabulary (`<unk>`) tokens.

---

### Tokenizer Pipeline Architecture
Below is the standard pipeline structure of a modern LLM tokenizer:

<div align="center" style="margin: 20px 0; max-width: 100%; overflow-x: auto;">
<svg viewBox="0 0 750 180" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-family: system-ui, -apple-system, sans-serif;">
  <!-- Nodes -->
  <!-- Raw Text -->
  <rect x="20" y="50" width="100" height="50" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5" />
  <text x="70" y="75" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">Raw Text</text>
  <text x="70" y="90" text-anchor="middle" font-size="10" fill="#64748b">"Learning LLMs!"</text>
  
  <path d="M 120 75 L 140 75" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-tok)" />
  
  <!-- Normalizer -->
  <rect x="140" y="50" width="110" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
  <text x="195" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">1. Normalization</text>
  <text x="195" y="90" text-anchor="middle" font-size="9" fill="#1d4ed8">Lowercase, NFKC</text>
  
  <path d="M 250 75 L 270 75" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-tok)" />
  
  <!-- Pre-Tokenizer -->
  <rect x="270" y="50" width="110" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
  <text x="325" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">2. Pre-Tokenize</text>
  <text x="325" y="90" text-anchor="middle" font-size="9" fill="#1d4ed8">Split by spaces/regex</text>
  
  <path d="M 380 75 L 400 75" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-tok)" />
  
  <!-- Model -->
  <rect x="400" y="50" width="110" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
  <text x="455" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">3. Subword Model</text>
  <text x="455" y="90" text-anchor="middle" font-size="9" fill="#1d4ed8">BPE / WordPiece</text>
  
  <path d="M 510 75 L 530 75" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-tok)" />
  
  <!-- Post-Processor -->
  <rect x="530" y="50" width="110" height="50" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
  <text x="585" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">4. Post-Process</text>
  <text x="585" y="90" text-anchor="middle" font-size="9" fill="#1d4ed8">Add BOS/EOS tags</text>
  
  <path d="M 640 75 L 660 75" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-tok)" />
  
  <!-- Output IDs -->
  <rect x="660" y="50" width="70" height="50" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5" />
  <text x="695" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#065f46">Token IDs</text>
  <text x="695" y="90" text-anchor="middle" font-size="9" fill="#047857">[1, 452, 98]</text>
  
  <text x="375" y="145" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">Standard Tokenizer Subsystem Pipeline</text>
  
  <!-- Arrow marker -->
  <defs>
    <marker id="arrow-tok" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
  </defs>
</svg>
</div>

---

## 2. Core Concepts & Mathematical Formulation

### BPE Merge Step Calculations (Dry Run)
Let's walk through the BPE merge loop algorithm using a simple mock corpus.

#### 1. Setup Initial Corpus
Assume our training corpus consists of the following words with frequencies:
1.  `"h u g _"` (Frequency = 4)
2.  `"r u g _"` (Frequency = 3)
3.  `"b u g _"` (Frequency = 2)

Our initial vocabulary is the set of all individual characters present:
$$\text{Vocab} = \{\text{"b"}, \text{"g"}, \text{"h"}, \text{"r"}, \text{"u"}, \text{"_"}\}$$

#### Step 1: Count All Adjacent Pairs (Bigrams)
Calculate frequencies of adjacent character pairs across the corpus:
*   Pair `('h', 'u')`: appears in `"hug_"` (4 times). Total = $4$.
*   Pair `('u', 'g')`: appears in `"hug_"` (4 times), `"rug_"` (3 times), and `"bug_"` (2 times). Total = $4 + 3 + 2 = 9$.
*   Pair `('g', '_')`: appears in `"hug_"` (4 times), `"rug_"` (3 times), and `"bug_"` (2 times). Total = $4 + 3 + 2 = 9$.
*   Pair `('r', 'u')`: appears in `"rug_"` (3 times). Total = $3$.
*   Pair `('b', 'u')`: appears in `"bug_"` (2 times). Total = $2$.

#### Step 2: Select the Most Frequent Pair
The highest frequency is $9$, tied between `('u', 'g')` and `('g', '_')`. 
Let's merge `('u', 'g')` first.

#### Step 3: Update Vocabulary and Corpus
*   New vocabulary token: `"ug"`.
*   $$\text{Vocab} = \{\text{"b"}, \text{"g"}, \text{"h"}, \text{"r"}, \text{"u"}, \text{"_"}, \text{"ug"}\}$$
*   **Updated Corpus:**
    1.  `"h ug _"` (Frequency = 4)
    2.  `"r ug _"` (Frequency = 3)
    3.  `"b ug _"` (Frequency = 2)

#### Step 4: Count Pairs for Next Merge
Compute bigram frequencies in the updated corpus:
*   Pair `('h', 'ug')`: appears in `"h ug _"` (4 times). Total = $4$.
*   Pair `('ug', '_')`: appears in `"h ug _"` (4 times), `"r ug _"` (3 times), and `"b ug _"` (2 times). Total = $4 + 3 + 2 = 9$.
*   Pair `('r', 'ug')`: appears in `"r ug _"` (3 times). Total = $3$.
*   Pair `('b', 'ug')`: appears in `"b ug _"` (2 times). Total = $2$.

The most frequent pair is `('ug', '_')` with $9$ occurrences. 
*   New vocabulary token: `"ug_"`.
*   $$\text{Vocab} = \{\text{"b"}, \text{"g"}, \text{"h"}, \text{"r"}, \text{"u"}, \text{"_"}, \text{"ug"}, \text{"ug_"}\}$$
*   **Updated Corpus:**
    1.  `"h ug_"` (Frequency = 4)
    2.  `"r ug_"` (Frequency = 3)
    3.  `"b ug_"` (Frequency = 2)

---

### Tensor & Shape Tracking
*   **Input string**: `str` (character sequence).
*   **Token IDs list**: `List[int]` of length $L$.
*   **Lookup Input**: `torch.Tensor([B, L])` (padded sequence).
*   **Embedding Output**: `torch.Tensor([B, L, d])`.

---

## 3. Implementation & Reference Code

Below is a self-contained Python implementation of the BPE merge training loop.

```python
from collections import defaultdict
import re

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges = {}
        
    def _get_stats(self, corpus: dict[str, int]) -> dict[tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
        
    def _merge_vocab(self, pair: tuple[str, str], corpus: dict[str, int]) -> dict[str, int]:
        new_corpus = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)
        for word, freq in corpus.items():
            new_word = pattern.sub(replacement, word)
            new_corpus[new_word] = freq
        return new_corpus
        
    def train(self, raw_corpus: dict[str, int], num_merges: int):
        # Format corpus to separate characters by space, add end-of-word marker '_'
        corpus = { ' '.join(list(word)) + ' _': freq for word, freq in raw_corpus.items() }
        
        # Initialize vocab
        for word in corpus.keys():
            self.vocab.update(word.split())
            
        print("Initial Vocab Size:", len(self.vocab))
        
        # Run merge steps
        for i in range(num_merges):
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            corpus = self._merge_vocab(best_pair, corpus)
            
            merged_token = ''.join(best_pair)
            self.vocab.add(merged_token)
            self.merges[best_pair] = merged_token
            print(f"Merge {i+1}: {best_pair} -> {merged_token} (Freq={pairs[best_pair]})")
            
        print("Final Vocab Size:", len(self.vocab))
        
# Verify execution
if __name__ == "__main__":
    raw_corpus = {
        "hug": 4,
        "rug": 3,
        "bug": 2
    }
    
    tokenizer = SimpleBPETokenizer()
    tokenizer.train(raw_corpus, num_merges=3)
```

---

## 4. Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
*   **Core Problem Solved:** Resolving the trade-off between out-of-vocabulary (`<unk>`) token loss and massive embedding parameters VRAM footprints.
*   **Why Introduced over Legacy Approaches:** Subword tokenizers allow open-vocabulary representations, handling out-of-training terms by falling back to constituent byte structures.
*   **Key Failure Modes & Limitations:** Tokenizers can partition numbers inconsistently (e.g. `"1000"` becomes `"10"` and `"00"`), making basic arithmetic difficult for models.

### 2. System Complexity & Scaling
*   **Time Complexity (FLOPs):** Tokenization search uses trie-structures running in $O(L_{\text{chars}})$. Training BPE scales as $O(M \cdot N_{\text{corpus}})$, where $M$ is target merge count.
*   **Space/Memory Footprint:** The vocabulary map stores $V$ keys. The model's embedding matrix parameter count is $V \times d$.
*   **Primary Bottleneck Type:** CPU-bound during pre-tokenization regex parsing and string lookups.
*   **Variable Legend:** $L_{\text{chars}}$ = Character Length of input, $V$ = Vocabulary Size, $d$ = Embedding Dimension, $M$ = Number of merges.

### 3. Production & Scalability
*   **Deployment Considerations:** Vocabulary mismatch is a common failure mode: feeding token IDs from a tokenizer vocabulary mismatching the target model's embedding table will completely break the model's output.
*   **Common Interviewer Follow-Up Questions:**
    1.  *Q:* What is the Byte-Fallback mechanism and why is it used in LLama tokenizers?
        *   *A:* Byte-fallback ensures that the tokenizer never outputs `<unk>` tokens. If BPE encounters a character outside its vocabulary (such as a rare emoji or foreign script), it decomposes the character into its raw UTF-8 byte values (0-255). Since all 256 individual byte tokens are pre-included in the vocabulary, BPE can represent any text stream as a sequence of byte tokens.
    2.  *Q:* Why do some tokenizers split digits into individual tokens?
        *   *A:* Tokenizers like Tiktoken (used in GPT-4) split numbers into single digits (e.g., `"348"` becomes `"3"`, `"4"`, `"8"`). If numbers are tokenized into subwords (like `"34"` and `"8"`), the model struggles to align values across arithmetic columns. Individual digit tokenization allows the model to process numbers digit-by-digit, improving arithmetic reasoning performance.
