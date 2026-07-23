# Module 05: Tokenization & Embeddings

This study guide explains the engineering design and mechanics of Tokenization and Embedding layers in LLMs, detailing vocabulary compression algorithms (BPE, WordPiece, SentencePiece), Unicode safety, and embedding similarity mappings.

---

## 1. The Tokenization Pipeline

Tokenization splits raw string inputs into a sequence of discrete integers (token IDs) queryable by the embedding layer.

```text
[Raw Input String] ──► [Subword Tokenizer] ──► [Token IDs] ──► [Embedding Layer]
```

### Paradigms:
1. **Character-level**: Splitting text into single characters (e.g. `c-a-t`).
   - *Pros*: Tiny vocabulary size ($|V| \approx 256$), zero OOV words.
   - *Cons*: High sequence length (depletes context VRAM), lacks direct semantic chunks.
2. **Word-level**: Splitting text by whitespaces and punctuation.
   - *Pros*: Simple, maps words to clean meaning.
   - *Cons*: Massive vocabulary ($|V| > 1\text{M}$), unable to handle grammatical modifications (e.g. `run` vs. `running` are separate vectors), crashes on Out-of-Vocabulary (OOV) terms.
3. **Subword-level (Modern Standard)**: Decomposes words into frequent subword chunks (e.g. `tokenization` = `token` + `ization`), balancing vocabulary size with sequence lengths.

---

## 2. Vocabulary Compression Algorithms

### Byte Pair Encoding (BPE)
BPE builds a vocabulary bottom-up, starting with single characters and iteratively merging the most frequent adjacent token pairs.
- **Used by**: GPT family (tiktoken), Llama, RoBERTa.

#### Step-by-Step Hand Calculation (BPE Vocab Expansion)
- **Scenario**: Tiny training corpus counts:
  - `"l o w </w>"`: 5 times
  - `"l o w e r </w>"`: 2 times
  - `"n e w e s t </w>"`: 6 times
  - `"w i d e s t </w>"`: 3 times
  - *Base Vocabulary*: `{'l', 'o', 'w', 'e', 'r', 'n', 'w', 'i', 'd', 's', 't', '</w>'}` (12 tokens).
- **Calculation**:
  1. Count adjacent pairs:
     - `e s`: occurs in `"newest"` ($6$) + `"widest"` ($3$) = $9$ times.
     - `s t`: occurs in `"newest"` ($6$) + `"widest"` ($3$) = $9$ times.
     - `t </w>`: occurs in `"newest"` ($6$) + `"widest"` ($3$) = $9$ times.
     - `l o`: occurs in `"low"` ($5$) + `"lower"` ($2$) = $7$ times.
  2. Merge the most frequent pair `e s` into `es`.
     - *Updated Corpus*:
       - `"l o w </w>"`: 5 times
       - `"l o w e r </w>"`: 2 times
       - `"n e w es t </w>"`: 6 times
       - `"w i d es t </w>"`: 3 times
     - *New Vocabulary*: Base characters + `{'es'}` (13 tokens).
  3. Re-count adjacent pairs:
     - `es t`: occurs in `"newest"` ($6$) + `"widest"` ($3$) = $9$ times.
     - `t </w>`: occurs in `"newest"` ($6$) + `"widest"` ($3$) = $9$ times.
     - `l o`: occurs in `"low"` ($5$) + `"lower"` ($2$) = $7$ times.
  4. Merge `es t` into `est`.
     - *Updated Corpus*:
       - `"l o w </w>"`: 5 times
       - `"l o w e r </w>"`: 2 times
       - `"n e w est </w>"`: 6 times
       - `"w i d est </w>"`: 3 times
     - *New Vocabulary*: Base characters + `{'es', 'est'}` (14 tokens).

### WordPiece
- **Concept**: Selects merges based on likelihood ratio maximization rather than raw pair frequency. It merges `A` and `B` if it maximizes:
  $$\text{Score}(A, B) = \frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$$
- **Used by**: BERT, DistilBERT.

### SentencePiece & Unigram
- **SentencePiece**: Fits tokenizers directly on raw bytes, treating spaces as normal characters (`_`), eliminating language-specific whitespace parser pre-processing.
- **Unigram**: Starts with a massive base vocabulary and iteratively *prunes* low-probability tokens using an EM entropy likelihood step until target $|V|$ is met.

---

## 3. Production Tokenizer Challenges

### Out-of-Vocabulary (OOV) Mitigation
If an unseen word occurs at runtime, traditional models fallback to an unknown token `<unk>`, losing input information.
- **Byte-level BPE**: Converts strings to UTF-8 bytes before running BPE. The base vocabulary holds the 256 possible byte values. This guarantees *any* string can be decomposed into bytes without ever needing the `<unk>` token.

### Unicode & Multi-Byte Character Handling
Unicode characters (e.g. emojis or non-Latin alphabets) occupy 2 to 4 bytes. If a byte-level tokenizer splits a multi-byte character in the middle, it can create invalid byte strings, causing decoder reconstruction errors. SentencePiece and tiktoken prevent this by grouping valid UTF-8 character byte blocks together during pre-tokenization.

---

## 4. Embedding Layer Mechanics

An embedding layer acts as a matrix lookup $\mathbf{W}_E \in \mathbb{R}^{|V| \times d}$. Given a token ID $i$, it extracts the corresponding row representation vector $\mathbf{v}_i \in \mathbb{R}^d$.
- **Static Embeddings (Word2Vec)**: Each token has a single representation vector, ignoring grammatical context.
- **Contextual Embeddings (Transformers)**: Raw token embedding vectors are combined with positional embeddings and transformed by attention blocks, outputting context-specific representations (e.g. resolving meaning variations of `bank`).

---

## 5. Interview Questions & Production Trade-offs

### What problem does this solve?
Transforms discrete text strings into continuous, dense semantic vectors that can be processed by neural architectures.

### Why was it introduced?
Subword tokenizers (like BPE) compile compact vocabularies ($|V| \approx 32\text{k} - 128\text{k}$), resolving word-level OOV bottlenecks and character-level sequence-length inflation.

### What are its limitations?
Tokenizers struggle with numbers (e.g. `12345` split into `12` and `345`), non-Latin scripts (requiring more tokens per word), and spelling anomalies.

### Computational Complexity (Time & Memory)
- **BPE encoding lookup**: $O(L \cdot \log |V|)$ using trie structures.
- **Embedding extraction**: $O(L \cdot d)$ memory bandwidth lookup.

### Component Variable Denotation Legend
- $L$: Input string length in characters.
- $|V|$: Vocabulary size (number of token entries).
- $d$: Embedding projection dimension.

### Production Use Cases:
- Web scraping pipelines using Byte-level BPE (tiktoken) to ingest raw text data containing emojis or symbols without OOV crashes.
- Vector search DBs calculating cosine similarity over token embeddings to find semantic duplicates.

### Follow-up Questions Interviewers Ask:
1. *Why does WordPiece merge tokens using the score $\frac{\text{Count}(A, B)}{\text{Count}(A) \times \text{Count}(B)}$ instead of BPE's raw frequency?*
   - **Answer**: Raw frequency BPE prioritizes common character merges (like `e s` or `t h`). WordPiece divides raw counts by individual frequencies. If `A` and `B` are highly common words (e.g., `of` and `the`), they are rarely merged. It only merges tokens that co-occur significantly more than expected by chance, capturing structural semantic chunks (like `un` + `happy`).
2. *Why do non-Latin languages (e.g. Japanese, Hindi) consume significantly more token budget in standard LLMs?*
   - **Answer**: Modern tokenizer vocabularies are trained on English-skewed corpora. English words map to single tokens (e.g. `apple`), whereas non-Latin words are split into multiple byte-level tokens because the combinations are rare in the corpus. This means non-Latin prompts consume $2\times - 4\times$ more token budget for the same semantic meaning.
3. *How does Byte-level BPE prevent the generation of invalid UTF-8 strings at decoder outputs?*
   - **Answer**: By restricting BPE merges. Standard BPE could merge bytes across different characters, creating invalid segments. To prevent this, the tokenizer enforces that merges cannot cross character boundaries (using regex split rules on character blocks), keeping all byte combinations valid UTF-8 sequences.
