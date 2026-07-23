# Instruction Tuning & SFT

Supervised Fine-Tuning (SFT) transitions a base next-token predictor into an assistant by training it on structured instruction-following sequences. Achieving high performance in production requires strict formatting guidelines, sequence efficiency, and target-only optimization.

---

## 1. Instruction-Following Paradigms & Data Formatting

Instruction datasets are structured to teach models specific behaviors. The two primary formats are:

1. **Alpaca Format (Single-turn)**:
   Useful for direct task completions (e.g., summarization, text extraction).
   ```json
   {
     "instruction": "Summarize the text below.",
     "input": "Large Language Models are...",
     "output": "LLMs are..."
   }
   ```
2. **ShareGPT Format (Multi-turn)**:
   Crucial for training chat assistants with conversational memory.
   ```json
   {
     "conversations": [
       {"from": "human", "value": "Explain gradient descent."},
       {"from": "gpt", "value": "Gradient descent is..."},
       {"from": "human", "value": "What is the learning rate?"},
       {"from": "gpt", "value": "The learning rate is..."}
     ]
   }
   ```

---

## 2. Chat Templates (Jinja2 & `apply_chat_template`)

To maintain consistency between training and inference, conversational datasets must format role boundaries using standard templates. Modern tokenizers utilize **Jinja2 chat templates** (e.g., Llama-3-Instruct or ChatML schemas) to generate standardized system-user-assistant sequences.

### ChatML Syntax Example
```text
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 is 4.<|im_end|>
```

In Python, this formatting is automated using Hugging Face's `apply_chat_template`:
```python
formatted_chat = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=False
)
```

---

## 3. Target-Only Loss Masking (`-100` Index)

During pre-training, causal models learn from every token in a document. However, during instruction tuning, our objective is to maximize output generation quality *conditioned* on the input instruction. If we calculate gradient updates on prompt tokens, the model's parameters drift to predict the user prompts, leading to sub-optimal response alignment and syntax degradation.

To prevent this, we apply **Target-Only Loss Masking** using PyTorch's `CrossEntropyLoss(ignore_index=-100)`. By setting target label indices of prompt tokens to `-100`, the loss calculations ignore these positions.

### Step-by-Step Numerical Walkthrough

Let's trace target labels creation for the sequence:
`"User: What is 2+2? Assistant: 4."`

#### 1. Tokenization
Suppose our tokenizer maps the text to 10 token IDs:
- **Prompt (User)**: `["User", ":", " What", " is", " 2", "+", "2", "?"]` $\rightarrow$ `[502, 29, 2043, 318, 122, 10, 122, 30]` (8 tokens)
- **Response (Assistant)**: `[" Assistant", ":", " 4", "<eos>"]` $\rightarrow$ `[4321, 29, 14, 2]` (4 tokens)
- **Combined input_ids**: `[502, 29, 2043, 318, 122, 10, 122, 30, 4321, 29, 14, 2]` (12 tokens)

#### 2. Standard Labels (Causal Shift-Left)
Normally, target labels are identical to input IDs shifted one position to the left (to predict the next token):
$$\text{input\_ids} = [502, 29, 2043, 318, 122, 10, 122, 30, 4321, 29, 14, 2]$$
$$\text{standard\_labels} = [29, 2043, 318, 122, 10, 122, 30, 4321, 29, 14, 2, \text{PAD}]$$

#### 3. SFT Target Masking Applied
We mask all prompt tokens (including role headers up to the assistant's response start) by setting their label values to `-100`.
- Prompt token indices are 0 through 9 (including `"Assistant"`, `":"` headers).
- Response token index starts at 10 (`"14"`) and ends at 11 (`"<eos>"`).

Therefore, the target labels tensor becomes:
$$\text{sft\_labels} = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 14, 2]$$

During training:
- Input token $30$ (`"?"`) predicts output $4321$ (`"Assistant"`). The loss for this step is skipped (label is `-100`).
- Input token $29$ (`":"`) predicts output $14$ (`"4"`). The loss is calculated and backpropagated.

---

## 4. Sequence Packing (FlashAttention Packing)

In a raw batch of sequences, sentences have variable lengths. Standard training pads shorter sequences to the maximum block length, leading to massive memory wastage on padding tokens.

```
Standard Padding:
[ Seq 1 (500 tokens) | Padding (1548 tokens) ] -> Block Size 2048
[ Seq 2 (1200 tokens) | Padding (848 tokens) ] -> Block Size 2048
```

**Sequence Packing** concatenates multiple samples into a single block of size 2048 or 4096, using special attention masking to prevent tokens in Sequence A from attending to tokens in Sequence B.

```
Packed Sequences:
[ Seq 1 (500 tokens) | Seq 2 (1200 tokens) | Seq 3 (300 tokens) | Pad (48 tokens) ] -> Block Size 2048
```

By passing a 1D attention boundary mask to FlashAttention, sequence packing completely avoids wasting GPU compute on pad tokens, improving SFT throughput by $2\times - 3\times$.

---

## 5. System, User, and Assistant Role Hygiene

When compiling instruction datasets, role hygiene must be enforced:
- **System Prompt Separation**: Ensure system prompts are only injected at the start of multi-turn dialogues, never repeated within user turns.
- **Empty Prompts**: Strip trailing empty lines or spaces around tokens, as they lead to generation artifacts during inference.
- **Out-of-Vocabulary (OOV) Special Tokens**: Ensure ChatML role markers (like `<|im_start|>`) are added as **special tokens** in the tokenizer configuration so they are treated as single atomic tokens rather than parsed as sub-words.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Instruction tuning converts raw next-token estimators into conversational partners, ensuring response generation matches user instructions rather than completing the query text.
- **Why was it introduced?**
  Pre-trained base models failed to understand instruction context natively, outputting continuation text (e.g. outputting another question) instead of the actual answer.
- **What are its limitations?**
  Over-alignment on specific prompts can degrade generalization capability. Packing sequences requires specialized 3D attention masks to avoid context cross-contamination.
- **Computational Complexity (Time & Memory)**
  - **Memory Efficiency (Packing)**: Saves $O(L_{\text{pad}})$ memory footprint by reducing token dimension length to the absolute minimum required.
  - **Loss computation**: Calculating cross-entropy only on target outputs reduces active loss backpropagation steps.
- **Component Variable Denotation Legend**
  - $L_{\text{pad}}$: Pad token count.
  - $N_{\text{seq}}$: Number of independent sequences packed together.
  - $L$: Sequence length of the target block size (e.g. 2048).
- **Production Use Cases**
  - Fine-tuning a customer support chatbot formatting strictly using assistant tokens.
  - Training models to output structured API arguments using target-only labels formatting.
- **Follow-up questions interviewers ask**
  - *How do you write a custom PyTorch collation function to mask out user instruction tokens dynamically?*
  - *If a packed batch contains 5 concatenated user conversations, how does the model prevent user context leakage between them?*
