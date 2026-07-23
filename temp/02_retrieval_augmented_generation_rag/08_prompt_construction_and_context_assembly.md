# Module 08: Prompt Construction & Context Assembly

This guide details Context Selection, "Lost in the Middle" mitigation, Grounded System Prompts, Automated Citation Mapping (`[Source 1: Doc A, Page 3]`), Context Truncation, and Answer Formatting rules.

> **Notebook Companion**: [08_prompt_construction_and_context_assembly.ipynb](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/02_retrieval_augmented_generation_rag/08_prompt_construction_and_context_assembly.ipynb)

---

## 1. Context Assembly & "Lost in the Middle" Mitigation

Self-attention layers exhibit highest recall for tokens located at the very beginning (top of prompt) and very end (bottom of prompt).

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 1. SYSTEM ROLE & GROUNDING RULES                                                │
│    (High Recall Zone - Top of Prompt)                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 2. MOST RELEVANT RETRIEVED CHUNK #1                                              │
│    (High Recall Zone)                                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 3. MODERATE RELEVANT CHUNKS #2, #3, #4                                           │
│    (ATTENTION DEGRADATION ZONE - Middle of Prompt)                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 4. MOST RELEVANT RETRIEVED CHUNK #2                                              │
│    (High Recall Zone)                                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 5. TARGET USER QUERY                                                             │
│    (High Recall Zone - Bottom of Prompt)                                        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Grounded System Prompt Engineering

To eliminate hallucinations, grounded prompts must enforce strict negative constraints:

```text
You are an Enterprise AI Assistant. Answer the user question STRICTLY using only the provided context passages below.
If the context does not contain sufficient facts to answer the question, state: "I cannot answer this question based on the provided enterprise documentation."
Do NOT use internal parametric knowledge or make assumptions.
```

---

## 3. Context Truncation & Token Budget Management

When injecting retrieved chunks, total context must not exceed model limits or latency SLAs:
- **Token Budget Equation**:
  $$\text{Budget}_{\text{context}} = \text{Limit}_{\text{max}} - (\text{Tokens}_{\text{system}} + \text{Tokens}_{\text{query}} + \text{Tokens}_{\text{generation}})$$
- **Dynamic Context Packing**: Sort retrieved chunks by relevance score and append iteratively until $\text{Budget}_{\text{context}}$ is reached.

---

## 4. Production LangChain Citation Mapping Implementation

```python
sources = [
    {"doc": "Medical_Record_A.pdf", "page": 3, "text": "Patient given 50mg Amoxicillin."},
    {"doc": "Intake_Form_B.pdf", "page": 1, "text": "No known allergies."}
]

context_text = ""
for idx, s in enumerate(sources, 1):
    context_text += f"[Source {idx}: {s['doc']}, Page {s['page']}]\n{s['text']}\n\n"

print("=== Assembled Grounded Prompt Context ===")
print(context_text)
```