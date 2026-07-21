# Workspace Customization Rules: DS & AI Engineer Interview Preparation

This document outlines the core background, architectural guidelines, code standards, and deliverable constraints for all AI agents assisting with study prep in this repository.

---

## 1. User Profile and Context

- **Experience Level:** AI Engineer with ~3 years of experience.
- **Primary Technical Domains:** 
  - **GenAI & Agentic AI:** Multi-agent orchestration, tool usage, planning, reflection, RAG (dense/sparse retrieval, reranking), prompt engineering, vector databases, and fine-tuning (PEFT/LoRA/QLoRA).
  - **Machine Learning & Deep Learning:** Classical ML algorithms, Transformer architectures, sequence models (RNN/LSTM/GRU), computer vision, embeddings, optimization algorithms, and model evaluation metrics.
- **Goal:** Prep for Data Science (DS), Applied AI Engineer, GenAI Engineer, and ML Engineer interviews at top companies (FAANG, tier-1 tech startups).
- **Target Knowledge Profile:** Focus on cracking high-bar interviews (systems design, production trade-offs, machine learning intuition, and debugging/refactoring) rather than academic derivations or pedantic schoolbook memorization.

---

## 2. Core Behavioral & Pedagogical Rules

1. **No Academic Filler:**
   - Skip introductory textbook summaries. Focus on the core mechanics, math intuition, system failure modes, and production realities tested in top tech interviews.
2. **Focus on Production Trade-offs:**
   - Every mathematical formula and system component must map directly to:
     - An evaluation choice (e.g., ROC-AUC vs. PR-AUC, BLEU vs. ROUGE, Precision vs. Recall, Perplexity).
     - A troubleshooting scenario (e.g., singular matrices, vanishing/exploding gradients, Softmax saturation, data drift, OOV handling).
     - A system constraint (e.g., latency budget, CPU vs. GPU memory footprint, INT8/FP16 quantization, batch size vs. throughput).
3. **Andrew Ng Coursera Style (Practical Mathematics):**
   - Do NOT skip core mathematics or formulas, but explain them with **simple, practical mathematical intuition and step-by-step hand-calculations on tiny sample datasets** (e.g., 3-sample datasets, 3-token matrix operations, scalar step calculations).
   - Follow this dual structure: **Intuitive Math Formula $\rightarrow$ Step-by-Step Hand Calculation on Tiny Sample $\rightarrow$ Production Code & Output Logs**.
4. **Notebooks and Code Style:**
   - Notebooks must reflect clean, production-grade, framework-agnostic Python code (PyTorch, Scikit-Learn, NLTK, Gensim, NumPy).
   - All code must be self-contained, fully executed with saved cell outputs, accompanied by clear log outputs and complexity analyses ($O(N)$ time & memory).

---

## 3. Formatting & Markdown Syntax Constraints

1. **KaTeX & LaTeX Math Block Compatibility:**
   - Use standard `$ ... $` for inline math and single-line `$$ ... $$` for display math blocks.
   - Avoid unescaped raw line breaks or unescaped characters inside math subscripts/superscripts (e.g., write `$$\mathcal{L}_{\text{loss}} = \dots$$` on a single line).
2. **Native GFM Markdown Tables (No Fenced Code Block Wrappers):**
   - NEVER wrap Markdown tables inside fenced backticks (` ```text ` or ` ``` `).
   - Always write native GFM Markdown tables (`| Header 1 | Header 2 |`) so renderers compile real HTML `<table>` elements with explicit grid cell boundaries and rendered KaTeX formulas.
3. **Code Syntax Highlighting & Background Transparency:**
   - HTML and PDF build scripts must enforce Pygments code syntax highlighting (`monokai` or dark slate theme).
   - Ensure CSS enforces transparent backgrounds and crisp white text defaults inside code blocks (`.codehilite, .codehilite pre, .codehilite code, .codehilite span { background-color: transparent !important; color: #f8fafc !important; }`) to prevent dark text on dark backgrounds or white box artifacts.

---

## 4. Standardized Study Guide Structure

Every module study guide (`.md`) across any topic must conclude with a standardized interview section:

```markdown
### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Production Use Cases
- Follow-up questions interviewers ask
```

---

## 5. Execution & Deliverable Workflow

1. **Sequential Module Refinement:**
   - Work module-by-module sequentially (update `.md` study guide + companion `.ipynb` notebook), executing and verifying cell outputs on the spot before proceeding.
2. **Comprehensive & Concise Deliverables:**
   - Maintain both a full comprehensive master PDF/HTML guide and a concise 1-page revision cheatsheet (`*_Interview_Cheatsheet.pdf`) for each major study topic.
