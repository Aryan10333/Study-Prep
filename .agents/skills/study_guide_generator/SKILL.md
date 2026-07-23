---
name: Study Guide Generator
description: Guidelines and constraints for generating high-quality Markdown study guides with practical mathematical derivations, GFM tables, and standard ending checklists.
---

# Study Guide Generator Skill

This skill defines the standardized pedagogy, formatting, and mathematical conventions for creating and refining Markdown study guides (`.md`) in this repository.

---

## 1. High-Level Conceptual Introduction

Every study guide and technical topic must begin with a short, high-level conceptual introduction that orients the reader. This introduction must explain the core engineering motivation and the specific issue the method resolves rather than skipping context entirely or jumping straight to equations.

---

## 2. Pedagogical Style: Andrew Ng Coursera Method

All technical study guides must follow a strict **Practical Mathematics** progression to make equations intuitive:

1. **State the Math Formula**: Express the mathematical formulation clearly using display math blocks.
2. **Step-by-Step Hand Calculation**: Break down the calculation on a tiny sample space (e.g. sequence length $L=2$, dimensions $d=2$, vocabulary size $|V|=3$). Work out intermediate sums, products, and exponents manually.
3. **Reference Code**: Write framework-agnostic Python/PyTorch code that computes and prints these values, showing exact consistency (to 4 decimal places) with the hand calculation.

---

## 3. Formatting & Syntax Constraints

### KaTeX & LaTeX Math Formatting:
- Use standard single dollar signs `$ ... $` for inline math.
- Use single-line double dollar signs `$$ ... $$` for display math blocks (avoid unescaped raw line breaks inside display blocks).
- Always escape percent signs inside math blocks (use `\%`) to prevent the KaTeX engine from interpreting them as line comments.
- Do not use unescaped subscripts/superscripts.

### Premium Headings Styling:
- **Heading 1 (`#`)**: Main titles. In compiled guides, styled as `#0f172a` with a `2px solid #3b82f6` bottom border.
- **Heading 2 (`##`)**: Topics. Styled as `#1e40af` (dark blue) with a `1px solid #e2e8f0` bottom border.
- **Heading 3 (`###`)**: Subtopics. Styled as `#0369a1` (light blue/cyan).

### Native GFM Tables:
- Never wrap Markdown tables in fenced backticks (` ```text ` or ` ``` `).
- Write native GitHub Flavored Markdown (GFM) tables directly:
  ```markdown
  | Header 1 | Header 2 |
  |---|---|
  | Cell 1 | Cell 2 |
  ```
  This allows Markdown compilers to output standard HTML `<table>` elements with clean borders and correct KaTeX rendering inside cells.

### Code Syntax Contrast:
- Fenced code blocks must compile using Pygments syntax highlighting (`monokai` or dark slate).
- Ensure styling overrides are present in target configurations to maintain transparent code block backgrounds and clean white text defaults, preventing dark-text-on-dark-background rendering errors.

### Native HTML/CSS Diagrams (No Mermaid or ASCII blocks):
- Do not use raw text-based ASCII flowcharts or Mermaid code blocks in study guides, as they do not render reliably in PDF compilers and require external CDN loading.
- Instead, construct diagrams using **pure inline HTML/CSS blocks** (using flexbox/grid containers, rounded borders, colored headers, and simple arrows). This guarantees that diagrams render instantly, print as vector-sharp shapes, and match the target A4 styles perfectly without external JavaScript.

---

## 4. Standardized Checklist Closure

Every module study guide (`.md`) across any topic must conclude with the following standardized section to connect theory to production:

```markdown
### Interview Questions & Production Trade-offs
- What problem does this solve?
- Why was it introduced?
- What are its limitations?
- Computational Complexity (Time & Memory)
- Component Variable Denotation Legend (Explicitly defining $N, L, |V|, d, m, K, T, C, P$)
- Production Use Cases
- Follow-up questions interviewers ask
```

---

## 5. Verification Check
Immediately after generating or modifying any markdown guide, perform these checks:
- **Incremental Creation**: Verify that the files were constructed incrementally (one section or file at a time or in logical batches) to ensure dense, high-quality information.
- **On-the-spot Review**: Inspect the material immediately after creation to confirm that it is self-sufficient, highly detailed, complete, and contains zero placeholders or "todo" shorthand.
- **Syntax Checks**:
  - Math block rendering and delimiters (inline and block KaTeX).
  - Native HTML table borders (no code blocks wrapping tables).
  - Code blocks for background transparency and text color contrast.
- **Completeness**: Ensure the standardized trade-offs checklist closes the module.

