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

1. **Limit Complex Math**: Limit dense academic notation and formal algebraic proofs (e.g., proof of speculative rejection sampling lossless properties, multi-index matrix tiling subscripts). Prune step-by-step calculus proofs (like multi-step partial derivative backpropagation chains or tedious matrix indexing algebra expansion) if they do not add direct value to conceptual interview situations. Replace them with clear conceptual descriptions, final structural mathematical equations, stability conditions (like eigenvalues or additive gradient flows), intuitive analogies, and comparative tables highlighting pros, cons, and production trade-offs.
2. **State the Essential Math Formula**: Express the mathematical formulation clearly using display math blocks (e.g. sizing/VRAM equations, quantization scaling, arithmetic intensity).
3. **Step-by-Step Hand Calculation**: Break down the calculation on a tiny sample space (e.g. sequence length $L=2$, dimensions $d=2$, vocabulary size $|V|=3$). Work out intermediate sums, products, and exponents manually.
4. **Reference Code**: Write framework-agnostic Python/PyTorch code that computes and prints these values, showing exact consistency (to 4 decimal places) with the hand calculation. If intermediate hand-calculation rounding causes a slight numerical offset (e.g. `27.8600` vs. `27.8592`), explicitly document both the rounded hand-calculation result and the exact unrounded floating-point values computed by the code to maintain 100% transparency.

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

### Premium HTML/CSS and Inline SVG Diagrams (No Mermaid or ASCII blocks):
- Do not use raw text-based ASCII flowcharts or Mermaid code blocks in study guides, as they do not render reliably in PDF compilers and require external CDN loading.
- Instead, construct diagrams using **pure inline HTML/CSS blocks** (using flexbox/grid containers, rounded borders, colored headers, and simple arrows) or **inline SVG markup** (defining paths, rects, text, and arrow markers). This guarantees that diagrams render instantly, print as vector-sharp shapes, and match the target A4 styles perfectly without external JavaScript.
- **Label Subscript Formatting**: Inside HTML/CSS/SVG diagram labels, do not use raw LaTeX/KaTeX subscripts (like `w_{t-2}`). Use native HTML `<sub>` and `<sup>` tags instead (e.g. `w<sub>t-2</sub>`, `v<sub>w<sub>t</sub></sub>`) to ensure they render beautifully and align correctly in headless browser PDF printers.
- **Long Equation Management**: To prevent horizontal overflow and clipping of long mathematical formulas in PDF/print sheets, split equations across multiple display blocks or separate lines in the markdown source.

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

**CRITICAL**: Do not leave these items as empty placeholder bullet points. You must replace them with comprehensive, customized answers, math complexity parameters, variable legends, and typical follow-up questions tailored specifically to the module's technical topic.

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

