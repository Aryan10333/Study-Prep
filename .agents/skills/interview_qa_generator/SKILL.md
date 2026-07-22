---
name: Interview QA Generator
description: Guidelines and instructions for generating high-quality interview preparation questions & answers in MD, HTML, and PDF formats, keeping them separate from theory documents.
---

# Interview Q&A Generation & Compilation Skill

This skill defines the standardized process for creating dedicated, high-impact **Interview Question & Answer** prep materials (Q&As, cheatsheets) and compiling them to independent HTML and PDF outputs, keeping them distinct from theoretical curriculum guides.

---

## 1. Context & Motivation

To crack high-bar technical interviews (AI Engineer, GenAI Engineer, ML Engineer), candidates need study guides that translate raw curriculum theory into quick screening responses, technical calculations, and production trade-offs. 

By separating **Theory Modules** (which contain deep conceptual text, diagrams, and code notebooks) from **Interview Q&As** (which focus on screening speed, buzzwords, and system-level trade-offs), we allow for:
1. Focused, high-intensity review sessions prior to interviews.
2. Lightweight standalone cheatsheets and PDFs (e.g., `*_Interview_Cheatsheet.pdf`) optimized for quick lookup.
3. Logical compilation where Q&A guides append as the final chapter in master guides.

---

## 2. Standardized Q&A Format

For every interview question generated, the output must strictly follow this structure:

```markdown
## Question [Number]: [Question Title]

### Short Interview Answer (30–60 seconds)
[Provide a concise, high-impact screening answer suitable for initial technical screens.]

### Key Interview Points
- [Keyword/Buzzword 1]
- [Keyword/Buzzword 2]
- [Keyword/Buzzword 3]

### Technical Intuition & Complexity (where applicable)
[Specify formulas, mathematical intuition, complexity classes (O(L^2), etc.), and tiny numerical step-by-step hand calculations on micro-samples (e.g. L=2, d=2) in KaTeX syntax.]

### Production Perspective & Trade-offs
[Analyze VRAM footprints, memory bandwidth limitations, latency budgets, GPU SM occupancy, and server bottlenecks.]

### Follow-up Questions
- **Follow-up**: [Question] -> [Answer]

### Common Mistakes
- [Explain standard misconceptions and incorrect answers candidates make]
```

### Global Selection Rule:
Not every question requires math formulas, numerical examples, complexity analysis, or code snippets. Include them only when they representationally improve the explanation (e.g., Self-Attention, KV Cache, RoPE, Scaling Laws, MoE). For purely conceptual questions, prioritize intuition, architecture, and production constraints.

---

## 3. Compilation Architecture

Compilation scripts (e.g., written in Python) must compile the Q&A documents separately from raw theory documents when building standalone cheatsheets, while also supporting consolidated master guides.

```mermaid
graph TD
    subgraph Raw Source Files
        T1[01_Theory.md]
        T2[02_Theory.md]
        QA[13_Interview_Questions.md]
    end

    subgraph Independent Compilation
        QA -->|compile_cheatsheet.py| CS_HTML[Cheatsheet.html]
        CS_HTML -->|msedge headless| CS_PDF[Cheatsheet.pdf]
    end

    subgraph Master Compilation
        T1 -->|compile_master.py| M_HTML[Master_Guide.html]
        T2 -->|compile_master.py| M_HTML
        QA -->|compile_master.py (Final Chapter)| M_HTML
        M_HTML -->|msedge headless| M_PDF[Master_Guide.pdf]
    end
```

### PDF & HTML Build Rules:
1. **Alert Parsing**: Convert GitHub alert styles (`> [!NOTE]`, etc.) into left-bordered HTML divs with background colors.
2. **Pygments Highlighting**: CSS must enforce dark slate or monokai code block syntax styling (`HtmlFormatter(style='monokai')`).
3. **Contrast Enforcement**: Prevent dark-on-dark text artifacts inside code blocks using strict CSS overrides:
   ```css
   .codehilite, .codehilite pre, .codehilite code, .codehilite span {
       background-color: transparent !important;
       color: #f8fafc !important;
   }
   .codehilite {
       background-color: #0f172a !important;
       border-radius: 6px;
       border: 1px solid #334155;
   }
   ```
4. **KaTeX Integration**: Embed standard KaTeX scripts and JS auto-renderers in headers to compile LaTeX math formulas dynamically:
   ```html
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
   <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
   <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
   ```
5. **Headless Browser Rendering**: Run headless Microsoft Edge command line printer to convert compiled HTML output to standard-aligned PDFs:
   ```python
   cmd = [
       "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
       f"--user-data-dir={temp_user_data}",
       "--headless",
       "--disable-gpu",
       "--run-all-compositor-stages-before-draw",
       "--no-pdf-header-footer",
       f"--print-to-pdf={pdf_out_path}",
       html_out_path
   ]
   ```

---

## 4. Verification Checklist

Before final delivery, verify:
- [ ] **Math delimiters**: Single `$` for inline math, double `$$` on a single line for display blocks.
- [ ] **GFM Tables**: Tables are written using native Markdown formatting and are never wrapped inside fenced code blocks.
- [ ] **No placeholders**: Every question has a complete answer without placeholders.
- [ ] **Variable consistency**: Variable notations ($L$, $d$, $b$, $N$, $C$) align 100% between complexity sheets and explanations.
