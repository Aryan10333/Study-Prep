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

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Compilation Pipeline Architecture</div>
    <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Left Box: Source -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center;">
            <div style="color: #475569; font-weight: bold; font-size: 11px; text-transform: uppercase;">Raw Source Files</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center;">01_Theory.md</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center;">02_Theory.md</div>
            <div style="background-color: #f5f3ff; color: #5b21b6; border: 1px solid #7c3aed; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">13_Interview_Questions.md</div>
        </div>
        <!-- Right Box: Formats -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 8px; align-items: center;">
            <div style="color: #475569; font-weight: bold; font-size: 11px; text-transform: uppercase;">Compilation Outputs</div>
            <div style="background-color: #eff6ff; color: #1e40af; border: 1px solid #2563eb; padding: 6px 12px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">Standalone Cheatsheet (PDF)</div>
            <div style="background-color: #ecfdf5; color: #065f46; border: 1px solid #059669; padding: 6px 12px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">Master Study Guide (PDF)</div>
        </div>
    </div>
</div>

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
