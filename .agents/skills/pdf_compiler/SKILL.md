---
name: PDF & HTML Master Compiler
description: Guidelines and scripts for compiling multiple Markdown modules into unified HTML and PDF master guides using python-markdown, pygments, and headless Microsoft Edge.
---

# PDF & HTML Master Compiler Skill

This skill defines the aggregation, preprocessing, and layout printing standards for generating master documentation (HTML, PDF) from Markdown source files.

---

## 1. Aggregation & Pre-processing Pipeline

To compile clean, professional PDF study guides, compilation scripts must follow these processing steps:

1. **Sequential Concatenation**: Read the markdown source files in logical curriculum order.
2. **Alert Formatting**: Preprocess custom alert tags (e.g. `> [!NOTE]`) and convert them to left-bordered HTML divs:
   - **NOTE**: Blue border (`#2563eb`), light blue background (`#eff6ff`), dark blue text (`#1e40af`).
   - **TIP**: Green border (`#059669`), light green background (`#ecfdf5`), dark green text (`#065f46`).
   - **IMPORTANT**: Purple border (`#7c3aed`), light purple background (`#f5f3ff`), dark purple text (`#5b21b6`).
   - **WARNING**: Orange border (`#d97706`), light orange background (`#fffbeb`), dark orange text (`#92400e`).
   - **CAUTION**: Red border (`#dc2626`), light red background (`#fef2f2`), dark red text (`#991b1b`).

---

## 2. Math & Code Protection Rules

During compilation, Markdown parsers can corrupt LaTeX syntax and code formats. Use the following techniques to safeguard them:

1. **Math Block Extraction**: Use regex to extract all inline math (`$ ... $`) and display blocks (`$$ ... $$`), replacing them with placeholders (e.g., `MATHPLACEHOLDER_0`) before calling the parser.
2. **Markdown Parsing**: Run `markdown.markdown()` with extensions:
   `['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']`
3. **Math Block Restoring**: Swap the LaTeX equations back in place of their corresponding placeholders.
4. **Pygments Code Styling**: Generate and append Pygments CSS styling using the `monokai` dark theme. Apply strict CSS overrides to avoid dark-text-on-dark-background errors:
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

## 3. Standard A4 Design System (from NLP Master Guide)

The compiled guide must use the standardized NLP master guide CSS and layout structure:
- **Page Dimensions**: Enforce standard A4 size with explicit margins:
  ```css
  @page {
      size: A4;
      margin: 18mm 15mm 18mm 15mm;
  }
  ```
- **Typography & Colors**:
  - Font: `'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif`
  - Font Size: `13px` base text size with `1.6` line height
  - Base Text Color: `#1e293b`
  - Header 1 Color: `#0f172a` with a `#3b82f6` blue bottom border
  - Header 2 Color: `#1e40af` with a `#e2e8f0` light grey bottom border
  - Header 3 Color: `#0369a1`
- **Cover Page Elements**: Include a metadata box detailing the target persona, core modules covered, and inclusion checklists (e.g. formulas, PyTorch code, Q&As).
- **Table of Contents Page**: Inject a dedicated Table of Contents page (`<div class="toc-page" style="page-break-after: always; padding-top: 20px;">`) with dashed-underlined lists linking to each module anchor.
- **HTML/CSS Inline Diagrams**: Ensure diagrams in guides use inline styled HTML/CSS elements to ensure clean, instant vector-perfect rendering without external JavaScript/CDN dependencies. Do not import Mermaid scripts in the HTML headers.

---

## 4. PDF Generation via Headless Microsoft Edge

Standard print engines often misalign KaTeX CSS layouts. To ensure perfect printing alignment, run headless Microsoft Edge to render HTML directly to PDF:

```python
import subprocess

cmd = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    f"--user-data-dir={temp_user_data_path}",
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=8000",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_output_path}",
    html_input_path
]
subprocess.run(cmd, capture_output=True, text=True)
```

---

## 5. Verification Check
- Verify that a cover page with center alignment, gradient borders, and the target persona box is present.
- Check that a Table of Contents page is compiled immediately following the cover page.
- Check that page breaks occur cleanly before each new module container.
- Ensure that KaTeX math renders correctly on the generated PDF pages.

