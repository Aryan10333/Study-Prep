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

## 3. Standard A4 Design System (from Master Guide)

The compiled guide must use the standardized master guide CSS and layout structure:
- **Page Dimensions**: Enforce standard A4 size with explicit margins:
  ```css
  @page {
      size: A4;
      margin: 20mm 18mm 20mm 18mm;
  }
  ```
- **Typography & Color Hierarchy**:
  - Font: `'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif`
  - Font Size: Consistent `14.5px` base text size for `body`, `p`, and `li` elements with `1.6` line height to ensure reading uniformity.
  - Base Text Color: Modern slate-grey `#334155`.
  - Inline Code size: Set to `13.5px` (using `!important`) to prevent size mismatch with body text.
  - Header 1 (`h1`): `22px` styled in `#0f172a` with a `#3b82f6` blue bottom border.
  - Header 2 (`h2`): `18px` styled in Royal Blue `#2563eb` with a light grey `#e2e8f0` bottom border.
  - Header 3 (`h3`): `15px` styled in Cyan-Teal `#0284c7`.
- **Premium Custom Bullets**: Disable native list bullets globally on content lists (`list-style: none !important`). Use CSS `::before` pseudo-elements to render custom styled bullets:
  - First-level lists: Royal blue bullet dots (`content: "•"` at `#3b82f6` with `font-size: 20px`).
  - Second-level (nested) lists: Purple open circles (`content: "◦"` at `#8b5cf6` with `font-size: 16px`).
  - Maintain absolute position layout on the bullet markers to prevent wrapping misalignment.
- **Layout Scale Preservation**: To prevent headless PDF printers from scaling down document fonts (auto-zooming down to fit oversized elements), all images, tables, and code pre-containers must be constrained horizontally:
  - Code blocks: Use `max-width: 100% !important; overflow-x: hidden !important;` on the wrapper, and `white-space: pre-wrap !important; word-wrap: break-word !important; word-break: break-word !important; font-size: 12px !important;` on the inner `pre` container.
  - Tables: Use `width: 100% !important; max-width: 100% !important; table-layout: auto !important;` with `word-wrap: break-word !important;` on cells.
  - Images: Use `max-width: 90% !important; height: auto !important; margin: 24px auto; display: block;`.
- **HTML/CSS Inline Diagrams**: Ensure diagrams in guides use inline styled HTML/CSS elements to ensure clean, instant vector-perfect rendering without external JavaScript/CDN dependencies. Do not import Mermaid scripts in the HTML headers.
- **Double Escaping in Python F-strings**: When defining the full HTML string template within a Python f-string (`f"""..."""`), all CSS curly braces must be double-escaped (e.g. `body {{ font-size: 15px; }}`) to prevent runtime formatting `KeyError` crashes.
- **Blockquote Alert Parsing & Inner Markdown**: Do not globally replace `>` in blockquote blocks (e.g. using `replace('>', '')`), as this deletes mathematical subscripts (like `<unk>`) and arrows (`->`). Instead, split the blockquote line-by-line and strip only the leading `>` from each line. Furthermore, compile the inner markdown of the blockquote to HTML first using `markdown.markdown()` before wrapping it in container divs, ensuring lists (`-`) and bold text (`**`) are parsed correctly.
- **Math Block HTML Escaping**: Inside extracted LaTeX mathematical blocks, convert all brackets `<` and `>` into `&lt;` and `&gt;` entities prior to inserting them back into the HTML body. This stops browsers from parsing subscripts (like `_{<t>}`) as hidden HTML tags, which breaks KaTeX rendering.

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

---

## 6. Standard Directory & Assets Reorganization Layout

To maintain a clean repository structure, folders must adhere to this standardized file layout:
- **`modules/` (Source Files)**: Contains all source Markdown files (e.g., theory sections, Q&A banks, checklists, walkthrough logs, and roadmap task lists).
- **`helpers/` (Scripts)**: Contains all executable scripts used for building notebooks, compiling documents, or generating plots.
- **`notebooks/`**: Holds executed, output-populated companion Jupyter Notebooks.
- **`plots/`**: Holds all generated graphical visualizations referenced in the markdown guides.
- **Root Directory**: Exclusively holds compiled master deliverables:
  - `*_master_study_guide.html` and `*_master_study_guide.pdf`
  - `*_interview_cheatsheet.html` and `*_interview_cheatsheet.pdf`

*Note: Ensure all paths within scripts are dynamically adjusted to read markdown from `modules/` and output HTML/PDF files to the root directory.*

