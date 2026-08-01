---
name: PDF & HTML Master Compiler
description: Guidelines and scripts for compiling multiple Markdown modules into unified HTML and PDF master guides using python-markdown, pygments, KaTeX, and headless Microsoft Edge or Chrome.
---

# PDF & HTML Master Compiler Skill

This skill defines the aggregation, preprocessing, CSS design system, and layout printing standards for generating master documentation (HTML, PDF) from Markdown source files in this repository.

---

## 1. Aggregation & Pre-processing Pipeline

To compile clean, professional PDF study guides from modular Markdown files, compilation scripts must follow these exact processing steps:

1. **Sequential Concatenation**: Read the source markdown files in `modules/` in logical curriculum order. Insert explicit page-break dividers (`<div class="page-break"></div>`) between major chapters.
2. **Alert Formatting**: Preprocess custom blockquote alert tags (e.g. `> [!NOTE]`) and convert them to left-bordered HTML divs:
   - **NOTE**: Blue border (`#2563eb`), light blue background (`#eff6ff`), dark blue text (`#1e40af`).
   - **TIP**: Green border (`#059669`), light green background (`#ecfdf5`), dark green text (`#065f46`).
   - **IMPORTANT**: Purple border (`#7c3aed`), light purple background (`#f5f3ff`), dark purple text (`#5b21b6`).
   - **WARNING**: Orange border (`#d97706`), light orange background (`#fffbeb`), dark orange text (`#92400e`).
   - **CAUTION**: Red border (`#dc2626`), light red background (`#fef2f2`), dark red text (`#991b1b`).
3. **Safe Blockquote Line Parsing**: Do not globally replace `>` in blockquotes, as this corrupts mathematical subscripts (`<unk>`) and arrows (`->`). Split blockquotes line-by-line, stripping only the leading `>`. Parse inner markdown to HTML before wrapping in container divs to preserve bold text and nested lists.

---

## 2. Math & Code Protection Rules

During compilation, standard Markdown parsers corrupt LaTeX mathematical syntax and Pygments code blocks. Use the following techniques to safeguard them:

1. **Math Block Extraction**: Use regex to extract all inline math (`$ ... $`) and display blocks (`$$ ... $$`), replacing them with unique string placeholders (e.g., `MATHPLACEHOLDER_0`) before calling the markdown parser.
2. **Math Bracket HTML Escaping**: Inside extracted LaTeX mathematical blocks, convert all brackets `<` and `>` into `&lt;` and `&gt;` entities prior to restoring them. This prevents headless browser engines from parsing LaTeX subscripts (like `_{<t>}`) as hidden HTML tags.
3. **Markdown Parsing**: Run `markdown.markdown()` with required extensions:
   `['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']`
4. **Math Block Restoring**: Swap the LaTeX equations back in place of their corresponding placeholders.
5. **Pygments Code Styling**: Append Pygments CSS styling using the `monokai` dark theme. Apply strict CSS overrides to avoid dark-text-on-dark-background rendering errors:
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

---

## 3. Standard A4 Design System & Layout Engine

The compiled master document must use this standardized CSS layout:

* **Page Dimensions**: Enforce standard A4 size with explicit margins:
```css
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
}

```


* **Typography & Color Hierarchy**:
* **Body Font**: `'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif`
* **Body Text**: `14.5px` base text size for `body`, `p`, and `li` elements with `1.6` line height; color: `#334155`.
* **Inline Code**: Set to `13.5px !important` to match body text height cleanly.
* **Header 1 (`h1`)**: `22px` styled in `#0f172a` with a `#3b82f6` blue bottom border.
* **Header 2 (`h2`)**: `18px` styled in Royal Blue `#2563eb` with a light grey `#e2e8f0` bottom border.
* **Header 3 (`h3`)**: `15px` styled in Cyan-Teal `#0284c7`.


* **Custom Bullet Formatting**: Disable native list bullets globally (`list-style: none !important`). Use CSS `::before` pseudo-elements:
* Level 1 lists: Royal blue bullet dots (`content: "•"` at `#3b82f6` with `font-size: 20px`).
* Level 2 lists: Purple open circles (`content: "◦"` at `#8b5cf6` with `font-size: 16px`).


* **Page-Break & Orphan Control**: Prevent orphan headers and split code blocks:
```css
h1, h2, h3 { page-break-after: avoid; break-after: avoid; }
.codehilite, blockquote, table, img { page-break-inside: avoid; break-inside: avoid; }
.page-break { page-break-before: always; break-before: page; }

```


* **Layout Scale Preservation**: Constrain elements to prevent PDF printer auto-zooming:
* **Code blocks**: Apply `max-width: 100% !important; overflow-x: hidden !important;` on wrapper, and `white-space: pre-wrap !important; word-wrap: break-word !important; font-size: 12px !important;` on `pre`.
* **Tables**: Apply `width: 100% !important; max-width: 100% !important; table-layout: auto !important;` with `word-wrap: break-word !important;`.
* **Images**: Apply `max-width: 90% !important; height: auto !important; margin: 24px auto; display: block;`.


* **KaTeX Script Inclusion**: The HTML template `<head>` MUST include KaTeX CSS and auto-render JS libraries to ensure LaTeX blocks render cleanly in the headless browser prior to printing:
```html
<link rel="stylesheet" href="[https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css](https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css)">
<script defer src="[https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css](https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css)"></script>
<script defer src="[https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js](https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js)"
        onload="renderMathInElement(document.body);"></script>

```


* **Double Escaping in Python F-strings**: When defining the full HTML template inside a Python f-string (`f"""..."""`), all CSS curly braces MUST be double-escaped (e.g. `body {{ font-size: 14.5px; }}`) to prevent runtime `KeyError` string formatting crashes.

---

## 4. Cross-Platform PDF Generation via Headless Browser

To prevent path failures across different machines, executable lookup must check common Edge and Chrome installation paths dynamically:

```python
import os
import shutil
import subprocess
import tempfile


def get_browser_path():
    """Dynamically locate Edge or Chrome browser binary across OS paths."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No Microsoft Edge or Google Chrome executable found."
    )


def compile_html_to_pdf(html_input_path, pdf_output_path):
    """Executes headless browser printing with temp profile cleanup."""
    browser_path = get_browser_path()

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            browser_path,
            f"--user-data-dir={temp_dir}",
            "--headless",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_output_path}",
            html_input_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

```

---

## 5. Directory & File Organization Layout

Folders must strictly adhere to this layout:

* **`modules/` (Source Files)**: Holds all source Markdown files (chapter modules, interview banks, roadmaps).
* **`helpers/` (Scripts)**: Holds compiler scripts (`compile_master.py`), notebook builders, and generator helpers.
* **`notebooks/`**: Holds executed, output-populated companion Jupyter Notebooks.
* **`plots/`**: Holds graphical diagrams referenced in the Markdown guides.
* **Root Directory**: Holds final output master deliverables exclusively:
* `*_master_study_guide.html` and `*_master_study_guide.pdf`
* `*_interview_cheatsheet.html` and `*_interview_cheatsheet.pdf`



---

## 6. Automated Verification Checklist

Immediately after compiling a master guide:

* [ ] Cover page with title, target persona box, and metadata present at top.
* [ ] Table of Contents renders correctly with working internal anchor links.
* [ ] KaTeX display equations render beautifully without broken bracket entities (`<` or `>`).
* [ ] Code blocks maintain dark slate `#0f172a` backgrounds with bright white/cyan readable text.
* [ ] Tables, code blocks, and images fit within A4 width without triggering page auto-scaling.
* [ ] Headings (`h1`, `h2`, `h3`) do not appear orphaned at the bottom of PDF pages.
