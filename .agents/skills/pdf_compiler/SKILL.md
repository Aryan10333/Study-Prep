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
4. **Math-Block Curly-Quote Fixing**: Straight double quotes (`"..."`) typed inside `\text{...}` commands within LaTeX blocks render as raw typewriter quotes in KaTeX, which look unprofessional next to the document's typographic quotes elsewhere. Before any other preprocessing, regex-match `\text{...}` spans inside `$...$`/`$$...$$` blocks and convert straight `"` pairs inside them to curly opening/closing quotes (`"` / `"`). This must run *before* math-block placeholder extraction (step 2 of Section 2), since it operates on the raw LaTeX content.
5. **Follow-Up Question Card Flattening**: Nested `Common Interviewer Follow-Up Questions` lists (numbered `*Q:*`/`*A:*` pairs, often indented under a bullet) render poorly as raw nested Markdown lists in the compiled PDF — inconsistent indentation and broken numbering. Detect this block pattern with a dedicated regex pass and re-emit it as flat, left-accented "card" divs (one bordered div per Q/A pair) instead of leaving it as nested Markdown for the generic list renderer to handle. This is a required transform, not an optional enhancement — compiled output without it has visibly degraded Q&A formatting.

---

## 2. Math & Code Protection Rules

During compilation, standard Markdown parsers corrupt LaTeX mathematical syntax and Pygments code blocks. Use the following techniques to safeguard them:

1. **Math Block Extraction**: Use regex to extract all inline math (`$ ... $`) and display blocks (`$$ ... $$`), replacing them with unique string placeholders (e.g., `MATHPLACEHOLDER_0`) BEFORE any preprocessing, custom regex formatting, or text substitutions take place. This guarantees that mathematical subscripts (underscores `_` or asterisks `*`) are never corrupted or converted into HTML italics/bold tags by markdown engines or preprocessors.
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

    # Verify the browser actually produced the file — a headless print can exit
    # 0 while silently failing to write output (e.g. locked profile dir, bad path).
    if not os.path.exists(pdf_output_path) or os.path.getsize(pdf_output_path) == 0:
        raise RuntimeError(f"PDF compilation did not produce a valid file at {pdf_output_path}")

```

**Confirmed in practice, not just theoretical:** headless Edge printing (and `--screenshot`) has been observed to exit 0 with no output file at all on the first attempt, then succeed immediately on an unmodified retry — a real, reproducible race condition, not a hypothetical edge case. Wrap the subprocess call + verification in a small retry loop (2-3 attempts, no backoff needed) rather than failing the whole compile on the first transient miss.

Use manual `tempfile.mkdtemp()` + `shutil.rmtree(..., ignore_errors=True)` for the profile directory, **not** `tempfile.TemporaryDirectory()`'s context manager — also confirmed in practice: Edge's Crashpad crash-handler subprocess can still hold a file lock inside the profile dir for a moment after the main Edge process exits, and `TemporaryDirectory`'s strict cleanup raises `OSError: [WinError 145] The directory is not empty` on that race, crashing the whole compile on an otherwise-successful PDF generation. A leftover profile dir is harmless (OS temp cleanup reclaims it later); a crashed compiler script is not.

```python
def compile_html_to_pdf(html_input_path, pdf_output_path, max_attempts=3):
    browser_path = get_browser_path()
    for attempt in range(1, max_attempts + 1):
        temp_dir = tempfile.mkdtemp(prefix="edge_pdf_")
        try:
            cmd = [
                browser_path, f"--user-data-dir={temp_dir}", "--headless", "--disable-gpu",
                "--run-all-compositor-stages-before-draw", "--virtual-time-budget=8000",
                "--no-pdf-header-footer", f"--print-to-pdf={pdf_output_path}", html_input_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(pdf_output_path) and os.path.getsize(pdf_output_path) > 0:
            return
    raise RuntimeError(f"PDF compilation did not produce a valid file at {pdf_output_path} after {max_attempts} attempts")
```

**If the retry loop above exhausts all attempts (not just an occasional single-attempt miss), check for orphaned `msedge.exe`/`chrome.exe` processes before assuming a code bug.** Confirmed in practice: repeated manual headless invocations during debugging (each spawning a main process plus Crashpad/GPU child processes that don't always terminate cleanly) can accumulate over a session to the point where *every* new headless launch silently produces no output file, with no error — indistinguishable from a code problem until you check `tasklist` (Windows) / `ps` (Unix) for a growing pile of `msedge.exe` processes. On Windows: `tasklist /FI "IMAGENAME eq msedge.exe"` to check, `taskkill /F /IM msedge.exe` to clear them. Do **not** build automatic process-killing into the compiler script itself — a user's real, open Edge browser windows share the same process name, and killing them unprompted would close the user's actual work. Treat this as a manual troubleshooting step, not something to automate.

**If the retry loop exhausts all attempts consistently (every single retry fails identically, not an intermittent miss) and `tasklist` shows no orphaned processes, suspect `--virtual-time-budget` being too short for the page's actual render time before diagnosing anything else.** Confirmed in practice: a 51-question interview cheatsheet (306KB HTML, dozens of KaTeX inline/display formulas per page) failed all 3 retry attempts at `--virtual-time-budget=8000` — msedge exited 0 every time with zero bytes written, no error output — while a much larger (1.2MB) but math-lighter master study guide compiled successfully first-try at the same 8000ms budget in the same run. The distinguishing factor was KaTeX auto-render workload (formula density), not file size. Manually invoking the same command with `--virtual-time-budget=30000` produced a valid PDF immediately. **Fix:** for any document with heavy math density (many formulas per page, not just a few scattered ones), raise the budget well above the 8000ms default — 20000-30000ms is a safe starting point — rather than assuming a 3x retry loop will paper over a budget that's systematically too short. A budget that's merely borderline shows up as the transient race described above (intermittent misses, succeeds on retry); a budget that's genuinely too short for the content shows up as this pattern instead (fails identically every single time, no amount of retrying helps).

### Never Hardcode Absolute Filesystem Paths

Compilation scripts must resolve every directory (`base_dir`, the `modules/`/`plots/`/`helpers/` paths, output paths) relative to the script's own location, never as a literal hardcoded absolute path (e.g. `r"d:\Study\Prep\..."`):

```python
# Correct: portable across machines, users, and repo relocations.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # helpers/ -> topic root

# Wrong: breaks the moment the repo is moved, renamed, cloned elsewhere,
# or run by a different user/agent session.
BASE_DIR = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
```

A hardcoded absolute `base_dir` doesn't just risk the compiler script breaking — it leaks directly into the compiled deliverable. If plot image references are rewritten to `file:///<base_dir>/plots/...` absolute URIs (required for the headless browser to resolve local images during PDF printing), those same literal paths persist inside the saved `*.html` file. The PDF itself stays portable (images are rasterized into it at compile time), but the standalone HTML deliverable will only render images on the exact machine and file path it was generated on. Prefer one of:
1. **Embed images as base64 `data:` URIs** in the HTML instead of `file:///` links, so the HTML deliverable is fully self-contained and portable, or
2. If keeping `file:///` links for simplicity, explicitly document in the deliverable (or to the user) that the `.html` output is machine-local only and the `.pdf` is the portable artifact.

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
* [ ] **Path portability**: `grep`/search the compiled `.html` output for the literal string `file:///` — if matches exist, confirm this was a deliberate choice (see Section 4) and not an accidental hardcoded `base_dir`. Search the compiler script itself for hardcoded absolute paths (e.g. `d:\`, `/home/`, `/Users/`) — none should exist outside of `os.path`-derived variables.
* [ ] **PDF output verified**: The compiler script asserted the output PDF file exists and is non-empty after the subprocess call (not just that the subprocess exited 0).

---

## 7. Standard Implementation Reference

For a complete, verified, and production-ready script that implements this compilation pipeline, refer to:
*   [sample_pdf_compiler.py](file:///d:/Study/Prep/.agents/scripts/sample_pdf_compiler.py)

> [!IMPORTANT]
> Always run compiler/builder scripts using the repository's active Python virtual environment (e.g. `.venv\Scripts\python.exe helpers/compile_<topic>.py` on Windows) to ensure libraries like `markdown` and `pygments` are correctly loaded.

