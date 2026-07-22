# Agentic Study Prep Workflow and Context Guide

This document defines the systematic operational workflow used to generate study guides, companion Jupyter Notebooks, HTML pages, and PDF deliverables for machine learning preparation.

---

## 1. Implementation Planning & Status Tracking
- **The Planning Phase**: Before creating files, we define a modular curriculum roadmap inside `implementation_plan.md` to avoid overlap and align with the user's focus (GenAI, sequence models, inference infrastructure).
- **Task List Tracker**: We instantiate `task.md` to map out every deliverable sequentially. The status is managed using custom notation:
  - `[ ]` for uncompleted modules.
  - `[/]` for active in-progress modules.
  - `[x]` for completed modules.

---

## 2. Technical Stack & Packages Used for Asset Creation

The generation pipeline relies on the following specific libraries and binaries to compile text, code, math, and prints:

| Asset Type | Primary Package / Binary | Extensions / Sub-components | Purpose |
|---|---|---|---|
| **Markdown parsing** | Python `markdown` library | `fenced_code`, `tables`, `toc`, `nl2br`, `sane_lists`, `codehilite` | Standardizes block code, registers GFM tables, and outputs formatted HTML templates. |
| **Code Highlighting** | Python `pygments` library | `HtmlFormatter(style='monokai')` | Generates CSS styles mapping token types to terminal syntax colors for coding boxes. |
| **Math Expressions** | `KaTeX` CSS/JS distribution | `katex.min.js`, `auto-render.min.js` | Embedded directly in HTML templates to render LaTeX formulas dynamically. |
| **Jupyter Notebooks** | Python `nbformat` library | Schema v4 `nbf.v4` components | Creates standard, valid `.ipynb` file structures programmatically. |
| **Notebook Execution** | Python `nbconvert` / `nbclient` | `ExecutePreprocessor(kernel_name='python3')` | Executes all cells in sequence using the local virtual environment kernel, capturing logs. |
| **PDF Deliverables** | `Microsoft Edge` | Headless execution CLI | Prints compiled HTML pages directly to print-to-pdf outputs, preserving fonts/SVG layouts. |

---

## 3. Writing Markdown Study Guides (`.md`)
Study guides must include a short, high-level conceptual introduction for each topic (orienting the reader and explaining the core engineering motivation and the issue it resolves) rather than skipping context entirely. We then directly target top-tier tech interview standards (FAANG, Tier-1 startups) by adhering to the following rules:

1. **Andrew Ng "Practical Math" Structure**:
   - State the math formula.
   - Perform a **step-by-step hand calculation** on a tiny sample space (e.g., $|V|=3$, sequence length $L=2$, dimensions $d=2$).
   - Execute PyTorch code that prints identical numbers to 4 decimal places.
2. **KaTeX & LaTeX Syntax Protection**:
   - Wrap inline math in single `$` and display blocks in double `$$`.
   - Always escape percent signs inside math environments (`\%`) to prevent KaTeX parsers from reading them as line comments.
3. **Standardized Module Ending Checklist**:
   Every study guide concludes with the following section:
   ```markdown
   ### Interview Questions & Production Trade-offs
   - What problem does this solve?
   - Why was it introduced?
   - What are its limitations?
   - Computational Complexity (Time & Memory)
   - Component Variable Denotation Legend
   - Production Use Cases
   - Follow-up questions interviewers ask
   ```

---

## 4. Jupyter Notebook (`.ipynb`) Creation & Execution
Instead of creating blank placeholders, all notebooks are compiled and executed programmatically:

1. **Programmatic Generation Script (`build_*_nb.py`)**:
   - Uses `nbformat` to construct a new notebook JSON structure.
   - Injects markdown explanation cells and PyTorch execution cells.
   - Loads specific matrices matching the hand calculations exactly.
2. **Mandatory Post-Execution Explanations**:
   - **Rule**: Every code execution cell inside a Jupyter Notebook must be immediately followed by a markdown cell that explains the printed outputs.
   - This explanation must describe the resulting tensors, shapes, loss numbers, or classification distributions, detailing why they are correct and verifying that they match the study guide's mathematical derivations exactly.
3. **Automated Verification Execution**:
   - We run the script using the workspace virtual environment python path:
     `d:\Study\Prep\.venv\Scripts\python.exe`
   - The script uses `nbconvert.preprocessors.ExecutePreprocessor` to run all cells in sequence, saving the executed notebook in-place.
   - Any runtime PyTorch assertions or value drift errors trigger script failures, acting as an automated testing suite.

---

## 5. Compiling Master HTML & PDF Guides
To create high-quality printable deliverables, we write compiling scripts (e.g., `build_llm_foundations_pdf.py`):

1. **Aggregation & Pre-processing**:
   - Sequentially read markdown files and add a styled cover page.
   - Parse custom GitHub alert tags (e.g. `> [!NOTE]`) and convert them to HTML div alert blocks with left borders and icons.
2. **Math & Code Syntax Protection**:
   - Regex-extract and protect math blocks during markdown-to-HTML conversion.
   - Convert Markdown tables into clean HTML `<table>` elements with explicit borders.
   - Generate Pygments syntax highlighting CSS (e.g., Monokai theme) and inject it into the header.
3. **PDF Generation via Headless Microsoft Edge**:
   - Edge headless is invoked via command line subprocesses to render the compiled HTML to PDF:
     ```python
     cmd = [
         "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
         "--user-data-dir=...",
         "--headless",
         "--disable-gpu",
         "--run-all-compositor-stages-before-draw",
         "--no-pdf-header-footer",
         "--print-to-pdf=<pdf_output_path>",
         "<html_input_path>"
     ]
     ```
   - This ensures all KaTeX math blocks and Pygments style codes print with perfect layout alignment, avoiding rendering artifacts.
