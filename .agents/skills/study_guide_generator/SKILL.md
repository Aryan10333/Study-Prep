---
name: Study Guide Generator
description: Guidelines and constraints for generating self-contained, interview-focused AI engineering Markdown study guides with tensor flow maps, GFM tables, and system trade-off closures.
---

# Study Guide Generator Skill

This skill defines the standardized pedagogy, formatting, and mathematical conventions for creating and refining Markdown study guides (`.md`) in this repository.

---

## 0. Pre-Flight Checkpoint: Implementation Plan

Before writing any raw Markdown study guide chapters, the agent **MUST** generate a detailed `implementation_plans/implementation_plan_study_guide.md` artifact (in the topic's `implementation_plans/` subfolder, created if it does not yet exist) detailing:
1.  The list of modules/chapters to create with their target file paths.
2.  The exact mathematical formulas to be retained and their planned step-by-step hand calculations.
3.  The visual diagrams and premium plots to be generated or saved locally.
4.  Any open design questions or dependencies.
The agent must wait for the user's explicit sign-off and approval on this implementation plan before executing.

---

## 1. Pedagogical Style: AI Engineer (Interview Prep Focus)

All technical study guides must follow a practical **AI Systems Engineering** progression. The goal is to build deep system-level intuition for technical coding, system design, and architecture interviews across any AI domain.

### A. Formatting & Structure Constraints

Each study guide must be structured using these natural, clean headings (avoiding rigid titles like "Foundational Motivation & First Principles" or "First Principles Mental Model"):

1.  **# [Topic Title]** (Main Title)
2.  **## 1. Introduction & Intuition**: Explain the real-world engineering bottleneck, high-level conceptual intuition, and baseline textbook definitions in plain language before showing any math or code.
3.  **## 2. Core Concepts & Mathematical Formulation**: Adhere strictly to these mathematical representation rules:
    *   **Formula Selection Constraint:** Only write out mathematical formula blocks for core, mostly-asked interview concepts (e.g., primary loss functions, fundamental vector similarity metrics, core evaluation statistics, sequential update loops, and standard drift telemetry indices).
    *   **Formula Accompaniment:** Every retained formula **must** be accompanied by:
        *   **Purpose & High-level Intuition:** A plain-language explanation of what the formula does, why it exists, and the specific production bottleneck it solves.
        *   **Hand Calculation on a Simple Example:** A concrete, step-by-step mathematical walk-through using small numbers (e.g., sequence length $L=2$, hidden size $d=2$, or a tiny mock dataset of 2-3 samples) showing how variables flow through the arithmetic to produce a final numerical result.
        *   **Tensor & Shape Tracking:** Map input shapes to output shapes.
    *   **Concept Simplification:** For all other secondary, auxiliary, or less-frequently-asked mathematical concepts (e.g., complex matrix-factorization loss functions, auxiliary probabilities, background power-law distributions, multi-step dynamic programming state transitions, or derivative backpropagation products), **strictly prohibit writing mathematical formula blocks**. Instead, replace the formula entirely with a description of its purpose, high-level intuition, and practical use in plain text with example if needed.
4.  **## 3. Implementation & Reference Code**: Provide self-contained, runnable code (e.g. in PyTorch/NumPy) annotated with shape comments (e.g., `# [B, L, H]`) and deterministic random seeds.
5.  **## 4. Interview Deep-Dive & System Trade-offs**: Complete the standard complexity tables and interview Q&A closure.

---

### B. Core Engineering Lens

* **Strictly Prohibit Academic Formalisms**: Omit formal mathematical proofs, lengthy calculus derivations, and purely theoretical convergence analysis.
* **Prioritize System Metrics & Hardware Awareness**: Evaluate every concept through real-world engineering constraints:
* **Resource & Memory Footprints**: Memory bandwidth, static vs. dynamic VRAM/RAM allocation, cache behaviors, and computational scaling.
* **System Bottlenecks**: Distinguish between memory-bandwidth-bound and compute-bound operations (e.g., Roofline Model dynamics).
* **Dimensionality & Flow**: Track tensor and matrix shapes explicitly at every stage of execution.

---

## 3. Formatting & Syntax Constraints

### KaTeX & LaTeX Math Formatting:

* Use standard single dollar signs `$ ... $` for inline math.
* Use double dollar signs isolated on their own lines for display math blocks:
$$\text{VRAM}_{\text{bytes}} = 2 \times N_{\text{params}} + 4 \times N_{\text{grads}}$$


* Always escape percent signs inside math blocks (`\%`) to prevent KaTeX line-comment syntax errors.

### Standard Tensor Dimension Notation:

Consistently use standard variable denotations across all sections:

* $B$: Batch Size
* $L$ or $S$: Sequence Length
* $H$ or $d$: Hidden / Model Dimension
* $C$: Channels
* $V$: Vocabulary Size
* $N$: Number of Nodes / GPUs / Layers

### Heading Hierarchy:

* **Heading 1 (`#`)**: Main Title.
* **Heading 2 (`##`)**: Major Topics / Sections.
* **Heading 3 (`###`)**: Subtopics, Hand Calculations, or Code.

### Native GFM Tables:

* Write standard GitHub Flavored Markdown (GFM) tables directly so KaTeX renders inside cells. Never wrap Markdown tables inside fenced code blocks:
| Parameter        | Memory Footprint   | Bottleneck Type        |
| ------------------| --------------------| ------------------------|
| $L$ (Seq Length) | $\mathcal{O}(L^2)$ | Memory Bandwidth (HBM) |



### Inline HTML/CSS & SVG Diagrams (No Mermaid/ASCII):

* Do not use raw ASCII flowcharts or Mermaid code blocks (they render unreliably in compiler/PDF tools).
* Construct visual diagrams using **responsive inline SVG** (`viewBox="0 0 W H" width="100%" height="auto"`) or **styled HTML flexbox containers** (`max-width: 100%; overflow-x: auto;`).
* **Never put `$...$`-delimited LaTeX inside an SVG `<text>` element — confirmed in practice to render as silently missing content, not visible broken text.** The compiler's KaTeX auto-render pass walks the whole document body and finds the `$` delimiters inside SVG text nodes same as anywhere else, but KaTeX's output is HTML (`<span>` elements), which cannot be inserted as a child of an SVG `<text>` node — the browser drops it, leaving a blank gap exactly where the math should be (e.g. a label reading "estimates $\hat{A}_t$" renders as "estimates )" with the math silently gone). Use plain Unicode characters/HTML entities instead (`&#952;` = θ, `&#955;` = λ, `&#960;` = π, `&#8722;` = −, `&#215;` = ×) with underscore notation for subscripts (`&#952;_base` reading as "θ_base") — this is the same "no raw LaTeX in SVG/HTML labels" rule as the `w<sub>t-1</sub>` guidance above, stated with the concrete failure mode so it's clear *why* it matters, not just that it's a style preference.
* Use HTML subscript/superscript tags inside SVG/HTML labels (`w<sub>t-1</sub>`) rather than raw LaTeX syntax.
* **After writing any inline SVG diagram, actually render and view it** (e.g. extract the `<svg>` block into a standalone HTML file and screenshot it) before considering the module complete — box/title collisions and off-canvas stray paths are easy to introduce with hand-computed coordinates and are not visible from reading the SVG source alone. This is the same discipline required for matplotlib diagrams in Section 6, point 7, extended to inline SVG.

---

## 4. Standardized Interview Deep-Dive & System Trade-Off Closure

Every module must conclude with this exact structured section, filled out with dense, topic-specific details tailored to technical interview questions:

```markdown
## Interview Deep-Dive & System Trade-offs

### 1. Architectural & Production Trade-offs
* **Core Problem Solved:** 
* **Why Introduced over Legacy Approaches:** 
* **Key Failure Modes & Limitations:** 

### 2. System Complexity & Scaling
* **Time Complexity (FLOPs):** 
* **Space/Memory Footprint:** 
* **Primary Bottleneck Type:** (Memory Bandwidth vs. Compute Bound)
* **Variable Legend:** 

### 3. Production & Scalability
* **Deployment Considerations:** (Quantization, Kernel Fusion, FP8/BF16 precision, Distributed Sharding)
* **Common Interviewer Follow-Up Questions:**
  1. *Q:* [Expected Follow-Up Question]
     * *A:* [Structured, high-signal answer]
  2. *Q:* [Expected System Design Follow-Up]
     * *A:* [Structured, high-signal answer]

```

---

## 5. Automated Verification Checklist

Immediately after generating or updating a guide, verify:

* [ ] YAML Frontmatter included with title, category, and prerequisites.
* [ ] First-principles conceptual motivation precedes any math or code.
* [ ] Operational math includes explicit tensor/array dimension shapes.
* [ ] Code is verified to be accurate, logically matching mathematical formulas and descriptions.
* [ ] Code is runnable, deterministic, and annotated with shape comments `# [B, L, H]`.
* [ ] KaTeX display blocks are line-isolated and `%` signs are escaped.
* [ ] Diagrams use responsive SVG or flexbox HTML (no Mermaid or raw ASCII).
* [ ] Section 4 interview deep-dive is fully populated without shorthand placeholders.

---

## 6. Standard Plot Generation Reference

For generating premium statistical charts, heatmaps, and metric timelines (to be saved under the topic's `plots/` directory), refer to the standard Matplotlib/Seaborn setup at:
*   [sample_plot_generator.py](file:///d:/Study/Prep/.agents/scripts/sample_plot_generator.py)

To ensure visual accuracy and cross-platform formatting compliance:
1.  **Plot Margins & Padding**: When defining axes limits in Matplotlib/Seaborn, always specify broad limits (e.g., `xlim` and `ylim` extending past coordinates/boxes by at least 10–15%) to prevent text labels and axis titles from being clipped by A4 document borders.
2.  **Flowchart Centering**: Ensure all flowcharts and schematics are horizontally centered on the page. Set exact `xlim` boundary limits in Matplotlib that match the bounding boxes of the drawn elements, eliminating empty whitespace margins on the right side of the figure.
3.  **Plot Placement & Context**: Always place each plot and its descriptive caption interpretation *directly* under the corresponding subsection explaining that specific concept, rather than grouping multiple unrelated plots or captions together.
4.  **Font & Character Compatibility**: Avoid using structural or mathematical symbols (like circled times `⊗`) or flag emojis inside Matplotlib/Seaborn labels. These symbols are frequently missing from the host system's default font caches, causing empty square fallbacks and rendering warnings. Use standard ASCII markers (like `*`) or LaTeX syntax instead.
5.  **LaTeX Operators Delimiting**: Wrap math relationship operators (like `\leftrightarrow`) inside LaTeX delimiters (`$ ... $` with text tags if necessary) rather than code backticks, to ensure they render as high-fidelity vector symbols.
6.  **Flowchart Visuals**: Avoid using complex HTML/CSS markup to build custom flowcharts in the raw Markdown source, as they render unreliably during Edge PDF printing. Flowcharts should be generated as clean Matplotlib graphics (`.png`) and referenced relatively.
7.  **Box-and-Arrow / State-Flow Diagram Label Clearance**: This diagram type (e.g. gated cell architectures, state transition lattices) has a distinct failure mode not shared by line/bar charts: text labels placed directly on top of a connector line's path render as struck-through and illegible (observed in a generated LSTM cell diagram where "Cell State" labels were crossed out by their own arrows). When drawing labeled connector arrows:
    *   Never anchor a text label's coordinates on the arrow's own path; offset the label above/below/beside the line with explicit clearance (e.g. `ha`/`va` offsets or a dedicated `xytext` distinct from the arrow's start/end points).
    *   Before treating the diagram as complete, verify every label drawn in the figure has a corresponding connecting arrow pointing to or from it — a label with no connecting line (e.g. an input variable named but never wired into the diagram) indicates an incomplete diagram, not a finished one.
    *   Do not rely solely on the script exiting without error. **Visually open and inspect the rendered PNG** after generation (e.g. via the Read tool) to catch overlap/clipping/dangling-label defects that only show up in the rasterized output, before referencing the image in a module.

> [!IMPORTANT]
> Always run plot generation scripts using the repository's active Python virtual environment (e.g. `.venv\Scripts\python.exe helpers/generate_<topic>_plots.py` on Windows) to ensure libraries like `matplotlib` and `seaborn` are correctly loaded.

