---
name: Study Guide Generator
description: Guidelines and constraints for generating self-contained, interview-focused AI engineering Markdown study guides with tensor flow maps, GFM tables, and system trade-off closures.
---

# Study Guide Generator Skill

This skill defines the standardized pedagogy, formatting, and mathematical conventions for creating and refining Markdown study guides (`.md`) in this repository.

---

## 1. High-Level Conceptual Motivation & First Principles

Every study guide must begin with a foundational, standalone introduction that builds intuition from first principles before introducing complex math or code:

1. **The Core Bottleneck**: Explain the real-world engineering failure mode, hardware wall, or computational bottleneck that existed prior (e.g., memory bandwidth limit, quadratic latency, vanishing gradient, communication overhead).
2. **First-Principles Mental Model**: Introduce an intuitive analogy or visual high-level narrative explaining how the technique overcomes this bottleneck.
3. **Core Glossary**: Define key domain terms and variables explicitly so the reader does not need to look up outside references.

---

## 2. Pedagogical Style: AI Engineer (Interview Prep Focus)

All technical study guides must follow a practical **AI Systems Engineering** progression. The goal is to build deep system-level intuition for technical coding, system design, and architecture interviews across any AI domain (NLP, Vision, Distributed Systems, RL, Inference Optimization, etc.).

### A. Core Engineering Lens

* **Strictly Prohibit Academic Formalisms**: Omit formal mathematical proofs, lengthy calculus derivations, and purely theoretical convergence analysis.
* **Prioritize System Metrics & Hardware Awareness**: Evaluate every concept through real-world engineering constraints:
* **Resource & Memory Footprints**: Memory bandwidth, static vs. dynamic VRAM/RAM allocation, cache behaviors, and computational scaling.
* **System Bottlenecks**: Distinguish between memory-bandwidth-bound and compute-bound operations (e.g., Roofline Model dynamics).
* **Dimensionality & Flow**: Track tensor and matrix shapes explicitly at every stage of execution.



### B. The 4-Step Engineering Progression

For every core algorithm, architecture, or optimization technique, follow this sequence:

1. **Production Motivation**: Frame the topic by identifying the real-world engineering bottleneck, latency issue, or hardware constraint it resolves.
2. **Minimal Operational Formulation**: Present the core equation(s) in KaTeX display math, paired with an explicit shape/dimension transformation layout mapping inputs to outputs.
3. **High-Level Flow Walkthrough**: Explain the computational path and logical flow of variables through the operations (e.g., query-key multiplication, softmax scaling, and value aggregation).
4. **Production-Style Reference Code**: Provide self-contained, framework-idiomatic code (e.g., PyTorch):
* Annotate operational shapes directly in comments (e.g., `# [B, C, H, W]`).
* Explicitly set deterministic random seeds (`torch.manual_seed(42)`) and isolate execution (e.g., `with torch.no_grad():`).



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
* Use HTML subscript/superscript tags inside SVG/HTML labels (`w<sub>t-1</sub>`) rather than raw LaTeX syntax.

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
