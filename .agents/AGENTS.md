# Agent Execution Protocol & Workspace Instructions

> **Purpose:** System instructions for AI Agents working in this repository (`Prep/`) to generate study material, companion notebooks, and interview cheatsheets tailored for **Aryan Chandra**.

---

## 1. Context & User Profile Anchoring
Before generating content, building notebooks, or executing scripts in this workspace, the agent **MUST** read and adhere to `.agents/USER_PROFILE.md`.

* **Target Persona:** Aryan Chandra (AI Engineer at Jio Platforms Ltd with ~3 years experience, targeting Applied AI / GenAI / AI Backend Engineer roles at tier-1 tech companies and high-growth AI startups).
* **Engineering Standard:** Content must emphasize production engineering, hardware constraints, system trade-offs, and whiteboard hand calculations over purely theoretical calculus derivations.

---

## 2. Skills Registry & Architectural Workflow

Auto-discovered agent skills are located in `.agents/skills/`. The agent MUST follow the guidelines specified in each skill directory:


```

```
                              +---------------------------------------+
                              |   1. study_guide_generator/           |
                              |   (Markdown Curriculum Modules)       |
                              +-------------------+-------------------+
                                                  |
                                                  v

```

+---------------------------------------+    +---------------------------------------+
| 2. notebook_generator/                |    |  4. interview_qa_generator/           |
| (Executable Companion Notebooks)      |    |  (Standalone Cheatsheets & Q&A)       |
+-------------------+-------------------+    +-------------------+-------------------+
|                                            |
+-------------------+------------------------+
|
v
+-----------------------------------+
|  3. pdf_compiler/                 |
|  (Pygments, KaTeX, Edge/Chrome)   |
+-------------------+---------------+
|
v
+-----------------------------------+
|      Final Output Deliverables     |
|  (*_master.pdf / *_cheatsheet.pdf)|
+-----------------------------------+

```

### Skill 1: Study Guide Generator (`.agents/skills/study_guide_generator/`)
* **Role:** Standardizes Markdown curriculum generation inside topic `modules/` folders.
* **Rules:**
  * Begin with first-principles motivation & real-world bottlenecks (e.g., memory wall, roofline bounds, latency limits).
  * Use standard variable notations ($B$, $L$, $H$, $d$, $V$, $N$) and line-isolated KaTeX display equations (`$$ ... $$`).
  * Include step-by-step micro hand calculations ($B=1, L=2, H=2$) matching deterministic PyTorch output (annotated with `# [B, L, H]`).
  * Diagrams MUST use inline responsive SVG or flexbox containers (strictly NO raw ASCII or Mermaid).
  * Always conclude with the mandatory **Interview Deep-Dive & System Trade-offs** template.

### Skill 2: Notebook Generator (`.agents/skills/notebook_generator/`)
* **Role:** Programmatically creates and executes production Jupyter Notebooks inside topic `notebooks/` folders via helper scripts (`helpers/build_*_notebooks.py`).
* **Rules:**
  * Must strictly follow the 3-cell sequence: **Markdown Heading** $\rightarrow$ **Code Implementation** $\rightarrow$ **### Output Explanation, Interpretation & Performance Analysis**.
  * Zero unexecuted cells (`In [ ]`) allowed in final `.ipynb` files.
  * Load API credentials dynamically using `python-dotenv` (`find_dotenv()`). Zero hardcoded secrets.


### Skill 3: Interview Q&A Generator (`.agents/skills/interview_qa_generator/`)
* **Role:** Generates standalone screening questions, cheatsheets, and Q&A modules.
* **Rules:**
  * Enforce the Q&A structure: **Short Answer (30–60s)**, **Key Buzzwords**, **Technical Intuition/Math**, **Production Perspective**, **Follow-ups**, and **Common Mistakes**.
  * Focus on screening speed, architectural clarity, and system trade-offs.

### Skill 4: PDF & HTML Master Compiler (`.agents/skills/pdf_compiler/`)
* **Role:** Compiles Markdown sources into unified HTML and print-ready A4 PDF master deliverables using helper scripts (`helpers/compile_*.py`).
* **Rules:**
  * Extract math blocks (`$ ... $` / `$$ ... $$`) and escape brackets (`<` $\rightarrow$ `&lt;`) prior to Markdown rendering to prevent KaTeX corruption.
  * Enforce Pygments Monokai dark styling with slate background (`#0f172a`) and CSS contrast overrides.
  * Execute headless printing via Microsoft Edge or Chrome with temporary user profile directories.

---

## 3. Topic Directory Structure Standard

Every study topic under `machine-learning-prep/` (e.g., `00_nlp_fundamentals/`) must strictly adhere to this 4-subfolder architecture:

```text
machine-learning-prep/<ai_discipline>/<topic_folder>/
├── README.md                          # High-level Syllabus of covered topics
├── modules/                           # [SOURCE FILES] Raw Markdown Chapters
│   ├── 01_topic_intro.md
│   └── 02_topic_advanced.md
├── notebooks/                         # [COMPANION CODE] 100% Executed Jupyter Notebooks
│   ├── 01_topic_intro.ipynb
│   └── 02_topic_advanced.ipynb
├── plots/                             # [ASSETS] Visual diagrams & exported charts
│   └── architecture_diagram.png
├── helpers/                           # [SCRIPTS] Compilation & generation utilities
│   ├── build_<topic>_notebooks.py
│   └── compile_<topic>.py
├── <topic>_master_study_guide.html    # [DELIVERABLE] Master curriculum HTML
├── <topic>_master_study_guide.pdf     # [DELIVERABLE] Print-ready curriculum PDF
├── <topic>_interview_cheatsheet.html  # [DELIVERABLE] Revision HTML cheatsheet
└── <topic>_interview_cheatsheet.pdf   # [DELIVERABLE] Print-ready revision PDF cheatsheet

```

---

## 4. Execution Pre-Flight Checklist

Before presenting any completed material or running compilation scripts, verify:

* [ ] Does this material reflect an Applied GenAI Engineer perspective (production engineering over pure academic research math)?
* [ ] Are tensor shapes (`# [B, L, H]`) annotated in PyTorch code blocks?
* [ ] Are micro hand calculations provided on small sample dimensions matching code outputs to 4 decimal places?
* [ ] Are system bottlenecks (memory-bandwidth vs. compute bound, roofline limits, HBM bounds) explicitly identified?
* [ ] Are all deliverables output at the topic directory root level (`<topic_folder>/`) while source files remain cleanly separated in `modules/`, `notebooks/`, `plots/`, and `helpers/`?