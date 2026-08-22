# Agent Execution Protocol & Workspace Instructions

> **Purpose:** System instructions for AI Agents working in this repository (`Prep/`) to generate study material, companion notebooks, and interview cheatsheets tailored for **Aryan Chandra**.

---

## 1. Context & User Profile Anchoring
Before generating content, building notebooks, or executing scripts in this workspace, the agent **MUST** read and adhere to `.agents/USER_PROFILE.md`.

* **Target Persona:** Aryan Chandra (AI Engineer at Jio Platforms Ltd with ~3 years experience, targeting Applied AI / GenAI / AI Backend Engineer roles at tier-1 tech companies and high-growth AI startups).
* **Engineering Standard:** Content must emphasize production engineering, hardware constraints, system trade-offs, and tensor dimension tracking over purely theoretical calculus derivations.

---

## 2. Skills Registry & Architectural Workflow

Auto-discovered agent skills are located in `.agents/skills/`. The agent MUST follow the guidelines specified in each skill's respective `SKILL.md` file:

```

                              +---------------------------------------+
                              |   1. study_guide_generator/           |
                              |   (Markdown Curriculum Modules)       |
                              +-------------------+-------------------+
                                                  |
                                                  v

```

+---------------------------------------+    +---------------------------------------+
| 2. notebook_generator/                |    |  3. interview_qa_generator/           |
| (Executable Companion Notebooks)      |    |  (Standalone Cheatsheets & Q&A)       |
+-------------------+-------------------+    +-------------------+-------------------+
|                                            |
+-------------------+------------------------+
|
v
+-----------------------------------+
|  4. pdf_compiler/                 |
|  (Pygments, KaTeX, Edge/Chrome)   |
+-------------------+---------------+
|
v
+-----------------------------------+
|      Final Output Deliverables     |
|  (*_master.pdf / *_cheatsheet.pdf)|
+-----------------------------------+

```

### Core Skills
- **Skill 1: [Syllabus Generator](file:///d:/Study/Prep/.agents/skills/syllabus_generator/SKILL.md)**
  - **Role:** Sets templates and filtering rules for drafting custom learning syllabus curricula in `README.md`.
- **Skill 2: [Study Guide Generator](file:///d:/Study/Prep/.agents/skills/study_guide_generator/SKILL.md)**
  - **Role:** Standardizes Markdown curriculum generation inside topic `modules/` folders.
- **Skill 3: [Notebook Generator](file:///d:/Study/Prep/.agents/skills/notebook_generator/SKILL.md)**
  - **Role:** Programmatically creates, executes, and profiles companion notebooks inside `notebooks/` folders.
- **Skill 4: [Interview Q&A Generator](file:///d:/Study/Prep/.agents/skills/interview_qa_generator/SKILL.md)**
  - **Role:** Generates standalone screening questions, cheatsheets, and Q&A modules.
- **Skill 5: [PDF & HTML Master Compiler](file:///d:/Study/Prep/.agents/skills/pdf_compiler/SKILL.md)**
  - **Role:** Compiles Markdown sources into unified HTML and print-ready A4 PDF master deliverables.

---

## 3. Decoupled Pipeline System Workflow

The workflow consists of an initial scoping phase followed by three independent parallel compilation tracks to prevent code-math synchronizing overhead and maintain high iteration speed:

*   **Scoping Phase: Syllabus Generation**
    *   **Objective:** Focus exclusively on defining the syllabus boundaries and creating the draft `README.md` at the topic's root folder.
    *   **Action Policy:** The agent must only generate the proposed syllabus (`README.md`) for user review during this initial phase. The implementation plan at this stage must NOT list the fine details of code implementation, module contents, or Q&A items; these details and planning steps must occur later during their respective execution tracks.
    *   **Files:** Generates the high-level syllabus at the topic folder's root `README.md`.
    *   *Checkpoint:* Explicit user review and sign-off required on `README.md` before starting any execution tracks.
*   **Track 1: Study Notes Modules (Interview Prep Focus)**
    *   **Objective:** Core theory, architecture diagrams, VRAM sizing formulas, parameter counts, and system trade-offs.
    *   **Files:** Raw markdown source chapters reside in `modules/` and compile directly into `<topic>_master_study_guide.pdf`.
    *   *Checkpoint:* **Implementation Plan Checkpoint:** The agent MUST generate a dedicated `implementation_plans/implementation_plan_study_guide.md` outlining the study notes modules, hand calculations, and visual diagrams, and obtain explicit user sign-off before writing any raw chapter files.
*   **Track 2: Production Notebooks & Code (Execution Focus)**
    *   **Objective:** Standalone, end-to-end executable examples profiling real-world parameters, hardware boundaries (VRAM, latency), and tensor contiguity.
    *   **Files:** Programmatically built and run inside `notebooks/` following the [sample_notebook_generator.py](file:///d:/Study/Prep/.agents/scripts/sample_notebook_generator.py) patterns.
    *   **Formatting Guidelines**:
        *   **Markdown Headings**: Always place a descriptive markdown heading cell (e.g. `## X. Step Name`) immediately before each code cell.
        *   **Execute-Before-Explain Policy**: When creating or modifying notebooks, the agent must **first write and execute the code cell to obtain the actual output, inspect the printed logs, and only then write/align the corresponding markdown explanation cell** to guarantee 100% numerical and conceptual accuracy. Do not write hypothetical explanations prior to execution.
    *   *Checkpoint:* **Implementation Plan Checkpoint:** The agent MUST generate a dedicated `implementation_plans/implementation_plan_notebook.md` outlining the datasets, profiling code steps, and analysis segments, and obtain explicit user sign-off before writing or executing any notebook builder scripts.
*   **Track 3: Standalone Interview Q&As (Screening Focus)**
    *   **Objective:** Fast screening responses, key buzzwords, technical intuition, and common mistakes.
    *   **Formatting Guideline**: 
        *   **Simple Question Titles**: Keep question headers simple and direct (no math equations or derivations in headers).
        *   **Essential vs. Deep Dive Split**: Format every Q&A with an *Essential* block (conversational/spoken-style first-person response, a 2–4 line intuitive example, and key buzzword definitions) and a *Deep Dive* block (formulas without long derivations, production constraints, short common mistakes, and 2–4 follow-ups).
        *   **One-Line Takeaway**: End each answer with a single-sentence takeaway.
        *   **Comparison Tables**: Proactively include comparison tables to contrast architectures, parameters, or algorithms (e.g. GPT vs. BERT vs. T5, LayerNorm vs. RMSNorm, MHA vs. GQA vs. MQA, Greedy vs. Sampling).
        *   **Final Revision Sheet**: Conclude each cheatsheet with a 2-page summary section containing a takeaway table, a key formula box, and a top follow-up Q&A index.
    *   **Batching Policy**: Large question sets (20+ questions) must be generated and appended in logical batches of 10–15 questions in sequence. To minimize interruptions, the agent will proceed from batch to batch automatically in a single session without pausing for review.
    *   **Files:** Reside in `modules/*_interview_questions.md` and compile to `<topic>_interview_cheatsheet.pdf`.
    *   *Checkpoint:* **Implementation Plan Checkpoint:** The agent MUST generate a dedicated `implementation_plans/implementation_plan_interview_qa.md` listing the proposed question list, and obtain explicit user sign-off before writing the Q&A sheet. Explicit user review is required twice: first for the question list draft, and second for the final completed Q&A sheet.

---

## 4. Topic Directory Structure Standard

Every study topic under `machine-learning-prep/` (e.g., `00_nlp_fundamentals/`) must strictly adhere to this 4-subfolder architecture:

```text
machine-learning-prep/<ai_discipline>/<topic_folder>/
├── README.md                          # High-level Syllabus of covered topics
├── implementation_plans/              # [PLANS] One file per track, all sign-off docs live here
│   ├── implementation_plan_study_guide.md   # Track 1 sign-off (study guide modules)
│   ├── implementation_plan_notebook.md      # Track 2 sign-off (companion notebooks)
│   └── implementation_plan_interview_qa.md  # Track 3 sign-off (interview Q&A)
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

## 5. Execution Pre-Flight Checklist

Before presenting any completed material or running compilation scripts, verify:

* [ ] Does this material reflect an Applied GenAI Engineer perspective (production engineering over pure academic research math)?
* [ ] Are tensor shapes (`# [B, L, H]`) annotated in PyTorch code blocks?
* [ ] Are variable transformations and dimension flows clearly explained?
* [ ] Are system bottlenecks (memory-bandwidth vs. compute bound, roofline limits, HBM bounds) explicitly identified?
* [ ] Are companion notebooks structured with markdown heading cells preceding each code cell, and are all explanation cells written only *after* reading actual executed cell outputs?
* [ ] Are all deliverables output at the topic directory root level (`<topic_folder>/`) while source files remain cleanly separated in `modules/`, `notebooks/`, `plots/`, and `helpers/`?

---

## 6. Automated Compliance Verification (Mandatory Last Step)

Every skill file in `.agents/skills/` lists an "Automated Verification Checklist" as markdown bullets. In practice, a checklist that only exists as prose is easy to silently skip — especially mid-batch on a large generation task, or when resuming work started in an earlier session. This has already produced real drift in this repository: a 50-question interview cheatsheet was fully written and left in place using none of the section structure its own skill mandated, and a compiler script embedded hardcoded absolute filesystem paths into a deliverable meant to be portable, and neither was caught until a later manual review.

To prevent this recurring, before presenting any track's output as complete (Study Notes, Notebooks, or Interview Q&As), run an explicit, scriptable compliance pass over the generated file(s) rather than eyeballing the prose checklist:
* **Interview Q&A track**: grep the generated `*_interview_questions.md` for the required section headings (`[ESSENTIAL]`, `[DEEP DIVE]`, `One-Line Takeaway`, `Final Revision Sheet`) per the check defined in [interview_qa_generator/SKILL.md](file:///d:/Study/Prep/.agents/skills/interview_qa_generator/SKILL.md) Section 5, and confirm counts match the question total.
* **PDF/HTML compiler track**: grep the compiled `.html` output and the compiler script itself for hardcoded absolute paths or stray `file:///` references per [pdf_compiler/SKILL.md](file:///d:/Study/Prep/.agents/skills/pdf_compiler/SKILL.md) Section 6, and confirm the output PDF file was actually written and is non-empty.
* **Study guide track**: after generating any box-and-arrow/state-flow diagram, visually inspect the rendered PNG (not just confirm the script exited cleanly) per [study_guide_generator/SKILL.md](file:///d:/Study/Prep/.agents/skills/study_guide_generator/SKILL.md) Section 6, point 7.

If a topic's `modules/`, `notebooks/`, or deliverables were generated before a skill file was last edited, do not assume they are still compliant — run the relevant check against the existing files before extending or re-compiling them.