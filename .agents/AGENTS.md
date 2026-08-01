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
- **Skill 1: [Study Guide Generator](file:///d:/Study/Prep/.agents/skills/study_guide_generator/SKILL.md)**
  - **Role:** Standardizes Markdown curriculum generation inside topic `modules/` folders.
- **Skill 2: [Notebook Generator](file:///d:/Study/Prep/.agents/skills/notebook_generator/SKILL.md)**
  - **Role:** Programmatically creates, executes, and profiles companion notebooks inside `notebooks/` folders.
- **Skill 3: [Interview Q&A Generator](file:///d:/Study/Prep/.agents/skills/interview_qa_generator/SKILL.md)**
  - **Role:** Generates standalone screening questions, cheatsheets, and Q&A modules.
- **Skill 4: [PDF & HTML Master Compiler](file:///d:/Study/Prep/.agents/skills/pdf_compiler/SKILL.md)**
  - **Role:** Compiles Markdown sources into unified HTML and print-ready A4 PDF master deliverables.
- **Skill 5: [Syllabus Generator](file:///d:/Study/Prep/.agents/skills/syllabus_generator/SKILL.md)**
  - **Role:** Sets templates and filtering rules for drafting custom learning syllabus curricula in `README.md`.

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
* [ ] Are variable transformations and dimension flows clearly explained?
* [ ] Are system bottlenecks (memory-bandwidth vs. compute bound, roofline limits, HBM bounds) explicitly identified?
* [ ] Are all deliverables output at the topic directory root level (`<topic_folder>/`) while source files remain cleanly separated in `modules/`, `notebooks/`, `plots/`, and `helpers/`?