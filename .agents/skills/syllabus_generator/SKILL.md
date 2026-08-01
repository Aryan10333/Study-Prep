---
name: Syllabus Generator
description: Guidelines and workflow for drafting a custom syllabus (README.md) for a target ML/AI topic, aligned with the candidate's career level, target company profiles, and curriculum focus weights.
---

# Syllabus Generator Skill

This skill defines the standardized process for defining, structuring, and generating a high-impact learning curriculum syllabus (in the `README.md` of a topic directory) before any study guides or notebooks are developed.

---

## 1. Context & Motivation

A syllabus acts as the blueprint for study notes and codebases. It ensures that the engineering depth matches the candidate's profile in [USER_PROFILE.md](file:///d:/Study/Prep/.agents/USER_PROFILE.md) (Mid-to-Senior Applied AI Engineer) and prevents "scope creep" (e.g. studying deep theoretical convergence proofs or irrelevant legacy academic algorithms).

---

## 2. Standardized Syllabus Structure

The generated `README.md` syllabus for any topic folder must strictly adhere to the following sections:

```markdown
# [Topic Title] Syllabus

## 1. Context & Alignment
* **Profile Focus:** [Explain how this topic aligns with the candidate's Mid/Senior AI Engineer target roles.]
* **Interview Frequency:** [High / Medium / Low - specify frequency in tier-1 tech and AI startup screens.]
* **Core Goal:** [What is the ultimate engineering goal, e.g., building a production RAG system or implementing custom transformers from scratch.]

## 2. Module Chapters & Conceptual Scope
List the sequential modules to be created under the `modules/` directory:
- **Module 01: [Title]**
  - *Key Concepts:* [Concept A, Concept B, etc.]
  - *System Bottlenecks & Focus:* [What engineering bounds will be addressed, e.g. memory footprint, computation limits, GPU walls.]
- **Module 02: [Title]**
  - ...
```

---

## 3. Curriculum Constraints & Filtering Rules

When generating the syllabus, the agent must check:
1. **Target Companies Alignment:** If the candidate targets Groq/AMD/NVIDIA, include infrastructure constraints (SRAM vs HBM, latency budgets). If they target OpenAI/Anthropic/VLLM providers, include API orchestration, latency profiling, and system evaluation.
2. **Applied focus:** Filter out PhD-level academic research papers or complex mathematical derivations unless they are essential to system architecture (e.g. FlashAttention-2 tiling mechanics, LoRA weight decomposition).
3. **Pre-flight Syllabus Checklist:**
   - [ ] Are modules grouped logically (dependencies first)?
   - [ ] Is every module focused on a practical, production engineering problem?
   - [ ] Are VRAM, latency, and hardware constraints explicitly targeted?
