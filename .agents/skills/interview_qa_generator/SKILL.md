---
name: Interview QA Generator
description: Guidelines and instructions for generating high-quality interview preparation questions & answers in MD, HTML, and PDF formats, keeping them separate from theory documents.
---

# Interview Q&A Generation & Compilation Skill

This skill defines the standardized process for creating dedicated, high-impact **Interview Question & Answer** prep materials (Q&As, cheatsheets) and compiling them to independent HTML and PDF outputs, keeping them distinct from theoretical curriculum guides.

---

## 0. Pre-Flight Checkpoint: Implementation Plan

Before writing any standalone Interview Q&A cheatsheets or question banks, the agent **MUST** generate a detailed `implementation_plans/implementation_plan_interview_qa.md` artifact (in the topic's `implementation_plans/` subfolder, created if it does not yet exist) detailing:
1.  The proposed list of technical screening questions grouped by concept categories.
2.  The target file paths for raw Markdown Q&A source documents and compiled deliverables.
3.  Any specific reference sources or question banks to incorporate.
The agent must wait for the user's explicit sign-off and approval on this implementation plan before executing.

---

## 1. Context & Motivation

To crack high-bar technical interviews (AI Engineer, GenAI Engineer, ML Engineer), candidates need study guides that translate raw curriculum theory into quick screening responses, technical calculations, and production trade-offs. 

By separating **Theory Modules** (which contain deep conceptual text, diagrams, and code notebooks) from **Interview Q&As** (which focus on screening speed, buzzwords, and system-level trade-offs), we allow for:
1. Focused, high-intensity review sessions prior to interviews.
2. Lightweight standalone cheatsheets and PDFs (e.g., `*_Interview_Cheatsheet.pdf`) optimized for quick lookup.
3. Logical compilation where Q&A guides append as the final chapter in master guides.

---

## 2. Standardized Q&A Format

For every interview question generated, the output must strictly follow this structure:

```markdown
## Question [Number]: [Question Title]

### [ESSENTIAL]

#### Conversational Answer
[Spoken-style interview response, written in a first-person/conversational flow (e.g. "I'd explain that...", "The core reason is...") rather than textbook passive paragraphs. Keep it punchy and direct.]

#### Intuitive Example
*   [A concrete, real-world example spanning 2–4 lines detailing how the concept applies to simple text or sequences.]

#### Key Interview Points
- **[Keyword/Buzzword 1]**: Short definition.
- **[Keyword/Buzzword 2]**: Short definition.
- **[Keyword/Buzzword 3]**: Short definition.

---

### [DEEP DIVE]

#### Technical Intuition & Key Formulas (No Derivations)
[Introduce the core formulas in KaTeX, with direct variable explanations. Remove all step-by-step mathematical proofs or variance derivations, keeping only the high-level intuition.]

#### Production Perspective & Trade-offs
[Analyze VRAM footprints, memory-bandwidth limits, latency budgets, GPU memory access, or server bottlenecks.]

#### Common Mistakes
*   **Common Mistakes**:
    1. [Top frequent misconception candidate makes - keep it short and focus on top 2-3 mistakes.]
    2. [Second frequent misconception.]

#### Common Follow-up Questions
1.  **Q: [Follow-up Question]?**
    *   **A**: [Conversational Answer].
2.  **Q: [Follow-up Question]?**
    *   **A**: [Conversational Answer].

#### One-Line Takeaway
> **Takeaway:** [One-sentence summary for fast revision.]
```

### Comparison Tables Rule:
Use comparison tables where appropriate to contrast competing architectures, methods, or parameters (e.g., GPT vs. BERT vs. T5, LayerNorm vs. RMSNorm, RoPE vs. ALiBi vs. Learned, MHA vs. GQA vs. MQA, Greedy vs. Beam Search vs. Sampling, Dense vs. MoE).

### Final Revision Sheet Rule:
Every interview cheatsheet must conclude with a dedicated 2-3 page revision sheet section:
`# [Topic] Interview Cheatsheet: Final Revision Sheet`
This section must contain:
1.  **Quick-Recall One-Line Takeaways Table**: A dense table mapping all questions to their respective one-line takeaways.
2.  **Essential Formula Cheat Sheet**: A list of core mathematical equations (e.g., Attention, Normalization, Scaling Laws, etc.) with KaTeX styling.
3.  **Top Follow-up Q&As**: A fast-review index of the most critical follow-up questions and answers.

### Global Selection Rule:
Not every question requires math formulas, numerical examples, complexity analysis, or code snippets. Include them only when they representationally improve the explanation (e.g., Self-Attention, KV Cache, RoPE, Scaling Laws, MoE). For purely conceptual questions, prioritize intuition, architecture, and production constraints.

### Simple Question Titles Rule:
Do not inject mathematical formulas or technical derivations directly into the `Question Title` itself. Keep the question title simple, clean, and direct (e.g. `Why divide attention scores by √d?` instead of `Why do we scale the attention score matrix by 1/√d_k? Derive the variance...`). Place all mathematical proofs, KaTeX calculations, and derivations exclusively in the *Technical Intuition & Complexity* or *Production Perspective* sections of the answer.

### Batch-Wise Generation Rule:
When generating answers for a large question bank (e.g. 20+ questions), do NOT write the entire set in a single turn. Instead, segment the list into logical batches of 10–15 questions. Write and append one batch at a time to the Markdown file sequentially. Do this automatically in sequence without pausing to ask for user approval between batches. A single final review and sign-off is requested at the very end when all batches are fully answered. This prevents context token limits and ensures maximum detail.

---

## 3. Compilation Architecture

Compilation scripts (e.g., written in Python) must compile the Q&A documents separately from raw theory documents when building standalone cheatsheets, while also supporting consolidated master guides.

<div class="custom-diagram" style="margin: 20px 0; background-color: #f8fafc; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; font-family: inherit; display: flex; flex-direction: column; align-items: center; gap: 15px;">
    <div style="font-weight: bold; color: #0f172a; font-size: 13px;">Compilation Pipeline Architecture</div>
    <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; width: 100%;">
        <!-- Left Box: Source -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 6px; align-items: center;">
            <div style="color: #475569; font-weight: bold; font-size: 11px; text-transform: uppercase;">Raw Source Files</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center;">01_Theory.md</div>
            <div style="background-color: #f1f5f9; color: #334155; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center;">02_Theory.md</div>
            <div style="background-color: #f5f3ff; color: #5b21b6; border: 1px solid #7c3aed; padding: 4px 10px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">13_Interview_Questions.md</div>
        </div>
        <!-- Right Box: Formats -->
        <div style="flex: 1; min-width: 200px; max-width: 250px; background-color: #ffffff; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 8px; align-items: center;">
            <div style="color: #475569; font-weight: bold; font-size: 11px; text-transform: uppercase;">Compilation Outputs</div>
            <div style="background-color: #eff6ff; color: #1e40af; border: 1px solid #2563eb; padding: 6px 12px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">Standalone Cheatsheet (PDF)</div>
            <div style="background-color: #ecfdf5; color: #065f46; border: 1px solid #059669; padding: 6px 12px; border-radius: 4px; font-size: 11px; width: 90%; text-align: center; font-weight: 600;">Master Study Guide (PDF)</div>
        </div>
    </div>
</div>

### PDF & HTML Build Rules:
1. **Alert Parsing**: Convert GitHub alert styles (`> [!NOTE]`, etc.) into left-bordered HTML divs with background colors.
2. **Pygments Highlighting**: CSS must enforce dark slate or monokai code block syntax styling (`HtmlFormatter(style='monokai')`).
3. **Contrast Enforcement**: Prevent dark-on-dark text artifacts inside code blocks using strict CSS overrides:
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
4. **KaTeX Integration**: Embed standard KaTeX scripts and JS auto-renderers in headers to compile LaTeX math formulas dynamically:
   ```html
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
   <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
   <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
   ```
5. **Headless Browser Rendering**: Run headless Microsoft Edge command line printer to convert compiled HTML output to standard-aligned PDFs:
   ```python
   cmd = [
       "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
       f"--user-data-dir={temp_user_data}",
       "--headless",
       "--disable-gpu",
       "--run-all-compositor-stages-before-draw",
       "--no-pdf-header-footer",
       f"--print-to-pdf={pdf_out_path}",
       html_out_path
   ]
   ```

---

## 4. Verification Checklist

Before final delivery, verify:
- [ ] **Math delimiters**: Single `$` for inline math, double `$$` on a single line for display blocks.
- [ ] **GFM Tables**: Tables are written using native Markdown formatting and are never wrapped inside fenced code blocks.
- [ ] **No placeholders**: Every question has a complete answer without placeholders.
- [ ] **Variable consistency**: Variable notations ($L$, $d$, $b$, $N$, $C$) align 100% between complexity sheets and explanations.

---

## 5. Mandatory Structural Compliance Check (Automated, Not Optional)

A checklist read by an agent is easy to silently skip under context pressure or when resuming a large batch job — this has previously produced a fully-written 50-question cheatsheet that used none of the required section structure (no `[ESSENTIAL]`/`[DEEP DIVE]` split, no takeaways, no Final Revision Sheet) without anyone catching it until a later review. To prevent this from recurring, the agent **MUST** run an explicit grep-based self-check against the finished `*_interview_questions.md` file before presenting it to the user or handing it to the compiler:

1. **Per-question structure**: Every `## Question N:` block must contain all of: `### [ESSENTIAL]`, `#### Conversational Answer`, `#### Intuitive Example`, `#### Key Interview Points`, `### [DEEP DIVE]`, `#### Technical Intuition & Key Formulas (No Derivations)`, `#### Production Perspective & Trade-offs`, `#### Common Mistakes`, `#### Common Follow-up Questions`, `#### One-Line Takeaway`. Count occurrences of each heading and confirm the count matches the total number of questions.
2. **No Derivations rule**: Scan each `[DEEP DIVE]` block for multi-step derivation chains (sequences of `$$...$$` blocks connected by "differentiating", "substituting", "therefore" prose). Formulas are allowed; step-by-step proofs are not — collapse them to the final formula plus a one-line intuition.
3. **Final Revision Sheet**: Confirm the file ends with the `# [Topic] Interview Cheatsheet: Final Revision Sheet` section, and that it contains all three required subsections (Quick-Recall Takeaways Table, Essential Formula Cheat Sheet, Top Follow-up Q&As).
4. **Report the check**: State explicitly to the user which of the four checks passed/failed before declaring the track complete. If any check fails, fix the file — do not present it as done with a caveat.

This check applies whenever a Q&A file is generated *or* edited, including resuming/appending batches to an existing file. If you are asked to review or extend an existing `*_interview_questions.md` file, run this check on it first — do not assume a pre-existing file is compliant.
