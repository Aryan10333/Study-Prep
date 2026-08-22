---
name: Jupyter Notebook Generator
description: Rules and workflow for programmatically creating, executing, profiling, and documenting real-world production Jupyter Notebooks (.ipynb) using nbformat and virtual environment kernels.
---

# Jupyter Notebook Generator Skill

This skill defines the guidelines for programmatically creating, executing, profiling, and documenting companion Jupyter Notebooks (`.ipynb`) in this repository.

---

## 0. Pre-Flight Checkpoint: Implementation Plan

Before writing or running any Jupyter notebook builder scripts (`build_*_notebooks.py`), the agent **MUST** generate a detailed `implementation_plans/implementation_plan_notebook.md` artifact (in the topic's `implementation_plans/` subfolder, created if it does not yet exist) detailing:
1.  The list of notebooks to generate with their target file paths.
2.  The real-world datasets and data ingestion APIs to be utilized.
3.  The specific engineering task pipelines, hardware assertions, and metrics profiling steps.
4.  Any open design questions or credentials required.
The agent must wait for the user's explicit sign-off and approval on this implementation plan before executing.

---

## 1. Production Scope & Real-World Engineering Tasks

Notebooks must completely avoid static toy mocks or trivial mathematical abstractions. Every notebook must serve as an end-to-end, production-grade engineering pipeline tailored for a Senior AI Systems Engineer preparing for technical interviews:

- **Real-World Datasets & APIs**: Process diverse, practical data (such as financial logs, IT ticket logs, web pages, Hugging Face Hub datasets, or live search queries) to address real engineering tasks.
- **Production AI Tasks**: Focus on advanced engineering patterns, including parent-child chunking, BM25/Vector RRF hybrid merging, Pydantic structured entity extraction, conversational memory indexing, state graph routing, custom Triton/CUDA flow, and multi-agent developer-reviewer loops.
- **Hardware & System Profiling**: Evaluate real-world hardware constraints in every pipeline—logging peak VRAM footprints (`torch.cuda.max_memory_allocated()`), computational throughput, execution latency (`time.perf_counter()`), and tensor memory layouts (`.is_contiguous()`).

---

## 2. Programmatic Creation & Execution Workflow

To guarantee zero unexecuted cells (`In [ ]`), empty outputs, or corrupted JSON structures, all notebooks MUST be generated and executed programmatically using a Python builder script (e.g., `build_*_nb.py`):

### A. Construction (`nbformat`)
1. **Use `nbformat` v4**: Construct notebook JSON structures programmatically using schema v4 components (`nbf.v4`).
2. **Logical Cell Splitting**: Never write large code pipelines in a single monolithic cell. Split code blocks logically (e.g., `Data Ingestion` $\rightarrow$ `Preprocessing` $\rightarrow$ `Pipeline Setup` $\rightarrow$ `Execution & Profiling` $\rightarrow$ `Validation`).
3. **Execution Block Cell Pattern**: Operational code cells MUST strictly follow this 3-cell sequence:
   - **Markdown Cell (Heading)**: Describes the step name and objective (e.g., `## 2. Implementing RRF Hybrid Search`). This cell **MUST** always precede the code cell to introduce the concept.
   - **Code Cell (Implementation)**: Self-contained, runnable Python code annotated with explicit tensor shape comments (e.g., `# [B, L, H]`), runtime assertions, and error handling.
   - **Markdown Cell (Output Explanation & Interpretation)**: Titled `### Output Explanation & Interpretation` (or `### Output Explanation: [Step Topic]`). See Section 3 below — its real content is **NOT** written at construction time; only a heading and, optionally, a `_(pending real output)_` placeholder go in during this pass. The actual explanation text is authored only in the separate Pass 2 after the notebook has been executed and its real output inspected.
   *(Note: Initial environment setups and `import` blocks are exempt from this Output Explanation requirement.)*

### B. Execution (`nbconvert` & Environment)
1. **Dynamic Environment Variables & API Keys**: If notebook execution requires API access, the builder script and notebook cells must load credentials dynamically using `python-dotenv`:
   ```python
   from dotenv import find_dotenv, load_dotenv

   load_dotenv(find_dotenv())
   ```

Sensitive credentials must **never** be hardcoded in code cells. The following environment variables are available for use:

* `GEMINI_API_KEY` (Google GenAI models)
* `GROQ_API_KEY` (Groq model endpoints)
* `OPENAI_API_KEY` (OpenAI models)
* `HF_TOKEN` (Hugging Face Hub access)
* `OLLAMA_BASE_URL` (Local Ollama endpoints)
* `SERPER_API_KEY` (Google search queries via Serper)
* `TAVILY_API_KEY` (Tavily search API)
* `GITHUB_TOKEN` (GitHub API integrations)

2. **Headless Execution Kernel**: Execute notebooks in place using `nbconvert.preprocessors.ExecutePreprocessor` targeting the active virtual environment Python kernel (`sys.executable`).
3. **Headless Plot Serialization**: When running the execution pipeline script headlessly, ensure `import matplotlib; matplotlib.use('Agg')` is called in the builder script prior to execution. Inside notebook code cells, include `%matplotlib inline` at the top and end plotting cells with `plt.show()` to serialize inline base64 plot strings directly into the `.ipynb` file. Do NOT call `plt.savefig()`.
4. **Sequential Execution Loop**: Never batch compile or execute multiple notebooks in a bulk run. Always programmatically define, generate, and execute **one notebook at a time sequentially**, following the strict two-pass build order in Section 3 (generate + execute with heading/code cells only, *then* author explanation cells from the real output) before proceeding to the next notebook.
5. **Assert Outputs**: Include explicit `assert` statements in code cells to verify tensor shapes, numerical bounds, and non-empty responses, catching runtime errors or numeric drift early.

---

## 3. Post-Execution Explanations & Numerical Alignment

**Rule**: Every operational code cell inside a companion notebook must be immediately followed by a markdown cell titled `### Output Explanation` or `### Output Explanation: [Step Topic]`. That markdown cell's content is authored in a **separate, later pass** than the code — never in the same pass that writes the code.

### 3.1 The Two-Pass Workflow (mandatory order)

**Pass 1 — Generate & Execute (code only).**
1. Write the builder function with all heading cells and code cells for the notebook. For the Output Explanation cells, either omit them entirely in this pass or insert a one-line placeholder (e.g. `"### Output Explanation: [Step Topic]\n_(pending real output)_"`) — never pre-write the real explanation text here, since no real output exists yet to base it on.
2. Execute the notebook end-to-end via `ExecutePreprocessor` and save it.
3. Read the **actual executed `.ipynb`** (its real cell outputs — printed numbers, tensor shapes, generated strings, plot captions), not what you expected it to print.

**Pass 2 — Author Explanations From Real Output.**
4. For every code cell, copy the literal values it actually printed and write the explanation cell around them. Do this either by editing the builder script's placeholder strings and re-running it, or by editing the executed `.ipynb` cells directly (e.g. with a notebook-cell-editing tool) once Pass 1's output is known — the second approach avoids re-execution drift on non-deterministic cells (sampling, live API calls, CUDA kernel ordering) and is preferred when the values must match an already-finalized run exactly.
5. Never go back and reuse Pass-1 boilerplate language ("the loss decreases as expected", "real GPU memory is used") as a substitute for the literal numbers — the explanation is not complete until it is re-derived from what Pass 1 actually printed.

### 3.2 Literal-Quote Requirement (what makes an explanation non-generic)

Every Output Explanation cell **MUST quote at least one literal value verbatim, in backticks, copied from that specific cell's real printed output** (a number, a tensor shape, a generated string, a loss sequence). A sentence that could be pasted unchanged into a *different* notebook run without becoming wrong is not acceptable — it means the explanation is describing the code's intent rather than its actual result.

| Generic (banned) | Literal (required) |
|---|---|
| "Loss decreases over training, confirming the model learns." | "Loss fell `4.0583 -> 3.4257 -> 2.9744 -> 2.5492 -> 2.1846` over 5 steps — a 46% relative drop." |
| "Real GPU memory usage is measured for the parameters." | "`124,439,808` params measured at `498.6 MB`, i.e. `4.01` bytes/param — matching fp32 almost exactly." |
| "The model generates plausible completions for the prompt." | "Completion `[0]`: `\"What's the big deal? When you are talking...\"` scored a real reward of `8.207`, the highest of the 4 sampled." |
| "Accuracy improves as training proceeds." | "Accuracy rose `0.25 -> 0.50 -> 0.62 -> 0.62` across steps 1/4/7/10, plateauing below 1.00 given only 8 training pairs." |

* **Thorough Inspection**: Analyze printed cell outputs in detail, quoting resulting tensor shapes, loss sequences, or probability distributions directly rather than summarizing them abstractly.
* **Floating-Point Precision Transparency**: When a measured number doesn't land exactly on the theoretical prediction (e.g. `16.36` measured vs. `16.0` predicted), state both the literal measured value and the theoretical one side by side, and explain the gap (allocator overhead, precision path, sampling variance) — don't silently round one to match the other.
* **Interpretation Insights**: Connect the *literal quoted values* to theoretical concepts — explain what those specific numbers represent, why they do or don't match expectations, and how the same pattern applies in production. The interpretation must be anchored to the numbers just quoted, not stated independently of them.

---

## 4. Automated Verification Checklist

Immediately after generating and executing any notebook, verify:

* [ ] **100% Executed State**: Every code cell has an explicit execution count (`In [1]`, `In [2]`) and populated output logs. No empty brackets (`In [ ]`) exist in the final notebook.
* [ ] **Real-World System Focus**: Pipeline operates on real data/APIs and includes metric checks.
* [ ] **Preceding Heading Cells**: Every code cell is preceded by a markdown heading cell.
* [ ] **Paired Analysis Cells**: Every operational code cell is paired with a corresponding Output Explanation markdown block immediately below it.
* [ ] **Executed Output Alignment**: The contents of all explanation cells have been aligned and verified against actual executed outputs of the notebook cells.
* [ ] **Literal Quotes, Not Generic Prose**: Every Output Explanation cell quotes at least one literal value (in backticks) copied verbatim from that cell's real output — per Section 3.2. Reread each explanation cell in isolation and confirm it would be *wrong*, not just imprecise, if pasted onto a different run's output.
* [ ] **Numerical Offset Documentation**: Any floating-point rounding or precision shifts in printed logs are explicitly explained in the analysis cell.
* [ ] **Environment Security**: Environment variables load dynamically via `find_dotenv()`. Zero hardcoded API keys or local file paths exist.
* [ ] **No Unnecessary PDF Compilation**: Modifying or generating companion notebooks does *not* trigger master HTML/PDF chapter compilation scripts (e.g., `compile_rag.py`, `compile_agents.py`), as notebook changes do not affect PDF text chapters.

---

## 5. Standard Implementation Reference

For a complete, verified, and production-ready script that programmatically builds and executes companion notebooks, refer to:
*   [sample_notebook_generator.py](file:///d:/Study/Prep/.agents/scripts/sample_notebook_generator.py)

> [!IMPORTANT]
> Always run notebook builder/generator scripts using the repository's active Python virtual environment (e.g. `.venv\Scripts\python.exe helpers/build_<topic>_notebooks.py` on Windows) to ensure libraries like `nbformat` and `nbconvert` are correctly loaded.

