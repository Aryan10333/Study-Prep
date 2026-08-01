---
name: Jupyter Notebook Generator
description: Rules and workflow for programmatically creating, executing, profiling, and documenting real-world production Jupyter Notebooks (.ipynb) using nbformat and virtual environment kernels.
---

# Jupyter Notebook Generator Skill

This skill defines the guidelines for programmatically creating, executing, profiling, and documenting companion Jupyter Notebooks (`.ipynb`) in this repository.

---

## 0. Pre-Flight Checkpoint: Implementation Plan

Before writing or running any Jupyter notebook builder scripts (`build_*_notebooks.py`), the agent **MUST** generate a detailed `implementation_plan.md` artifact detailing:
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
2. **Logical Cell Splitting**: Never write large code pipelines in a single monolithic cell. Split code blocks logically (e.g., `Data Ingestion` $\rightarrow$ `Preprocessing` $\rightarrow$ `Pipeline Setup` $\rightarrow$ `Execution & Profiling` $\rightarrow$ `Validation`)3. **Execution Block Cell Pattern**: Operational code cells MUST strictly follow this 3-cell sequence:
   - **Markdown Cell (Heading)**: Describes the step name and objective (e.g., `## 2. Implementing RRF Hybrid Search`). This cell **MUST** always precede the code cell to introduce the concept.
   - **Code Cell (Implementation)**: Self-contained, runnable Python code annotated with explicit tensor shape comments (e.g., `# [B, L, H]`), runtime assertions, and error handling.
   - **Markdown Cell (Output Explanation & Interpretation)**: Titled `### Output Explanation & Interpretation` (or `### Output Explanation: [Step Topic]`), explaining printed tensor shapes, losses, matrix outputs, and providing a clear conceptual and practical interpretation.
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
4. **Sequential Execution Loop**: Never batch compile or execute multiple notebooks in a bulk run. Always programmatically define, generate, and execute **one notebook at a time sequentially**. Inspect its printed cell outputs, confirm metrics, and write/align markdown cell explanations before proceeding to the next notebook.
5. **Assert Outputs**: Include explicit `assert` statements in code cells to verify tensor shapes, numerical bounds, and non-empty responses, catching runtime errors or numeric drift early.

---

## 3. Post-Execution Explanations & Numerical Alignment

**Rule**: Every operational code cell inside a companion notebook must be immediately followed by a markdown cell titled `### Output Explanation` or `### Output Explanation: [Step Topic]`.

* **Execute First, Explain Second Policy**: When writing or updating notebooks, the agent **MUST first compile and execute the code cell to obtain the actual output logs in the notebook, read those printed outputs from the executed file, and only then write or refine the explanation cell** based directly on the actual results. Do not write hypothetical explanations using pre-execution assumptions.
* **Thorough Inspection**: Analyze printed cell outputs in detail, detailing resulting tensor shapes, loss outputs, or probability distributions.
* **Floating-Point Precision Transparency**: Detail why these numbers and metrics are correct. If intermediate floating-point execution or hardware precision differences cause slight numerical shifts, explicitly document both expected theoretical values and exact floating-point outputs to maintain 100% transparency.
* **Interpretation Insights**: Connect outputs directly to theoretical concepts, explaining what the data represents, why the results match expectations, and how they apply in production.

---

## 4. Automated Verification Checklist

Immediately after generating and executing any notebook, verify:

* [ ] **100% Executed State**: Every code cell has an explicit execution count (`In [1]`, `In [2]`) and populated output logs. No empty brackets (`In [ ]`) exist in the final notebook.
* [ ] **Real-World System Focus**: Pipeline operates on real data/APIs and includes metric checks.
* [ ] **Preceding Heading Cells**: Every code cell is preceded by a markdown heading cell.
* [ ] **Paired Analysis Cells**: Every operational code cell is paired with a corresponding Output Explanation markdown block immediately below it.
* [ ] **Executed Output Alignment**: The contents of all explanation cells have been aligned and verified against actual executed outputs of the notebook cells.
* [ ] **Numerical Offset Documentation**: Any floating-point rounding or precision shifts in printed logs are explicitly explained in the analysis cell.
* [ ] **Environment Security**: Environment variables load dynamically via `find_dotenv()`. Zero hardcoded API keys or local file paths exist.
* [ ] **No Unnecessary PDF Compilation**: Modifying or generating companion notebooks does *not* trigger master HTML/PDF chapter compilation scripts (e.g., `compile_rag.py`, `compile_agents.py`), as notebook changes do not affect PDF text chapters.

---

## 5. Standard Implementation Reference

For a complete, verified, and production-ready script that programmatically builds and executes companion notebooks, refer to:
*   [sample_notebook_generator.py](file:///d:/Study/Prep/.agents/scripts/sample_notebook_generator.py)

> [!IMPORTANT]
> Always run notebook builder/generator scripts using the repository's active Python virtual environment (e.g. `.venv\Scripts\python.exe helpers/build_<topic>_notebooks.py` on Windows) to ensure libraries like `nbformat` and `nbconvert` are correctly loaded.

