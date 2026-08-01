---
name: Jupyter Notebook Generator
description: Rules and workflow for programmatically creating, executing, profiling, and documenting real-world production Jupyter Notebooks (.ipynb) using nbformat and virtual environment kernels.
---

# Jupyter Notebook Generator Skill

This skill defines the guidelines for programmatically creating, executing, profiling, and documenting companion Jupyter Notebooks (`.ipynb`) in this repository.

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
   - **Markdown Cell (Heading)**: Describes the step name and objective (e.g., `## 2. Implementing RRF Hybrid Search & Memory Profiling`).
   - **Code Cell (Implementation)**: Self-contained, runnable Python code annotated with explicit tensor shape comments (e.g., `# [B, L, H]`), runtime assertions, and error handling.
   - **Markdown Cell (Output & System Analysis)**: Titled `### Output & Performance Analysis`, explaining printed tensor shapes, memory footprints, execution latency, and practical trade-offs.
   *(Note: Initial environment setups and `import` blocks are exempt from the Output Analysis cell requirement.)*

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

**Rule**: Every operational code cell inside a companion notebook must be immediately followed by a markdown cell titled `### Output & Performance Analysis`.

* **Thorough Inspection**: Analyze printed cell outputs in detail, detailing resulting tensor shapes, gradients, memory allocation, loss outputs, or probability distributions.
* **Floating-Point Precision Transparency**: Detail why these numbers and metrics are correct. If intermediate floating-point execution or hardware precision differences (e.g., FP16/BF16 vs. FP32) cause slight numerical shifts (e.g., `27.8600` vs. `27.8592`), explicitly document both expected theoretical values and exact floating-point outputs to maintain 100% transparency.
* **System Insights**: Connect system logs directly to real-world performance—highlighting memory bottlenecks, latency trade-offs, and scalability implications.

---

## 4. Automated Verification Checklist

Immediately after generating and executing any notebook, verify:

* [ ] **100% Executed State**: Every code cell has an explicit execution count (`In [1]`, `In [2]`) and populated output logs. No empty brackets (`In [ ]`) exist in the final notebook.
* [ ] **Real-World System Focus**: Pipeline operates on real data/APIs and includes hardware, memory, or latency profiling blocks.
* [ ] **Paired Analysis Cells**: Every operational code cell is paired with a corresponding `### Output & Performance Analysis` markdown block immediately below it.
* [ ] **Numerical Offset Documentation**: Any floating-point rounding or precision shifts in printed logs are explicitly explained in the analysis cell.
* [ ] **Environment Security**: Environment variables load dynamically via `find_dotenv()`. Zero hardcoded API keys or local file paths exist.
* [ ] **No Unnecessary PDF Compilation**: Modifying or generating companion notebooks does *not* trigger master HTML/PDF chapter compilation scripts (e.g., `compile_rag.py`, `compile_agents.py`), as notebook changes do not affect PDF text chapters.
