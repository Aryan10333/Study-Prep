---
name: Jupyter Notebook Generator
description: Rules and workflow for programmatically creating, executing, and explaining companion Jupyter Notebooks (.ipynb) using nbformat and virtual environment kernels.
---

# Jupyter Notebook Generator Skill

This skill defines the guidelines for creating, executing, and documenting Jupyter Notebooks (`.ipynb`) in this repository.

---

## 1. Programmatic Creation Workflow

To avoid empty placeholders or unexecuted notebook states, all companion notebooks must be generated programmatically using a builder script (e.g., `build_*_nb.py`):

1. **Use `nbformat`**: Construct a new notebook JSON structure programmatically using schema v4 components (`nbf.v4`).
2. **Inject Markdown Explanations**: Interleave markdown explanation cells detailing equations, setups, and matrix parameters. **Limit Complex Math**: Avoid dense mathematical derivations or algebraic proofs inside notebooks. Keep equations clear and focus on parameters intuition, pros, cons, and production trade-offs.
3. **Inject PyTorch/NumPy Executable Cells**: Add code blocks containing matrices and vectors that exactly match the hand calculations in the study guide.
4. **Save Draft**: Save the unexecuted notebook to disk.
5. **Diverse Real-World Datasets & Practical Examples**: Notebooks must avoid static toy mocks or trivial mathematical abstractions. They must load and process diverse, real-world data (such as financial logs, IT ticket logs, web pages, or files from Hugging Face/scraping) to address practical engineering problems (e.g. parent-child chunking, BM25/Vector RRF hybrid merging, Pydantic entity extraction, conversational memory indexing, state graph routing, and multi-agent developer-reviewer loops) which mirror senior AI engineer production tasks.

---

## 2. Programmatic Execution & Verification

After generating the draft, the script must execute the notebook in place:

1. **Local Kernel**: Use `nbconvert.preprocessors.ExecutePreprocessor` to run the cells in sequence.
2. **Virtual Environment**: Execute using the local python executable:
   `d:\Study\Prep\.venv\Scripts\python.exe`
3. **One-by-One Sequential Execution Loop**: Never batch compile or execute multiple notebooks in a single bulk run. Always programmatically define, generate, and execute **one notebook at a time sequentially**. Inspect its printed cell outputs, confirm numerical metrics, and write/align the markdown cell explanations before proceeding to generate and execute the next notebook.
4. **Matplotlib Agg Backend & Inline Plots**: When running the execution pipeline script headlessly, ensure that `import matplotlib; matplotlib.use('Agg')` is called in the builder script **prior** to running execution prep. Inside the notebook code cells themselves, do not call `plt.savefig()`. Instead, include `%matplotlib inline` at the top of the plotting cells and end the cell with `plt.show()` to ensure that drawn plots are serialized directly as inline base64 string outputs inside the `.ipynb` file.
5. **Multi-Cell Structuring & Sequence**: Never write large code pipelines in a single cell. Split code blocks logically (e.g., Data Loading -> Preprocessing -> Model Setup -> Execution -> Validation) and structure each section strictly using this three-part cell sequence:
   - **Markdown Cell (Heading)**: Describes the step name and objective (e.g., `## 1. Step Name`).
   - **Code Cell (Implementation)**: Self-contained python code executing that specific step.
   - **Markdown Cell (Output Explanation)**: Titled `### Output Analysis: ...` explaining the printed values and shapes immediately following the code cell.
6. **Assert Outputs**: Include assertions in python code cells to verify calculation bounds, catching any runtime PyTorch or numeric drift errors.
7. **Environment Variables & API Keys**: If the execution of the notebook requires API access, the script or notebook cells must load keys dynamically using `python-dotenv` (i.e., `from dotenv import load_dotenv; load_dotenv()`) from the **root `.env` file** (located at the root of the repository: `d:\Study\Prep\.env`). Sensitive credentials must never be hardcoded in code cells. The following environment variables are available for use:
   - `GEMINI_API_KEY` (for Google GenAI models)
   - `GROQ_API_KEY` (for Groq model endpoints)
   - `HF_TOKEN` (for Hugging Face Hub downloads/uploads)
   - `OLLAMA_BASE_URL` (for local Ollama endpoints)
   - `OPENAI_API_KEY` (for OpenAI models)
   - `SERPER_API_KEY` (for Google search queries via Serper)
   - `TAVILY_API_KEY` (for Tavily search API)
   - `GITHUB_TOKEN` (for GitHub API integrations)
8. **Save Executed State**: Save the final notebook with cell outputs populated.

---

## 3. Mandatory Post-Execution Explanations

**Rule**: Every code execution cell inside a companion Jupyter Notebook must be immediately followed by a markdown cell explaining the printed outputs.

- Read cell outputs in detail, and write a thorough explanation of why the values are correct.
- Detail the resulting tensor shapes, gradients, loss outputs, or probability distributions.
- Explain why these numbers are correct.
- Cross-reference printed logs with the corresponding math study guide to ensure **100% numerical consistency** (matching values, similarity scores, and classification distribution metrics to 4 decimal places). If there is a slight numerical shift due to rounding in intermediate hand-calculation steps (e.g. `27.8600` vs. `27.8592`), clarify this in the explanation cell so that the printed output and explanation align perfectly.

---

## 4. Verification Check
- Ensure all cells are executed and output logs are preserved.
- Double-check that no empty brackets (`In [ ]`) exist in the final notebook.
- Verify that every code cell is paired with a corresponding explanation block below it.
- **No PDF Recompilation on Notebook Changes**: Modifying or generating companion notebooks does *not* affect the text study chapters. Do NOT trigger or run the master HTML/PDF compilation scripts (e.g. `compile_rag.py`, `compile_agents.py`) after notebook changes, as it has no effect on the resulting PDF guides.

