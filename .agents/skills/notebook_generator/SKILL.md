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
2. **Inject Markdown Explanations**: Interleave markdown explanation cells detailing equations, setups, and matrix parameters.
3. **Inject PyTorch/NumPy Executable Cells**: Add code blocks containing matrices and vectors that exactly match the hand calculations in the study guide.
4. **Save Draft**: Save the unexecuted notebook to disk.

---

## 2. Programmatic Execution & Verification

After generating the draft, the script must execute the notebook in place:

1. **Local Kernel**: Use `nbconvert.preprocessors.ExecutePreprocessor` to run the cells in sequence.
2. **Virtual Environment**: Execute using the local python executable:
   `d:\Study\Prep\.venv\Scripts\python.exe`
3. **Assert Outputs**: Include assertions in python code cells to verify calculation bounds, catching any runtime PyTorch or numeric drift errors.
4. **Environment Variables & API Keys**: If the execution of the notebook requires API access, the script or notebook cells must load keys dynamically using `python-dotenv` (i.e., `from dotenv import load_dotenv; load_dotenv()`) from the **root `.env` file** (located at the root of the repository: `d:\Study\Prep\.env`). Sensitive credentials must never be hardcoded in code cells. The following environment variables are available for use:
   - `GEMINI_API_KEY` (for Google GenAI models)
   - `GROQ_API_KEY` (for Groq model endpoints)
   - `HF_TOKEN` (for Hugging Face Hub downloads/uploads)
   - `OLLAMA_BASE_URL` (for local Ollama endpoints)
   - `SERPER_API_KEY` (for Google search queries via Serper)
   - `TAVILY_API_KEY` (for Tavily search API)
   - `GITHUB_TOKEN` (for GitHub API integrations)
5. **Save Executed State**: Save the final notebook with cell outputs populated.

---

## 3. Mandatory Post-Execution Explanations

**Rule**: Every code execution cell inside a companion Jupyter Notebook must be immediately followed by a markdown cell explaining the printed outputs.

- Detail the resulting tensor shapes, gradients, loss outputs, or probability distributions.
- Explain why these numbers are correct.
- Cross-reference printed logs with the corresponding math study guide to ensure **100% numerical consistency** (matching values, similarity scores, and classification distribution metrics to 4 decimal places).

---

## 4. Verification Check
- Ensure all cells are executed and output logs are preserved.
- Double-check that no empty brackets (`In [ ]`) exist in the final notebook.
- Verify that every code cell is paired with a corresponding explanation block below it.
