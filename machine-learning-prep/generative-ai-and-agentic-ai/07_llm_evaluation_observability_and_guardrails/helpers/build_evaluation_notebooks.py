import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")


def run_and_save(nb, notebook_out_path, timeout=900):
    os.makedirs(os.path.dirname(notebook_out_path), exist_ok=True)
    nb["metadata"] = {
        "kernelspec": {"display_name": "prep-venv", "language": "python", "name": "prep-venv"},
        "language_info": {"name": "python"},
    }
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="prep-venv")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_out_path) or "."}})
    with open(notebook_out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved: {notebook_out_path}")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================================
# Notebook 01: Real Automated Metrics vs. Real Human-Style Correctness Labels
# ============================================================================

def build_01_automated_metrics_vs_correctness():
    cells = []

    cells.append(md(
        "# Notebook 01: Real Automated Metrics vs. Real Human-Style Correctness Labels\n"
        "\n"
        "`[REAL]` Companion to Modules 01-02. Real `gpt-4o-mini` calls generating real, varied-phrasing answers "
        "to a real, small set of factual questions, scored under real BLEU-1/ROUGE-1 and a real, rigorous, "
        "pre-stated correctness protocol -- testing whether Module 02's small hand-worked counterexample "
        "(n-gram overlap can favor a wrong answer over a correct rephrasing) generalizes across a larger real sample.\n"
        "\n"
        "**Correctness labeling protocol (stated before any generation, per the signed-off plan):** each question "
        "has one authoritative real reference fact, fixed in advance. A real generated answer is labeled correct "
        "**only if** the authoritative fact string (case-insensitive) appears literally in the generated text -- "
        "an explicit, reproducible, defensible real rule, not a subjective judgment applied after seeing results.\n"
        "\n"
        "**Sample-size caveat, stated upfront:** this notebook's real sample (8 questions x 2 real generations "
        "each = 16 items) is small. Any correlation finding below is reported as **exploratory**, not a "
        "statistically robust conclusion -- Module 02's own small, hand-verified counterexample remains the "
        "real, load-bearing evidence regardless of how this larger sample's trend comes out."
    ))

    cells.append(code(
        "import os\n"
        "import math\n"
        "from collections import Counter\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "MODEL = \"gpt-4o-mini\"\n"
        "print(f\"OpenAI client ready. Model: {MODEL}\")"
    ))

    cells.append(md(
        "## 1. Real Question Set with Pre-Fixed Authoritative Facts\n"
        "\n"
        "`[REAL]` 8 real factual questions, each with one authoritative real fact string and a real reference "
        "sentence (for BLEU/ROUGE scoring) -- both fixed **before** any real generation happens."
    ))

    questions_cell_index = len(cells)
    cells.append(code(
        "QUESTIONS = [\n"
        "    {\"q\": \"What is the capital of France?\", \"fact\": \"Paris\", \"reference\": \"The capital of France is Paris.\"},\n"
        "    {\"q\": \"What is the chemical symbol for gold?\", \"fact\": \"Au\", \"reference\": \"The chemical symbol for gold is Au.\"},\n"
        "    {\"q\": \"Who wrote Romeo and Juliet?\", \"fact\": \"Shakespeare\", \"reference\": \"Romeo and Juliet was written by William Shakespeare.\"},\n"
        "    {\"q\": \"What is the largest planet in our solar system?\", \"fact\": \"Jupiter\", \"reference\": \"The largest planet in our solar system is Jupiter.\"},\n"
        "    {\"q\": \"In what year did World War II end?\", \"fact\": \"1945\", \"reference\": \"World War II ended in 1945.\"},\n"
        "    {\"q\": \"What is the boiling point of water in Celsius at sea level?\", \"fact\": \"100\", \"reference\": \"Water boils at 100 degrees Celsius at sea level.\"},\n"
        "    {\"q\": \"What is the currency of Japan?\", \"fact\": \"yen\", \"reference\": \"The currency of Japan is the yen.\"},\n"
        "    {\"q\": \"Who painted the Mona Lisa?\", \"fact\": \"da Vinci\", \"reference\": \"The Mona Lisa was painted by Leonardo da Vinci.\"},\n"
        "]\n"
        "print(f\"Real question set fixed: {len(QUESTIONS)} questions, each with an authoritative fact and reference.\")\n"
        "for item in QUESTIONS:\n"
        "    print(f\"  Q: {item['q']!r} -> fact={item['fact']!r}\")"
    ))

    cells.append(md(
        "## 2. Real Generation: Two Real Answers Per Question, Varied Phrasing\n"
        "\n"
        "`[REAL]` Two real `gpt-4o-mini` calls per question -- one at low temperature (direct), one at higher "
        "temperature with an explicit real instruction to phrase the answer differently -- to produce genuine "
        "paraphrase variety, extending Module 01's own 5-paraphrase illustration."
    ))

    generation_cell_index = len(cells)
    cells.append(code(
        "def generate_answer(question, temperature, style_instruction):\n"
        "    prompt = f\"{question} {style_instruction} Keep it to one sentence.\"\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL,\n"
        "        messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=temperature,\n"
        "        max_tokens=60,\n"
        "    )\n"
        "    return resp.choices[0].message.content.strip()\n"
        "\n"
        "generated_items = []\n"
        "for item in QUESTIONS:\n"
        "    direct = generate_answer(item[\"q\"], temperature=0.0, style_instruction=\"Answer directly and simply.\")\n"
        "    varied = generate_answer(item[\"q\"], temperature=0.9, style_instruction=\"Answer in a full, differently-phrased sentence, varying your wording.\")\n"
        "    generated_items.append({**item, \"answer\": direct, \"variant\": \"direct\"})\n"
        "    generated_items.append({**item, \"answer\": varied, \"variant\": \"varied\"})\n"
        "    print(f\"Q: {item['q']!r}\")\n"
        "    print(f\"  direct: {direct!r}\")\n"
        "    print(f\"  varied: {varied!r}\")\n"
        "\n"
        "print(f\"\\nTotal real generated items: {len(generated_items)}\")\n"
        "print(\"\\n(pending real scoring)\")"
    ))

    cells.append(md(
        "## 3. Real BLEU-1/ROUGE-1 Scoring and Real Correctness Labeling\n"
        "\n"
        "`[COMPUTED FROM REAL DATA]` Scoring each real generated answer with Module 02's own BLEU-1 formula "
        "against its question's real reference, and applying the pre-stated correctness protocol (Section "
        "intro) -- both computed directly from this notebook's own real generated text, not simulated."
    ))

    scoring_cell_index = len(cells)
    cells.append(code(
        "def tokenize(s):\n"
        "    return s.lower().rstrip(\".\").split()\n"
        "\n"
        "def bleu1_precision(candidate, reference):\n"
        "    cand_tokens = tokenize(candidate)\n"
        "    ref_counts = Counter(tokenize(reference))\n"
        "    cand_counts = Counter(cand_tokens)\n"
        "    clipped = sum(min(c, ref_counts[w]) for w, c in cand_counts.items())\n"
        "    return clipped / len(cand_tokens) if cand_tokens else 0.0\n"
        "\n"
        "def is_correct(answer, fact):\n"
        "    \"\"\"Pre-stated protocol: correct iff the authoritative fact string appears literally\n"
        "    (case-insensitive) in the real generated answer.\"\"\"\n"
        "    return fact.lower() in answer.lower()\n"
        "\n"
        "for item in generated_items:\n"
        "    item[\"bleu1\"] = bleu1_precision(item[\"answer\"], item[\"reference\"])\n"
        "    item[\"correct\"] = is_correct(item[\"answer\"], item[\"fact\"])\n"
        "    print(f\"[{item['variant']:6s}] BLEU-1={item['bleu1']:.3f}, correct={item['correct']}, answer={item['answer']!r}\")\n"
        "\n"
        "n_correct = sum(1 for i in generated_items if i[\"correct\"])\n"
        "print(f\"\\nReal correctness rate: {n_correct}/{len(generated_items)} = {n_correct/len(generated_items)*100:.1f}%\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 4. Real, Exploratory Correlation: BLEU-1 vs. Correctness\n"
        "\n"
        "`[COMPUTED FROM REAL DATA]` Comparing real mean BLEU-1 scores between the real correct and real "
        "incorrect groups -- reported as an exploratory finding given the small real sample size, per the "
        "signed-off plan's explicit caveat."
    ))

    correlation_cell_index = len(cells)
    cells.append(code(
        "correct_bleu = [i[\"bleu1\"] for i in generated_items if i[\"correct\"]]\n"
        "incorrect_bleu = [i[\"bleu1\"] for i in generated_items if not i[\"correct\"]]\n"
        "\n"
        "print(f\"Real correct-group BLEU-1: n={len(correct_bleu)}, mean={sum(correct_bleu)/len(correct_bleu):.3f}\" if correct_bleu else \"No correct items\")\n"
        "print(f\"Real incorrect-group BLEU-1: n={len(incorrect_bleu)}, mean={sum(incorrect_bleu)/len(incorrect_bleu):.3f}\" if incorrect_bleu else \"No incorrect items\")\n"
        "\n"
        "print(f\"\\nSample size: {len(generated_items)} items -- EXPLORATORY finding only, not a statistically robust result.\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "01_automated_metrics_vs_correctness.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "questions_cell_index": questions_cell_index,
        "generation_cell_index": generation_cell_index,
        "scoring_cell_index": scoring_cell_index,
        "correlation_cell_index": correlation_cell_index,
    }


# ============================================================================
# Notebook 02: Real LLM-as-Judge: Position Bias & Calibration
# ============================================================================

def build_02_llm_as_judge_bias_and_calibration():
    cells = []

    cells.append(md(
        "# Notebook 02: Real LLM-as-Judge -- Position Bias & Calibration\n"
        "\n"
        "`[REAL]` Companion to Module 03. Real `gpt-4o-mini` judge calls testing position bias and calibration.\n"
        "\n"
        "**Judge-independence, stated per the signed-off plan:** the response sets scored/compared below are "
        "**manually constructed, fixed, deterministic text** (written directly, not LLM-generated), with a "
        "real, objective quality ground truth -- the count of real, verifiably-correct facts each response "
        "contains, checked by direct string matching. This keeps calibration genuinely independent of any "
        "LLM's judgment: the ground truth the judge is checked against was never produced by another model."
    ))

    cells.append(code(
        "import os\n"
        "import re\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "MODEL = \"gpt-4o-mini\"\n"
        "print(f\"OpenAI client ready. Model: {MODEL}\")"
    ))

    cells.append(md(
        "## 1. Real, Manually-Constructed Response Sets with a Deterministic Quality Ground Truth\n"
        "\n"
        "`[REAL]` Three real questions, each with 4 manually-written responses containing 0, 1, 2, or 3 out of "
        "3 real, correct facts -- a real, objective, judge-independent quality score computed by direct string "
        "matching, not any model's opinion."
    ))

    responses_cell_index = len(cells)
    cells.append(code(
        "QUESTION_SETS = [\n"
        "    {\n"
        "        \"question\": \"Name three primary colors.\",\n"
        "        \"facts\": [\"red\", \"blue\", \"yellow\"],\n"
        "        \"responses\": {\n"
        "            0: \"Green, purple, and orange are primary colors.\",\n"
        "            1: \"Red, green, and purple are primary colors.\",\n"
        "            2: \"Red, blue, and purple are primary colors.\",\n"
        "            3: \"Red, blue, and yellow are primary colors.\",\n"
        "        },\n"
        "    },\n"
        "    {\n"
        "        \"question\": \"Name three noble gases.\",\n"
        "        \"facts\": [\"helium\", \"neon\", \"argon\"],\n"
        "        \"responses\": {\n"
        "            0: \"Oxygen, nitrogen, and hydrogen are noble gases.\",\n"
        "            1: \"Helium, oxygen, and nitrogen are noble gases.\",\n"
        "            2: \"Helium, neon, and oxygen are noble gases.\",\n"
        "            3: \"Helium, neon, and argon are noble gases.\",\n"
        "        },\n"
        "    },\n"
        "    {\n"
        "        \"question\": \"Name three programming paradigms.\",\n"
        "        \"facts\": [\"functional\", \"object-oriented\", \"imperative\"],\n"
        "        \"responses\": {\n"
        "            0: \"Compiled, interpreted, and scripted are programming paradigms.\",\n"
        "            1: \"Functional, compiled, and interpreted are programming paradigms.\",\n"
        "            2: \"Functional, object-oriented, and compiled are programming paradigms.\",\n"
        "            3: \"Functional, object-oriented, and imperative are programming paradigms.\",\n"
        "        },\n"
        "    },\n"
        "]\n"
        "\n"
        "def real_objective_quality(response_text, facts):\n"
        "    \"\"\"Real, deterministic ground truth: count of real facts present via direct string match --\n"
        "    never an LLM's opinion.\"\"\"\n"
        "    return sum(1 for f in facts if f.lower() in response_text.lower())\n"
        "\n"
        "for qs in QUESTION_SETS:\n"
        "    print(f\"Q: {qs['question']!r}\")\n"
        "    for level, text in qs[\"responses\"].items():\n"
        "        measured = real_objective_quality(text, qs[\"facts\"])\n"
        "        assert measured == level, f\"Response quality-level mismatch: expected {level}, measured {measured}\"\n"
        "        print(f\"  [{level}/3 real facts] {text!r}\")\n"
        "print(\"\\nAll response quality levels verified to match their real, deterministic fact-count exactly.\")"
    ))

    cells.append(md(
        "## 2. Real Judge Scoring and Calibration (Spearman Correlation vs. Objective Ground Truth)\n"
        "\n"
        "`[REAL]` A real `gpt-4o-mini` judge call scores each of the 12 real responses 1-10 for quality. "
        "Calibration is computed as real Spearman correlation between these real judge scores and the real, "
        "judge-independent objective fact-count ground truth from Section 1 -- not against another LLM's ranking."
    ))

    calibration_cell_index = len(cells)
    cells.append(code(
        "def judge_score(question, response):\n"
        "    prompt = (\n"
        "        f\"Question: {question}\\nResponse: {response}\\n\\n\"\n"
        "        \"Rate the quality of this response on a scale from 1 to 10, where 10 is a fully correct, \"\n"
        "        \"excellent answer. Reply with ONLY the integer score, nothing else.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=5,\n"
        "    )\n"
        "    match = re.search(r\"\\d+\", resp.choices[0].message.content)\n"
        "    return int(match.group()) if match else None\n"
        "\n"
        "scored_items = []\n"
        "for qs in QUESTION_SETS:\n"
        "    for level, text in qs[\"responses\"].items():\n"
        "        score = judge_score(qs[\"question\"], text)\n"
        "        scored_items.append({\"question\": qs[\"question\"], \"response\": text,\n"
        "                              \"objective_quality\": level, \"judge_score\": score})\n"
        "        print(f\"[{level}/3 real facts] judge_score={score} -- {text!r}\")\n"
        "\n"
        "print(f\"\\nTotal real judge-scored items: {len(scored_items)}\")\n"
        "print(\"\\n(pending real correlation computation)\")"
    ))

    correlation_cell_index = len(cells)
    cells.append(code(
        "def spearman_correlation(x, y):\n"
        "    def rank(values):\n"
        "        sorted_vals = sorted(values, reverse=True)\n"
        "        return [sorted_vals.index(v) + 1 for v in values]\n"
        "\n"
        "    x_ranks = rank(x)\n"
        "    y_ranks = rank(y)\n"
        "    n = len(x)\n"
        "    d_sq_sum = sum((xr - yr) ** 2 for xr, yr in zip(x_ranks, y_ranks))\n"
        "    return 1 - (6 * d_sq_sum) / (n * (n**2 - 1))\n"
        "\n"
        "objective_scores = [i[\"objective_quality\"] for i in scored_items]\n"
        "judge_scores = [i[\"judge_score\"] for i in scored_items]\n"
        "\n"
        "rho = spearman_correlation(judge_scores, objective_scores)\n"
        "print(f\"Real objective quality (fact count): {objective_scores}\")\n"
        "print(f\"Real judge scores (1-10):            {judge_scores}\")\n"
        "print(f\"\\nReal Spearman correlation (judge vs. judge-independent objective ground truth): {rho:.4f}\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 3. Real Position-Bias Flip Rate\n"
        "\n"
        "`[REAL]` For each question, real pairwise judge comparisons between the (0/3, 3/3) and (1/3, 2/3) "
        "response pairs -- each judged once in original order and once with presentation order swapped -- "
        "a real, live measurement of Module 03's position-bias flip rate, not a constructed example."
    ))

    position_bias_cell_index = len(cells)
    cells.append(code(
        "def judge_pick(question, resp_a, resp_b):\n"
        "    \"\"\"Returns 'A' or 'B' -- which response the real judge picks as better.\"\"\"\n"
        "    prompt = (\n"
        "        f\"Question: {question}\\n\\nResponse A: {resp_a}\\n\\nResponse B: {resp_b}\\n\\n\"\n"
        "        \"Which response is better? Reply with ONLY the single letter A or B.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=3,\n"
        "    )\n"
        "    text = resp.choices[0].message.content.strip().upper()\n"
        "    return \"A\" if \"A\" in text else (\"B\" if \"B\" in text else None)\n"
        "\n"
        "position_bias_trials = []\n"
        "for qs in QUESTION_SETS:\n"
        "    for pair in [(0, 3), (1, 2)]:\n"
        "        low_text = qs[\"responses\"][pair[0]]\n"
        "        high_text = qs[\"responses\"][pair[1]]\n"
        "\n"
        "        # Original order: low quality = A, high quality = B\n"
        "        pick_original = judge_pick(qs[\"question\"], low_text, high_text)\n"
        "        original_identity = \"low\" if pick_original == \"A\" else (\"high\" if pick_original == \"B\" else None)\n"
        "\n"
        "        # Swapped order: high quality = A, low quality = B\n"
        "        pick_swapped = judge_pick(qs[\"question\"], high_text, low_text)\n"
        "        swapped_identity = \"high\" if pick_swapped == \"A\" else (\"low\" if pick_swapped == \"B\" else None)\n"
        "\n"
        "        flipped = original_identity != swapped_identity\n"
        "        position_bias_trials.append({\n"
        "            \"question\": qs[\"question\"], \"pair\": pair,\n"
        "            \"original_identity\": original_identity, \"swapped_identity\": swapped_identity,\n"
        "            \"flipped\": flipped,\n"
        "        })\n"
        "        print(f\"{qs['question']!r} pair{pair}: original_pick={original_identity}, \"\n"
        "              f\"swapped_pick={swapped_identity}, flipped={flipped}\")\n"
        "\n"
        "n_flips = sum(1 for t in position_bias_trials if t[\"flipped\"])\n"
        "print(f\"\\nReal position-bias flip rate: {n_flips}/{len(position_bias_trials)} = \"\n"
        "      f\"{n_flips/len(position_bias_trials)*100:.1f}%\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "02_llm_as_judge_bias_and_calibration.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "responses_cell_index": responses_cell_index,
        "calibration_cell_index": calibration_cell_index,
        "correlation_cell_index": correlation_cell_index,
        "position_bias_cell_index": position_bias_cell_index,
    }


# ============================================================================
# Notebook 03: Real LLM-Rater Agreement Across Two Independent LLM Raters
# ============================================================================

def build_03_llm_rater_agreement():
    cells = []

    cells.append(md(
        "# Notebook 03: Real LLM-Rater Agreement Across Two Independent LLM Raters\n"
        "\n"
        "`[REAL]` Companion to Module 04. Two real, independently-prompted `gpt-4o-mini` \"rater\" calls "
        "labeling the same real 20 responses pass/fail against a fixed rubric.\n"
        "\n"
        "**Naming discipline, per the signed-off plan:** this notebook measures real **LLM-rater agreement** "
        "-- never called \"inter-annotator agreement\" anywhere in this notebook. Two independent LLM raters "
        "measure a genuinely different real thing than human inter-annotator agreement (Module 04's actual "
        "real scope: real human annotators). This is an honest, real, adjacent exercise in agreement "
        "measurement using the same real Cohen's kappa formula, not a claimed proxy or substitute for human "
        "annotation -- no human-annotation pipeline is available in this environment, and this notebook does "
        "not pretend otherwise."
    ))

    cells.append(code(
        "import os\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "MODEL = \"gpt-4o-mini\"\n"
        "print(f\"OpenAI client ready. Model: {MODEL}\")"
    ))

    cells.append(md(
        "## 1. Real, Fixed 20-Item Rating Set\n"
        "\n"
        "`[REAL]` 20 real question/answer pairs -- a deliberate real mix of clearly correct, clearly "
        "incorrect, and genuinely ambiguous/partial answers, so real rater disagreement has a real chance "
        "to occur (unlike Notebook 01's all-correct result)."
    ))

    items_cell_index = len(cells)
    cells.append(code(
        "RATING_ITEMS = [\n"
        "    (\"What is 7*8?\", \"56\"),\n"
        "    (\"What is 7*8?\", \"54\"),\n"
        "    (\"What is the capital of Italy?\", \"Rome\"),\n"
        "    (\"What is the capital of Italy?\", \"Milan\"),\n"
        "    (\"Name two prime numbers less than 10.\", \"2 and 3\"),\n"
        "    (\"Name two prime numbers less than 10.\", \"4 and 6\"),\n"
        "    (\"What is the freezing point of water in Celsius?\", \"0 degrees Celsius\"),\n"
        "    (\"What is the freezing point of water in Celsius?\", \"32 degrees Celsius\"),\n"
        "    (\"Who was the first president of the United States?\", \"George Washington\"),\n"
        "    (\"Who was the first president of the United States?\", \"Thomas Jefferson\"),\n"
        "    (\"What is the powerhouse of the cell?\", \"The mitochondria\"),\n"
        "    (\"What is the powerhouse of the cell?\", \"The nucleus\"),\n"
        "    (\"List the three primary colors.\", \"Red, blue, and yellow\"),\n"
        "    (\"List the three primary colors.\", \"Red and blue\"),\n"
        "    (\"What year did the Titanic sink?\", \"1912\"),\n"
        "    (\"What year did the Titanic sink?\", \"Sometime in the early 1900s\"),\n"
        "    (\"What is the chemical formula for water?\", \"H2O\"),\n"
        "    (\"What is the chemical formula for water?\", \"Water is made of hydrogen and oxygen atoms\"),\n"
        "    (\"Name the largest ocean on Earth.\", \"The Pacific Ocean\"),\n"
        "    (\"Name the largest ocean on Earth.\", \"The Atlantic Ocean, which is the biggest\"),\n"
        "]\n"
        "RUBRIC = \"PASS if the response fully and correctly answers the question. FAIL if it is incorrect, incomplete, or does not address the question.\"\n"
        "print(f\"Real rating set fixed: {len(RATING_ITEMS)} items.\")"
    ))

    cells.append(md(
        "## 2. Two Real, Independently-Prompted LLM Raters\n"
        "\n"
        "`[REAL]` Rater 1 and Rater 2 apply the identical real rubric, but through differently-worded, "
        "independently-written prompts -- two real, separate live API calls per item, not one call reused."
    ))

    rating_cell_index = len(cells)
    cells.append(code(
        "def rater_1(question, answer):\n"
        "    prompt = (\n"
        "        f\"Rubric: {RUBRIC}\\n\\nQuestion: {question}\\nAnswer: {answer}\\n\\n\"\n"
        "        \"Apply the rubric. Reply with ONLY the single word PASS or FAIL.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=3,\n"
        "    )\n"
        "    text = resp.choices[0].message.content.strip().upper()\n"
        "    return \"PASS\" if \"PASS\" in text else \"FAIL\"\n"
        "\n"
        "def rater_2(question, answer):\n"
        "    prompt = (\n"
        "        f\"You are grading a student's answer against this real standard: {RUBRIC}\\n\\n\"\n"
        "        f\"Student was asked: {question}\\nStudent answered: {answer}\\n\\n\"\n"
        "        \"Grade the answer against the standard above. Respond with exactly one word: PASS or FAIL.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=3,\n"
        "    )\n"
        "    text = resp.choices[0].message.content.strip().upper()\n"
        "    return \"PASS\" if \"PASS\" in text else \"FAIL\"\n"
        "\n"
        "ratings = []\n"
        "for question, answer in RATING_ITEMS:\n"
        "    r1 = rater_1(question, answer)\n"
        "    r2 = rater_2(question, answer)\n"
        "    ratings.append({\"question\": question, \"answer\": answer, \"rater_1\": r1, \"rater_2\": r2, \"agree\": r1 == r2})\n"
        "    print(f\"Q: {question!r} A: {answer!r} -> rater_1={r1}, rater_2={r2}, agree={r1==r2}\")\n"
        "\n"
        "n_agree = sum(1 for r in ratings if r[\"agree\"])\n"
        "print(f\"\\nReal raw agreement: {n_agree}/{len(ratings)} = {n_agree/len(ratings)*100:.1f}%\")\n"
        "print(\"\\n(pending real kappa computation)\")"
    ))

    cells.append(md(
        "## 3. Real Cohen's Kappa on the Real 2-Rater Confusion Table\n"
        "\n"
        "`[COMPUTED FROM REAL DATA]` Building the real confusion table from Section 2's real ratings and "
        "computing Cohen's kappa directly, per Module 04's own formula."
    ))

    kappa_cell_index = len(cells)
    cells.append(code(
        "pass_pass = sum(1 for r in ratings if r[\"rater_1\"] == \"PASS\" and r[\"rater_2\"] == \"PASS\")\n"
        "pass_fail = sum(1 for r in ratings if r[\"rater_1\"] == \"PASS\" and r[\"rater_2\"] == \"FAIL\")\n"
        "fail_pass = sum(1 for r in ratings if r[\"rater_1\"] == \"FAIL\" and r[\"rater_2\"] == \"PASS\")\n"
        "fail_fail = sum(1 for r in ratings if r[\"rater_1\"] == \"FAIL\" and r[\"rater_2\"] == \"FAIL\")\n"
        "total = pass_pass + pass_fail + fail_pass + fail_fail\n"
        "print(f\"Real confusion table: PASS-PASS={pass_pass}, PASS-FAIL={pass_fail}, FAIL-PASS={fail_pass}, FAIL-FAIL={fail_fail}, total={total}\")\n"
        "\n"
        "p_o = (pass_pass + fail_fail) / total\n"
        "r1_pass = (pass_pass + pass_fail) / total\n"
        "r1_fail = (fail_pass + fail_fail) / total\n"
        "r2_pass = (pass_pass + fail_pass) / total\n"
        "r2_fail = (pass_fail + fail_fail) / total\n"
        "p_e = r1_pass * r2_pass + r1_fail * r2_fail\n"
        "kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else float(\"nan\")\n"
        "\n"
        "print(f\"Real p_o (observed agreement): {p_o:.4f}\")\n"
        "print(f\"Real p_e (expected chance agreement): {p_e:.4f}\")\n"
        "print(f\"Real Cohen's kappa: {kappa:.4f}\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "03_llm_rater_agreement.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "items_cell_index": items_cell_index,
        "rating_cell_index": rating_cell_index,
        "kappa_cell_index": kappa_cell_index,
    }


# ============================================================================
# Notebook 04: Real RAG Faithfulness/Context Precision-Recall + Agent Efficiency
# ============================================================================

def build_04_rag_faithfulness_and_agent_efficiency():
    cells = []

    cells.append(md(
        "# Notebook 04: Real RAG Faithfulness/Context Precision-Recall + Agent Efficiency\n"
        "\n"
        "`[REAL]` Companion to Module 05. A real, small embedding-based RAG pipeline (`text-embedding-3-small` "
        "+ `gpt-4o-mini`) and a real, minimal tool-using agent loop.\n"
        "\n"
        "**Ground-truth relevance, defined before retrieval, per the signed-off plan:** every chunk in the "
        "real fixed corpus below is pre-labeled relevant/non-relevant against each real query *before* any "
        "retrieval call runs -- context precision/recall are computed against this real, explicit set, not "
        "inferred from whatever gets retrieved."
    ))

    cells.append(code(
        "import os\n"
        "import time\n"
        "import math\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "CHAT_MODEL = \"gpt-4o-mini\"\n"
        "EMBED_MODEL = \"text-embedding-3-small\"\n"
        "print(f\"OpenAI client ready. Chat: {CHAT_MODEL}, Embeddings: {EMBED_MODEL}\")"
    ))

    cells.append(md(
        "## 1. Real Fixed Corpus with Pre-Defined Ground-Truth Relevance\n"
        "\n"
        "`[REAL]` 6 real document chunks; 2 real queries, each with a real, pre-defined relevant-chunk set "
        "fixed before any retrieval runs."
    ))

    corpus_cell_index = len(cells)
    cells.append(code(
        "CORPUS = {\n"
        "    \"C1\": \"Mars is the fourth planet from the Sun and is known as the Red Planet due to iron oxide on its surface.\",\n"
        "    \"C2\": \"Jupiter is the largest planet in the solar system, a gas giant primarily composed of hydrogen and helium.\",\n"
        "    \"C3\": \"Mars has two small moons, Phobos and Deimos, which are thought to be captured asteroids.\",\n"
        "    \"C4\": \"The Great Red Spot on Jupiter is a giant storm that has been raging for centuries.\",\n"
        "    \"C5\": \"Earth's moon is the fifth largest moon in the solar system and stabilizes Earth's tilt.\",\n"
        "    \"C6\": \"Saturn is known for its extensive ring system made mostly of ice particles and rocky debris.\",\n"
        "}\n"
        "\n"
        "QUERIES = [\n"
        "    {\"query\": \"What do we know about Mars?\", \"ground_truth_relevant\": {\"C1\", \"C3\"}},\n"
        "    {\"query\": \"Tell me about Jupiter's storms and composition.\", \"ground_truth_relevant\": {\"C2\", \"C4\"}},\n"
        "]\n"
        "\n"
        "print(f\"Real corpus fixed: {len(CORPUS)} chunks.\")\n"
        "for q in QUERIES:\n"
        "    print(f\"  Query: {q['query']!r} -> pre-defined ground-truth relevant: {q['ground_truth_relevant']}\")"
    ))

    cells.append(md(
        "## 2. Real Embedding-Based Retrieval + Real Context Precision/Recall\n"
        "\n"
        "`[REAL]` Real cosine-similarity retrieval using real `text-embedding-3-small` embeddings, top-k=3 "
        "chunks per query. Precision/recall computed against Section 1's real, pre-defined ground truth -- "
        "denominators as Module 05 explicitly defines them (precision: retrieved count; recall: total real "
        "relevant count)."
    ))

    retrieval_cell_index = len(cells)
    cells.append(code(
        "def embed(text):\n"
        "    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])\n"
        "    return resp.data[0].embedding\n"
        "\n"
        "def cosine_sim(a, b):\n"
        "    dot = sum(x * y for x, y in zip(a, b))\n"
        "    norm_a = math.sqrt(sum(x * x for x in a))\n"
        "    norm_b = math.sqrt(sum(y * y for y in b))\n"
        "    return dot / (norm_a * norm_b)\n"
        "\n"
        "corpus_embeddings = {cid: embed(text) for cid, text in CORPUS.items()}\n"
        "print(\"Real corpus embeddings computed.\")\n"
        "\n"
        "def retrieve(query, top_k=3):\n"
        "    q_emb = embed(query)\n"
        "    sims = [(cid, cosine_sim(q_emb, c_emb)) for cid, c_emb in corpus_embeddings.items()]\n"
        "    sims.sort(key=lambda x: x[1], reverse=True)\n"
        "    return [cid for cid, _ in sims[:top_k]]\n"
        "\n"
        "for q in QUERIES:\n"
        "    retrieved = retrieve(q[\"query\"])\n"
        "    q[\"retrieved\"] = retrieved\n"
        "    retrieved_relevant = set(retrieved) & q[\"ground_truth_relevant\"]\n"
        "    precision = len(retrieved_relevant) / len(retrieved)\n"
        "    recall = len(retrieved_relevant) / len(q[\"ground_truth_relevant\"])\n"
        "    q[\"precision\"] = precision\n"
        "    q[\"recall\"] = recall\n"
        "    print(f\"Query: {q['query']!r}\")\n"
        "    print(f\"  Retrieved: {retrieved}\")\n"
        "    print(f\"  Ground truth relevant: {q['ground_truth_relevant']}\")\n"
        "    print(f\"  Real precision: {precision:.3f}, Real recall: {recall:.3f}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 3. Real RAG Generation + Real Claim-by-Claim Faithfulness Check\n"
        "\n"
        "`[REAL]` A real `gpt-4o-mini` generation call answering each query using only its real retrieved "
        "chunks as context, followed by a real, separate claim-extraction-and-verification call checking each "
        "claim in the generated answer against that same real retrieved context."
    ))

    faithfulness_cell_index = len(cells)
    cells.append(code(
        "def generate_rag_answer(query, retrieved_chunk_ids):\n"
        "    context = \"\\n\".join(f\"- {CORPUS[cid]}\" for cid in retrieved_chunk_ids)\n"
        "    prompt = f\"Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer using only the context above, in 2-3 sentences.\"\n"
        "    resp = client.chat.completions.create(\n"
        "        model=CHAT_MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=150,\n"
        "    )\n"
        "    return resp.choices[0].message.content.strip()\n"
        "\n"
        "def check_faithfulness(answer, retrieved_chunk_ids):\n"
        "    context = \"\\n\".join(f\"- {CORPUS[cid]}\" for cid in retrieved_chunk_ids)\n"
        "    prompt = (\n"
        "        f\"Context:\\n{context}\\n\\nAnswer to check: {answer}\\n\\n\"\n"
        "        \"List each distinct factual claim in the answer on its own line, formatted exactly as: \"\n"
        "        \"'CLAIM: <claim text> | SUPPORTED: yes' or 'CLAIM: <claim text> | SUPPORTED: no' -- \"\n"
        "        \"where SUPPORTED is yes only if the claim is directly stated in the context above.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=CHAT_MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=300,\n"
        "    )\n"
        "    lines = [l for l in resp.choices[0].message.content.strip().split(\"\\n\") if l.startswith(\"CLAIM:\")]\n"
        "    supported = sum(1 for l in lines if l.strip().lower().endswith(\"yes\"))\n"
        "    return {\"claims\": lines, \"total_claims\": len(lines), \"supported_claims\": supported}\n"
        "\n"
        "for q in QUERIES:\n"
        "    answer = generate_rag_answer(q[\"query\"], q[\"retrieved\"])\n"
        "    faithfulness = check_faithfulness(answer, q[\"retrieved\"])\n"
        "    q[\"answer\"] = answer\n"
        "    q[\"faithfulness\"] = faithfulness\n"
        "    score = faithfulness[\"supported_claims\"] / faithfulness[\"total_claims\"] if faithfulness[\"total_claims\"] else None\n"
        "    q[\"faithfulness_score\"] = score\n"
        "    print(f\"Query: {q['query']!r}\")\n"
        "    print(f\"  Real answer: {answer!r}\")\n"
        "    for line in faithfulness[\"claims\"]:\n"
        "        print(f\"    {line}\")\n"
        "    print(f\"  Real faithfulness score: {score}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    cells.append(md(
        "## 4. Real Minimal Tool-Using Agent: Efficiency Logging Across Two Real Task Runs\n"
        "\n"
        "`[REAL]` A minimal real agent loop with a `search_docs` tool (reusing Section 2's real retrieval) "
        "completing two real tasks, logging real tool-call count, real token usage, real wall-clock latency, "
        "and real cost (at OpenAI's stated real `gpt-4o-mini` rate: $0.150/1M input, $0.600/1M output tokens)."
    ))

    agent_cell_index = len(cells)
    cells.append(code(
        "INPUT_RATE_PER_TOKEN = 0.150 / 1_000_000\n"
        "OUTPUT_RATE_PER_TOKEN = 0.600 / 1_000_000\n"
        "\n"
        "def run_agent_task(task_query, max_tool_calls=2):\n"
        "    start = time.perf_counter()\n"
        "    tool_calls = 0\n"
        "    total_input_tokens = 0\n"
        "    total_output_tokens = 0\n"
        "\n"
        "    retrieved = retrieve(task_query, top_k=2)\n"
        "    tool_calls += 1\n"
        "    context = \"\\n\".join(f\"- {CORPUS[cid]}\" for cid in retrieved)\n"
        "\n"
        "    prompt = f\"Context:\\n{context}\\n\\nTask: {task_query}\\n\\nAnswer in 1-2 sentences using only the context.\"\n"
        "    resp = client.chat.completions.create(\n"
        "        model=CHAT_MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=100,\n"
        "    )\n"
        "    total_input_tokens += resp.usage.prompt_tokens\n"
        "    total_output_tokens += resp.usage.completion_tokens\n"
        "\n"
        "    elapsed = time.perf_counter() - start\n"
        "    cost = total_input_tokens * INPUT_RATE_PER_TOKEN + total_output_tokens * OUTPUT_RATE_PER_TOKEN\n"
        "    return {\n"
        "        \"task\": task_query, \"answer\": resp.choices[0].message.content.strip(),\n"
        "        \"tool_calls\": tool_calls, \"input_tokens\": total_input_tokens, \"output_tokens\": total_output_tokens,\n"
        "        \"latency_s\": elapsed, \"cost_usd\": cost,\n"
        "    }\n"
        "\n"
        "AGENT_TASKS = [\"What do we know about Mars?\", \"Tell me about Jupiter's storms and composition.\"]\n"
        "agent_runs = [run_agent_task(t) for t in AGENT_TASKS]\n"
        "for run in agent_runs:\n"
        "    print(f\"Task: {run['task']!r}\")\n"
        "    print(f\"  Real answer: {run['answer']!r}\")\n"
        "    print(f\"  tool_calls={run['tool_calls']}, input_tok={run['input_tokens']}, output_tok={run['output_tokens']}, \"\n"
        "          f\"latency={run['latency_s']:.3f}s, cost=${run['cost_usd']:.8f}\")\n"
        "\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "04_rag_faithfulness_and_agent_efficiency.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "corpus_cell_index": corpus_cell_index,
        "retrieval_cell_index": retrieval_cell_index,
        "faithfulness_cell_index": faithfulness_cell_index,
        "agent_cell_index": agent_cell_index,
    }


# ============================================================================
# Notebook 05: Real Hallucination Detection: Self-Consistency vs. Grounded Verification
# ============================================================================

def build_05_hallucination_self_consistency_vs_grounded():
    cells = []

    cells.append(md(
        "# Notebook 05: Real Hallucination Detection -- Self-Consistency vs. Grounded Verification\n"
        "\n"
        "`[REAL]` Companion to Module 06. Real $k$-sample self-consistency checks via live `gpt-4o-mini` "
        "calls, followed by real grounded verification against the live Wikipedia REST API -- a real, "
        "independent source, not the same generation model.\n"
        "\n"
        "**Pre-stated \"wrong but self-consistent\" criterion (fixed before any run, per the signed-off "
        "plan):** a real trial counts as this failure pattern only if (a) self-consistency agreement across "
        "$k=5$ samples is $\\geq 0.7$, AND (b) real grounded verification against Wikipedia returns a "
        "contradiction against the majority-agreed answer. Both criteria are checked mechanically after all "
        "real data collection completes -- no example is selected or excluded after seeing results."
    ))

    cells.append(code(
        "import os\n"
        "import re\n"
        "from collections import Counter\n"
        "import requests\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "MODEL = \"gpt-4o-mini\"\n"
        "WIKI_HEADERS = {\"User-Agent\": \"StudyPrepResearchBot/1.0 (aryan.chandra.compcoding@gmail.com)\"}\n"
        "AGREEMENT_THRESHOLD = 0.7  # pre-stated, per the signed-off plan -- not tuned after seeing results\n"
        "print(f\"OpenAI client ready. Model: {MODEL}. Pre-stated agreement threshold: {AGREEMENT_THRESHOLD}\")"
    ))

    cells.append(md(
        "## 1. Real Question Set, Including One Genuine Real \"Trick Question\"\n"
        "\n"
        "`[REAL]` 5 real factual questions, each with a real, independently-verifiable authoritative fact. "
        "Question 1 is a real, well-documented case where LLMs are known to sometimes answer incorrectly "
        "(defaulting to Mount Everest instead of the real correct answer, Mauna Kea) -- included specifically "
        "to give this notebook's central hypothesis a genuine, real chance to manifest, not to bias the result."
    ))

    questions_cell_index = len(cells)
    cells.append(code(
        "QUESTIONS = [\n"
        "    {\"question\": \"What is the tallest mountain in the world measured from base to peak, not sea level?\",\n"
        "     \"wiki_title\": \"Mauna Kea\", \"real_fact\": \"Mauna Kea\"},\n"
        "    {\"question\": \"In what year did the Eiffel Tower open to the public?\",\n"
        "     \"wiki_title\": \"Eiffel Tower\", \"real_fact\": \"1889\"},\n"
        "    {\"question\": \"How many moons does Mars have?\",\n"
        "     \"wiki_title\": \"Moons of Mars\", \"real_fact\": \"2\"},\n"
        "    {\"question\": \"In what year was the Great Fire of London?\",\n"
        "     \"wiki_title\": \"Great Fire of London\", \"real_fact\": \"1666\"},\n"
        "    {\"question\": \"What is generally considered the driest place on Earth, excluding polar regions?\",\n"
        "     \"wiki_title\": \"Atacama Desert\", \"real_fact\": \"Atacama\"},\n"
        "]\n"
        "print(f\"Real question set fixed: {len(QUESTIONS)} questions.\")"
    ))

    cells.append(md(
        "## 2. Real $k$-Sample Self-Consistency\n"
        "\n"
        "`[REAL]` 5 real live `gpt-4o-mini` samples per question at temperature 0.7 -- computing the real "
        "self-consistency agreement rate on the majority answer."
    ))

    self_consistency_cell_index = len(cells)
    cells.append(code(
        "def sample_answer(question):\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": f\"{question} Answer in a few words only.\"}],\n"
        "        temperature=0.7, max_tokens=20,\n"
        "    )\n"
        "    return resp.choices[0].message.content.strip()\n"
        "\n"
        "K = 5\n"
        "for item in QUESTIONS:\n"
        "    samples = [sample_answer(item[\"question\"]) for _ in range(K)]\n"
        "    counts = Counter(s.lower() for s in samples)\n"
        "    majority_answer, majority_count = counts.most_common(1)[0]\n"
        "    agreement = majority_count / K\n"
        "    item[\"samples\"] = samples\n"
        "    item[\"majority_answer\"] = majority_answer\n"
        "    item[\"agreement\"] = agreement\n"
        "    print(f\"Q: {item['question']!r}\")\n"
        "    print(f\"  Real samples: {samples}\")\n"
        "    print(f\"  Real majority answer: {majority_answer!r}, real agreement: {agreement:.2f}\")\n"
        "\n"
        "print(\"\\n(pending real grounded verification)\")"
    ))

    cells.append(md(
        "## 3. Real Grounded Verification Against an Independent Source (Wikipedia)\n"
        "\n"
        "`[REAL]` For every question, a real, live Wikipedia REST API fetch retrieves independent real "
        "reference content -- genuinely separate from the generation model's own parametric memory. The real "
        "entailment/contradiction judgment then compares the majority answer against this real, independently-"
        "sourced text (a real NLI-style step over real external evidence, not the generation model "
        "introspecting on its own prior answer from memory)."
    ))

    grounded_cell_index = len(cells)
    cells.append(code(
        "def fetch_wikipedia_extract(title):\n"
        "    resp = requests.get(\"https://en.wikipedia.org/w/api.php\", params={\n"
        "        \"action\": \"query\", \"prop\": \"extracts\", \"exintro\": True, \"explaintext\": True,\n"
        "        \"titles\": title, \"format\": \"json\",\n"
        "    }, headers=WIKI_HEADERS, timeout=15)\n"
        "    pages = resp.json()[\"query\"][\"pages\"]\n"
        "    for _, page in pages.items():\n"
        "        return page.get(\"extract\", \"\")\n"
        "    return \"\"\n"
        "\n"
        "def grounded_verify(question, majority_answer, wiki_extract):\n"
        "    prompt = (\n"
        "        f\"Independent reference text:\\n{wiki_extract[:1500]}\\n\\n\"\n"
        "        f\"Question: {question}\\nProposed answer: {majority_answer}\\n\\n\"\n"
        "        \"Does the independent reference text support the proposed answer, contradict it, or say \"\n"
        "        \"nothing relevant (neutral)? Reply with ONLY one word: entailed, contradicted, or neutral.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=5,\n"
        "    )\n"
        "    text = resp.choices[0].message.content.strip().lower()\n"
        "    for verdict in [\"entailed\", \"contradicted\", \"neutral\"]:\n"
        "        if verdict in text:\n"
        "            return verdict\n"
        "    return \"neutral\"\n"
        "\n"
        "for item in QUESTIONS:\n"
        "    extract = fetch_wikipedia_extract(item[\"wiki_title\"])\n"
        "    verdict = grounded_verify(item[\"question\"], item[\"majority_answer\"], extract)\n"
        "    item[\"wiki_extract_len\"] = len(extract)\n"
        "    item[\"grounded_verdict\"] = verdict\n"
        "    print(f\"Q: {item['question']!r}\")\n"
        "    print(f\"  Real majority answer: {item['majority_answer']!r}, real agreement: {item['agreement']:.2f}\")\n"
        "    print(f\"  Real Wikipedia extract length: {len(extract)} chars\")\n"
        "    print(f\"  Real grounded verdict: {verdict}\")\n"
        "\n"
        "print(\"\\n(pending real criterion check)\")"
    ))

    cells.append(md(
        "## 4. Real, Mechanical Check of the Pre-Stated \"Wrong but Self-Consistent\" Criterion\n"
        "\n"
        "`[REAL]` Applying Section intro's pre-stated criterion mechanically to all 5 real trials -- reported "
        "honestly whichever way it comes out."
    ))

    criterion_cell_index = len(cells)
    cells.append(code(
        "wrong_but_consistent_trials = [\n"
        "    item for item in QUESTIONS\n"
        "    if item[\"agreement\"] >= AGREEMENT_THRESHOLD and item[\"grounded_verdict\"] == \"contradicted\"\n"
        "]\n"
        "\n"
        "print(f\"Pre-stated criterion: agreement >= {AGREEMENT_THRESHOLD} AND grounded_verdict == 'contradicted'\\n\")\n"
        "for item in QUESTIONS:\n"
        "    meets = item[\"agreement\"] >= AGREEMENT_THRESHOLD and item[\"grounded_verdict\"] == \"contradicted\"\n"
        "    print(f\"Q: {item['question']!r}: agreement={item['agreement']:.2f}, verdict={item['grounded_verdict']}, \"\n"
        "          f\"meets_criterion={meets}\")\n"
        "\n"
        "print(f\"\\nReal count of trials meeting the pre-stated 'wrong but self-consistent' criterion: \"\n"
        "      f\"{len(wrong_but_consistent_trials)}/{len(QUESTIONS)}\")\n"
        "print(\"\\n(pending real interpretation)\")"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "05_hallucination_self_consistency_vs_grounded.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "questions_cell_index": questions_cell_index,
        "self_consistency_cell_index": self_consistency_cell_index,
        "grounded_cell_index": grounded_cell_index,
        "criterion_cell_index": criterion_cell_index,
    }


# ============================================================================
# Notebook 06: Real Guardrail Classifier + Production Capstone
# ============================================================================

def build_06_guardrail_classifier_and_production_capstone():
    cells = []

    cells.append(md(
        "# Notebook 06: Real Guardrail Classifier + Production Capstone\n"
        "\n"
        "`[REAL]` Companion to Modules 07-09. A real, local Hugging Face toxicity classifier run on the RTX "
        "4060, evaluated for real precision/recall/F1 across a real swept threshold range with a principled, "
        "stated threshold-selection method. Real measured sequential-vs-parallel guardrail latency using two "
        "genuinely independent checks. A two-part capstone: (A) a real per-request pipeline trace with root-"
        "cause localization (Module 07's own function, reused verbatim), and (B) a separate, explicitly "
        "`[SIMULATION]`-labeled aggregate evaluation-set-versioning comparison (Module 09's own functions, "
        "reused verbatim) -- kept as two distinct experiments answering two different real questions, per the "
        "signed-off plan."
    ))

    cells.append(code(
        "import os\n"
        "import re\n"
        "import time\n"
        "from dataclasses import dataclass\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "\n"
        "import torch\n"
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
        "from dotenv import load_dotenv, find_dotenv\n"
        "from openai import OpenAI\n"
        "\n"
        "load_dotenv(find_dotenv())\n"
        "client = OpenAI(api_key=os.environ[\"OPENAI_API_KEY\"])\n"
        "GEN_MODEL = \"gpt-4o-mini\"\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "print(f\"OpenAI client ready. Generation model: {GEN_MODEL}. Local classifier device: {DEVICE}\")"
    ))

    cells.append(md(
        "## 1. Real Local Toxicity Classifier + Real Labeled Test Set\n"
        "\n"
        "`[REAL]` A real, small pretrained Hugging Face toxicity classifier (`unitary/toxic-bert`), loaded and "
        "run locally on the RTX 4060 -- more representative of real production guardrail deployment than a "
        "remote API call.\n"
        "\n"
        "**Labeling protocol (stated before scoring, mirroring Notebook 01's own correctness protocol):** an "
        "item is labeled toxic only if it contains a direct hostile insult or threat aimed at a person; it is "
        "labeled non-toxic if it is neutral, informational, or positive in content. This is a real, explicit, "
        "reproducible rule, not a subjective per-item judgment."
    ))

    classifier_setup_cell_index = len(cells)
    cells.append(code(
        "TOXICITY_MODEL_NAME = \"unitary/toxic-bert\"\n"
        "tox_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)\n"
        "tox_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_NAME).to(DEVICE)\n"
        "tox_model.eval()\n"
        "TOXIC_LABEL_INDEX = [i for i, lbl in tox_model.config.id2label.items() if lbl == \"toxic\"][0]\n"
        "print(f\"Real classifier loaded: {TOXICITY_MODEL_NAME} on {DEVICE}. \"\n"
        "      f\"Real label set: {list(tox_model.config.id2label.values())}\")\n"
        "\n"
        "def toxicity_score(text):\n"
        "    inputs = tox_tokenizer(text, return_tensors=\"pt\", truncation=True).to(DEVICE)\n"
        "    with torch.no_grad():\n"
        "        logits = tox_model(**inputs).logits\n"
        "    probs = torch.sigmoid(logits)[0]\n"
        "    return probs[TOXIC_LABEL_INDEX].item()\n"
        "\n"
        "TEST_SET = [\n"
        "    (\"You are a complete idiot and everyone hates you.\", 1),\n"
        "    (\"Get lost, nobody wants you here, you worthless piece of trash.\", 1),\n"
        "    (\"I will find you and make you regret this, you disgusting coward.\", 1),\n"
        "    (\"Shut up, you stupid moron, nobody cares what you think.\", 1),\n"
        "    (\"You're pathetic and everyone can see how useless you are.\", 1),\n"
        "    (\"Go away, you filthy idiot, and never come back.\", 1),\n"
        "    (\"You're an absolute waste of space and everyone is sick of you.\", 1),\n"
        "    (\"Nobody asked for your stupid opinion, keep it to yourself, loser.\", 1),\n"
        "    (\"The weather today is sunny with a light breeze.\", 0),\n"
        "    (\"Thank you so much for your help, I really appreciate it.\", 0),\n"
        "    (\"The quarterly report is due on Friday at 5pm.\", 0),\n"
        "    (\"Could you please pass the salt?\", 0),\n"
        "    (\"I think this restaurant has great pasta.\", 0),\n"
        "    (\"The train departs from platform 4 at 9:15am.\", 0),\n"
        "    (\"Congratulations on your promotion, well deserved!\", 0),\n"
        "    (\"I disagree with this approach, but I respect your reasoning.\", 0),\n"
        "]\n"
        "\n"
        "scored_test_set = [(text, label, toxicity_score(text)) for text, label in TEST_SET]\n"
        "for text, label, score in scored_test_set:\n"
        "    print(f\"label={label}  real_score={score:.4f}  {text!r}\")"
        ))

    cells.append(md(
        "## 2. Real Threshold Sweep + Principled Threshold Selection\n"
        "\n"
        "`[REAL]` Real precision/recall/F1 (Module 08's own `precision_recall_f1`, reused verbatim) computed "
        "across a real swept range of decision thresholds on Section 1's real scored test set. **Threshold "
        "selection method, stated explicitly:** the threshold used in later sections is the one that maximizes "
        "real F1 -- a principled, stated criterion, not an arbitrary choice."
    ))

    threshold_sweep_cell_index = len(cells)
    cells.append(code(
        "def precision_recall_f1(tp, fp, fn):\n"
        "    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n"
        "    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n"
        "    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0\n"
        "    return {\"precision\": precision, \"recall\": recall, \"f1\": f1}\n"
        "\n"
        "THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]\n"
        "sweep_results = []\n"
        "for t in THRESHOLDS:\n"
        "    tp = sum(1 for _, label, score in scored_test_set if label == 1 and score >= t)\n"
        "    fp = sum(1 for _, label, score in scored_test_set if label == 0 and score >= t)\n"
        "    fn = sum(1 for _, label, score in scored_test_set if label == 1 and score < t)\n"
        "    metrics = precision_recall_f1(tp, fp, fn)\n"
        "    sweep_results.append({\"threshold\": t, \"tp\": tp, \"fp\": fp, \"fn\": fn, **metrics})\n"
        "    print(f\"t={t:.1f}  tp={tp} fp={fp} fn={fn}  \"\n"
        "          f\"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}\")\n"
        "\n"
        "CHOSEN_THRESHOLD = max(sweep_results, key=lambda r: r[\"f1\"])[\"threshold\"]\n"
        "print(f\"\\nReal F1-maximizing threshold selected: {CHOSEN_THRESHOLD}\")"
    ))

    cells.append(md(
        "## 3. Real Sequential vs. Parallel Guardrail Latency (Genuinely Independent Checks)\n"
        "\n"
        "`[REAL]` Two genuinely independent checks with no data dependency between them: the real local "
        "toxicity classifier (GPU) and a real, deterministic regex-based PII detector (CPU, no model weights, "
        "no shared computation) -- satisfying the plan's independence requirement, unlike two calls to the "
        "same classifier. Real wall-clock latency measured for both a sequential loop and a "
        "`ThreadPoolExecutor`-parallel run across the same real inputs."
    ))

    latency_cell_index = len(cells)
    cells.append(code(
        "PII_PATTERNS = {\n"
        "    \"email\": re.compile(r\"[\\w.+-]+@[\\w-]+\\.[\\w.-]+\"),\n"
        "    \"phone\": re.compile(r\"\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b\"),\n"
        "    \"ssn\": re.compile(r\"\\b\\d{3}-\\d{2}-\\d{4}\\b\"),\n"
        "}\n"
        "\n"
        "def detect_pii(text):\n"
        "    for kind, pattern in PII_PATTERNS.items():\n"
        "        if pattern.search(text):\n"
        "            return True, kind\n"
        "    return False, None\n"
        "\n"
        "LATENCY_TEST_TEXTS = [text for text, _ in TEST_SET[:8]] + [\n"
        "    \"Please call me back at 555-234-7788 when you get a chance.\",\n"
        "    \"You can reach the sales desk at sales@example.com anytime.\",\n"
        "]\n"
        "\n"
        "def run_sequential(text):\n"
        "    t0 = time.perf_counter()\n"
        "    _ = toxicity_score(text)\n"
        "    _ = detect_pii(text)\n"
        "    return time.perf_counter() - t0\n"
        "\n"
        "def run_parallel(text):\n"
        "    t0 = time.perf_counter()\n"
        "    with ThreadPoolExecutor(max_workers=2) as pool:\n"
        "        fut_tox = pool.submit(toxicity_score, text)\n"
        "        fut_pii = pool.submit(detect_pii, text)\n"
        "        fut_tox.result()\n"
        "        fut_pii.result()\n"
        "    return time.perf_counter() - t0\n"
        "\n"
        "sequential_times = [run_sequential(t) for t in LATENCY_TEST_TEXTS]\n"
        "parallel_times = [run_parallel(t) for t in LATENCY_TEST_TEXTS]\n"
        "\n"
        "mean_sequential_ms = sum(sequential_times) / len(sequential_times) * 1000\n"
        "mean_parallel_ms = sum(parallel_times) / len(parallel_times) * 1000\n"
        "savings_pct = (mean_sequential_ms - mean_parallel_ms) / mean_sequential_ms * 100\n"
        "\n"
        "print(f\"Real mean sequential latency: {mean_sequential_ms:.2f}ms\")\n"
        "print(f\"Real mean parallel latency:   {mean_parallel_ms:.2f}ms\")\n"
        "print(f\"Real savings: {savings_pct:.1f}%\")"
    ))

    cells.append(md(
        "## 4. Capstone Part A: Real Per-Request Trace + Root-Cause Localization\n"
        "\n"
        "`[REAL]` `Span`, `total_latency_ms`, and `localize_root_cause` are reused verbatim from Module 07. A "
        "real, small 3-step pipeline (retrieve -> generate -> guardrail check, reusing this notebook's own "
        "real Section 1-3 classifier/detector functions) runs for 3 real requests against a real, tiny fixed "
        "document set, with real per-span timing and status logged."
    ))

    capstone_a_cell_index = len(cells)
    cells.append(code(
        "@dataclass\n"
        "class Span:\n"
        "    name: str\n"
        "    latency_ms: float\n"
        "    status: str\n"
        "    detail: str\n"
        "\n"
        "def total_latency_ms(spans):\n"
        "    return sum(s.latency_ms for s in spans)\n"
        "\n"
        "def localize_root_cause(spans):\n"
        "    for span in spans:\n"
        "        if span.status != \"ok\":\n"
        "            return span\n"
        "    return None\n"
        "\n"
        "DOCS = [\n"
        "    {\"id\": \"returns\", \"keywords\": [\"return\", \"refund\", \"electronics\"],\n"
        "     \"text\": \"Our return policy allows electronics to be returned within 30 days of purchase for a full refund, provided the original packaging is included.\"},\n"
        "    {\"id\": \"support\", \"keywords\": [\"contact\", \"support\", \"help\"],\n"
        "     \"text\": \"For further assistance, contact our support team at 555-201-4488 or email support@example.com.\"},\n"
        "    {\"id\": \"hours\", \"keywords\": [\"hours\", \"open\", \"store\"],\n"
        "     \"text\": \"Our stores are open Monday through Saturday, 9am to 8pm, and Sunday 10am to 6pm.\"},\n"
        "]\n"
        "\n"
        "def retrieve(query):\n"
        "    query_words = set(re.findall(r\"[a-z']+\", query.lower()))\n"
        "    best_doc, best_overlap = None, 0\n"
        "    for doc in DOCS:\n"
        "        overlap = len(query_words & set(doc[\"keywords\"]))\n"
        "        if overlap > best_overlap:\n"
        "            best_doc, best_overlap = doc, overlap\n"
        "    return best_doc\n"
        "\n"
        "def generate(query, context_text):\n"
        "    prompt = (\n"
        "        f\"Context: {context_text or '(no relevant context found)'}\\n\\n\"\n"
        "        f\"Question: {query}\\n\\n\"\n"
        "        \"Answer using ONLY the context above. Be specific and include any relevant contact details. \"\n"
        "        \"If the context does not contain the answer, say you don't have that information.\"\n"
        "    )\n"
        "    resp = client.chat.completions.create(\n"
        "        model=GEN_MODEL, messages=[{\"role\": \"user\", \"content\": prompt}],\n"
        "        temperature=0.0, max_tokens=80,\n"
        "    )\n"
        "    return resp.choices[0].message.content.strip()\n"
        "\n"
        "def guardrail_check(answer):\n"
        "    score = toxicity_score(answer)\n"
        "    has_pii, pii_kind = detect_pii(answer)\n"
        "    if score >= CHOSEN_THRESHOLD:\n"
        "        return \"flagged\", f\"toxicity score {score:.3f} >= threshold {CHOSEN_THRESHOLD}\"\n"
        "    if has_pii:\n"
        "        return \"flagged\", f\"PII detected: {pii_kind}\"\n"
        "    return \"ok\", f\"toxicity score {score:.3f}, no PII detected\"\n"
        "\n"
        "def run_traced_request(query):\n"
        "    spans = []\n"
        "\n"
        "    t0 = time.perf_counter()\n"
        "    doc = retrieve(query)\n"
        "    retrieve_ms = (time.perf_counter() - t0) * 1000\n"
        "    if doc is None:\n"
        "        spans.append(Span(\"retrieve\", retrieve_ms, \"error\", \"no relevant document found\"))\n"
        "        context_text = \"\"\n"
        "    else:\n"
        "        spans.append(Span(\"retrieve\", retrieve_ms, \"ok\", f\"matched doc: {doc['id']}\"))\n"
        "        context_text = doc[\"text\"]\n"
        "\n"
        "    t0 = time.perf_counter()\n"
        "    answer = generate(query, context_text)\n"
        "    generate_ms = (time.perf_counter() - t0) * 1000\n"
        "    spans.append(Span(\"generate\", generate_ms, \"ok\", answer[:80]))\n"
        "\n"
        "    t0 = time.perf_counter()\n"
        "    guard_status, guard_detail = guardrail_check(answer)\n"
        "    guard_ms = (time.perf_counter() - t0) * 1000\n"
        "    spans.append(Span(\"guardrail_check\", guard_ms, guard_status, guard_detail))\n"
        "\n"
        "    return spans, answer\n"
        "\n"
        "REQUESTS = [\n"
        "    \"What is your return policy for electronics?\",\n"
        "    \"What is the CEO's home address?\",\n"
        "    \"How do I contact your support team?\",\n"
        "]\n"
        "\n"
        "traces = []\n"
        "for query in REQUESTS:\n"
        "    spans, answer = run_traced_request(query)\n"
        "    root_cause = localize_root_cause(spans)\n"
        "    traces.append({\"query\": query, \"spans\": spans, \"answer\": answer, \"root_cause\": root_cause})\n"
        "    print(f\"Query: {query!r}\")\n"
        "    print(f\"  Real answer: {answer!r}\")\n"
        "    for s in spans:\n"
        "        print(f\"  Span: {s.name:<18} status={s.status:<7} latency={s.latency_ms:.1f}ms  detail={s.detail}\")\n"
        "    print(f\"  Real total latency: {total_latency_ms(spans):.1f}ms\")\n"
        "    print(f\"  Real localized root cause: {root_cause.name if root_cause else 'None (request succeeded)'}\")\n"
        "    print()"
    ))

    cells.append(md(
        "## 5. Capstone Part B: Real Aggregate Evaluation-Set-Versioning Comparison `[SIMULATION]`\n"
        "\n"
        "`[SIMULATION]` `accuracy` and `diagnose_versioning_failure` are reused verbatim from Module 09. This "
        "is a deliberately constructed scenario -- two different real reference-answer sets are authored on "
        "purpose to demonstrate the versioning-failure pattern -- kept explicitly separate from Part A above: "
        "Part A diagnoses one real request's failure; Part B evaluates aggregate real system behavior across a "
        "small evaluation run. Real model outputs as inputs do not make this a real observed production event."
    ))

    capstone_b_cell_index = len(cells)
    cells.append(code(
        "def accuracy(correct, total):\n"
        "    return correct / total\n"
        "\n"
        "def diagnose_versioning_failure(acc_v1, acc_v2, model_output_changed):\n"
        "    if not model_output_changed and acc_v1 != acc_v2:\n"
        "        return \"evaluation-pipeline versioning failure (NOT a real system-quality change)\"\n"
        "    return \"real system-quality change (or no change)\"\n"
        "\n"
        "EVAL_QUERIES = [\n"
        "    \"What is your return policy for electronics?\",\n"
        "    \"What are your store hours?\",\n"
        "    \"How do I contact your support team?\",\n"
        "]\n"
        "eval_answers = [generate(q, retrieve(q)[\"text\"] if retrieve(q) else \"\") for q in EVAL_QUERIES]\n"
        "for q, a in zip(EVAL_QUERIES, eval_answers):\n"
        "    print(f\"Q: {q!r}\\n  Real answer: {a!r}\")\n"
        "\n"
        "REFERENCE_V1 = {\n"
        "    \"What is your return policy for electronics?\": [\"30 days\", \"refund\"],\n"
        "    \"What are your store hours?\": [\"9am\", \"8pm\"],\n"
        "    \"How do I contact your support team?\": [\"555-201-4488\", \"support@example.com\"],\n"
        "}\n"
        "REFERENCE_V2 = {\n"
        "    \"What is your return policy for electronics?\": [\"30 days\", \"refund\", \"original packaging\"],\n"
        "    \"What are your store hours?\": [\"9am\", \"8pm\", \"Sunday\", \"10am\", \"6pm\"],\n"
        "    \"How do I contact your support team?\": [\"555-201-4488\", \"support@example.com\"],\n"
        "}\n"
        "\n"
        "def score_against_reference(answers, queries, reference):\n"
        "    correct = 0\n"
        "    for q, a in zip(queries, answers):\n"
        "        required = reference[q]\n"
        "        if all(term.lower() in a.lower() for term in required):\n"
        "            correct += 1\n"
        "    return correct, len(queries)\n"
        "\n"
        "correct_v1, total_v1 = score_against_reference(eval_answers, EVAL_QUERIES, REFERENCE_V1)\n"
        "correct_v2, total_v2 = score_against_reference(eval_answers, EVAL_QUERIES, REFERENCE_V2)\n"
        "acc_v1 = accuracy(correct_v1, total_v1)\n"
        "acc_v2 = accuracy(correct_v2, total_v2)\n"
        "\n"
        "print(f\"\\nReal SAME model outputs scored under Reference V1: {correct_v1}/{total_v1} = {acc_v1:.4f}\")\n"
        "print(f\"Real SAME model outputs scored under Reference V2: {correct_v2}/{total_v2} = {acc_v2:.4f}\")\n"
        "\n"
        "diagnosis = diagnose_versioning_failure(acc_v1, acc_v2, model_output_changed=False)\n"
        "print(f\"Real diagnosis: {diagnosis}\")"
    ))

    cells.append(md(
        "## 6. Real Interpretation\n"
        "\n"
        "_(pending real interpretation)_"
    ))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = os.path.join(NOTEBOOKS_DIR, "06_guardrail_classifier_and_production_capstone.ipynb")
    run_and_save(nb, out_path)
    return out_path, {
        "classifier_setup_cell_index": classifier_setup_cell_index,
        "threshold_sweep_cell_index": threshold_sweep_cell_index,
        "latency_cell_index": latency_cell_index,
        "capstone_a_cell_index": capstone_a_cell_index,
        "capstone_b_cell_index": capstone_b_cell_index,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "01"
    if target == "01":
        path, indices = build_01_automated_metrics_vs_correctness()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "02":
        path, indices = build_02_llm_as_judge_bias_and_calibration()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "03":
        path, indices = build_03_llm_rater_agreement()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "04":
        path, indices = build_04_rag_faithfulness_and_agent_efficiency()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "05":
        path, indices = build_05_hallucination_self_consistency_vs_grounded()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
    elif target == "06":
        path, indices = build_06_guardrail_classifier_and_production_capstone()
        print(f"\nBuilt {path}")
        print(f"Cell indices for Pass 2 explanation edits: {indices}")
