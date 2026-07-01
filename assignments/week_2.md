# Week 2 Engineering Assignment: Search Re-ranking & Conversion A/B Experimentation Engine

In this assignment, you will act as a Senior Machine Learning Engineer at an e-commerce platform. Your task is to build and evaluate the search optimization layer:
1. **Product Re-ranking Engine (XGBoost & ONNX):** Optimize product search results by training a Gradient Boosting model to predict purchase conversion, handling feature cardinality bias, and compiling the model to ONNX for sub-millisecond production latency.
2. **Conversion A/B Experimentation Engine (Hypothesis Testing):** Evaluate layout changes and prompt templates using statistical power analysis, conversion Z-tests, and family-wise error rate corrections.

You will implement these models, compile them for real-time serving, design statistical test parameters, and protect against false positive code rollouts.

---

## 1. Business Context and Dataset

### The Business Problem
Search conversion is the primary revenue driver.
- A poor ranking model displays irrelevant search results, leading to lost sales and session abandonment.
- In real-time search scoring, latency is critical: every $10\text{ms}$ of API delay drops conversion.
- Marketing wants to test $10$ different prompt templates and layout variations. Without proper statistical controls, running these tests in parallel yields false positive winners due to random chance, leading to expensive rollouts of useless layouts.

### The Dataset
You will generate a synthetic dataset of search sessions containing:
- **Features:** `session_duration` (continuous), `page_views` (continuous), `random_user_id` (high-cardinality noise column), `historical_click_rate` (continuous), and `discount_ratio` (continuous).
- **Targets:**
  - `purchased` (Binary: `1` for purchase conversion, `0` for none) for the XGBoost model.
  - Variant conversion rates for the A/B testing simulation.

---

## 2. Saturday Sprint: High-Performance Search Re-ranker (≤3 Hours)

**Objective:** Build an XGBoost ranking pipeline, diagnose feature importance bias, use early stopping, and compile the model to ONNX runtime for sub-millisecond serving.

### Task 1: Expose Gini MDI Bias & Calculate Permutation Importance (45 Mins)
- **Action:** Train a Random Forest model on the generated dataset which includes the high-cardinality `random_user_id` column. Calculate Gini Mean Decrease in Impurity (MDI) feature importances. Then, run Permutation Feature Importance on a validation split.
- **Deliverable:** Importance comparison table or bar chart code.
- **Success Criteria:** 
  - Proved that Gini MDI incorrectly ranks the 100% noise `random_user_id` feature near the top.
  - Proved that Permutation Importance correctly assigns `random_user_id` an importance score of $0.0$, identifying true predictors.

### Task 2: XGBoost Training & Early Stopping (45 Mins)
- **Action:** Train an `XGBClassifier` to predict search purchase probability. Use an validation split (`eval_set`) and set `early_stopping_rounds` to stop training when log-loss stalls.
- **Deliverable:** Learning curves plot showing log-loss over training iterations.
- **Success Criteria:**
  - Plotted train vs. validation log-loss.
  - Programmatically extracted the optimal tree count (`model.best_iteration`) and verified that the model does not include overfitted trailing trees.

### Task 3: ONNX Model Compilation & Latency Benchmarking (60 Mins)
- **Action:** Convert the trained XGBoost model to **ONNX** format. Load the model using `onnxruntime` and write a latency benchmark script comparing single-sample prediction speed between native XGBoost (`model.predict_proba`) and Compiled ONNX Runtime.
- **Deliverable:** Compilation script and execution log reporting mean latencies.
- **Success Criteria:**
  - Model successfully serialized to `.onnx`.
  - Single-sample inference latency measured over 1,000 runs.
  - Proved that ONNX Runtime runs significantly faster than native Python, reducing latencies to the sub-50 microsecond range.

### Task 4: Diagnostics and Feature Split Extraction (30 Mins)
- **Action:** Programmatically parse the XGBoost booster tree dump (`model.get_booster().get_dump()`).
- **Deliverable:** Print of the first tree showing features, split points, gains, and leaf cover parameters.
- **Success Criteria:**
  - Documented what the "cover" parameter represents for binary log-loss classification and how it relates to sample counts.

---

## 3. Sunday Sprint: Conversion A/B Experimentation Engine (≤3 Hours)

**Objective:** Design an A/B test sample plan, run conversion hypothesis Z-tests, and apply the Bonferroni correction for multi-variant layouts.

### Task 1: Sample Size Calculation (Power Analysis) (45 Mins)
- **Action:** Solve for the required sample size per variant before launching the layout test.
  - Baseline Conversion: $6\%$.
  - Minimum Detectable Effect (MDE): $1.2\%$ absolute (detecting a lift from $6\%$ to $7.2\%$).
  - Significance Level ($\alpha$): $0.05$.
  - Target Statistical Power ($1 - \beta$): $80\%$.
- **Deliverable:** Calculation code using `statsmodels.stats.power.NormalIndPower`.
- **Success Criteria:**
  - Correct required sample size per variant outputted.
  - Plotted statistical power curves for MDEs of $1.0\%$, $1.5\%$, and $2.0\%$ over sample sizes up to 20,000.

### Task 2: Two-Sample Proportion Z-Test Execution (45 Mins)
- **Action:** Suppose your A/B test ran for the required sample size:
  - **Control (A):** $n_A = 12000$, conversions $x_A = 720$.
  - **Treatment (B):** $n_B = 12000$, conversions $x_B = 912$.
- **Deliverable:** Calculation code implementing standard error, Z-statistic, and two-tailed p-value.
- **Success Criteria:**
  - Outputted the correct pooled conversion rate ($p_c$), standard error, and Z-statistic by hand or python implementation.
  - Stated whether the layout lift is statistically significant at $\alpha = 0.05$.

### Task 3: Bonferroni Correction for Multiple Prompt Variants (45 Mins)
- **Action:** You run $k = 10$ different prompt variations in parallel against the control.
  - Calculate the FWER (Family-Wise Error Rate) if you Naively evaluate each test at $\alpha = 0.05$.
  - Apply the Bonferroni Correction to correct raw p-values and protect against false positives.
- **Deliverable:** Correction code using `statsmodels` or manual scaling.
- **Success Criteria:**
  - Outputted FWER inflation probability ($40.1\%$).
  - Corrected raw p-values. Stated which of the 10 variants remain significant under the adjusted threshold.

### Task 4: Documentation & Walkthrough (45 Mins)
- **Action:** Document your results, structure the repository, and push to your GitHub vault.
- **Deliverable:** Pushed Git commit with a clean repository structure.
- **Success Criteria:**
  - A comprehensive `README.md` explaining the re-ranker latency speeds and A/B test recommendations.
  - All paths are relative, dependencies are locked in `requirements.txt`.

---

## 4. Expected Project Structure

```
study-prep/
└── assignments/
    └── week_2/
        ├── README.md                 # Latency reports, A/B recommendations, QA
        ├── requirements.txt          # xgboost, onnxruntime, onnxmltools, statsmodels, scipy
        ├── notebooks/
        │   ├── 01_search_reranker.ipynb  # Saturday: MDI vs Permutation, early stopping, ONNX benchmark
        │   └── 02_ab_experiment.ipynb   # Sunday: Power curves, Z-tests, Bonferroni simulation
        └── src/
            ├── benchmark_onnx.py     # Latency comparison script
            └── stats_helper.py       # Z-test and power functions
```

---

## 5. Interview Questions to Answer in your README

To prepare for your upcoming interviews, answer these 5 technical questions based on your assignment implementation:
1. Explain mathematically why Gini Importance (MDI) is biased toward high-cardinality random keys (like unique user IDs). Why does Permutation Importance resolve this?
2. In Gradient Boosting, why does training sequential estimators with a low learning rate (shrinkage $\nu = 0.1$) outperform training a single large tree ($\nu = 1.0$) even if the total boosting depth is equivalent?
3. How do XGBoost's L2 regularization parameter ($\lambda$) and split penalty ($\gamma$) control tree structures, and what happens to similarity scores if $\lambda \to \infty$?
4. What is the Family-Wise Error Rate (FWER) and why does testing 10 layout variations in parallel inflate it? How does the Bonferroni correction mathematically fix this?
5. Why does compiling a tree model to ONNX runtime reduce single-sample API serving latency compared to native Python wrappers? Detail the architectural differences.
