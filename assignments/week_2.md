# Week 2 Engineering Assignment: Conversion Modeling & Marketing A/B Experiments

In this assignment, you will act as a Senior Machine Learning Engineer at a digital advertising agency. Your task is to build and evaluate the user conversion scoring pipeline:
1. **Conversion Prediction Engine (XGBoost & ONNX):** Train a Gradient Boosting model to predict customer conversion, diagnose high-cardinality feature importance bias using the raw user IDs, and compile the model to ONNX for low-latency scoring.
2. **Marketing Campaign A/B Tester (Hypothesis Testing):** Design and evaluate the statistical lift of the ad campaign comparing users who saw ads vs. those who saw a public service announcement (PSA), correcting for multiple-testing false positives.

You will implement these models, compile them for real-time serving, design statistical test parameters, and protect against false positive code rollouts.

---

## 1. Business Context and Dataset

### The Business Problem
Digital ad budgets are easily wasted on users who would have converted anyway. 
- You need a machine learning classifier to predict purchase conversion probability so you can target high-probability users.
- In ad exchanges, latency is critical: bidding decisions must occur within **10–15ms**.
- Marketing runs multi-variant experiments comparing different ad prompts. Without statistical controls, rolling out layouts based on naive significance thresholds results in false positive wins.

### The Dataset
You will use the **Kaggle Marketing A/B Testing Dataset**.
- **Kaggle Link:** [Marketing A/B Testing Dataset](https://www.kaggle.com/datasets/faviolasolano/marketing-ab-testing)
- **Target Variable:** `converted` (Binary: `True` if user converted, `False` if not).
- **Core Features:**
  - `user id`: Unique identifier of the user (100% random high-cardinality noise).
  - `test group`: Treatment indicator (`ad` for those who saw ads, `psa` for those who saw control PSAs).
  - `total ads`: Total number of ads seen by the user.
  - `most active hour`: Hour of the day when the user saw the most ads.
  - `most active day`: Day of the week when the user saw the most ads.

---

## 2. Saturday Sprint: High-Performance Conversion Classifier (≤3 Hours)

**Objective:** Build an XGBoost conversion pipeline, diagnose feature importance bias, use early stopping, and compile the model to ONNX runtime for sub-millisecond serving.

### Task 1: Expose Gini MDI Bias & Calculate Permutation Importance (45 Mins)
- **Action:** Train a Random Forest model to predict `converted = True`. Include the raw `user id` column as a numeric feature. Extract Gini Mean Decrease in Impurity (MDI) feature importances. Then, run Permutation Feature Importance on the validation split.
- **Deliverable:** Bar chart code comparing MDI vs. Permutation importance scores side-by-side.
- **Success Criteria:** 
  - Proved that Gini MDI incorrectly ranks the random `user id` column near the top.
  - Proved that Permutation Importance correctly assigns `user id` an importance score of $0.0$, identifying `total ads` as the true predictor.

### Task 2: XGBoost Training & Early Stopping (45 Mins)
- **Action:** Train an `XGBClassifier` to predict conversion. Use a validation split (`eval_set`) and set `early_stopping_rounds` to stop training when log-loss stalls.
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

## 3. Sunday Sprint: Marketing Campaign A/B Tester (≤3 Hours)

**Objective:** Design an A/B test sample plan, run conversion hypothesis Z-tests, and apply the Bonferroni correction for multi-variant layouts.

### Task 1: Sample Size Calculation (Power Analysis) (45 Mins)
- **Action:** Solve for the required sample size per variant before launching the layout test.
  - Baseline Conversion: $2.5\%$.
  - Minimum Detectable Effect (MDE): $0.5\%$ absolute (detecting a lift from $2.5\%$ to $3.0\%$).
  - Significance Level ($\alpha$): $0.05$.
  - Target Statistical Power ($1 - \beta$): $80\%$.
- **Deliverable:** Calculation code using `statsmodels.stats.power.NormalIndPower`.
- **Success Criteria:**
  - Correct required sample size per variant outputted.
  - Plotted statistical power curves for MDEs of $0.3\%$, $0.5\%$, and $0.7\%$ over sample sizes up to 30,000.

### Task 2: Two-Sample Proportion Z-Test Execution (45 Mins)
- **Action:** Extract the actual counts from the Kaggle dataset's `test group` and `converted` columns. Compute conversion rates for the treatment group (`ad`) and control group (`psa`).
- **Deliverable:** Calculation code implementing standard error, Z-statistic, and two-tailed p-value.
- **Success Criteria:**
  - Outputted the correct pooled conversion rate ($p_c$), standard error, and Z-statistic for the dataset's actual groups.
  - Stated whether showing ads leads to a statistically significant lift compared to control PSAs at $\alpha = 0.05$.

### Task 3: Bonferroni Correction for Multiple Ad Creatives (45 Mins)
- **Action:** Suppose marketing wants to test $k = 10$ different ad creatives in parallel against the control.
  - Calculate the FWER (Family-Wise Error Rate) if you Naively evaluate each test at $\alpha = 0.05$.
  - Apply the Bonferroni Correction to correct raw p-values and protect against false positives.
- **Deliverable:** Correction code using `statsmodels` or manual scaling.
- **Success Criteria:**
  - Outputted FWER inflation probability ($40.1\%$).
  - Corrected raw p-values. Stated which of the 10 variants remain significant under the adjusted threshold.

### Task 4: Documentation & Walkthrough (45 Mins)
- **Action:** Document your findings, structure the repository, and push to your GitHub vault.
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

## 5. Expected Questions to Answer in your README

To prepare for your upcoming interviews, answer these 5 technical questions based on your assignment implementation:
1. Explain mathematically why Gini Importance (MDI) is biased toward high-cardinality random keys (like unique user IDs). Why does Permutation Importance resolve this?
2. In Gradient Boosting, why does training sequential estimators with a low learning rate (shrinkage $\nu = 0.1$) outperform training a single large tree ($\nu = 1.0$) even if the total boosting depth is equivalent?
3. How do XGBoost's L2 regularization parameter ($\lambda$) and split penalty ($\gamma$) control tree structures, and what happens to similarity scores if $\lambda \to \infty$?
4. What is the Family-Wise Error Rate (FWER) and why does testing 10 layout variations in parallel inflate it? How does the Bonferroni correction mathematically fix this?
5. Why does compiling a tree model to ONNX runtime reduce single-sample API serving latency compared to native Python wrappers? Detail the architectural differences.
