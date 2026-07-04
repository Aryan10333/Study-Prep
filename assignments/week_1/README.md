# Project Report: End-to-End Regularized Credit Risk Pricing Engine

This report outlines the step-by-step machine learning pipeline engineered to predict loan interest rates (`loan_int_rate`) using the Kaggle Credit Risk dataset. The project transitions from raw, noisy data into a highly predictive, production-ready linear regression engine.

---

## Executive Summary

* **Objective:** Build an end-to-end regularized Linear Regression framework to automate and predict credit risk pricing.
* **Baseline Accuracy ($R^2$):** **0.9071** (The engine successfully explains **90.71%** of variance in interest rates out-of-sample).
* **Average Deviation (MAE):** **$\pm$0.7922 percentage points**, offering a tight, reliable margin for commercial loan pricing.

---

## Step 1: Data Splitting & Leakage Guardrails

Before inspecting or transforming any data, a strict splitting architecture was established to prevent **data leakage** (accidental sharing of evaluation data variances during training):

1. **Train/Val/Test Split:** The dataset was partitioned into an **80/10/10** structure.
2. **Parameters Isolation:** All statistical benchmarks—including the imputation median and standard scaling parameters $(\mu, \sigma)$—were calculated **strictly on the training set** and simply reused to transform the validation and test datasets.

---

## Step 2: Exploratory Data Analysis (EDA) & Cleaning Pipeline

### Finding 1: Numerical Missing Values

* **Observation:** The feature `person_emp_length` was found to have **661 missing values** in the training partition. The column was heavily right-skewed.
* **Action taken:** Imputed missing values using the **training median (4.0 years)** rather than the mean to prevent extreme outliers from biasing the baseline replacement value.

### Finding 2: Impossible Human Anomalies (Data Entry Errors)

* **Observation:** Statistical summaries and pairplots highlighted physically impossible thresholds: a maximum `person_age` of **144** and a maximum `person_emp_length` of **123.0 years**. Calculating differences also revealed an applicant who had been working for 102 years longer than they had been alive.
* **Action taken:** Engineered a logical boolean mask to strictly filter out corrupt rows. The criteria restricted the training data to realistic human horizons:

$$\text{person\_age} \le 100 \quad \text{and} \quad \text{person\_emp\_length} \le 40$$



A secondary cross-check ensured employment history never exceeded age minus a minimum legal working buffer:

$$(\text{person\_age} - \text{person\_emp\_length}) \ge 14$$



### Finding 3: Explosive Distribution Skewness

* **Observation:** Running a mathematical skewness audit revealed that `person_income` possessed an extreme right-skew score of **9.64**, with a maximum income stretching up to $2.03M while 99% of applicants made under $225,000.
* **Action taken:** Applied a natural log transformation using `np.log1p()` on `person_income`. This effectively compressed the extreme right tail, dropping the skewness score down to a symmetric **0.129**, mapping beautifully to linear modeling assumptions.

### Finding 4: Magnitude Unit Mismatches

* **Observation:** Features were on entirely different scales (e.g., age hovered in the 20s-80s, while loan amounts scaled up to $35,000).
* **Action taken:** Applied `StandardScaler` to put all 6 continuous variables onto a standardized playing field (Mean = 0, Std = 1). Categorical flags (one-hot encoded values) were explicitly isolated and **left unscaled** to preserve their binary interpretation.

---

## Step 3: Diagnostic Auditing & Multicollinearity

### Finding 5: High Feature Redundancy

* **Observation:** Visual pairplots displayed an aggressive linear path between `person_age` and `cb_person_cred_hist_length`.
* **Action taken:** Ran a **Variance Inflation Factor (VIF)** analysis across the design matrix. The VIF scores for age (4.23) and credit history length (4.11) safely cleared the standard threshold rule of thumb ($\text{VIF} < 5.0$), proving that while they share data, they don't introduce enough severe multicollinearity to destabilize the model's coefficients.

---

## Step 4: Model Training, Regularization, & Sparsity

Three separate configurations were trained on the optimized matrix to test regularized constraint stability against an Ordinary Least Squares (OLS) baseline. For Ridge and Lasso, **5-Fold Cross-Validation** (`RidgeCV` and `LassoCV`) searched an array of 100 log-spaced alpha ($\alpha$) candidates to find the mathematical optimum.

### Weight Vector Analysis & Automated Feature Selection

By pulling the coefficients side-by-side, we analyzed how the models penalized features:

* **OLS vs. Ridge ($\alpha \approx 0.024$):** Because multicollinearity was low, Ridge selected a tiny penalty, matching OLS coefficients almost identically. Both models confirmed that **Loan Grade** was the single most dominant driver of interest rates (e.g., shifting from Grade A to Grade G automatically adds $\approx 12.85\%$ to the baseline interest rate).
* **Lasso ($\alpha = 0.001$):** Lasso ($L_1$ penalty) acted as an automated feature selector. It successfully recognized that minor variances in specific intent types did not contribute meaningful unique signal, driving their coefficients to **exactly 0.000000** and effectively dropping them:
* ❌ `loan_intent_EDUCATION` (Dropped)
* ❌ `loan_intent_MEDICAL` (Dropped)
* ❌ `loan_intent_VENTURE` (Dropped)



---

## Step 5: Test Evaluation & Pricing Diagnostics

The models were evaluated out-of-sample on the held-out test dataset, ensuring an identical preprocessing pipeline via our reusable `clean()` function.

### Master Test Performance Matrix

| Performance Metric | OLS Baseline | Ridge ($L_2$) | Lasso ($L_1$) | Unit of Measurement |
| --- | --- | --- | --- | --- |
| **Mean Squared Error (MSE)** | 1.0039 | 1.0039 | 1.0054 | Percentage Points Squared ($\%^2$) |
| **Root Mean Squared Error (RMSE)** | 1.0019 | 1.0020 | 1.0027 | Percentage Points ($\%$) |
| **Mean Absolute Error (MAE)** | **0.7922** | **0.7922** | **0.7924** | Percentage Points ($\%$) |
| **Coefficient of Determination ($R^2$)** | 0.9071 | 0.9071 | 0.9070 | Proportion ($0.0$ to $1.0$) |
| **Adjusted $R^2$ Score** | 0.9064 | 0.9064 | 0.9063 | Proportion ($0.0$ to $1.0$) |

### Pricing Diagnostic: Why the Business Prefers MAE over RMSE

When explaining the pricing engine's error rates to non-technical business stakeholders, **MAE is heavily favored over RMSE** for reporting:

1. **Direct Financial Interpretation:** MAE maps linearly to the native metric. Saying *"Our model misses the true market interest rate by an average of 0.79 percentage points"* directly translates to clear financial planning and margin calculations. RMSE squares the errors, resulting in abstract units that cannot be cleanly mapped back to real-world pricing.
2. **Robustness to Production Outliers:** Because the validation and test splits preserved raw data points to simulate real-world data feeds, a few extreme anomalous rows produced isolated, massive residuals. Because RMSE squares errors before averaging, these few extreme points disproportionately drag RMSE up to **1.00%**. MAE treats errors linearly, providing a much cleaner and realistic depiction of the model's true day-to-day baseline performance on 99% of normal loan applications.