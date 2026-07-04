# Week 1 Engineering Assignment: Underwriting and Risk Pricing Engine

In this assignment, you will act as a Risk and ML Engineer at a fintech lending platform. Your sprint task is to build the core statistical engine for a new lending product:
1. **Interest Rate Pricing Model (Regression):** Predict the optimal risk-adjusted interest rate to charge a customer based on their credit profile.
2. **Default Probability Classifier (Classification):** Predict the probability that a borrower will default on their loan.

You will implement these models, diagnose their bias-variance profiles, handle production preprocessing realities, and align performance metrics with business outcomes.

---

## 1. Business Context and Dataset

### The Business Problem
Lending platforms make money by charging interest, but they lose money when borrowers default. 
- Undercharging interest on a high-risk borrower leads to defaults and capital loss.
- Overcharging interest on a low-risk borrower causes them to reject the loan offer, leading to customer acquisition waste.
Your engine must balance this by predicting loan pricing (Interest Rate) and risk (Default Probability).

### The Dataset
You will use the **Kaggle Credit Risk Dataset** (or an equivalent Lending Club CSV).
- **Kaggle Link:** [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Target Variables:**
  - `loan_int_rate` (Continuous: Interest rate of the loan, for the pricing engine).
  - `loan_status` (Binary: `0` for non-default, `1` for default, for the risk engine).
- **Core Features:** Age, annual income, employment length, loan amount, home ownership status, and historical default flag.

---

## 2. Saturday Sprint: Interest Rate Pricing Engine (≤3 Hours)

**Objective:** Build a regularized Linear Regression pipeline using Scikit-Learn to predict `loan_int_rate`.

### Task 1: Exploratory Data Analysis & Cleaning (45 Mins)
- **Action:** Inspect the distribution of the target variable `loan_int_rate` and input features. Identify missing values (e.g., in `person_emp_length` and `loan_int_rate`) and extreme outliers (e.g., impossible employment lengths or ages).
- **Deliverable:** Jupyter notebook code performing imputations and outlier clipping.
- **Success Criteria:** 
  - Missing values imputed using the median calculated *strictly on the training split* to prevent leakage.
  - Outliers handled (e.g., clipping employment length at 40 years).
  - Categorical variables encoded using target encoding or one-hot encoding without falling into the dummy variable trap.

### Task 2: Multicollinearity & Feature Engineering (45 Mins)
- **Action:** Calculate the Variance Inflation Factor (VIF) for all continuous features. Identify multicollinear columns and engineer interaction features (e.g., `loan_to_income_ratio`).
- **Deliverable:** Matrix correlation analysis and a VIF report.
- **Success Criteria:** 
  - VIF scores calculated. All features in the final matrix have $\text{VIF} < 5.0$.
  - Interaction feature `loan_to_income_ratio` created and validated.

### Task 3: OLS vs. Ridge/Lasso Model Training (60 Mins)
- **Action:** Split data into Train/Validation/Test (80/10/10). Train an OLS Linear Regression model, a Ridge ($L_2$) model, and a Lasso ($L_1$) model.
- **Deliverable:** Model training code and coefficient comparisons.
- **Success Criteria:** 
  - Standard scaling parameters ($\mu, \sigma$) calculated *only* on the training set.
  - Hyperparameter tuning ($L_1$/$L_2$ penalty $\alpha$) performed using 5-Fold Cross-Validation.
  - Analyzed the weights vector to prove Lasso forced coefficients to exactly $0$, identifying dropped features.

### Task 4: Evaluation and Pricing Diagnostics (30 Mins)
- **Action:** Compute and compare MSE, RMSE, MAE, and Adjusted $R^2$ on the test set.
- **Deliverable:** Markdown performance summary table.
- **Success Criteria:** 
  - All metrics reported with correct units.
  - Explained in the documentation why MAE is preferred over RMSE for reporting baseline pricing deviations to business stakeholders.

---

## 3. Sunday Sprint: Default Risk Classifier & Tuning (≤3 Hours)

**Objective:** Build a Logistic Regression classifier to predict default probability (`loan_status`), calibrate output thresholds, and diagnose learning curves.

### Task 1: Logistic Regression and Class Imbalance (45 Mins)
- **Action:** Train a Logistic Regression model to predict binary default. Since default rates are highly imbalanced ($\approx 20\%$ default, $80\%$ non-default), apply cost-sensitive class weights.
- **Deliverable:** Classification pipeline code.
- **Success Criteria:** 
  - Model trained with `class_weight='balanced'` in Scikit-Learn.
  - Compared the log-loss cost of the weighted model vs. the unweighted model.

### Task 2: Threshold Tuning for Financial Decisions (45 Mins)
- **Action:** By default, classification uses a $0.5$ threshold. Slide this threshold to align with business costs:
  - Cost of missing a default (False Negative): Loss of loan principal ($\$10,000$).
  - Cost of a false alarm (False Positive): Lost interest margin ($\$500$).
- **Deliverable:** Precision-Recall curve plot and a threshold optimization table.
- **Success Criteria:** 
  - Plotted Precision, Recall, and F1-score across thresholds from $0.0$ to $1.0$.
  - Calculated and selected the optimal threshold that minimizes the total financial loss rather than maximizing raw accuracy.

### Task 3: Bias-Variance Learning Curve Diagnostics (45 Mins)
- **Action:** Plot training and validation Log-Loss against training set sizes (from $100$ samples to the full dataset size).
- **Deliverable:** Learning curve visualization plot.
- **Success Criteria:** 
  - Clearly diagnosed if the model suffers from a high-bias (underfitting) or high-variance (overfitting) bottleneck.
  - Documented concrete recommendations based on the curve (e.g., whether to collect more rows or engineer polynomial features).

### Task 4: Walkthrough & Git Push (45 Mins)
- **Action:** Document your findings, structure the repository, and push to your GitHub vault.
- **Deliverable:** Pushed Git commit with a clean repository structure.
- **Success Criteria:**
  - A comprehensive `README.md` explaining the pricing and risk engine results.
  - Reproducible code execution (all paths are relative, dependencies are locked).

---

## 4. Optional Stretch Goals

1. **Implement Scratch Solvers (NumPy):**
   Implement your own regularized Linear Regression (using the closed-form Normal Equation) and Logistic Regression solver (using gradient descent updates) from scratch using only raw NumPy. Compare the coefficients and prediction execution latency against Scikit-Learn.
2. **Log-Odds Probability Recalibration:**
   Manually downsample the majority class to $50/50$ balance, train the model, and implement the log-odds recalibration formula to correct prediction probabilities for production serving:
   $$\text{logit}_{\text{calibrated}} = \text{logit}_{\text{model}} - \log\left(\frac{p_{\text{downsample}}}{1 - p_{\text{downsample}}}\right)$$

---

## 5. Expected Project Structure

```
study-prep/
└── assignments/
    └── week_1/
        ├── README.md                 # Project summary and business decisions
        ├── requirements.txt          # Python packages (locked versions)
        ├── notebooks/
        │   ├── 01_pricing_engine.ipynb   # Saturday: EDA, VIF, OLS/Ridge/Lasso
        │   └── 02_risk_classifier.ipynb  # Sunday: Logistic, threshold tuning, curves
        └── src/
            ├── preprocessing.py      # Data cleaning and feature pipeline
            └── models.py             # Scratch implementations (if attempted)
```

---

## 6. Interview Questions to Answer in your README

To prepare for your upcoming interviews, answer these 5 technical questions based on your assignment implementation:
1. When you normalized features for the pricing engine, why did you fit the scaler only on the training split, and what mathematical error occurs if you scale the entire dataset first?
2. If your learning curve showed a high-bias profile, why would collecting 50,000 additional customer records fail to improve validation log-loss? What would you do instead?
3. In your default risk classifier, why did raw classification accuracy fail as an evaluation metric, and how did you select your final decision threshold?
4. If a feature like `person_income` perfectly separated defaults from non-defaults during training, what happens to the weights vector of an unregularized Logistic Regression model? How does Ridge ($L_2$) regularization stabilize this?
5. How does the Lasso ($L_1$) model act as an embedded feature selector during model training? Explain the geometric difference that causes it to set weights to exactly zero compared to Ridge ($L_2$).
