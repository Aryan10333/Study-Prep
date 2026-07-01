# Interview Flashcards: Linear Regression

This document contains 10 core, production-focused interview questions and answers designed to prepare you for senior ML engineering interviews by referencing our real-world scenarios.

---

### Q1: What happens to gradient descent without feature scaling?
**Answer:**
Without feature scaling, gradient descent becomes extremely slow and mathematically unstable.
- **The Visual and Math:** If Feature A (e.g., house size in sq ft) ranges from $1000$ to $5000$, and Feature B (e.g., number of bedrooms) ranges from $1$ to $5$, the gradients for the weights of Feature A will be massive compared to those of Feature B. This elongates the cost contours of $J(w,b)$ into a narrow valley.
- **The Optimization Failure:** The gradient updates will bounce back and forth across the valley walls (oscillating) rather than pointing toward the minimum. To prevent divergence, you must set an extremely small learning rate $\alpha$, stalling convergence. Scaling features (e.g., via standard Z-scoring) centers the data and rounds the contours into symmetric circles, allowing gradient updates to point directly toward the global minimum.

---

### Q2: Why do we leave the bias term 'b' unregularized?
**Answer:**
We do not regularize the bias scalar $b$ because doing so has no impact on preventing overfitting and can introduce massive systematic bias.
- **The Conceptual Intuition:** Regularization ($L_1$/$L_2$) restricts the weights vector $w$ to prevent the model from over-indexing on high-variance noise in your input features. The bias term $b$ does not scale with any feature; it simply controls the intercept, shifting the model's baseline prediction up or down.
- **Concrete Example:** If you are predicting house prices where the minimum market entry price is \$200k, forcing $b \rightarrow 0$ via a regularization penalty forces the regression plane closer to the origin $(0,0)$. The model will be forced to make massive errors on lower-end homes, dragging down overall accuracy for no generalization benefit.

---

### Q3: What are the 2 main data layout reasons that cause the Normal Equation matrix to crash (non-invertibility), and how do you fix them?
**Answer:**
The Normal Equation $\theta = (X^T X)^{-1} X^T y$ crashes when the matrix $X^T X$ is singular (not invertible). This is caused by:
1. **Perfect Multicollinearity (Redundant Columns):** Like the **Dummy Variable Trap** (e.g., One-Hot encoding both `Desktop` and `Mobile` device type flags alongside a bias intercept column). Since `Desktop + Mobile = 1` (the bias vector), the columns are linearly dependent.
2. **Underdetermined Systems ($m \le n$):** Having fewer training rows ($m$) than features ($n$) (e.g., predicting LTV with 10 rows of historical customer data but 100 features).

**How to Fix:**
- Remove the redundant dummy column (e.g., drop the `Desktop` column, leaving `Mobile` to act as the baseline indicator).
- Add L2 Regularization (Ridge Regression), which adds a small penalty $\lambda I$ along the diagonal: $(X^T X + \lambda I)^{-1}$. This shifts the eigenvalues, guaranteeing invertibility.

---

### Q4: In plain English, why can standard $R^2$ deceive you when adding new features, and how does Adjusted $R^2$ help?
**Answer:**
- **Standard $R^2$ Deception:** Standard $R^2$ measures the proportion of variance explained by features. OLS optimization always tries to minimize training residuals. If you add a completely random column of noise (e.g., `tokyo_temperature` to predict `chicago_housing_prices`), the optimizer will still assign it a small non-zero weight to fit whatever noise exists in the training set. This artificially reduces the training residual sum of squares ($\text{SS}_{\text{res}}$), forcing standard $R^2$ to increase.
- **How Adjusted $R^2$ Helps:** It scales the $R^2$ score by the number of features $n$ relative to sample size $m$:
  $$\text{Adjusted } R^2 = 1 - \left( 1 - R^2 \right) \frac{m - 1}{m - n - 1}$$
  If you add a useless noise feature, $n$ increases, which shrinks the denominator $m - n - 1$, magnifying the error multiplier. If the improvement in $R^2$ is not large enough to offset this penalty, the Adjusted $R^2$ score will drop, alerting you to feature bloat.

---

### Q5: What does a "megaphone" or "funnel" shape in a residual plot tell you about your model's variance?
**Answer:**
It tells you that the model violates the assumption of **homoscedasticity**, indicating **heteroscedasticity** (non-constant variance of errors).
- **The Concept:** The spread of residuals increases or decreases systematically as the predicted values grow. For example, in our daily sales forecasting scenario, predicting sales on normal days is highly stable (small residuals), but predicting sales on hot days is highly volatile (large residuals), creating a funnel shape.
- **The Danger (Why standard errors break):** Under heteroscedasticity, OLS parameter estimates ($\hat{w}$) remain unbiased, but the standard formula for the covariance matrix of the coefficients:
  $$\text{Var}(\hat{w}) = s^2 (X^T X)^{-1}$$
  is no longer valid. Standard errors of the coefficients are typically underestimated. This inflates the t-statistic ($t = \hat{w}_j / \text{SE}(\hat{w}_j)$), leading to false positives (Type I errors) where you conclude a feature is highly significant ($p < 0.05$) when it is actually just noise. Confidence intervals also become too narrow and untrustworthy.
- **How the Fix Solves It:** Applying a log transformation (`np.log1p()`) to the target variable $y$ compresses the scale of larger values. In many economic datasets, variance scales proportionally with magnitude (e.g., variance of a \$100k store's sales is smaller than a \$10M store's sales). The log transform converts multiplicative scaling into additive scaling, stabilizing the absolute error variance to a constant ribbon in log space. This restores homoscedasticity, correcting the standard error calculations.

---

### Q6: How do you handle a production model tracking severe macroeconomic data drift without changing your codebase architecture?
**Answer:**
Instead of modifying model architecture or retraining immediately, you can resolve macroeconomic drift (like high inflation or interest rate spikes) at the **pipeline ingestion layer**:

*   **Example Scenario:** A credit scoring model predicting loan default risk trained in a low-inflation year (e.g., 2019) is deployed during a high-inflation period (e.g., 2022), causing raw borrower income and debt distributions to shift upward.

**Ingestion Layer Fixes:**
1. **Ratio Features:** Replace raw monetary amounts with ratios (e.g., transform raw `debt` and `income` columns into a stable `debt_to_income_ratio`).
   - *How it solves the drift:* A borrower earning \$100k with \$10k debt has the same $0.10$ ratio as someone earning \$200k with \$20k debt. Ratios normalize monetary expansion, keeping the feature range stable and preserving the linear relationship with default risk.
2. **Deflation Indexing:** Apply a real-time Consumer Price Index (CPI) lookup table to deflate raw currency features before feeding them to the model.
   - *How it solves the drift:* Divide nominal incoming features by the current CPI factor: $x_{\text{real}} = x_{\text{nominal}} / \text{CPI}$. This maps incoming nominal dollars back to real 2019 baseline purchasing power, ensuring the model's coefficients (like $\beta_{\text{income}}$) operate on the scale they were trained on.
3. **Rolling Standardization:** Scale features using a Z-score calculated over a rolling window (e.g., Z-scoring feature inputs based on the mean and standard deviation of the last 30 days).
   - *How it solves the drift:* If macro factors cause credit scores to drop across the board during a recession, rolling Z-scoring:
     $$x_{\text{scaled}} = \frac{x_t - \mu_{30}}{\sigma_{30}}$$
     scales the distribution so that a credit score represents a customer's relative rank within the current economic climate (e.g., $+1.5$ standard deviations above average), rather than their absolute value, mitigating score drift.

---

### Q7: Why is a simple vectorized linear model preferred over deep learning for high-throughput, low-latency API systems?
**Answer:**
It comes down to computational complexity, cost, and latency budgets:
- **Low Latency:** A vectorized linear model requires only a single dot product ($w \cdot x + b$), which compiles to CPU register SIMD instructions. In high-frequency ad-tech bidding systems, this dot product runs in **microseconds** (sub-millisecond). Deep learning models require processing millions of weights across multiple layer matrices, taking tens of milliseconds.
- **High Throughput:** Because dot products consume minimal CPU cycles, a single API server can handle tens of thousands of requests per second without needing expensive, high-maintenance GPU hardware.

---

### Q8: How do you protect an automated production regression pipeline against massive data outliers causing unstable predictions?
**Answer:**
You protect the pipeline by implementing robust scaling, robust loss functions, and input capping:
1. **Robust Scaling:** Standardize features using the median and Interquartile Range (IQR) via `RobustScaler` instead of the mean and variance, which are easily skewed by outliers.
2. **Huber Loss:** Train the model using Huber Loss instead of MSE. Huber loss is a piecewise function defined as:
   $$L_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta \left( |y - \hat{y}| - \frac{1}{2}\delta \right) & \text{otherwise} \end{cases}$$
   - *Under the hood:* The hyperparameter $\delta$ is the transition threshold. Within the threshold ($|y - \hat{y}| \le \delta$), the loss is quadratic (MSE), yielding gradients that scale linearly with the error to ensure smooth convergence. Outside the threshold ($|y - \hat{y}| > \delta$), the loss becomes linear (MAE) with a constant gradient sign of $\pm \delta$. Even if a customs delay creates a massive error of $100$ days, the gradient update is capped at $\pm \delta$ rather than exploding to $100$. This prevents outlier samples from dominating the parameter updates.
3. **Winsorization:** Apply feature clipping at ingestion (e.g., capping values at the 1st and 99th percentiles of the training distribution). This limits extreme input values from warping the prediction vector.

---

### Q9: Why does raw One-Hot Encoding on high-cardinality attributes break a linear model's stability, and what is an engineering alternative?
**Answer:**
- **Why it breaks:** If a categorical feature has $5000$ unique values (e.g., zip codes), One-Hot encoding it adds $5000$ columns of mostly zeros. This makes the design matrix $X$ extremely sparse. In OLS, this causes $X^T X$ to be close to singular (non-invertible) or highly unstable. The weights for these rare categories will overfit heavily.
- **Engineering Alternatives:**
  - **Target Encoding:** Replace the category with the average target value for that category in the training set, applying smoothing to prevent target leakage.
  - **Feature Hashing:** Use the "hashing trick" to hash the categories into a fixed number of bins (e.g., $128$ columns). This bounds the dimensionality and gracefully handles new categories at inference.

---

### Q10: If you don't get ground-truth labels back for 30 days, how do you know if your live regression model is degrading?
**Answer:**
You monitor proxy metrics that reflect data health and distribution shifts:
1. **Monitor Prediction Drift ($P(\hat{y})$) via PSI:** Compute the **Population Stability Index (PSI)** daily to compare the distribution of the model's active predictions against the baseline validation predictions during training:
   $$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$
   - *Interpret Thresholds:*
     - $\text{PSI} < 0.1$: Stable (no action).
     - $0.1 \le \text{PSI} < 0.25$: Moderate drift (flag for monitoring and validation split checks).
     - $\text{PSI} \ge 0.25$: Significant drift (triggers immediate model retraining or rollback).
2. **Monitor Feature Drift ($P(X)$) via Kolmogorov-Smirnov (KS) Test:**
   - Run a two-sample KS test comparing the distribution of incoming active features over a rolling window against the baseline training distribution. A low p-value indicates covariate shift, warning that incoming inputs no longer match the training assumptions.
3. **Monitor Pipeline Health (Null Rates & Imputation Collapse):**
   - Track the percentage of default/imputed values. If an upstream database schema change or API outage causes a feature (e.g., `user_income`) to drop out, the pipeline will fall back to imputing the median value for 100% of cases. The input variance collapses to 0, causing the model to output degraded, flatline predictions. Monitoring null spikes alerts you before predictions drift.
