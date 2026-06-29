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
- **The Danger:** OLS assumes constant variance. Under heteroscedasticity, OLS weights remain unbiased, but the standard error calculations of the coefficients are wrong. This renders confidence intervals and feature selection $p$-values completely untrustworthy.
- **The Fix:** Apply a log transformation (`np.log1p()`) to the target sales variable to compress the variance scale at higher values.

---

### Q6: How do you handle a production model tracking severe macroeconomic data drift without changing your codebase architecture?
**Answer:**
Instead of modifying model architecture or retraining immediately, you can resolve macroeconomic drift (like inflation or interest rate spikes) at the **pipeline ingestion layer**:
1. **Ratio Features:** Replace raw monetary amounts with ratios (e.g., transform raw `debt` and `income` columns into a stable `debt_to_income_ratio`).
2. **Deflation Indexing:** Apply a real-time CPI (Consumer Price Index) lookup table to deflate raw currency features before feeding them to the model.
3. **Rolling Standardization:** Scale features using a Z-score calculated over a rolling window (e.g., standardizing feature inputs based on the mean and standard deviation of the last 30 days).

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
2. **Huber Loss:** Train the model using Huber Loss instead of MSE. Huber loss treats small errors quadratically (like MSE) but treats large errors linearly (like MAE). This prevents a single extreme outlier (like a customs-delayed order) from generating a massive gradient update that warps the entire model.
3. **Winsorization:** Apply feature clipping at ingestion (e.g., capping values at the 1st and 99th percentiles of the training distribution).

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
1. **Monitor Prediction Drift ($P(\hat{y})$):** Compute the **Population Stability Index (PSI)** or **Jensen-Shannon Divergence** daily to compare the distribution of the model's active predictions against the validation predictions during training. If the distribution shifts, the model's behavior has drifted.
2. **Monitor Feature Drift ($P(X)$):** Check if the distribution of incoming features has shifted relative to training (e.g., tracking feature means/standard deviations).
3. **Monitor Pipeline Health (Null Rates):** Track the percentage of `NaN` values per feature. A sudden spike in missing values means the pipeline is substituting features with default/median values, causing the model to output degraded, flatline predictions.
