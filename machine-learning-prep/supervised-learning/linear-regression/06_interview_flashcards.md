# Interview Flashcards: Linear Regression

This document contains 10 core, production-focused interview questions and answers designed to test an applied Machine Learning Engineer's practical understanding of Linear Regression.

---

### Q1: What happens to gradient descent without feature scaling?
**Answer:**
Without feature scaling, gradient descent becomes extremely slow and mathematically unstable.
- **Visual Intuition:** If Feature A ranges from $0$ to $1$ (e.g., number of bedrooms) and Feature B ranges from $0$ to $1,000,000$ (e.g., annual income), the contours of the cost function $J(w,b)$ will look like highly elongated, narrow ellipses. 
- **The Optimization Bug:** The gradient vectors will point almost perpendicular to the direction of the global minimum. If you use a normal learning rate $\alpha$, the updates will oscillate wildly back and forth across the narrow valley walls rather than traveling down the center toward the minimum. To prevent this divergence, you are forced to set an extremely small $\alpha$, making training painfully slow.
- **The Solution:** Standardizing features (e.g., Z-score normalization) transforms the cost contours into symmetric, concentric circles, allowing gradient updates to point directly toward the global minimum.

---

### Q2: Why do we leave the bias term 'b' unregularized?
**Answer:**
We do not regularize the bias scalar $b$ because doing so has no impact on preventing overfitting and can severely degrade model performance.
- **The Purpose of Regularization:** Regularization ($L_1$/$L_2$) restricts the weights vector $w$ to prevent the model from relying too heavily on complex combinations of features or fitting high-variance noise in the data.
- **The Role of Bias:** The bias $b$ simply shifts the regression line/plane up or down to align with the mean of the target variable $y$. It does not control the sensitivity of the model to input feature changes.
- **The Consequence of Regularizing $b$:** If we penalize $b$ and force it toward $0$, we force our regression line to pass closer to the origin $(0,0)$. For example, if predicting house prices where the average price is $300\text{k}$, forcing $b \rightarrow 0$ would force the model to start predicting near-zero prices for houses with small features, severely biasing the model's predictions.

---

### Q3: What are the 2 main data layout reasons that cause the Normal Equation matrix to crash (non-invertibility), and how do you fix them?
**Answer:**
The Normal Equation calculation $\theta = (X^T X)^{-1} X^T y$ crashes when the matrix $X^T X$ is singular (non-invertible). The two primary reasons are:
1. **Perfect Multicollinearity (Redundant Columns):** If you have two features that are linearly dependent (e.g., weight in kilograms and weight in pounds, or you fall into the "Dummy Variable Trap" by One-Hot encoding all categories without dropping a baseline column). One column is a scalar multiple of another, making the matrix rank-deficient.
2. **Too Few Rows ($m \le n$):** You have more features than training examples (e.g., $5$ rows of data but $10$ features). The model is underdetermined, meaning there are infinite solutions, so the unique inverse does not exist.

**How to Fix:**
- **Remove Redundant Columns:** Screen columns with a correlation matrix or drop the baseline dummy variable.
- **Apply $L_2$ Regularization:** Switch to Ridge Regression. The mathematical update adds a diagonal identity matrix $\lambda I$ to $X^T X$. The matrix $(X^T X + \lambda I)$ is guaranteed to be invertible.
- **Use Pseudo-inverse:** Calculate the Moore-Penrose pseudo-inverse (`np.linalg.pinv` in NumPy) which uses SVD under the hood to find a stable solution without a standard inversion.

---

### Q4: In plain English, why can standard $R^2$ deceive you when adding new features, and how does Adjusted $R^2$ help?
**Answer:**
- **Standard $R^2$ Deception:** Standard $R^2$ measures the percentage of variance explained by the model's features. The Ordinary Least Squares (OLS) algorithm is designed to minimize training residuals. If you add a completely random, useless column of noise, the optimizer will still assign it a tiny, non-zero weight to fit whatever noise exists in the training set. Because this marginally reduces the residual sum of squares ($\text{SS}_{\text{res}}$), standard $R^2$ will **always increase (or stay the same)**, making the model look better than it actually is.
- **How Adjusted $R^2$ Helps:** Adjusted $R^2$ introduces a penalty term that scale with the number of features $n$. If the new feature does not improve the model's performance significantly enough to offset this penalty, the Adjusted $R^2$ will **decrease**. It acts as an automatic filter for feature bloat.

---

### Q5: What does a "megaphone" or "funnel" shape in a residual plot tell you about your model's variance?
**Answer:**
A funnel shape indicates **heteroscedasticity**, which means the variance of the model's errors is not constant.
- **The Diagnostic:** The spread of the residuals increases (or decreases) systematically as the predicted values (fitted values) increase.
- **The Danger:** It violates a core OLS assumption. While the model's weights remain unbiased, the calculated standard errors of those weights are wrong. This means any confidence intervals or feature selection $p$-values are completely untrustworthy.
- **The Fix:** Apply a variance-stabilizing transformation to the target variable $y$ (like a log or Box-Cox transform), or switch to **Weighted Least Squares (WLS)**, which downweights samples that have high variance.

---

### Q6: How do you handle a production model tracking severe macroeconomic data drift without changing your codebase architecture?
**Answer:**
Instead of modifying model code or retraining with complex structures, you can handle macroeconomic drift (like inflation or shifting interest rates) in the **data engineering/feature pipeline** layer before it hits the model:
1. **Ratio / Relative Features:** Instead of feeding raw monetary values (like `monthly_rent` and `monthly_income`), transform them into relative metrics (like `rent_to_income_ratio`). Ratios remain highly stable even during inflation.
2. **Deflation / Scaling:** Feed the raw currency features through a real-time inflation-adjustment step (deflating values using a monthly Consumer Price Index (CPI) index table lookup).
3. **Rolling Standardization:** Apply standard Z-score scaling ($x_{\text{scaled}} = \frac{x - \mu}{\sigma}$) using a rolling window (e.g., calculating $\mu$ and $\sigma$ over the last 30 days) to keep input distributions constant for the model.

---

### Q7: Why is a simple vectorized linear model preferred over deep learning for high-throughput, low-latency API systems?
**Answer:**
It comes down to computational complexity, cost, and latency budgets:
- **Low Latency:** In a vectorized linear model, predicting $\hat{y}$ is a single dot product ($w \cdot x + b$). This translates to a single CPU vector instruction (SIMD) that runs in **microseconds** (sub-millisecond). Deep learning models require millions of parameters, matrix multiplications across multiple layers, and activation function evaluations, which take tens of milliseconds.
- **High Throughput / Low Cost:** Dot products are computationally cheap. A single server can process tens of thousands of requests per second for a linear model without needing expensive GPUs. A deep learning model requires dedicated GPU nodes, increasing infrastructure costs and latency overhead.

---

### Q8: How do you protect an automated production regression pipeline against massive data outliers causing unstable predictions?
**Answer:**
You can protect the pipeline at the scaling, loss function, and ingestion layers:
1. **Robust Scaling:** Avoid Standard Scaling (mean-variance based) which is heavily skewed by outliers. Use `RobustScaler` (median-IQR based) to normalize features.
2. **Huber Loss:** Train the model using **Huber Loss** instead of MSE. Huber loss acts quadratically (like MSE) for small errors, but linearly (like MAE) for large errors. This bounds the gradient impact of extreme outliers so they don't drag the regression plane away from the majority of the data.
3. **Winsorization (Clipping):** Implement an ingestion transformer that clips values outside a specific percentile (e.g., capping values at the 1st and 99th percentiles of the training distribution).

---

### Q9: Why does raw One-Hot Encoding on high-cardinality attributes break a linear model's stability, and what is an engineering alternative?
**Answer:**
- **Why it breaks:** If a categorical feature has $5,000$ unique values (e.g., zip codes), One-Hot encoding it adds $5,000$ columns of mostly zeros. This causes the design matrix $X$ to become extremely sparse, and the matrix $X^T X$ becomes unstable or singular. The model will overfit heavily on categories with only one or two occurrences, assigning them wild, extreme weights.
- **Engineering Alternatives:**
  - **Target Encoding:** Replace each category with the average target value for that category in the training set, applying smoothing to prevent target leakage.
  - **Feature Hashing:** Use the "hashing trick" to hash the categories into a fixed number of bins (e.g., $128$ columns). This bounds the dimensionality and gracefully handles new categories at inference.

---

### Q10: If you don't get ground-truth labels back for 30 days, how do you know if your live regression model is degrading?
**Answer:**
You monitor proxy metrics that reflect data health and distribution shifts:
1. **Monitor Prediction Drift ($P(\hat{y})$):** Compute the **Population Stability Index (PSI)** or **Jensen-Shannon Divergence** daily to compare the distribution of the model's active predictions against the validation predictions during training. If the distribution shifts, the model's behavior has drifted.
2. **Monitor Feature Drift ($P(X)$):** Check if the distribution of incoming features has shifted relative to training (e.g., tracking feature means/standard deviations).
3. **Monitor Pipeline Health (Null Rates):** Track the percentage of `NaN` values per feature. A sudden spike in missing values means the pipeline is substituting features with default/median values, causing the model to output degraded, flatline predictions.
