# Assumptions and Diagnostics

For linear regression predictions to be stable and for its coefficients to be interpretable, the underlying data must satisfy several statistical assumptions. In production, violating these assumptions leads to unstable predictions, inflated errors on unseen data, and misleading feature importances.

---

## 1. The 5 Core Assumptions and Visual Diagnostics

Instead of relying on rigid academic hypothesis tests, applied ML engineers use **visual diagnostic plots** to identify model issues.

```
Residuals vs. Fitted
  Residuals
   ^
   |    .   .  .     .
   |  .   .      .      .
 --+----------------------> Fitted Values (Predictions)
   |  .   .      .      .
   |    .   .  .     .
   v
   (Healthy: Even ribbon of points, no patterns)
```

### Assumption 1: Linearity
- **What it means:** The relationship between the independent variables $x$ and the dependent variable $y$ is linear.
- **Diagnostic Plot:** **Residuals vs. Fitted plot**. The residuals (errors) should be randomly scattered around the horizontal line $y=0$. If you see a curved U-shape or S-shape, the relationship is non-linear.
- **Production Fix:** 
  - Add polynomial features (e.g., $x^2$, $x^3$).
  - Apply mathematical transformations to features (e.g., $\log(x)$, $\sqrt{x}$).

### Assumption 2: Independence of Residuals (No Autocorrelation)
- **What it means:** The residuals of one prediction should not depend on the residuals of another. This is particularly critical in time-series or spatial datasets.
- **Diagnostic Plot:** **Residuals vs. Time/Row Index**. If you see a wave-like pattern, or a sequence of consistently positive errors followed by negative errors, the residuals are autocorrelated.
- **Production Fix:**
  - Introduce lagged target variables as features (e.g., $y_{t-1}$).
  - Use time-series-specific models (e.g., ARIMA or temporal features) rather than standard linear regression.

### Assumption 3: Homoscedasticity (Constant Variance of Residuals)
- **What it means:** The variance of the residuals must remain constant across all levels of predictions.
- **Diagnostic Plot:** **Residuals vs. Fitted plot**. Look for a "funnel" or "megaphone" shape where the scatter of residuals spreads out wider as the predicted value increases.
  
  ```
  Heteroscedasticity (Funnel Pattern)
    Residuals
     ^
     |         .     .   .
     |       .   .  .  .
   --+----------------------> Fitted Values
     |       .   .  .  .
     |         .     .   .
     v
  ```
- **Production Fix:**
  - Apply a log-transform or Box-Cox transform to the target variable $y$. This compresses the scale of large values and stabilizes the variance.
  - Use **Weighted Least Squares (WLS)** to penalize errors on high-variance predictions less heavily.

### Assumption 4: Normality of Residuals
- **What it means:** The residuals should be normally distributed. (Note: The *features* do not need to be normal; only the *errors* do).
- **Diagnostic Plot:** **Quantile-Quantile (Q-Q) Plot**. The plotted points (sample quantiles vs. theoretical normal quantiles) should lie along a straight diagonal line. Deviations at the tails indicate heavy-tailed distributions or outliers.
- **Production Fix:**
  - Transform highly skewed features/targets (e.g., using $\log(x)$ or power transforms).
  - Robustly identify and remove or clip extreme training outliers, as they distort the residual distribution.

### Assumption 5: No Multicollinearity
- **What it means:** The features $x_j$ should not be highly correlated with one another.
- **Diagnostic Tool:** 
  - **Correlation Matrix Heatmap:** Look for feature pairs with correlation coefficients $|r| > 0.8$.
  - **Variance Inflation Factor (VIF):** A feature with a $\text{VIF} > 5$ or $10$ indicates high multicollinearity.
- **Production Fix:**
  - Drop one of the highly correlated features (e.g., if you have `temp_celsius` and `temp_fahrenheit`).
  - Use dimensionality reduction (e.g., PCA) to project collinear features into orthogonal space.
  - Apply $L_2$ Regularization (Ridge Regression), which stabilizes coefficients in the presence of multicollinearity.

---

## 2. Feature Selection: What does $p < 0.05$ actually mean?

When training a linear regression model using libraries like `statsmodels` in Python, each feature $x_j$ is assigned a $p$-value.

### Applied Definition
Under the null hypothesis, we assume that feature $x_j$ has no linear relationship with the target $y$ (i.e., its true weight $w_j = 0$). 
A $p$-value of **$< 0.05$** means:
> **"If the feature truly had zero effect, the probability of observing a coefficient as large as ours by pure random chance is less than 5%."**

Therefore, we reject the null hypothesis and keep the feature.

### Production Engineering Limitations of $p$-values
In real-world ML systems, relying blindly on $p < 0.05$ for feature selection is dangerous for two reasons:

1. **The Large-$m$ Effect (Sample Size Bias):**
   If your production dataset has millions of rows ($m \rightarrow \infty$), the standard error of your coefficients shrinks to near zero. Consequently, almost **every feature**—even random noise with a tiny correlation—will have a $p$-value of $0.000$ and register as "statistically significant."
   - *Engineering Rule:* Look at the **effect size** (the actual magnitude of $w_j$) and the business context, not just the $p$-value.
2. **Multicollinearity Inflation:**
   If two features are highly correlated, their standard errors become inflated. This drives their $p$-values up ($p > 0.05$), making them appear "insignificant" individually, even though they are highly predictive.
   - *Engineering Rule:* Solve multicollinearity (e.g., using VIF or Ridge) before using $p$-values for feature pruning.
