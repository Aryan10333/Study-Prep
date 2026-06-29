# Overfitting vs. Underfitting Profiles

When building machine learning pipelines, we must balance model complexity to avoid two failure modes: **underfitting** and **overfitting**. This guide details these profiles, analyzes parametric vs. non-parametric tradeoffs, and maps the degrees of freedom frontier using a concrete query latency estimation scenario.

---

## 1. Scenario: Database Query Latency Estimation

You are building a system to predict **database query latency (in milliseconds)** based on the size of the query (number of rows retrieved).

### The Training Dataset ($m = 4$ database operations)

| Query ($i$) | Rows ($x$, in thousands) | Latency ($y$, ms) |
| :--- | :--- | :--- |
| 1 | 1.0 | **10.0** |
| 2 | 2.0 | **30.0** |
| 3 | 3.0 | **20.0** |
| 4 | 4.0 | **80.0** |

---

## 2. The Underfitting Profile: High Bias, Low Variance

Underfitting occurs when a model is too simple to capture the underlying structure of the data.

### The Mathematics & Behavior
- **High Structural Bias:** The functional form we assume is far more rigid than the true function: $\text{Bias}(\hat{f}(x)) \gg 0$.
- **Model Behavior:** High training error and high test error.
- **Low Variance:** If you retrain the model on different datasets, the weights will barely change: $\text{Var}(\hat{f}(x)) \approx 0$.

### Pricing Scenario Context
We train a simple parametric linear model: $\hat{f}(x) = w x + b$.
- The OLS solver finds a line (e.g., $w=20, b=-15$).
- The training predictions are far from the actual targets. For a test query of $4,000$ rows ($x=4.0$), the model predicts a latency of $65$ ms, compared to the true value of $80$ ms. It fails to fit the curve.

---

## 3. The Overfitting Profile: Low Bias, High Variance

Overfitting occurs when a model is overly complex and starts learning the random noise in the training set rather than just the signal.

### The Mathematics & Behavior
- **High Variance:** The model has too much flexibility, over-indexing on whatever random fluctuations are present in the training set: $\text{Var}(\hat{f}(x)) \gg 0$.
- **Model Behavior:** Low training error (often 0) but high test error.
- **Low Bias:** The model is flexible enough to represent the true function: $\text{Bias}(\hat{f}(x)) \approx 0$.

### Pricing Scenario Context: 1-Nearest Neighbor (1-NN)
We train a non-parametric **1-Nearest Neighbor (1-NN)** model: the predicted latency for a new query is equal to the latency of the closest query in the training set.

#### Why 1-NN Achieves Zero Training Error
For any point already in the training dataset, its nearest neighbor is itself (distance = 0):
- If we query the model with $x = 2.0$, it returns the exact latency of Query 2: **30.0 ms**.
- The prediction matches the target perfectly, yielding a **Training MSE = 0.0**.

#### Why 1-NN Fails on Test Data (High Variance)
Suppose a background system backup occurred during Query 4, making its latency spike to **80.0 ms** (outlier noise).
- If we run a test query at $x_{\text{test}} = 3.999$, the nearest neighbor is $x_4 = 4.0$. The model predicts **80.0 ms**.
- In reality, a query of $3,999$ rows under normal conditions takes **50.0 ms**. 
- Our prediction error is $80.0 - 50.0 = 30.0$ ms. The model memorized the backup noise and mapped it onto our test predictions, yielding high test error.

---

## 4. Parametric vs. Non-Parametric Tradeoffs

Following the ISLR Chapter 2 framework, we compare our two latency estimators:

| Trade-off Dimension | Linear Model (Parametric) | 1-NN (Non-Parametric) |
| :--- | :--- | :--- |
| **Structural Flexibility** | Low (rigid straight line) | High (can model any step boundary) |
| **Active Degrees of Freedom** | Fixed at $n+1$ parameters | Proportional to training set size $m$ |
| **Training Error** | High (underfitting) | **0.0** (fits every point perfectly) |
| **Sensitivity to Noise** | Low (averages out outliers) | **High** (memorizes outliers and noise) |
| **Generalization Error (Test)** | High (due to Bias) | High (due to Variance) |
| **Data Requirement** | Small (stable with few rows) | Large (requires high density to smooth boundaries) |

---

## 5. Mapping the Tradeoff Frontier

As you increase the **flexibility** of your model, bias decreases while variance increases. This yields the classical U-shaped expected test MSE curve:

```
    Error / MSE
     ^
     |        \            /   <-- Expected Test MSE (U-Shape)
     |         \  _..---''
     |          \/        
     |          /\         .---'' <-- Variance
     |         /  `''''---'
     |        /___________            <-- Bias^2
     +-----------------------------------> Flexibility / Complexity
            Underfitting     Overfitting
```
- **Optimal Complexity:** The minimum point of the expected test MSE curve, representing the sweet spot where bias and variance are perfectly balanced.
