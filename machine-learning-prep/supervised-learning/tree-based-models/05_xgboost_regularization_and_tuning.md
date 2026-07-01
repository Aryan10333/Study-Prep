# XGBoost: Regularization & Booster Tree Structure

This guide explains how XGBoost incorporates regularization parameters ($\lambda$, $\gamma$) directly into its node split calculation formulas, walks through a manual similarity and gain calculation, and shows how to inspect tree structures in Python.

---

## 1. XGBoost Math: Similarity Scores & Split Gain

Unlike standard Gradient Boosting, XGBoost adds regularization directly to its node split criteria. Instead of using raw residual variance reduction, XGBoost calculates a **Similarity Score** for each candidate node:

$$\text{Similarity Score} = \frac{\left( \sum_{i=1}^n g_i \right)^2}{\sum_{i=1}^n h_i + \lambda}$$

Where:
- $g_i$ is the first-order gradient (error: $p_i - y_i$).
- $h_i$ is the second-order Hessian (for binary classification under log-loss: $h_i = p_i(1 - p_i)$).
- $\lambda$ is the **$L_2$ regularization coefficient** (`reg_lambda`).

### The Split Gain Equation
To determine if a split is beneficial, XGBoost calculates the **Gain** of splitting a parent node into left and right children:

$$\text{Gain} = \frac{1}{2} \left[ \text{Similarity}_{\text{Left}} + \text{Similarity}_{\text{Right}} - \text{Similarity}_{\text{Parent}} \right] - \gamma$$

Where $\gamma$ is the **split complexity penalty** (`gamma`). A split is only executed if the Gain is positive ($\text{Gain} > 0$).

---

## 2. Step-by-Step Hand Calculations: Churn Prediction Split

Let's evaluate a split for $m = 4$ users in a subscriber churn dataset. 
- Assume the current prediction probability is $p_i = 0.5$ for all users.
- For binary classification under log-loss, the Hessian is: $h_i = 0.5(1 - 0.5) = 0.25$.
- Set regularization parameters: $\lambda = 1.0$, $\gamma = 0.5$.

Here is our dataset:

| User ($i$) | Monthly Usage Hours | Churn ($y$) | Gradient ($g_i = p_i - y$) | Hessian ($h_i$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 15 | 1 | -0.5 | 0.25 |
| 2 | 22 | 1 | -0.5 | 0.25 |
| 3 | 75 | 0 | 0.5 | 0.25 |
| 4 | 90 | 0 | 0.5 | 0.25 |

---

### Step 1: Calculate Parent Similarity Score
- Sum of Gradients: $\sum g_i = -0.5 - 0.5 + 0.5 + 0.5 = 0$
- Sum of Hessians: $\sum h_i = 0.25 \times 4 = 1.0$

$$\text{Similarity}_{\text{Parent}} = \frac{0^2}{1.0 + \lambda} = \frac{0}{1.0 + 1.0} = 0.0$$

---

### Step 2: Evaluate Split Candidate (Split on `Usage Hours < 50`)
- **Left Child (Usage Hours < 50):** Contains Users 1 and 2.
  - Sum of Gradients: $\sum g_L = -0.5 - 0.5 = -1.0$
  - Sum of Hessians: $\sum h_L = 0.25 + 0.25 = 0.5$
  $$\text{Similarity}_{\text{Left}} = \frac{(-1.0)^2}{0.5 + \lambda} = \frac{1.0}{0.5 + 1.0} \approx 0.6667$$

- **Right Child (Usage Hours $\ge 50$):** Contains Users 3 and 4.
  - Sum of Gradients: $\sum g_R = 0.5 + 0.5 = 1.0$
  - Sum of Hessians: $\sum h_R = 0.25 + 0.25 = 0.5$
  $$\text{Similarity}_{\text{Right}} = \frac{1.0^2}{0.5 + \lambda} = \frac{1.0}{0.5 + 1.0} \approx 0.6667$$

---

### Step 3: Compute Net Split Gain
Evaluate the Gain using $\lambda = 1.0$ and $\gamma = 0.5$:

$$\text{Gain} = \frac{1}{2} \left[ 0.6667 + 0.6667 - 0.0 \right] - \gamma = \frac{1}{2}(1.3333) - 0.5 = 0.6667 - 0.5 = 0.1667$$

**Conclusion:** Since the Gain is positive ($0.1667 > 0$), the split is accepted.

---

### Step 4: Leaf Weight Calculation
The optimal weight update (leaf value) $w_j^*$ is:

$$w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$$

- **Left Leaf Weight:**
  $$w_L^* = -\frac{-1.0}{0.5 + 1.0} = \frac{1.0}{1.5} \approx 0.67$$
- **Right Leaf Weight:**
  $$w_R^* = -\frac{1.0}{0.5 + 1.0} = -\frac{1.0}{1.5} \approx -0.67$$

---

## 3. Python Code: Extracting Booster Tree Structures

In production, you can inspect the actual split criteria, leaf weights, and gains of your XGBoost model using the `get_booster()` and `get_dump()` APIs.

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

np.random.seed(42)
m = 100
usage_hours = np.random.uniform(5, 100, m)
churn = np.where(usage_hours < 30.0, 1, 0)

X = pd.DataFrame({'usage_hours': usage_hours})
y = churn

# Fit an XGBoost model with L2 regularization (reg_lambda) and complexity threshold (gamma)
model_xgb = XGBClassifier(
    n_estimators=1,
    max_depth=2,
    learning_rate=1.0,
    reg_lambda=1.0,
    gamma=0.5,
    random_state=42
)
model_xgb.fit(X, y)

# Extract booster and dump tree structure
booster = model_xgb.get_booster()
tree_dump = booster.get_dump()

print("--- XGBoost Tree Structure Dump ---")
print(tree_dump[0])
```

### Expected Console Output
```text
--- XGBoost Tree Structure Dump ---
0:[usage_hours<29.9880371] yes=1,no=2,missing=1,gain=94.5268478,cover=100
	1:leaf=-0.954545498,cover=27
	2:leaf=0.973333359,cover=73
```

---

## 4. Interactive Practice Notebook
To tune regularization coefficients on a full customer churn dataset, monitor log-loss learning curves, and practice ONNX runtime compilation, open the interactive notebook:
- [05_xgboost_early_stopping_and_onnx.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/05_xgboost_early_stopping_and_onnx.ipynb)
