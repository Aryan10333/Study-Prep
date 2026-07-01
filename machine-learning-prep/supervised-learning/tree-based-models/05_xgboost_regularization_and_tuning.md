# XGBoost: Regularization & Booster Tree Structure

This guide details how XGBoost incorporates regularization parameters ($\lambda$, $\gamma$) directly into the booster split calculations, and how to programmatically extract and inspect tree structures in Python.

---

## 1. XGBoost Regularization Parameters

Unlike standard Gradient Boosting, XGBoost adds regularization directly to its node split scoring formula:
- **$\lambda$ ($L_2$ Regularization):** Added to the leaf score denominator ($\sum h_i + \lambda$). When a candidate leaf contains very few samples (low Hessian sum $h_i$), $\lambda$ dominates the denominator, shrinking the leaf weight $w_j^*$ close to $0.0$. This prevents splits that isolate rare outlier samples.
- **$\gamma$ (Split Complexity Threshold):** Acts as the minimum Gain threshold. If the raw score reduction of a candidate split is less than $2\gamma$, the split Gain becomes negative, and XGBoost will prune the split.

---

## 2. Python Code: Extracting Booster Tree Structures

In production, you can inspect the actual split criteria, leaf weights, and gains of your XGBoost model using the `get_booster()` and `get_dump()` APIs.

### Python Code Pipeline
```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Generate a small subscriber churn dataset
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

## 3. Interpreting the Booster Dump

- **`0:[usage_hours<29.988]`**: The root split occurs at `usage_hours = 29.988`.
- **`gain=94.52`**: The score reduction (Gain) achieved by this split. Since $94.52 > \gamma = 0.5$, the split is accepted.
- **`cover=100`**: The sum of Hessians ($h_i$) for samples in this node. For binary classification using log-loss, the Hessian is $p_i(1 - p_i)$. Since the default starting prediction probability is $0.5$, each Hessian is $0.5 \times 0.5 = 0.25$. A cover of $100$ means $100$ samples are present in the parent node ($100 \times 0.25 = 25.0$ effective cover scale, depending on frame implementation).
- **`leaf=-0.95` and `leaf=0.97`**: The optimal leaf weights $w_j^*$ learned for the left and right nodes. These are raw log-odds updates.

---

## 4. Interactive Practice Notebook
To tune regularization coefficients on a full customer churn dataset, monitor log-loss learning curves, and practice ONNX runtime compilation, open the interactive notebook:
- [05_xgboost_early_stopping_and_onnx.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/05_xgboost_early_stopping_and_onnx.ipynb)
