# Gradient Boosting: Residual Fitting & Stage Predictions

This guide explains how Gradient Boosting models fit continuous targets by training weak regression trees to predict the residual errors of the prior ensemble, walks through a manual residual fitting calculation, and shows how to track this in Python.

---

## 1. Residual Fitting: Fitting the Error

Unlike AdaBoost, which re-weights data points, Gradient Boosting treats residual errors as target values. 

Under Mean Squared Error (MSE) loss, the negative gradient of the loss function is simply the raw residual (prediction error):
$$-\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = y_i - F(x_i) = \text{residual } r_{i}$$

By fitting each new tree to the residuals of the previous trees, the model takes a step in the direction of steepest descent, minimizing the overall loss.

### The Algorithm Step-by-Step
1. **Base Prediction ($F_0(x)$):** Initialize the ensemble prediction with a constant value (for MSE loss, this is the mean of the target values).
2. **Compute Residuals ($r_{it}$):** Compute the current residual for each sample: $r_{it} = y_i - F_{t-1}(x_i)$.
3. **Train Weak Tree ($h_t(x)$):** Train a regression tree $h_t(x)$ using the residuals $r_{it}$ as the target variable.
4. **Update Ensemble ($F_t(x)$):** Update the predictions using a learning rate (shrinkage) $\nu$:
   $$F_t(x) = F_{t-1}(x) + \nu \cdot h_t(x)$$

---

## 2. Step-by-Step Hand Calculations: Housing Prices

Let's predict house price ($y$, in thousands) based on Square Footage ($x_1$). We have $m = 3$ houses.

---

### Step 1: Base Initialization ($F_0$)
Initialize the prediction with the mean of the house prices:
- $y = [200, 300, 400]$
- $F_0 = \text{mean}(y) = 300$

| House ($i$) | Sq Footage ($x_1$) | Price ($y$) | $F_0$ | Initial Residual ($r_{i1} = y_i - F_0$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1000 | 200 | 300 | -100 |
| 2 | 1500 | 300 | 300 | 0 |
| 3 | 2000 | 400 | 300 | 100 |

---

### Step 2: Fit Regression Tree ($h_1$) on Residuals $r_{i1}$
Train a single regression split $h_1(x)$ on the target variable $r_{i1} = [-100, 0, 100]$ using the threshold **Sq Footage $\le 1250$**:
- For House 1 ($x_1 = 1000 \le 1250$): Falls into Left Leaf. Prediction $h_1(1000) = -100$.
- For Houses 2 & 3 ($x_1 > 1250$): Fall into Right Leaf. Average residual is:
  $$\text{Mean}(0, 100) = 50$$
  So, prediction $h_1(1500) = 50$, $h_1(2000) = 50$.

---

### Step 3: Update Ensemble Prediction ($F_1$)
Apply learning rate shrinkage $\nu = 0.1$:
- **House 1:**
  $$F_1(1000) = F_0 + 0.1 \cdot h_1(1000) = 300 + 0.1(-100) = 290$$
- **House 2:**
  $$F_1(1500) = F_0 + 0.1 \cdot h_1(1500) = 300 + 0.1(50) = 305$$
- **House 3:**
  $$F_1(2000) = F_0 + 0.1 \cdot h_1(2000) = 300 + 0.1(50) = 305$$

---

### Step 4: Compute Next Residuals ($r_{i2}$)
Calculate the new residual values:
- House 1: $r_{12} = y_1 - F_1(1000) = 200 - 290 = -90$
- House 2: $r_{22} = y_2 - F_1(1500) = 300 - 305 = -5$
- House 3: $r_{32} = y_3 - F_1(2000) = 400 - 305 = 95$

**Result:** The residuals have shrunk from $[-100, 0, 100]$ to $[-90, -5, 95]$. The next tree $h_2(x)$ will be trained to fit these new residuals.

---

## 3. Python Code Walkthrough: Staged Predictions

We can step through the predictions of each tree in a `GradientBoostingRegressor` using the `staged_predict` generator. This allows us to inspect how residuals shrink stage-by-stage.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Simple house pricing dataset matching our calculation
df = pd.DataFrame({
    'sq_ft': [1000, 1500, 2000],
    'price': [200, 300, 400]
})

X = df[['sq_ft']]
y = df['price']

# Train Gradient Boosting Regressor with learning rate = 0.1
model = GradientBoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=1, random_state=42)
model.fit(X, y)

F_0 = np.mean(y)
print(f"Base Prediction (F_0): {F_0:.2f}")
print(f"Initial Residuals: {list(y - F_0)}\n")

for t, F_t in enumerate(model.staged_predict(X)):
    print(f"Ensemble predictions after Tree {t+1} (F_{t+1}):")
    print(f"  Predictions: {list(np.round(F_t, 2))}")
    print(f"  New Residuals: {list(np.round(y - F_t, 2))}")
```

### Expected Console Output
```text
Base Prediction (F_0): 300.00
Initial Residuals: [-100.0, 0.0, 100.0]

Ensemble predictions after Tree 1 (F_1):
  Predictions: [290.0, 305.0, 305.0]
  New Residuals: [-90.0, -5.0, 95.0]
Ensemble predictions after Tree 2 (F_2):
  Predictions: [281.0, 309.5, 309.5]
  New Residuals: [-81.0, -9.5, 90.5]
Ensemble predictions after Tree 3 (F_3):
  Predictions: [272.9, 313.55, 313.55]
  New Residuals: [-72.9, -13.55, 86.45]
```

### Diagnostic Visual (Staged Residual Reduction)
The 3-panel plot below shows how the ensemble prediction line (red) shifts from a flat line (Tree 1) toward the true target patterns, while the residual error spreads shrink with each boosting step:

![Staged Residual Reduction](images/staged_residual_reduction.png)

---

## 4. Interactive Practice Notebook
To run convergence curves and work with multi-estimator residual reductions, open the interactive notebook:
- [05_xgboost_early_stopping_and_onnx.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/05_xgboost_early_stopping_and_onnx.ipynb)
