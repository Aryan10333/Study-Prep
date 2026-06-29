# Metrics and Regularization

To deploy linear regression models safely, we must measure their performance accurately and prevent them from overfitting. This guide compares standard evaluation metrics and details the mathematics and intuition behind Lasso ($L_1$) and Ridge ($L_2$) regularization.

---

## 1. Metrics in Action: MSE vs. RMSE vs. MAE

Evaluating a regression model requires measuring the distance between predictions $f_{w,b}(x)$ and ground-truth values $y$.

- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left| y^{(i)} - \hat{y}^{(i)} \right|$$

---

## 2. Logistics Scenario: E-Commerce Shipping Duration

You are building an ML pipeline to predict **shipping delivery time (in days)** for an e-commerce platform.

### The Dataset with a Customs Outlier
Here is a sample of 5 shipments:

| Shipment ($i$) | True Duration ($y$, days) | Predicted Duration ($\hat{y}$, days) | Error ($y - \hat{y}$) | Absolute Error | Squared Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3.0 | 2.5 | 0.5 | 0.5 | 0.25 |
| 2 | 2.0 | 3.0 | -1.0 | 1.0 | 1.00 |
| 3 | 4.0 | 3.5 | 0.5 | 0.5 | 0.25 |
| 4 | 3.0 | 2.8 | 0.2 | 0.2 | 0.04 |
| 5 (Customs Delay) | **45.0** | **5.0** | **40.0** | **40.0** | **1600.00** |

### Step-by-Step Metric Calculations
Let's calculate the metrics explicitly on this table:
- **MSE:**
  $$\text{MSE} = \frac{0.25 + 1.0 + 0.25 + 0.04 + 1600.00}{5} = \frac{1601.54}{5} \approx 320.31 \text{ days}^2$$
- **RMSE:**
  $$\text{RMSE} = \sqrt{320.31} \approx 17.90 \text{ days}$$
- **MAE:**
  $$\text{MAE} = \frac{0.5 + 1.0 + 0.5 + 0.2 + 40.0}{5} = \frac{42.2}{5} \approx 8.44 \text{ days}$$

### The Business Context
- **When to choose MAE ($8.44$ days):** If reporting typical deviations to customer support. For 4 out of 5 shipments, the model was accurate within 1 day. MAE is robust and does not allow the customs anomaly to distort baseline metrics.
- **When to choose RMSE ($17.90$ days):** If planning warehouse logistics buffers. Because RMSE squares errors, the 40-day error dominates, forcing the system to account for shipping bottleneck risks.

---

## 3. $R^2$ vs. Adjusted $R^2$

### Standard $R^2$ (Coefficient of Determination)
$R^2$ measures the proportion of variance in target $y$ explained by the features:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2}{\sum_{i=1}^{m} \left( y^{(i)} - \bar{y} \right)^2}$$

### Why Standard $R^2$ Deceives You
If you add a random column of noise (e.g., `random_numbers()`) to your training data, standard $R^2$ will **never decrease** and almost always **increase**.
- **Reason:** The OLS optimizer will find a non-zero weight $w_{\text{noise}}$ to fit the noise in the training set. This reduces the training residual sum of squares ($\text{SS}_{\text{res}}$), artificially inflating $R^2$ without adding real predictive power.

### The Fix: Adjusted $R^2$
Adjusted $R^2$ penalizes feature bloat by factoring in the feature count $n$ relative to sample size $m$:

$$\text{Adjusted } R^2 = 1 - \left( 1 - R^2 \right) \frac{m - 1}{m - n - 1}$$

- **How it works:** If you add a useless feature, $n$ increases, decreasing the denominator $m - n - 1$. This increases the multiplier fraction. If the improvement in $R^2$ is not large enough to offset this penalty, **Adjusted $R^2$ will decrease**.

---

## 4. Regularization: Preventing Overfitting

Regularization adds a penalty term to the cost function to restrict weights from growing too large.

### Ridge Regression ($L_2$ Regularization)
Ridge adds the sum of squared weights to the cost:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

### Lasso Regression ($L_1$ Regularization)
Lasso adds the sum of absolute weights to the cost:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} |w_j|$$

---

## 5. Why Lasso Drops Weights to 0 While Ridge Shrinks Them

### 1. Geometric Intuition (Constraint Boundaries)
We can think of regularization as minimizing the MSE cost subject to a budget constraint:
- **Ridge ($L_2$):** $\sum_{j=1}^{n} w_j^2 \le t$ (a circular/spherical boundary).
- **Lasso ($L_1$):** $\sum_{j=1}^{n} |w_j| \le t$ (a diamond boundary with sharp corners along the axes).

```
   Ridge Constraint (L2)              Lasso Constraint (L1)
           w2                                  w2
           |   _.._                            |   /\
           | .      .                          |  /  \
        ---+--------+--- w1                 ---+--/----\-- w1
           | .      .                          |  \  /
           |   `''`                            |   \/
           |                                   |
```

When OLS optimization contours (ellipses representing levels of $J(w,b)$) expand outward to touch the constraint boundary:
- In **Ridge**, the ellipse is highly likely to touch the circle at a smooth tangent point where neither weight is zero.
- In **Lasso**, because of the sharp corners on the axes, the ellipse is highly likely to intersect the diamond at a **corner** (where one of the feature weights is exactly 0).

### 2. Gradient / Derivative Intuition
Let's look at the gradient of the penalty terms with respect to weight $w_j$:

- **Ridge ($L_2$) Penalty Gradient:** $\frac{\partial}{\partial w_j} (w_j^2) = 2 w_j$
  As $w_j$ approaches $0$, the gradient approaches $0$. The regularization pressure becomes weaker. Thus, the weight is merely shrunk but rarely reaches absolute zero.
- **Lasso ($L_1$) Penalty Gradient:** $\frac{\partial}{\partial w_j} (|w_j|) = \text{sign}(w_j) \in \{-1, 1\}$
  The regularization force is constant ($\lambda/m$) regardless of how small the weight is. Even if $w_j = 0.0001$, it is pulled down with the same constant force, driving it to **exactly 0**.

---

## 6. Regularization under Multicollinearity (Highly Correlated Features)

Suppose we include both `miles_to_destination` and `kilometers_to_destination` as features in our shipping model.

| Feature / Behavior | Ridge ($L_2$) Regularization | Lasso ($L_1$) Regularization |
| :--- | :--- | :--- |
| **Penalty Geometry** | Circular boundary ($w_1^2 + w_2^2 \le t$) | Diamond boundary ($|w_1| + |w_2| \le t$) |
| **Weight Shrinkage** | Shrinks weights smoothly: $w \rightarrow 0.001$. | Shrinks weights to **exactly zero**: $w \rightarrow 0$. |
| **Handling Correlated Features** | Splits the weight coefficient evenly between `miles` and `kilometers`. Both stay in the model with small weights. | Selects one feature (e.g., `miles`) and drops the other (`kilometers` weight set to exactly $0.0$). |
| **Embedded Feature Selection** | **No.** All features remain in the model, requiring computation. | **Yes.** Drops uninformative or redundant features, reducing the prediction pipeline's computational footprint. |
