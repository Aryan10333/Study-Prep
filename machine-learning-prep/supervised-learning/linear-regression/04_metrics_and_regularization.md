# Metrics and Regularization

To deploy linear regression models safely, we must measure their performance accurately and prevent them from overfitting. This guide compares standard evaluation metrics and details the mechanics of $L_1$ and $L_2$ regularization.

---

## 1. Metrics in Action: MSE vs. RMSE vs. MAE

Evaluating a regression model requires measuring the distance between predictions $f_{w,b}(x)$ and ground-truth values $y$.

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$
- **Characteristics:** Errors are squared. Units are squared (e.g., if predicting housing prices in USD, MSE is in $\text{USD}^2$).
- **Use Case:** Good for mathematical optimization because it is differentiable.

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2}$$
- **Characteristics:** Puts the error back into the original target units.
- **Use Case:** Highly sensitive to outliers. A single error of $100$ will weigh far more heavily than ten errors of $10$. Choose RMSE when large errors are extremely costly (e.g., predicting delivery delay times where a 5-hour delay ruins customer satisfaction far more than five 1-hour delays).

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left| f_{w,b}(x^{(i)}) - y^{(i)} \right|$$
- **Characteristics:** Linear error scale. Outliers are not disproportionately penalized.
- **Use Case (Business Context):** Choose MAE when outliers represent normal business variations and should not warp the general baseline performance. E.g., predicting daily retail store sales where a massive holiday spike shouldn't skew the model's performance metrics for standard operating days.

---

## 2. $R^2$ vs. Adjusted $R^2$

### The $R^2$ Metric (Coefficient of Determination)
$R^2$ measures the proportion of variance in the target variable $y$ that is explained by the model's features:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{m} \left( y^{(i)} - f_{w,b}(x^{(i)}) \right)^2}{\sum_{i=1}^{m} \left( y^{(i)} - \bar{y} \right)^2}$$

Where $\bar{y}$ is the mean of the target variable $y$.

### Why Standard $R^2$ Deceives You
If you add a random column of noise (e.g., `random_number_generator()`) to your training data, standard $R^2$ will **never decrease** and will almost always **increase**.
- **Reason:** The Ordinary Least Squares (OLS) algorithm will find *some* non-zero weight $w_{\text{noise}}$ to fit the noise in the training set. This reduces the training residual sum of squares ($\text{SS}_{\text{res}}$), artificially inflating $R^2$ without adding real predictive power.

### The Fix: Adjusted $R^2$
Adjusted $R^2$ penalizes feature bloat by factoring in the number of features $n$ relative to the sample size $m$:

$$\text{Adjusted } R^2 = 1 - \left( 1 - R^2 \right) \frac{m - 1}{m - n - 1}$$

- **How it works:** If you add a useless feature, $n$ increases, which decreases the denominator $m - n - 1$. This increases the multiplier $\frac{m - 1}{m - n - 1}$. If the improvement in $R^2$ is not large enough to offset this penalty, **Adjusted $R^2$ will decrease**.
- **Production Implication:** Always use Adjusted $R^2$ (or cross-validation) in feature selection to verify if a new feature actually adds value.

---

## 3. Regularization: Preventing Overfitting

Regularization adds a penalty term to the cost function to prevent weights $w$ from growing too large.

### Ridge Regression ($L_2$ Regularization)
Ridge adds the sum of squared weights to the cost function:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- **Behavior:** Shrinks all weights $w_j$ toward zero, but **never forces them to exactly zero**.
- **When to use:** Use when you have many features, most of which contribute a small amount of predictive signal, or when features are highly correlated (multicollinearity).

### Lasso Regression ($L_1$ Regularization)
Lasso adds the sum of absolute weights to the cost function:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} |w_j|$$

- **Behavior:** Shrinks weights to **exactly zero**, effectively removing features from the model.
- **When to use:** Use when you want a sparse model for high interpretability, or when you suspect only a small subset of your features are actually predictive.

---

## 4. Why Lasso Drops Weights to 0 While Ridge Shrinks Them

This is a classic ML engineering interview question. There are two ways to explain the intuition:

### 1. Geometric Intuition (Constraint Boundaries)
We can think of regularization as minimizing the MSE cost subject to a budget constraint:
- Ridge constraint: $\sum_{j=1}^{n} w_j^2 \le t$ (a circle/hypersphere in parameter space).
- Lasso constraint: $\sum_{j=1}^{n} |w_j| \le t$ (a diamond/polytope with sharp corners along the axes).

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
- In **Ridge**, the ellipse is highly likely to touch the circle at a smooth point where neither weight is zero.
- In **Lasso**, because of the sharp corners on the axes, the ellipse is highly likely to intersect the diamond at a **corner** (where one of the coordinates is exactly 0).

### 2. Gradient / Derivative Intuition
Let's look at the gradient of the penalty terms with respect to weight $w_j$:

- **Ridge ($L_2$) Penalty Gradient:** $\frac{\partial}{\partial w_j} (w_j^2) = 2 w_j$
  As $w_j$ approaches $0$, the gradient approaches $0$. The regularization pressure becomes weaker and weaker. Thus, the weight is merely shrunk but rarely reaches absolute zero.
  
- **Lasso ($L_1$) Penalty Gradient:** $\frac{\partial}{\partial w_j} (|w_j|) = \text{sign}(w_j) \in \{-1, 1\}$
  The regularization force is constant ($\lambda/m$) regardless of how small the weight is. Even if $w_j = 0.0001$, it is pulled down with the same constant force, driving it to **exactly 0**.
