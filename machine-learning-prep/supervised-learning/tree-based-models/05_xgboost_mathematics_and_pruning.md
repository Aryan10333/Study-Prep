# XGBoost: Objective Function & Pruning Math

This document details the mathematical derivation of the XGBoost objective function, second-order Taylor expansion approximations, leaf weight calculations, and split pruning criteria.

---

## 1. The Regularized Objective Function

Unlike standard Gradient Boosting, XGBoost incorporates $L_1$ and $L_2$ regularization directly into the step-wise objective function to control model complexity and prevent overfitting:

$$\text{Obj}^{(t)} = \sum_{i=1}^m L\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

Where:
- $L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))$ is the loss of the model prediction after adding the $t$-th tree $f_t(x_i)$.
- $\Omega(f_t)$ is the complexity penalty of the $t$-th tree, defined as:
  $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
  Where $T$ is the number of leaf nodes, $w_j$ is the weight (prediction value) of leaf node $j$, $\gamma$ is the split complexity threshold, and $\lambda$ / $\alpha$ are the $L_2$ and $L_1$ leaf weight regularization coefficients.

---

## 2. Second-Order Taylor Expansion

To make the objective solvable for arbitrary loss functions, XGBoost applies a second-order Taylor expansion approximation to the loss function:
$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2$$

Substituting this into the objective:
$$\text{Obj}^{(t)} \approx \sum_{i=1}^m \left[ L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

Where $g_i$ (gradient) and $h_i$ (Hessian) are the first and second-order derivatives of the loss function with respect to the prior prediction:
$$g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$$

### Simplifying the Objective
Removing the constant term $L(y_i, \hat{y}_i^{(t-1)})$ (which does not depend on the current tree $f_t$):
$$\tilde{\text{Obj}}^{(t)} = \sum_{i=1}^m \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

We can rewrite the sum over samples by grouping them according to the leaf nodes they fall into. Let $I_j = \{i | x_i \in \text{leaf } j\}$ represent the set of samples in leaf $j$:
$$\tilde{\text{Obj}}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T$$

---

## 3. Optimal Leaf Weights and Score Calculation

For a fixed tree structure, we find the optimal leaf weight $w_j^*$ by taking the partial derivative of the simplified objective with respect to $w_j$, setting it to $0$:
$$\frac{\partial \tilde{\text{Obj}}^{(t)}}{\partial w_j} = \left(\sum_{i \in I_j} g_i\right) + \left(\sum_{i \in I_j} h_i + \lambda\right) w_j = 0$$

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Substituting $w_j^*$ back into the objective function yields the optimal score (which measures the quality of the tree structure):
$$\text{Obj}^* = -\frac{1}{2} \sum_{j=1}^T \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

---

## 4. Split Finding and Gain Formula

To find the optimal split, we compute the reduction in objective value (the Gain) achieved by splitting a parent node into left ($L$) and right ($R$) children:

$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\)2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

Where:
- The first term represents the score of the left child.
- The second term represents the score of the right child.
- The third term represents the score of the undivided parent.
- $\gamma$ is the complexity penalty for creating a new split.

---

## 5. Real-World Scenario: Streaming Service Customer Churn

### The Problem
We are predicting subscriber cancellations (`1` vs. `0`).
Let's evaluate a candidate split in our tree:
- Let the $L_2$ regularization coefficient be $\lambda = 0.2$.
- Let the complexity penalty be $\gamma = 1.0$.

Suppose the parent node contains $4$ users, resulting in the following gradients ($g_i$) and Hessians ($h_i$):

| User ($i$) | True Churn ($y$) | Pred Probability ($\hat{p}$) | Gradient ($g_i = \hat{p}_i - y_i$) | Hessian ($h_i = \hat{p}_i(1-\hat{p}_i)$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 0.2 | -0.8 | 0.16 |
| 2 | 1 | 0.2 | -0.8 | 0.16 |
| 3 | 0 | 0.2 | 0.2 | 0.16 |
| 4 | 0 | 0.2 | 0.2 | 0.16 |

---

### Step-by-Step Split Calculations

#### 1. Evaluate Parent Node Score
- $\sum g_i = -0.8 - 0.8 + 0.2 + 0.2 = -1.2$
- $\sum h_i = 0.16 \times 4 = 0.64$

$$S_{\text{parent}} = \frac{(-1.2)^2}{0.64 + 0.2} = \frac{1.44}{0.84} \approx 1.7143$$

---

#### 2. Evaluate Left Child Node (Users 1 & 2: Churners)
- $\sum g_L = -0.8 - 0.8 = -1.6$
- $\sum h_L = 0.16 \times 2 = 0.32$

$$S_{\text{left}} = \frac{(-1.6)^2}{0.32 + 0.2} = \frac{2.56}{0.52} \approx 4.9231$$

---

#### 3. Evaluate Right Child Node (Users 3 & 4: Active Users)
- $\sum g_R = 0.2 + 0.2 = 0.4$
- $\sum h_R = 0.16 \times 2 = 0.32$

$$S_{\text{right}} = \frac{(0.4)^2}{0.32 + 0.2} = \frac{0.16}{0.52} \approx 0.3077$$

---

#### 4. Compute Split Gain
$$\text{Gain} = \frac{1}{2} \left( S_{\text{left}} + S_{\text{right}} - S_{\text{parent}} \right) - \gamma$$
$$\text{Gain} = \frac{1}{2} \left( 4.9231 + 0.3077 - 1.7143 \right) - 1.0 = \frac{1}{2} \left( 3.5165 \right) - 1.0 = 1.7583 - 1.0 = 0.7583$$

Since the Gain is **positive** ($0.7583 > 0$), splitting the node reduces the objective loss, and the split is accepted.

---

### Post-hoc Pruning
Suppose we set $\gamma = 2.0$ instead of $1.0$:
$$\text{Gain} = 1.7583 - 2.0 = -0.2417$$
Since the Gain is now **negative**, the split is rejected and pruned.

XGBoost builds trees to their maximum depth first, then works backward from the bottom up to prune splits that do not meet the minimum gain threshold $\gamma$.
