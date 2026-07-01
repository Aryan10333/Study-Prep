# Gradient Boosting: Optimization & Pseudo-Residual Mechanics

This document details the mathematical formulation of Gradient Boosting, the optimization of loss functions via negative gradients, and the role of learning rate shrinkage.

---

## 1. Gradient Descent in Function Space

While standard gradient descent updates parameter weights ($w \leftarrow w - \alpha \nabla_w J$), Gradient Boosting performs gradient descent in **function space**:
- Instead of learning model weights, it adds new functions (trees) to the model sequentially:
  $$F_t(x) = F_{t-1}(x) + \nu h_t(x)$$
- Each new weak learner $h_t(x)$ is trained to predict the negative gradient of the loss function evaluated at the prior model's predictions.

---

## 2. Gradient Boosting Algorithm (Regression with MSE Loss)

Given a dataset $(x_i, y_i)$ and a differentiable loss function $L(y, F(x))$. Under Mean Squared Error (MSE) loss:
$$L(y_i, F(x_i)) = \frac{1}{2}\left(y_i - F(x_i)\right)^2$$

### Step 1: Initialize Constant Model
Initialize the model with a constant value that minimizes the overall loss:
$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^m L(y_i, \gamma)$$

For MSE loss, taking the derivative with respect to $\gamma$ and setting it to $0$ shows that the optimal starting value is simply the average of the targets:
$$F_0(x) = \frac{1}{m} \sum_{i=1}^m y_i$$

---

### Step 2: Training Trees Sequentially
For $t = 1, 2, \dots, T$:

1. **Calculate Pseudo-Residuals ($r_{it}$):**
   Compute the negative gradient of the loss function with respect to the current predictions:
   $$r_{it} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{t-1}(x)}$$
   
   - *For MSE Loss:*
     $$\frac{\partial}{\partial F(x_i)} \left( \frac{1}{2} (y_i - F(x_i))^2 \right) = -(y_i - F(x_i))$$
     $$r_{it} = y_i - F_{t-1}(x_i)$$
     *(Under MSE, the negative gradients are exactly the raw residuals).*

2. **Fit Weak Learner:** Train a regression tree $h_t(x)$ to fit the pseudo-residuals $r_{it}$ as target features.
3. **Compute Leaf Terminal Values ($\gamma_{jt}$):**
   For each leaf region $R_{jt}$ of tree $h_t$:
   $$\gamma_{jt} = \arg\min_{\gamma} \sum_{x_i \in R_{jt}} L(y_i, F_{t-1}(x_i) + \gamma)$$
   *(For MSE loss, $\gamma_{jt}$ is simply the average of the pseudo-residuals in leaf region $R_{jt}$).*
4. **Update Predictions with Shrinkage ($\nu$):**
   $$F_t(x) = F_{t-1}(x) + \nu h_t(x)$$
   The learning rate shrinkage parameter $\nu$ (typically $0.01$ to $0.1$) scales the predictions of each new tree, preventing the model from overfitting.

---

## 3. Real-World Scenario: Real Estate Housing Price Estimation

### The Problem
We want to predict home sale prices (in thousands of dollars) based on `Square Footage`.
We have a training set of $m = 3$ houses.

---

### Step-by-Step Hand Calculations

#### 1. Initialize Constant Model ($F_0$)
Our training set:

| House ($i$) | Square Footage ($x$) | True Price ($y$) |
|:---:|:---:|:---:|
| 1 | 1000 | 200 |
| 2 | 1500 | 300 |
| 3 | 2000 | 400 |

Initial constant prediction $F_0(x)$:
$$F_0(x) = \text{mean}(200, 300, 400) = 300$$

---

#### 2. Iteration 1 ($t = 1$)

##### A. Compute Pseudo-Residuals ($r_{i1}$)
$$r_{i1} = y_i - F_0(x_i)$$
- **House 1:** $r_{11} = 200 - 300 = -100$
- **House 2:** $r_{21} = 300 - 300 = 0$
- **House 3:** $r_{31} = 400 - 300 = 100$

Our target residuals for the first tree are: `[-100, 0, 100]`

---

##### B. Fit weak learner ($h_1$)
A regression tree splits on the threshold **Square Footage $\le 1250$**:
- **Left Leaf ($R_{11}$):** Contains House 1 (Size 1000).
  $$\gamma_{11} = \text{mean}(-100) = -100$$
- **Right Leaf ($R_{21}$):** Contains Houses 2 and 3 (Sizes 1500, 2000).
  $$\gamma_{21} = \text{mean}(0, 100) = 50$$

---

##### C. Update predictions with shrinkage ($\nu = 0.1$)
$$F_1(x_i) = F_0(x_i) + \nu h_1(x_i)$$
- **House 1:** $F_1(1000) = 300 + 0.1(-100) = 290$ (Closer to true price 200)
- **House 2:** $F_1(1500) = 300 + 0.1(50) = 305$ (Closer to true price 300)
- **House 3:** $F_1(2000) = 300 + 0.1(50) = 305$ (Closer to true price 400)

By adding the first tree, the residual errors shrink, and the predictions begin converging toward the target prices.

---

## 4. Production Realities: Sequential Bottlenecks

A major limitation of Gradient Boosting in high-throughput systems is **sequential training dependencies**:
- Unlike Random Forests, where 500 trees can be trained simultaneously in parallel across 64 CPU cores (using bagging independence), each tree in a Gradient Boosting model must wait for tree $t-1$ to finish training and calculate its pseudo-residuals.
- This serial bottleneck increases overall training time on massive datasets.
