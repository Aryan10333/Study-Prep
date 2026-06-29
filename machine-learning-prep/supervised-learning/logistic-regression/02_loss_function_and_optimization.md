# Loss Function and Optimization

To train a logistic regression model, we need a cost function that evaluates the quality of its probability predictions. Standard Mean Squared Error (MSE) fails in classification, requiring us to use **Log-Loss** (also known as Binary Cross-Entropy Loss). This guide outlines the mathematics and optimization of Log-Loss using a concrete click fraud scenario.

---

## 1. The Log-Loss (Binary Cross-Entropy) Cost Function

For a dataset of $m$ training examples, the cost function $J(w,b)$ is defined as:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{w,b}(x^{(i)})\right) + (1-y^{(i)}) \log\left(1-f_{w,b}(x^{(i)})\right) \right]$$

Where:
- $y^{(i)} \in \{0, 1\}$ is the true label.
- $f_{w,b}(x^{(i)}) \in [0, 1]$ is the model's predicted probability of $y=1$.
- $\log$ is the natural logarithm.

### Single-Example Loss Breakdown
We can break down the loss for a single training example $L\left(f_{w,b}(x), y\right)$ as:

$$L\left(f_{w,b}(x), y\right) = -y \log\left(f_{w,b}(x)\right) - (1-y) \log\left(1-f_{w,b}(x)\right)$$

This conditional formula evaluates based on the actual ground truth $y$:
- **If $y = 1$:** The second term disappears, leaving $L = -\log\left(f_{w,b}(x)\right)$.
- **If $y = 0$:** The first term disappears, leaving $L = -\log\left(1 - f_{w,b}(x)\right)$.

---

## 2. Scenario: Click Fraud Classification

You are building a real-time system to classify ad clicks as genuine ($y=0$) or click fraud bots ($y=1$).

### The Training Example Table
Here is a sample of 4 incoming clicks:

| Click ($i$) | True Label ($y$) | Predicted Churn Prob ($f(x)$) | Loss Calculation | Error / Loss Contribution |
| :--- | :--- | :--- | :--- | :--- |
| 1 (Normal Click) | 0 (Genuine) | 0.05 | $-\log(1 - 0.05)$ | **0.051** (Low) |
| 2 (Typical Bot) | 1 (Bot) | 0.90 | $-\log(0.90)$ | **0.105** (Low) |
| 3 (Uncertain Click) | 0 (Genuine) | 0.40 | $-\log(1 - 0.40)$ | **0.510** (Moderate) |
| 4 (Confident Mistake) | **1 (Bot)** | **0.01** | $-\log(0.01)$ | **4.605** (Extreme Penalty) |

### Deconstructing the "Confident Mistake" Penalty
Let's look at the mathematical impact of **Click 4** (the bot click we predicted as only 1% likely to be a bot):
- **Normal click (Click 1) error:**
  $$\text{Loss}_1 = -\log(1 - 0.05) = -\log(0.95) \approx 0.051$$
- **Confident mistake (Click 4) error:**
  $$\text{Loss}_4 = -\log(0.01) \approx 4.605$$

**The Engineering Impact:** The loss for the single confident mistake (4.605) is **90 times larger** than the loss for the correct click (0.051). If your model outputs near-zero probabilities for a true positive during training, this single term will generate massive gradient updates, forcing the weights vector to adjust aggressively.

---

## 3. Why Not Mean Squared Error (MSE) for Classification?

If we plug the non-linear sigmoid prediction $f_{w,b}(x) = g(w \cdot x + b)$ into the quadratic MSE cost function, the resulting loss landscape is **non-convex**. 

Mathematically, because of the non-linear sigmoid activation, the second derivative (Hessian matrix) of this cost function is no longer positive semi-definite. Geometrically, this introduces multiple local minima, saddle points, and flat plateaus.

```text
       MSE Loss Surface (Non-Convex)         Log-Loss Surface (Convex)
            Loss                                  Loss
             ^                                     ^
             |    /\     /\                        |     \       /
             |   /  \___/  \                       |      \_____/
             |  /           \                      |         ^
             +-------------------->                +-------------------->
                Local Minima Traps                     Global Minimum Only
```

- **The Danger:** If we run gradient descent on a non-convex surface, the model is highly likely to get trapped in a suboptimal local minimum, making convergence highly sensitive to initial weights.
- **The Log-Loss Solution:** Log-loss guarantees a **convex** loss surface with a single global minimum, making optimization stable and predictable.

---

## 4. Gradient Descent Optimization

To minimize the Log-Loss cost function, we update parameters iteratively:

$$w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \quad \text{for } j = 1, \dots, n$$
$$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$$

### Vectorized Update
$$w = w - \alpha \frac{1}{m} X^T \left( f_{w,b}(X) - y \right)$$

### Production Training Logs
During training, monitoring the loss logs helps diagnose convergence behavior:

```text
[Epoch 01] Train Loss: 0.6931  (Initial random entropy)
[Epoch 10] Train Loss: 0.4512
[Epoch 20] Train Loss: 0.2319
[Epoch 30] Train Loss: 0.1205  (Convergence achieved)
```
- **Similarity to OLS:** On the surface, the update rule looks identical to linear regression. However, the calculation of the prediction vector is non-linear ($f_{w,b}(X) = g(X w + b)$). This equivalence is a mathematical consequence of the sigmoid derivative canceling out log-loss denominators.
