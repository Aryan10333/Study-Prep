# Deep Learning: Loss Functions & Metric Learning

This guide outlines standard loss functions, focal loss optimization for heavy class imbalances, and triplet/contrastive loss for metric representation learning.

---

## 1. Classical Loss Functions & Deployment Criteria

| Loss Function | Formula ($L$) | Target Type | Properties & Trade-offs |
| :--- | :--- | :--- | :--- |
| **Mean Squared Error (MSE / $L_2$)** | $$\frac{1}{m}\sum (y_i - \hat{y}_i)^2$$ | Continuous | penalizes large errors quadratically; highly sensitive to outliers. |
| **Mean Absolute Error (MAE / $L_1$)** | $$\frac{1}{m}\sum |y_i - \hat{y}_i|$$ | Continuous | Constant gradient; highly robust to outliers; non-differentiable at $0$. |
| **Binary Cross-Entropy (BCE)** | $$-\frac{1}{m}\sum [y \ln(p) + (1-y)\ln(1-p)]$$ | Binary Classification | Optimizes log-odds of occurrence. Paired with Sigmoid output. |
| **Categorical Cross-Entropy (CE)** | $$-\frac{1}{m}\sum \sum Y_{ik} \ln(A_{ik})$$ | Multi-class Classification | Maximizes probability of true class. Paired with Softmax output. |

---

## 2. Focal Loss: Handling Severe Class Imbalance

In production datasets (e.g., ad click prediction, anomaly detection), positive instances are extremely rare ($< 1\%$). Under standard BCE, the aggregate loss is dominated by the massive volume of easy-to-classify negative samples.

**Focal Loss** adds a modulating factor $(1 - p_t)^\gamma$ to BCE:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ is the model's predicted probability for the true class.
- $\alpha_t$ balances class weights.
- $\gamma$ is the **focusing parameter** (typically set to $2.0$).

### How it Works:
- **Easy Sample ($p_t = 0.9$):** If the model predicts $90\%$ confidence for a correct sample, the modulating factor is $(1 - 0.9)^2 = 0.01$. The loss contribution is scaled down by **$100\text{x}$**.
- **Hard Sample ($p_t = 0.1$):** If the model is confident in an incorrect prediction, the factor is $(1 - 0.1)^2 = 0.81$. The loss contribution is barely reduced.

*Result:* The optimizer ignores easy background samples, forcing the network to focus on learning hard, rare positive samples.

---

## 3. Metric Learning: Triplet & Contrastive Loss

Metric learning trains a neural network (e.g., a Siamese network) to map inputs into a low-dimensional embedding space where geometric distances correspond to semantic similarity.

### Triplet Loss
We train the network on a triplet consisting of:
- **Anchor ($a$):** A base reference image/text.
- **Positive ($p$):** A different sample of the *same* class.
- **Negative ($n$):** A sample of a *different* class.

The goal is to ensure the anchor is closer to the positive sample than to the negative sample by at least a specified **margin**:

$$L(a, p, n) = \max\left( 0, \, d(a, p) - d(a, n) + \text{margin} \right)$$

Where $d(x, y) = \|f(x) - f(y)\|^2$ is the Euclidean distance in embedding space.

```text
       Before Training:                     After Training:
      [Negative]                           [Negative]
         \
          \ (Far)
           \
  [Anchor]-----(Close)----[Positive]     [Anchor]--[Positive]      [Negative] (Far)
```

- **Loss = 0:** Occurs when $d(a, n) > d(a, p) + \text{margin}$. The anchor-negative distance is sufficiently large, so no parameter updates are made.
- **Loss > 0:** Occurs when the negative sample is too close to the anchor. Backpropagation pushes the positive embedding closer to the anchor and drives the negative embedding farther away.
