# Production Realities: Multiclass and Imbalance

Deploying classification systems requires extending binary models to handle multiple classes, managing class imbalance, and addressing numeric stability errors. This guide covers these strategies using a concrete LLM routing scenario.

---

## 1. Multiclass Classification: OvR vs. Softmax

Binary logistic regression can be extended to handle multiclass classification (where target $y \in \{1, 2, \dots, K\}$). There are two primary architectures:

### One-vs-Rest (OvR) / One-vs-All (OvA)
- **How it works:** Train $K$ independent binary logistic regression classifiers. Classifier $k$ is trained to predict class $k$ (positive) vs. all other classes combined (negative).
- **Inference:** Assign the final class label matching the classifier that outputs the highest probability:
  $$\hat{y} = \arg\max_{k} f_{w^{(k)},b^{(k)}}(x)$$

### Softmax Regression (Multinomial Logistic Regression)
- **How it works:** Instead of multiple models, we output a vector of $K$ mutually exclusive probabilities that sum to exactly $1.0$ by computing a linear score $z_k = w^{(k)} \cdot x + b^{(k)}$ for each class and passing them through the **Softmax function**:
  $$P(y=k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$
- **Use Case:** Best when classes are mutually exclusive (e.g., classifying a digit as exactly one of $0, 1, \dots, 9$).

---

## 2. Scenario: Multi-LLM Gateway Router

You are building a high-throughput router that classifies user queries into one of three classes to optimize query costs:
- **Class 0:** Route to Llama-3 (Simple text, cheap).
- **Class 1:** Route to Claude-3 (Coding query, moderate).
- **Class 2:** Route to GPT-4 (Complex reasoning, expensive).

### One-vs-Rest (OvR) Python Implementation

```python
import numpy as np

# Simulate query features: [word_count, contains_code_symbols (0/1)]
features = np.array([
    [10, 0],   # Query 1: Llama candidate
    [150, 1]   # Query 2: Claude candidate
])

# Weights and biases learned by our 3 binary classifiers during training
# Model 0: Llama vs. Rest
w0 = np.array([-0.05, -2.0])
b0 = 0.5

# Model 1: Claude vs. Rest
w1 = np.array([0.02, 3.5])
b1 = -1.0

# Model 2: GPT-4 vs. Rest
w2 = np.array([0.08, 0.5])
b2 = -3.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def route_query(x):
    # Calculate probabilities from each model
    p0 = sigmoid(np.dot(w0, x) + b0)  # P(Llama)
    p1 = sigmoid(np.dot(w1, x) + b1)  # P(Claude)
    p2 = sigmoid(np.dot(w2, x) + b2)  # P(GPT-4)
    
    probs = [p0, p1, p2]
    assigned_class = np.argmax(probs)
    print(f"Probabilities: Llama={p0:.3f}, Claude={p1:.3f}, GPT-4={p2:.3f}")
    print(f"Routed to Class: {assigned_class}\n")
    return assigned_class

# Run inference
route_query(features[0])  # Outputs Llama
route_query(features[1])  # Outputs Claude
```

---

## 3. Handling Class Imbalance in Production Pipelines

In real-world applications (such as ad click-through rate prediction or fraud transaction screening), the positive class is highly rare (e.g., $< 1\%$). Training a logistic regression model on highly imbalanced datasets will cause the decision boundary to collapse, predicting the majority class for all instances.

---

### Mitigation A: Class Weights (Cost-Sensitive Learning)

We modify the Log-Loss cost function to assign a higher penalty to mistakes made on the minority class:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ w_{\text{pos}} y^{(i)} \log\left(f_{w,b}(x^{(i)})\right) + w_{\text{neg}} (1-y^{(i)}) \log\left(1-f_{w,b}(x^{(i)})\right) \right]$$

#### The Balanced Weight Calculation
Scikit-Learn computes the default `class_weight='balanced'` parameters using:
$$w_k = \frac{m}{K \times m_k}$$

Where:
- $m$ is the total number of samples.
- $K$ is the number of classes (for binary classification, $K=2$).
- $m_k$ is the number of samples belonging to class $k$.

#### Implementation Example:
```python
from sklearn.linear_model import LogisticRegression

# Option 1: Automatic balanced weights
model_auto = LogisticRegression(class_weight='balanced')

# Option 2: Custom manual weights (e.g., penalize positive class errors 99x more)
model_custom = LogisticRegression(class_weight={0: 1.0, 1: 99.0})
```

---

### Mitigation B: Downsampling (with Log-Odds Calibration)

To reduce compute cost, we can downsample the majority class (e.g., retaining only $10\%$ of negative instances). However, downsampling artificially increases the proportion of positive cases in training, making the model's raw output probabilities too high.

#### The Mathematical Proof of Recalibration
Logistic regression predicts log-odds:
$$\text{logit}_{\text{model}} = \log\left(\frac{P_{\text{train}}(y=1 | x)}{1 - P_{\text{train}}(y=1 | x)}\right) = w \cdot x + b_{\text{train}}$$

Let the downsampling rate of the majority class ($y=0$) be $p_{\text{down}}$. The ratio of positive to negative classes in our training split is scaled:
$$\frac{P_{\text{train}}(y=1)}{P_{\text{train}}(y=0)} = \frac{P_{\text{true}}(y=1)}{p_{\text{down}} \times P_{\text{true}}(y=0)}$$

Taking the logarithm:
$$\log\left(\frac{P_{\text{train}}(y=1)}{P_{\text{train}}(y=0)}\right) = \log\left(\frac{P_{\text{true}}(y=1)}{P_{\text{true}}(y=0)}\right) - \log(p_{\text{down}})$$
$$\text{logit}_{\text{train}} = \text{logit}_{\text{true}} - \log(p_{\text{down}})$$

To calibrate the model output logits at inference time back to the true population prior, we must adjust the log-odds:
$$\text{logit}_{\text{calibrated}} = \text{logit}_{\text{model}} + \log(p_{\text{down}})$$

Since $p_{\text{down}} < 1$, $\log(p_{\text{down}})$ is negative, which shifts the decision threshold down to represent the true population.

#### Python Recalibration Code Example:
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Suppose we downsampled class 0 by retaining only 10% of samples
p_down = 0.10

# Train model on the downsampled training dataset
model = LogisticRegression().fit(X_train_downsampled, y_train_downsampled)

# Inference on new raw inputs
raw_logits = model.decision_function(X_new)  # outputs w.x + b

# Apply calibration adjustment
calibrated_logits = raw_logits + np.log(p_down)

# Convert calibrated logits back to probabilities using sigmoid
calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
```

---

### Mitigation C: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates synthetic examples of the minority class by interpolating between nearest neighbors:
1. Select a minority class sample $x_i$.
2. Find its $k$-nearest neighbors in the minority class.
3. Choose one neighbor $x_j$, and generate a new sample:
   $$x_{\text{new}} = x_i + \lambda (x_j - x_i) \quad \text{where } \lambda \sim \text{Uniform}(0, 1)$$

#### Production Limitations:
- **Computational footprint:** Calculating nearest neighbors on millions of samples scales quadratically $O(m^2)$, bottlenecking training pipelines.
- **Overlapping regions:** If the minority class is highly noisy, SMOTE generates synthetic points inside the majority class distribution, degrading classification boundary precision.

---

## 4. The Danger of "Perfect Separation"

Perfect separation occurs when one or more features perfectly split the positive and negative targets in the training set (e.g., `contains_python_keyword` is only present in Claude queries).

### The Training Log (Unregularized)
To drive log-loss to zero, the optimizer tries to output predictions of exactly $1.0$ and $0.0$, pushing the weights toward infinity:

```text
[Iter 01] Loss: 0.6931 | Weight w_python: 0.50
[Iter 10] Loss: 0.1504 | Weight w_python: 12.45
[Iter 50] Loss: 0.0021 | Weight w_python: 284.12
[Iter 100] Loss: 0.0000 | Weight w_python: 15302.94
Warning: Optimization failed to converge. Gradient overflow.
```
- **The Solution:** $L_2$ Regularization (Ridge) adds the penalty $\frac{\lambda}{2m} \|w\|_2^2$ to the cost, preventing the weight from expanding to infinity.

```python
from sklearn.linear_model import LogisticRegression

# Violates stability (No regularization, C is set extremely high)
model_unstable = LogisticRegression(penalty='l2', C=1e10) 
model_unstable.fit(X_train, y_train)

# Production-safe: Add L2 regularization (C=1.0)
model_stable = LogisticRegression(penalty='l2', C=1.0)
model_stable.fit(X_train, y_train)
```
