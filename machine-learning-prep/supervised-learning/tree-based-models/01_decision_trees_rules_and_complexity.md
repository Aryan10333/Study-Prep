# Decision Trees: Boundary Rules & Impurity Math

This guide explains the mathematical intuition behind Decision Tree splitting criteria (Gini Impurity vs. Entropy) using simple numerical examples, walks through a manual split calculation, and shows how to inspect rules in Python.

---

## 1. Split Criteria: Gini Impurity vs. Entropy

A decision tree splits data by asking binary questions (e.g., `page_views > 6`). To choose the best split, the algorithm evaluates how "pure" the resulting child nodes are.

For a node containing samples from $K$ classes, let $p_k$ represent the fraction of samples belonging to class $k$.

### Gini Impurity
Gini measures the probability that a randomly selected sample from the node would be incorrectly labeled if it were randomly classified according to the node's class distribution:
$$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$

- **Intuition:**
  - If a node is **perfectly pure** (e.g., all 100 samples are class 1): $p_1 = 1.0$, $p_0 = 0.0$.
    $$I_G = 1 - (1.0^2 + 0.0^2) = 0.0 \quad \text{(Minimum Impurity)}$$
  - If a node is **maximally impure** (e.g., 50 samples class 1, 50 samples class 0): $p_1 = 0.5$, $p_0 = 0.5$.
    $$I_G = 1 - (0.5^2 + 0.5^2) = 0.5 \quad \text{(Maximum Impurity for 2 classes)}$$

### Entropy
Entropy comes from Information Theory and measures the average uncertainty (or "surprise") in a node's class distribution:
$$H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$$

- **Comparison:** Both metrics behave similarly. However, Gini is faster to calculate in production because it only requires squaring numbers, whereas Entropy requires computing logarithms, which is more CPU-intensive.

---

## 2. Information Gain: How Split Decisions Are Made

Information Gain measures the reduction in impurity after splitting a parent node $T$ into left ($T_L$) and right ($T_R$) children:

$$\text{Information Gain} = \text{Impurity}(T) - \left[ \frac{|T_L|}{|T|} \text{Impurity}(T_L) + \frac{|T_R|}{|T|} \text{Impurity}(T_R) \right]$$

The algorithm searches all features and split thresholds, choosing the split that maximizes this Gain.

---

## 3. Step-by-Step Hand Calculations: E-Commerce Purchase Split

Let's predict if a user will complete a purchase (`1`) or leave (`0`) based on:
- **Page Views ($x_1$):** Continuous.
- **Discount Applied ($x_2$):** Binary ($1 = \text{Yes}$, $0 = \text{No}$).

We have a sample dataset of $m = 6$ sessions:

| Session ($i$) | Page Views ($x_1$) | Discount Applied ($x_2$) | Purchased ($y$) |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 0 | 0 |
| 2 | 5 | 1 | 1 |
| 3 | 1 | 0 | 0 |
| 4 | 7 | 1 | 1 |
| 5 | 3 | 0 | 0 |
| 6 | 4 | 0 | 1 |

---

### Step 1: Calculate Parent Node Gini
Out of the $6$ sessions, $3$ resulted in a purchase ($y=1$) and $3$ did not ($y=0$):
- $p_{\text{purchased}} = 3/6 = 0.5$
- $p_{\text{left}} = 3/6 = 0.5$

$$I_{G,\text{parent}} = 1 - (0.5^2 + 0.5^2) = 0.50$$

---

### Step 2: Evaluate Split Candidate A (Split on `Discount Applied` $x_2$)
- **Left Child ($T_L$: Discount = 1):** Contains Sessions 2 and 4.
  - Class counts: $2$ Purchased, $0$ Not Purchased ($p_1 = 1.0$, $p_0 = 0.0$).
  - Gini Impurity:
    $$I_{G,L} = 1 - (1.0^2 + 0.0^2) = 0.0$$
- **Right Child ($T_R$: Discount = 0):** Contains Sessions 1, 3, 5, and 6.
  - Class counts: $1$ Purchased (Session 6), $3$ Not Purchased ($p_1 = 0.25$, $p_0 = 0.75$).
  - Gini Impurity:
    $$I_{G,R} = 1 - (0.25^2 + 0.75^2) = 1 - (0.0625 + 0.5625) = 0.375$$

- **Weighted Gini of Split:**
  $$I_{G,\text{split}} = \frac{2}{6}(0.0) + \frac{4}{6}(0.375) = 0.25$$
- **Information Gain:**
  $$\text{Gain} = 0.50 - 0.25 = 0.25$$

---

### Step 3: Evaluate Split Candidate B (Split on `Page Views` threshold $x_1 \le 3.5$)
- **Left Child ($T_L$: Page Views $\le 3.5$):** Contains Sessions 1, 3, 5.
  - Class counts: $0$ Purchased, $3$ Not Purchased ($p_1 = 0.0$, $p_0 = 1.0$).
  - Gini Impurity:
    $$I_{G,L} = 1 - (0.0^2 + 1.0^2) = 0.0$$
- **Right Child ($T_R$: Page Views $> 3.5$):** Contains Sessions 2, 4, 6.
  - Class counts: $3$ Purchased, $0$ Not Purchased ($p_1 = 1.0$, $p_0 = 0.0$).
  - Gini Impurity:
    $$I_{G,R} = 1 - (1.0^2 + 0.0^2) = 0.0$$

- **Weighted Gini of Split:**
  $$I_{G,\text{split}} = \frac{3}{6}(0.0) + \frac{3}{6}(0.0) = 0.0$$
- **Information Gain:**
  $$\text{Gain} = 0.50 - 0.0 = 0.50$$

**Conclusion:** The algorithm selects the split **Page Views $\le 3.5$** because its Information Gain ($0.50$) is higher than the Discount split ($0.25$).

---

## 4. Python Implementation: Rule Extraction

In production, you can inspect the exact rules learned by your tree model using Scikit-Learn's `export_text` function. This is highly useful for validating that the model has not learned biased split rules.

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Generate e-commerce customer transaction session data
np.random.seed(42)
m = 100
page_views = np.random.randint(1, 15, m)
discount_applied = np.random.choice([0, 1], m, p=[0.7, 0.3])

y = np.where((page_views > 6) & (discount_applied == 1), 1, 0)
y[12] = 1 # Add outlier noise

X = pd.DataFrame({'page_views': page_views, 'discount_applied': discount_applied})

# Fit a constrained Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
model.fit(X, y)

# Export the learned rules as a text tree
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
```

### Expected Console Output
```text
|--- page_views <= 6.50
|   |--- class: 0
|--- page_views >  6.50
|   |--- discount_applied <= 0.50
|   |   |--- class: 0
|   |--- discount_applied >  0.50
|   |   |--- class: 1
```

---

## 5. Overfitting & Complexity Sweeps

To find the optimal balance between bias and variance, you can run a complexity sweep on `max_depth` to track training vs. validation accuracy.

- **Underfitting (High Bias):** A `max_depth` of 1 or 2 splits on too few features, failing to capture the interaction between `page_views` and `discount_applied`.
- **Overfitting (High Variance):** A `max_depth` of 10 or more isolates individual noisy samples (like User 12) into custom leaf nodes, collapsing training error to 0 but degrading generalization on unseen test data.

### Diagnostic Visual (Depth Complexity Sweep)
![Tree Complexity Sweeps](images/tree_complexity_overfitting.png)

---

## 6. Interactive Practice Notebook
To run the depth sweep, plot the 2D decision boundary surfaces, and practice tuning decision tree parameters, open the interactive notebook:
- [01_decision_trees_complexity_and_rules.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/01_decision_trees_complexity_and_rules.ipynb)
