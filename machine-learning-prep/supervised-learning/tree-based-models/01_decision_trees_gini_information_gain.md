# Decision Trees: Splitting Criteria & Impurity Math

This document details the mathematical mechanics, impurity measures, and split selection logic for Classification and Regression Trees (CART) in production.

---

## 1. Classification vs. Regression Split Objectives

Decision trees are non-parametric models that recursively partition the feature space into hyper-rectangles. The split objective differs between tasks:
- **Classification Trees:** Partition the space to maximize **node purity** (clustering class labels into distinct nodes) using impurity metrics like Gini or Entropy.
- **Regression Trees:** Partition the space to minimize the **residual sum of squares (RSS)** or variance of the continuous target variable.

---

## 2. Impurity Metrics: Gini vs. Entropy

For a node $T$ containing samples from $K$ distinct classes, let $p_k$ represent the empirical probability of a sample in $T$ belonging to class $k$:
$$p_k = \frac{1}{|T|} \sum_{i \in T} I(y_i = k)$$

### Gini Impurity
Gini Impurity measures the probability of misclassifying a randomly chosen element from the node if it were randomly labeled according to the class distribution in the node:
$$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$

- **Limits:** Max value is $1 - 1/K$ (perfectly uniform distribution, high impurity). Min value is $0.0$ (all samples belong to a single class, perfect purity).
- **Computation:** Fast. Requires only multiplication and subtraction, making it the default in CART algorithms.

### Entropy (Information Theory)
Entropy measures the average rate of information (or surprise) in the node's class distribution:
$$H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$$

- **Limits:** Max value is $\log_2(K)$ (uniform distribution). Min value is $0.0$ (perfect purity).
- **Computation:** Slower than Gini because calculating log base 2 requires floating-point transcendental CPU instructions.

---

## 3. Information Gain (IG)

When split criterion $a$ divides parent node $T$ into left child $T_L$ and right child $T_R$, **Information Gain** measures the reduction in impurity:
$$\text{Gain}(T, a) = \text{Impurity}(T) - \left( \frac{|T_L|}{|T|} \text{Impurity}(T_L) + \frac{|T_R|}{|T|} \text{Impurity}(T_R) \right)$$

The tree search algorithm evaluates all possible features and split thresholds, selecting the split that yields the highest Information Gain.

---

## 4. Regression Splits (Variance Reduction)

For a continuous target $y$, the tree searches for a split feature $j$ and split threshold $s$ that divides the parent region into child regions $R_1 = \{x | x_j \le s\}$ and $R_2 = \{x | x_j > s\}$ to minimize the sum of squared residuals:
$$\text{RSS}(j, s) = \sum_{i: x_i \in R_1} (y_i - c_1)^2 + \sum_{i: x_i \in R_2} (y_i - c_2)^2$$

Where $c_1$ and $c_2$ are the mean target values of the samples inside regions $R_1$ and $R_2$:
$$c_1 = \frac{1}{|R_1|} \sum_{i: x_i \in R_1} y_i, \quad c_2 = \frac{1}{|R_2|} \sum_{i: x_i \in R_2} y_i$$

---

## 5. Real-World Scenario: E-Commerce Customer Purchase Prediction

### Dataset
Suppose we want to predict if a user will complete a purchase (`1`) or leave (`0`) based on two features:
- **Page Views ($x_1$):** Continuous.
- **Discount Applied ($x_2$):** Binary ($1 = \text{Yes}$, $0 = \text{No}$).

We have a sample training set of $m = 6$ users:

| User | Page Views ($x_1$) | Discount Applied ($x_2$) | Purchased ($y$) |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 0 | 0 |
| 2 | 5 | 1 | 1 |
| 3 | 1 | 0 | 0 |
| 4 | 7 | 1 | 1 |
| 5 | 3 | 0 | 0 |
| 6 | 4 | 0 | 1 |

---

### Step-by-Step Split Calculations

#### Step 1: Calculate Parent Node Impurity
In the parent node, $3$ users purchased and $3$ did not:
- $p_0 = 3/6 = 0.5$ (No Purchase)
- $p_1 = 3/6 = 0.5$ (Purchase)

Parent Gini Impurity:
$$I_{G,\text{parent}} = 1 - (p_0^2 + p_1^2) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.50$$

---

#### Step 2: Evaluate Split Candidate A: Split on `Discount Applied` ($x_2 = 1$ vs. $x_2 = 0$)
- **Left Child Node ($T_L$: Discount = 1):** Contains Users 2 and 4.
  - Class counts: $2$ Purchased, $0$ Not Purchased.
  - Probabilities: $p_1 = 2/2 = 1.0$, $p_0 = 0/2 = 0.0$
  - Gini Impurity:
    $$I_{G,L} = 1 - (1.0^2 + 0.0^2) = 0.0$$
- **Right Child Node ($T_R$: Discount = 0):** Contains Users 1, 3, 5, and 6.
  - Class counts: $1$ Purchased (User 6), $3$ Not Purchased (Users 1, 3, 5).
  - Probabilities: $p_1 = 1/4 = 0.25$, $p_0 = 3/4 = 0.75$
  - Gini Impurity:
    $$I_{G,R} = 1 - (0.25^2 + 0.75^2) = 1 - (0.0625 + 0.5625) = 1 - 0.625 = 0.375$$

- **Weighted Gini Impurity of Split:**
  $$I_{G,\text{split}} = \frac{|T_L|}{|T|} I_{G,L} + \frac{|T_R|}{|T|} I_{G,R} = \frac{2}{6}(0.0) + \frac{4}{6}(0.375) = 0.0 + 0.25 = 0.25$$

- **Information Gain:**
  $$\text{Gain} = I_{G,\text{parent}} - I_{G,\text{split}} = 0.50 - 0.25 = 0.25$$

---

#### Step 3: Evaluate Split Candidate B: Split on `Page Views` (Threshold $x_1 \le 3.5$)
- **Left Child Node ($T_L$: Page Views $\le 3.5$):** Contains Users 1, 3, and 5 (Page views: 2, 1, 3).
  - Class counts: $0$ Purchased, $3$ Not Purchased.
  - Gini Impurity:
    $$I_{G,L} = 1 - (0.0^2 + 1.0^2) = 0.0$$
- **Right Child Node ($T_R$: Page Views $> 3.5$):** Contains Users 2, 4, and 6 (Page views: 5, 7, 4).
  - Class counts: $3$ Purchased, $0$ Not Purchased.
  - Gini Impurity:
    $$I_{G,R} = 1 - (1.0^2 + 0.0^2) = 0.0$$

- **Weighted Gini Impurity of Split:**
  $$I_{G,\text{split}} = \frac{3}{6}(0.0) + \frac{3}{6}(0.0) = 0.0$$

- **Information Gain:**
  $$\text{Gain} = I_{G,\text{parent}} - I_{G,\text{split}} = 0.50 - 0.0 = 0.50$$

---

### Conclusion
Comparing the two split candidates:
- **Discount Applied Gain:** $0.25$
- **Page Views ($\le 3.5$) Gain:** $0.50$

The algorithm selects the split **Page Views $\le 3.5$** because it maximizes Information Gain, yielding two perfectly pure child nodes.

---

## 6. Production Realities & Overfitting Profiles

### The Split Search Bottleneck
For a dataset with $n$ features and $m$ rows, finding a split on a continuous feature requires:
1. Sorting the feature values: $O(m \log m)$ complexity.
2. Checking every midpoint threshold between adjacent values: $O(m)$ calculations.
This is highly expensive on large tables. Modern systems (like XGBoost or LightGBM) pre-bin continuous features into histograms (e.g., 256 bins) to speed up split evaluations to $O(\text{bins})$.

### Overfitting Profile
- **Underfitting (High Bias):** Shallow trees (low `max_depth`) split on very few general features, failing to capture non-linear feature interactions.
- **Overfitting (High Variance):** Unconstrained deep trees grow until every leaf node contains exactly 1 sample (purity = 1.0). The model memorizes noisy outliers (e.g., User 6 who purchased despite having no discount). This collapses training error to $0$ but causes test error to explode.
- **Remedies:** Constraint parameters like `max_depth`, `min_samples_split`, and cost-complexity pruning.
