# Random Forests & Bagging (Bootstrap Aggregation)

This document details the ensemble mechanics of bagging, the statistical properties of tree decorrelation, and the variance reduction equations of Random Forests.

---

## 1. Bagging (Bootstrap Aggregation) Mechanics

Bootstrap Aggregation (Bagging) is an ensemble method designed to reduce the variance of high-variance estimators (like deep decision trees).

### Bootstrapping
Given a training dataset of $m$ rows, we generate $B$ unique bootstrap datasets by sampling $m$ rows **with replacement**.

#### The $63.2\%$ Coverage Intuition
In a dataset of size $m$, the probability of a specific row **not** being selected in a single random draw is:
$$p_{\text{not}} = 1 - \frac{1}{m}$$

For a bootstrap sample of $m$ independent draws, the probability that the row is completely excluded is:
$$p_{\text{exclude}} = \left(1 - \frac{1}{m}\right)^m$$

As $m \rightarrow \infty$, this limit converges to:
$$\lim_{m \rightarrow \infty} \left(1 - \frac{1}{m}\right)^m = \frac{1}{e} \approx 0.368$$

Thus, the probability of any given row being **included** in a bootstrap sample is:
$$p_{\text{include}} = 1 - 0.368 = 0.632 \text{ or } 63.2\%$$

On average, each individual tree is trained on only $63.2\%$ of the unique training samples.

---

## 2. Out-of-Bag (OOB) Validation

The remaining $36.8\%$ of samples not selected for a tree's training set are called **Out-of-Bag (OOB)** samples. 

### Performance Estimation
Because a tree $T_b$ was not trained on its corresponding OOB sample set, we can use these samples as an independent test set for that specific tree:
1. For each sample $x_i$ in the original dataset, predict its value using only the subset of trees that did **not** contain $x_i$ in their bootstrap training sets.
2. Aggregate these predictions to compute the **OOB Error Rate**.
3. OOB error serves as an unbiased estimator of generalization error, removing the need for a separate train/validation split during tuning.

---

## 3. Feature Bagging (Random Subspaces)

Standard bagging on decision trees has a limitation: if the dataset contains one highly dominant predictive feature (e.g., historical payment delays), almost all bootstrap trees will select this feature for their root split. This makes the trees highly structurally similar and correlated.

### Tree Decorrelation
Random Forests resolve this by introducing **Feature Bagging**:
- When splitting a node inside a tree, the algorithm is restricted to selecting from a random subset of $k$ features out of the $n$ total features.
- Typically:
  - For classification: $k = \sqrt{n}$
  - For regression: $k = n / 3$
- By forcing trees to split on alternative features, the correlation between tree predictions is minimized, allowing weak secondary features to build independent trees.

---

## 4. The Math Intuition of Variance Reduction

If we average $B$ identically distributed, but correlated, estimators (trees) $T_i(x)$, each with variance $\sigma^2$ and positive pairwise correlation coefficient $\rho$:

$$\text{Var}\left( \frac{1}{B} \sum_{i=1}^B T_i(x) \right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

### The Variance Floor
As the ensemble size $B$ grows:
$$\lim_{B \rightarrow \infty} \left( \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2 \right) = \rho \sigma^2$$

- The second term $\frac{1 - \rho}{B} \sigma^2$ represents the variance component that decays to $0$ by adding more estimators.
- The first term $\rho \sigma^2$ represents the **variance floor**. No matter how many trees you add ($B \rightarrow \infty$), you cannot reduce the variance below this limit.
- **The Role of Feature Bagging:** The only way to lower this variance floor is to reduce the tree correlation $\rho$. Feature bagging shrinks $\rho$, making the ensemble far more stable than standard bagging.

---

## 5. Real-World Scenario: Credit Card Default Risk Prediction

### The Problem
We are predicting whether a credit card holder will default on their next payment (`1`) or pay (`0`). The dataset has features like:
- `pay_delay_months` (highly dominant feature).
- `credit_limit`, `bill_amount`, `age`.

### Single Tree vs. Random Forest Behavior

```text
Single Tree:
[pay_delay_months > 2] ---> Overfits to noise in billing limits ---> High Variance

Random Forest:
Tree 1: Splits on [pay_delay_months] ---> Predicts Default
Tree 2: Splits on [credit_limit] -------------> Predicts Default
Tree 3: Splits on [bill_amount] --------------> Predicts No Default
Ensemble Consensus: Predicts Default (Averages out the noise)
```

- **Single Decision Tree:** If a customer has a minor late billing dispute, a deep single tree will split on this outlier and immediately predict default. It is highly sensitive to training variance.
- **Random Forest:** Feature bagging forces some trees to split on `credit_limit` and `bill_amount` instead of `pay_delay_months`. The final prediction is a consensus average of these decorrelated trees, smoothing out the noise and preventing over-prediction of defaults.

---

## 6. Production Constraints

- **Inference Latency:** While a linear model is a simple vector dot product, a Random Forest requires traversing $B$ independent trees (e.g., 500 trees). This requires significant CPU cache jumps, introducing latency bottlenecks that make it slow for sub-millisecond real-time APIs.
- **Memory Footprint:** Storing 500 deep decision trees in RAM can require hundreds of megabytes, which is expensive for containerized serverless deployments compared to compact weights vectors.
- **No Online Updates:** Random Forests cannot be trained incrementally (online learning). If new user transaction patterns emerge, you must rebuild the entire bootstrap ensemble from scratch.
