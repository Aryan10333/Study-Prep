# Interview Flashcards: 40 Tree-Based Model & Ensemble Questions

This comprehensive guide contains answers to the 40 core interview questions on Decision Trees, Bias-Variance, Random Forests, Boosting, XGBoost, LightGBM, CatBoost, and Feature Importance.

---

## Part 1: Decision Trees

### Q1: Explain how a Decision Tree works. How does it decide where to split?
- **How it works:** A decision tree recursively partitions the feature space into orthogonal regions (leaves) by applying sequential binary rules (e.g., `feature_A > 10`).
- **Where to split:** At each node, the algorithm performs a greedy search across all features and all candidate split thresholds. It evaluates each split point and selects the one that maximizes the **Information Gain** (i.e., the largest reduction in child node impurity).

### Q2: How does a Decision Tree choose the best split for a classification problem? How is it different for a regression problem?
- **Classification:** Impurity is measured using Gini Impurity or Entropy. The algorithm chooses the split that yields the largest drop in categorical uncertainty.
- **Regression:** Impurity is measured using Mean Squared Error (MSE) or Mean Absolute Error (MAE). The algorithm chooses the split that minimizes the sum of squared residuals:
  $$\text{RSS} = \sum_{i \in \text{Left}} (y_i - \bar{y}_L)^2 + \sum_{i \in \text{Right}} (y_i - \bar{y}_R)^2$$
  Each leaf output is predicted as the **mean** (for MSE) or **median** (for MAE) of the targets in that partition.

### Q3: What are Gini Impurity, Entropy, and Information Gain? Which one would you prefer and why?
- **Gini Impurity:** Measures the probability of misclassifying a randomly chosen sample if it were labeled according to the class distribution:
  $$I_G = 1 - \sum p_k^2$$
- **Entropy:** Measures the average information surprise (uncertainty):
  $$H = -\sum p_k \log_2(p_k)$$
- **Information Gain:** The difference in impurity between the parent node and the weighted child nodes.
- **Preference:** **Gini Impurity** is preferred in production because it does not require computing logarithms (which are CPU-expensive), leading to faster training times.

### Q4: Why is a Decision Tree considered a greedy algorithm?
- **Greedy Nature:** At each node, the tree-growing algorithm makes the locally optimal split to maximize immediate Information Gain. It does not look ahead to evaluate if a sub-optimal split now would enable a much better split downstream, making it vulnerable to local minima.

### Q5: Why are Decision Trees prone to overfitting? How can you prevent it?
- **Why it overfits:** Without constraints, a tree can grow until every single training sample occupies its own pure leaf node, completely memorizing noise.
- **Prevention:** Use **pre-pruning** hyperparameters (limit `max_depth`, increase `min_samples_leaf`, require a `min_samples_split` size) or apply **post-pruning** (cost-complexity pruning).

### Q6: Explain pre-pruning and post-pruning. Which hyperparameters are commonly used to control tree growth?
- **Pre-Pruning:** Stopping tree growth during training (e.g., if a node contains fewer than $N$ samples). Parameters: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`.
- **Post-Pruning:** Allowing the tree to grow fully, then merging leaves back up if the split gain does not justify a penalty based on tree size:
  $$\text{Cost} = \text{Impurity} + \alpha |T|$$

### Q7: How do Decision Trees handle missing values, categorical variables, outliers, and feature scaling?
- **Missing Values:** Routed to whichever child node yields the highest split gain during training (surrogate splits).
- **Categoricals:** Evaluated by testing subsets of categories (e.g., $x \in \{\text{A}, \text{B}\}$).
- **Outliers:** Robust. Since splits are boundary thresholds, extreme values are routed identically to standard values on that side of the threshold.
- **Scaling:** Completely invariant.

### Q8: Why are tree-based models invariant to feature scaling?
- **Rank-Order Splitting:** Splits are based on threshold comparisons ($x_j > \theta$). Any monotonic scale transformation (e.g., multiplying by 100 or taking log) preserves the relative sorting order of the samples. The split search will find the equivalent threshold, producing identical partitions.

---

## Part 2: Bias-Variance

### Q9: Why do Decision Trees have low bias but high variance?
- **Low Bias:** A fully grown tree can create highly complex, non-linear decision boundaries to fit the training set perfectly.
- **High Variance:** Because splits are greedy and sequential, changing a single sample near the root split changes the threshold, altering every downstream split and producing a completely different tree.

### Q10: Explain the bias-variance tradeoff using Decision Trees, Random Forests, and Boosting.
- **Decision Trees:** Low bias, high variance (fully grown).
- **Random Forest:** Reduces variance by averaging predictions across parallel, independent, low-bias deep trees (Bagging).
- **Boosting:** Starts with high bias (shallow stumps) and sequentially reduces bias by fitting models to residual errors. Early stopping keeps variance in check.

---

## Part 3: Random Forest

### Q11: What is Bagging? Why does it reduce variance?
- **Bagging (Bootstrap Aggregating):** Training $B$ models in parallel on independent bootstrap samples of the data. 
- **Variance Reduction:** Averaging $B$ independent estimators, each with variance $\sigma^2$, reduces the overall variance to $\frac{1}{B}\sigma^2$. If the models are correlated ($\rho$), the variance floor becomes:
  $$\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

### Q12: Explain how Random Forest works from scratch.
1. Generate $B$ bootstrap datasets by sampling $m$ rows with replacement.
2. Train a decision tree on each dataset.
3. At each split node in each tree, select a random subset of $k$ features (usually $k = \sqrt{D}$) and search only among them for the best split.
4. Predict by taking the average (regression) or majority vote (classification) across all $B$ trees.

### Q13: How is Random Forest different from a Bagging ensemble of Decision Trees?
- **Feature Bagging:** Standard bagging searches all available features at every split. Random Forest restricts the search to a random subset of features at each node, decorrelating the trees and reducing the variance floor.

### Q14: Why does selecting a random subset of features at each split improve Random Forest?
- **Decorrelation:** If a dataset has one dominant predictive feature, standard bagging trees will all split on it first, making the trees highly correlated. Restricting features forces some trees to split on secondary features, lowering the tree-to-tree correlation $\rho$ and reducing ensemble variance.

### Q15: What is Out-of-Bag (OOB) error? How is it useful?
- **OOB Error:** The average validation error computed on training samples using only the trees that did not contain those samples in their bootstrap training set.
- **Utility:** Provides an unbiased estimate of generalization error on the fly, eliminating the need for a separate validation split or $K$-Fold cross-validation.

### Q16: When would you choose Random Forest over a single Decision Tree?
- Choose Random Forest when predictive accuracy is critical and you want a stable model that is robust to label noise without requiring extensive hyperparameter tuning.

---

## Part 4: Boosting

### Q17: Explain the difference between Bagging and Boosting.
- **Bagging:** Parallel, independent trees; reduces variance; averages predictions.
- **Boosting:** Sequential, dependent trees; reduces bias; sums predictions scaled by a learning rate.

### Q18: Explain how Gradient Boosting works. Can you walk me through the algorithm?
1. **Initialize:** Predict a constant base value $F_0(x) = \text{mean}(y)$ (for MSE).
2. **Loop $t = 1 \dots M$ trees:**
   a. Compute pseudo-residuals $r_{it} = y_i - F_{t-1}(x_i)$.
   b. Fit a regression tree $h_t(x)$ to predict these residuals.
   c. Update the ensemble: $F_t(x) = F_{t-1}(x) + \nu \cdot h_t(x)$ (where $\nu$ is the learning rate).

### Q19: Why does Gradient Boosting fit residuals instead of the target variable?
- **Gradient Descent:** Under MSE, the negative gradient of the loss function is the raw residual $y_i - F(x_i)$. Fitting residuals is equivalent to performing gradient descent in function space, moving predictions closer to the true targets.

### Q20: What is the role of the learning rate in Gradient Boosting?
- **Shrinkage:** The learning rate ($\nu$) scales the contribution of each tree. Setting $\nu$ low ($0.01 - 0.1$) requires more trees to converge but prevents the model from overshooting the minimum, yielding lower validation variance.

### Q21: Why is Boosting more prone to overfitting than Bagging?
- **Error Correction:** Boosting sequentially fits remaining residuals. If the dataset contains mislabeled outliers, boosting will eventually focus its capacity on fitting these noise points perfectly, leading to overfitting.

---

## Part 5: XGBoost

### Q22: How is XGBoost different from traditional Gradient Boosting?
- **Regularization:** Built-in $L_1$ and $L_2$ leaf penalties.
- **Optimization:** Uses a second-order Taylor expansion (gradients and Hessians).
- **Missing Values:** Learns default split directions natively.
- **Hardware:** Cache-aware block structures for parallel tree construction.

### Q23: What makes XGBoost faster and more accurate than Gradient Boosting?
- **Speed:** Employs approximate split finding (using histograms/percentiles) rather than evaluating all exact split thresholds.
- **Accuracy:** The second-order Taylor expansion provides a closer local approximation of loss, and built-in regularization directly controls overfitting.

### Q24: How does XGBoost handle missing values?
- **Sparsity-Aware Splitting:** During training, XGBoost evaluates sending missing values to the left child and then the right child, choosing the default direction that yields the highest gain.

### Q25: What are the most important hyperparameters in XGBoost? How do they affect model performance?
- `max_depth` (tree complexity), `eta`/`learning_rate` (step size), `subsample` (row bagging), `colsample_bytree` (column bagging), `reg_lambda` ($L_2$ regularization), `min_child_weight` (leaf Hessian sum limit).

### Q26: How does XGBoost prevent overfitting?
- Early stopping, $L_1$/$L_2$ leaf weight regularization, column/row sub-sampling, split complexity pruning ($\gamma$), and leaf Hessian thresholds (`min_child_weight`).

---

## Part 6: LightGBM

### Q27: How is LightGBM different from XGBoost?
- **Tree Growth:** Level-wise (XGBoost) vs. Leaf-wise (LightGBM).
- **Split Strategy:** Histogram-based split searching using GOSS and Exclusive Feature Bundling (EFB), yielding faster training speeds.

### Q28: What is the difference between leaf-wise and level-wise tree growth?
- **Level-wise:** Splits all nodes at a given tree depth in parallel.
- **Leaf-wise:** Evaluates all candidate leaf splits across the entire tree and splits only the single leaf that yields the maximum loss reduction. It grows deeper, more asymmetric trees that converge faster but overfit more easily on small datasets.

### Q29: What are GOSS and Exclusive Feature Bundling (EFB)? Why do they make LightGBM faster?
- **GOSS (Gradient-based One-Side Sampling):** Keeps samples with large gradients (errors) and randomly down-samples samples with small gradients, accelerating split evaluation.
- **EFB (Exclusive Feature Bundling):** Bundles mutually exclusive sparse features (e.g., one-hot features that are rarely non-zero at the same time) into a single feature, reducing the feature space dimension.

### Q30: When would you choose LightGBM over XGBoost?
- Choose LightGBM when working with large datasets ($>100k$ samples), when training time is constrained, or when memory availability is limited.

---

## Part 7: CatBoost

### Q31: Why is CatBoost particularly effective for datasets with categorical features?
- **Native Encoding:** Handles categorical features natively by calculating target statistics (mean target encoding) across randomized permutations of the dataset without requiring one-hot or ordinal encoding.

### Q32: What is ordered boosting, and how does it prevent target leakage?
- **Ordered Boosting:** Traditional target encoding uses the target of sample $i$ to compute its own feature value, introducing leakage. CatBoost prevents this by sorting samples randomly and calculating the target statistics for sample $i$ using only samples that precede it in the permutation sequence.

### Q33: When would you choose CatBoost over XGBoost or LightGBM?
- Choose CatBoost when the dataset contains high-cardinality categorical features and you want high out-of-the-box performance without manual preprocessing.

---

## Part 8: Feature Importance & Interpretability

### Q34: How do tree-based models compute feature importance?
- **MDI (Gini Importance):** Sums the split gains achieved by feature $j$ across all nodes in all trees, weighted by the number of samples passing through those nodes.
- **Permutation Importance:** Measures the validation score drop after randomly shuffling feature $j$'s values.

### Q35: What are the limitations of impurity-based feature importance? When would you use permutation importance instead?
- **MDI Limit:** Biased toward high-cardinality features because they offer more split points.
- **When to use Permutation:** Use Permutation Importance when features have varying cardinalities or when you need an unbiased importances metric on a validation dataset.

---

## Part 9: Practical & Scenario-Based Questions

### Q36: Can Decision Trees or Random Forests extrapolate beyond the range of the training data? Why or why not?
- **No Extrapolation:** Tree splits partition space. Any test sample with values larger than the training maximum will fall into the outermost boundary leaf, returning a constant value (the average target of the training samples in that leaf).

### Q37: If your Random Forest is overfitting, what steps would you take to improve generalization?
- Reduce `max_depth`, decrease `max_features` (feature bagging), increase `min_samples_leaf`, or decrease `max_samples` (row bootstrapping).

### Q38: If your XGBoost model is overfitting, which hyperparameters would you tune first, and why?
- Decrease `max_depth` (limit tree complexity), increase `min_child_weight` (prevent splits on small sample groups), decrease `learning_rate` (with early stopping), or decrease `subsample`/`colsample_bytree` to add bagging noise.

### Q39: Which tree-based model would you choose for each of the following scenarios, and why?
- **Very small dataset:** **Random Forest** (robust to overfitting without parameter tuning).
- **Very large dataset:** **LightGBM** (fast training via GOSS/EFB).
- **Many categoricals:** **CatBoost** (native ordered target statistics prevent leakage).
- **Highly imbalanced:** **XGBoost** (leverages `scale_pos_weight` and custom focal loss).
- **Many missing values:** **XGBoost** (sparsity-aware default routing).
- **Model interpretability:** **Single Decision Tree** (inspectable split rules).

### Q40: Compare Decision Tree, Random Forest, XGBoost, LightGBM, and CatBoost. What are the strengths, weaknesses, and ideal use cases for each?

| Model | Strengths | Weaknesses | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | Highly interpretable, fast training. | High variance, prone to overfitting. | Simple baselines, rule-extraction pipelines. |
| **Random Forest** | Robust to overfitting, requires little tuning. | Large file sizes, slow real-time inference. | Noisy tabular datasets with clean signal. |
| **XGBoost** | High accuracy, regularized split search. | Sensitive to noise, slow training on huge data. | Standard tabular datasets, regression/classification. |
| **LightGBM** | Extremely fast training, low memory. | Prone to overfitting on small datasets. | Large-scale tabular datasets ($>100k$ samples). |
| **CatBoost** | Best native categorical handling. | Slow training on continuous-heavy datasets. | Tabular data with high-cardinality categoricals. |
