# Production Scenarios: Feature Importances & Ensemble Tuning

This document details feature importance biases, hyperparameter tuning profiles, and production performance optimization for tree-based ensembles.

---

## 1. Feature Importance Bias: MDI vs. Permutation

In production, identifying which features drive predictions is critical for model interpretability. However, the default feature importance calculations in tree packages (like Scikit-Learn's `feature_importances_`) can be highly misleading.

### Mean Decrease in Impurity (MDI) (Gini Importance)
MDI measures the total reduction in impurity (Gini or Entropy) contributed by splits on feature $j$, averaged across all trees:

$$\text{MDI}(x_j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in \text{nodes}(T_b)} I(v(t) = j) \frac{|t|}{|T|} \left[ \text{Impurity}(t) - \text{Weighted Child Impurity} \right]$$

#### The High-Cardinality Bias Bug:
MDI is highly biased toward continuous or high-cardinality categorical features (e.g., customer IDs, timestamps, transaction amounts). 
- **Why?** At each split search, a feature with many unique values offers many candidate split midpoints. The greedy splitting algorithm is more likely to select this feature by chance to partition noise in the training set, even if it has no actual predictive signal.
- **The Result:** The model splits on high-cardinality noise features frequently, causing their MDI scores to look artificially high, while stable binary features (like `has_discount`) look unimportant.

---

### Permutation Feature Importance (The Production Fix)
Permutation importance is a model-agnostic technique that measures the drop in a model's score after shuffling the values of a single feature:
1. Train the model on the training set and compute its baseline score (e.g., F1-score) on a validation set.
2. Shuffle the values of feature $j$ randomly across the validation set (this breaks the relationship between feature $j$ and target $y$ while preserving the overall feature distribution).
3. Pass the shuffled dataset to the model to predict and compute the new validation score.
4. The difference between the baseline score and the shuffled score is the **Permutation Importance** of feature $j$.

$$\text{Importance}(x_j) = \text{Score}_{\text{baseline}} - \text{Score}_{\text{shuffled}(j)}$$

- **Why it solves the bias:** If a high-cardinality feature is just random noise, shuffling its values has no impact on predictions, yielding a permutation importance score of $0.0$.

---

## 2. XGBoost Hyperparameter Tuning Profiles

When tuning XGBoost in production, hyperparameters must be configured to balance the bias-variance tradeoff:

| Hyperparameter | Default | Parameter Role in Bias-Variance Tradeoff |
|:---:|:---:|:---|
| `max_depth` | `6` | Maximum tree depth. High values increase complexity, reducing bias but increasing variance (leads to overfitting). |
| `min_child_weight` | `1` | Minimum sum of Hessians ($h_i$) needed in a child leaf. High values prevent the model from splitting on small, noisy leaf nodes, reducing variance. |
| `gamma` ($\gamma$) | `0` | Minimum split gain needed to create a new node. Increasing $\gamma$ forces conservative splitting, reducing variance. |
| `subsample` | `1` | Subsample ratio of the training instances. Lower values (e.g., `0.8`) decorrelate trees by training on different row partitions, reducing variance. |
| `colsample_bytree` | `1` | Subsample ratio of columns when constructing each tree. Lower values (e.g., `0.8`) prevent dominant features from overshadowing weaker features, reducing variance. |
| `learning_rate` ($\eta$) | `0.3` | Step-size shrinkage applied to updates. Lower values (e.g., `0.05`) prevent optimization oscillations, requiring a larger `n_estimators` to converge. |
| `reg_lambda` ($\lambda$) | `1` | $L_2$ leaf weight regularization. Higher values penalize large leaf weights, shrinking predictions toward the average, reducing variance. |
| `reg_alpha` ($\alpha$) | `0` | $L_1$ leaf weight regularization. Higher values drive unimportant leaf weights to exactly $0.0$, creating a sparse model. |

---

## 3. Production Post-Mortem: The Case of the Lagging Re-ranker

### Context
A search retrieval system uses a Gradient Boosting Decision Tree (GBDT) model to re-rank products returned by Elasticsearch. The target latency budget is **50ms**. In production, the API latency spiked to **250ms**, violating the SLA and causing checkout drop-offs.

### Root Cause Analysis
The re-ranking model was built with `n_estimators=1000` and `max_depth=12`. 
- **Tree Depth Latency:** A depth of $12$ results in trees with up to $2^{12} = 4096$ leaf nodes. Traversing 1,000 of these deep trees for thousands of candidate products during search queries consumed excessive CPU clock cycles.
- **Diminishing Returns:** Reviewing training metrics showed that validation accuracy plateaued around tree $150$. The trailing 850 trees contributed less than $0.1\%$ to the NDCG metric but consumed $80\%$ of the inference compute time.

---

### Code Migration: Optimizing Latency vs. Performance

#### Before (Slow, Unconstrained Model)
```python
from xgboost import XGBClassifier

# Oversized, slow model configuration
model = XGBClassifier(
    n_estimators=1000,
    max_depth=12,
    learning_rate=0.3,
    tree_method='exact'  # Slow exact split finding
)
model.fit(X_train, y_train)
```

#### After (Fast, Constrained, and Calibrated Model)
```python
from xgboost import XGBClassifier

# Constrained, fast model using early stopping and histogram-based splits
model = XGBClassifier(
    n_estimators=150,     # Reduced tree count
    max_depth=5,          # Shallowed trees (max 32 leaves)
    learning_rate=0.08,   # Smaller learning rate for stable shallow updates
    tree_method='hist',   # Fast histogram-based binning
    reg_lambda=1.5        # Increased L2 regularization to control variance
)

# Fit using early stopping: halt if validation loss does not improve for 10 rounds
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

# Save compact model payload (small file size, fast load times)
model.save_model("optimized_reranker.json")
```

### Production Results
- **Latency:** Decreased from **250ms** to **8ms** (sub-millisecond per candidate).
- **Model Size:** Shrinkage from **45MB** to **1.2MB**, reducing container memory requirements.
- **NDCG Score:** The normalized discounted cumulative gain dropped by a negligible **0.08%**, keeping search quality stable.
