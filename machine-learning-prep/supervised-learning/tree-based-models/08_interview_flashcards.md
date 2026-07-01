# Interview Flashcards: Tree-Based Models & A/B Testing

This document contains 10 core, high-frequency interview questions and answers on tree-based models, boosting, and statistical hypothesis testing, designed to prepare you for senior machine learning engineering rounds.

---

### Q1: Compare Random Forest and Gradient Boosted Decision Trees (GBDTs) in terms of bagging vs. boosting, parallel vs. sequential training, bias-variance profiles, and overfitting sensitivity.
**Answer:**
- **Bagging vs. Boosting:** Random Forest uses **Bagging** (bootstrap aggregating) to build independent, deep trees in parallel. GBDT uses **Boosting** to build shallow trees sequentially, where each tree is trained to correct the errors (residuals) of the prior ensemble.
- **Bias-Variance Profile:** 
  - Random Forest reduces **variance** by averaging high-variance, low-bias deep trees. 
  - GBDTs reduce **bias** by iteratively summing low-variance, high-bias shallow trees.
- **Overfitting Sensitivity:** Random Forest is highly robust to overfitting: adding more trees ($B \to \infty$) never increases overfitting; it only stabilizes the ensemble variance. GBDTs are highly sensitive to overfitting: adding too many estimators ($N$) will eventually cause the model to memorize noise, requiring early stopping and regularization.

---

### Q2: How do Decision Trees and XGBoost handle missing values natively during training and inference? How does this compare to linear models?
**Answer:**
- **Decision Trees (CART):** Standard trees route missing values by calculating the split gain for all missing samples going left, and then all going right, choosing the direction that yields the highest information gain.
- **XGBoost (Sparsity-Aware Split Finding):** During training, at each node split, XGBoost evaluates sending all missing values to the left child and then to the right child. It learns a **default direction** (left or right) that maximizes the split Gain. During inference, if a feature value is missing, it is automatically routed to this learned default direction.
- **Comparison to Linear Models:** Linear models cannot handle missing values natively. If an input feature $x_j$ is `NaN`, the dot product $w^T x$ evaluates to `NaN`, requiring explicit imputation (e.g., median, mean, or model-based imputation) in the preprocessing pipeline.

---

### Q3: Your XGBoost model is overfitting on the validation set. What hyperparameters would you tune, and in what direction, to stabilize validation loss?
**Answer:**
To reduce overfitting (high variance) in XGBoost, tune these parameters:
- **`max_depth` (Decrease):** Shallows the individual trees (typically set between $3$ and $7$) to limit the complexity of individual split rules.
- **`min_child_weight` (Increase):** Represents the minimum sum of Hessians (instance weights) required in a leaf. Increasing this prevents splits that isolate very few samples, suppressing noise splits.
- **`subsample` and `colsample_bytree` (Decrease):** Adds bagging randomness by sub-sampling training rows and features (typically set to $0.6 - 0.8$), reducing correlation between trees.
- **`reg_lambda` ($L_2$) and `reg_alpha` ($L_1$) (Increase):** Adds regularization penalties to shrink leaf weights.
- **`gamma` (Increase):** Increases the minimum gain threshold required to make a split.
- **`learning_rate` (Decrease) & `n_estimators` (Increase with Early Stopping):** Slows down gradient updates, allowing smoother convergence.

---

### Q4: Explain how averaging $B$ trees reduces variance in a Random Forest, and show how tree-to-tree correlation $\rho$ sets the variance floor.
**Answer:**
If we model each tree in a Random Forest as a random variable $T_i(x)$ with variance $\sigma^2$ and positive pairwise correlation $\rho$, the variance of the average prediction is:
$$\text{Var}\left(\frac{1}{B}\sum_{i=1}^B T_i(x)\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
- **The Decay:** As the number of trees $B$ increases, the second term $\frac{1-\rho}{B}\sigma^2$ decays to $0$. This represents the random noise that is averaged out by the ensemble.
- **The Variance Floor:** The remaining term $\rho \sigma^2$ is the variance floor. No matter how many trees you add ($B \to \infty$), you cannot reduce the variance below this floor.
- **The Actionable Insight:** To build a more stable model, you must decrease the correlation $\rho$ between trees. This is achieved in Random Forests by **feature bagging** (restricting each split search to a random subset of features like $\sqrt{n}$), ensuring trees are highly diverse and decorrelated.

---

### Q5: Compare Gini Importance (MDI), Permutation Feature Importance, and SHAP values. What are their biases, computational costs, and when would you use each?
**Answer:**
- **Gini Importance (MDI):**
  - *Mechanism:* Sum of Gini impurity drops across all tree splits on feature $j$.
  - *Bias/Cost:* Highly biased toward continuous or high-cardinality categorical features (e.g., unique customer IDs) because they offer more candidate split points. Virtually free computationally (calculated during training).
- **Permutation Importance:**
  - *Mechanism:* Measures validation score drop after shuffling feature $j$'s values.
  - *Bias/Cost:* Unbiased. However, if features are highly collinear (multicollinear), it over-penalizes them because shuffling one while keeping the other intact creates unrealistic feature combinations. Medium computational cost.
- **SHAP (SHapley Additive exPlanations):**
  - *Mechanism:* Based on cooperative game theory, calculates marginal contributions of feature $j$ across all feature subsets.
  - *Bias/Cost:* Mathematically consistent, unbiased, handles interactions correctly. High computational cost (though optimized for trees via TreeSHAP). Use SHAP when exact, sample-level explanations are required for business logic.

---

### Q6: Why do tree-based models and ensembles not require normalization or standardization of continuous input features?
**Answer:**
- **Rank-Order Splitting:** Decision trees split feature space using orthogonal cuts (e.g., $x_j > \theta$). The split point $\theta$ is determined by sorting the values of feature $x_j$ and evaluating split thresholds.
- **Monotonic Transformation Invariance:** Any monotonic transformation of a feature (e.g., taking the logarithm $\log(x_j)$, scaling by a constant $1000 \cdot x_j$, or converting to Z-scores) preserves the relative rank order of the samples. The split search algorithm will find the exact same sample partition, yielding identical trees.
- **Contrast with Distance Models:** Unlike linear regression or K-Means, which calculate distance metrics ($w^T x$ or Euclidean distances), trees look at feature values individually and are unaffected by scale disparities.

---

### Q7: What is Out-of-Bag (OOB) validation in a Random Forest? Is OOB error an unbiased estimator of generalization error, and what is its advantage over K-Fold Cross-Validation?
**Answer:**
- **How it works:** Because bootstrapping samples with replacement, approximately $36.8\%$ of the training samples are omitted from any individual tree's training split. To compute the OOB prediction for sample $i$, we pass it through *only* the subset of trees that did not use sample $i$ during their training step.
- **Unbiased Estimate:** Yes, OOB error is an unbiased estimator of the generalization (test) error because each sample is evaluated only on trees that never saw it during training.
- **Production Advantage:** K-Fold Cross-Validation requires training $K$ separate models (e.g., training 5 models for 5-fold CV). OOB validation provides a rigorous validation metric on the fly in a single model training run, saving $5\text{x}$ compute time and resources on large datasets.

---

### Q8: In Gradient Boosting for binary classification, what is the base prediction $F_0$, and what are the residuals (pseudo-residuals) that the first tree is trained on?
**Answer:**
- **Base Prediction ($F_0$):** GBDTs predict log-odds. The base prediction $F_0$ is initialized with the constant log-odds of the positive class in the training set:
  $$F_0 = \ln\left( \frac{\sum y_i}{\sum (1 - y_i)} \right)$$
- **Pseudo-Residuals ($r_{i1}$):** Under log-loss, the negative gradient of the loss function with respect to the raw log-odds prediction simplifies to:
  $$r_{i1} = -\frac{\partial L(y_i, F_0)}{\partial F_0} = y_i - p_i$$
  Where $p_i = \sigma(F_0)$ is the predicted probability (using the sigmoid function).
- **Result:** The first tree ($h_1$) is trained to predict these raw residuals ($y_i - p_i$), which represent the prediction errors (the difference between the actual binary label $0$ or $1$ and the current probability $p_i$).

---

### Q9: How do you determine the sample size for an A/B test? Explain how sample size $m$ scales with metric variance $\sigma^2$ and Minimum Detectable Effect (MDE $\Delta$).
**Answer:**
- **Determination:** Sample size is computed using a power analysis, which relates the significance level ($\alpha$, Type I error rate), statistical power ($1-\beta$, target probability of detecting a true effect), baseline rate, and Minimum Detectable Effect (MDE, $\Delta$).
- **Scaling Relationship:** The required sample size per variant $m$ scales as:
  $$m \propto \frac{\sigma^2}{\Delta^2}$$
- **Implications:**
  - **Quadratic scaling with MDE ($\Delta^2$):** If you halve the MDE (e.g., trying to detect a $0.5\%$ conversion lift instead of a $1.0\%$ lift), you need **four times ($4\text{x}$)** the sample size.
  - **Linear scaling with variance ($\sigma^2$):** Continuous metrics with high variance (e.g., purchase cart values or latency spikes) require larger sample sizes to separate the true campaign lift from statistical noise.

---

### Q10: What is the Multiple Comparisons Trap in A/B testing? How does the Bonferroni correction address this, and what is its main drawback?
**Answer:**
- **The Trap (Type I Error Inflation):** If you run $k$ independent hypothesis tests (e.g., testing 10 different layout variants or prompt templates against a single control) each at a significance level of $\alpha = 0.05$, the probability of committing at least one Type I error (declaring a false positive winner by random chance) across the entire experiment is:
  $$FWER = 1 - (1 - 0.05)^{10} \approx 40.1\%$$
- **The Bonferroni Fix:** To control the Family-Wise Error Rate (FWER) back to $5\%$, adjust the significance threshold for each of the $k$ individual tests:
  $$\alpha_{\text{adjusted}} = \frac{\alpha}{k} = \frac{0.05}{10} = 0.005$$
  A variant is only declared a winner if its p-value is $< 0.005$.
- **The Drawback:** The Bonferroni correction assumes all tests are independent and is highly conservative. It severely reduces statistical power ($1-\beta$), leading to high Type II error rates (false negatives) and requiring significantly longer experiment runtimes to confirm genuine winners.
