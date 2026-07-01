# Interview Flashcards: Tree-Based Models & A/B Testing

This document contains 10 core interview questions and answers, blending direct conceptual questions with scenario-based diagnostic questions to prepare you for senior ML engineering rounds.

---

### Q1 [Direct - GBDT vs. Random Forest]: Compare Random Forest and GBDT in terms of bagging vs. boosting, parallel vs. sequential building, variance vs. bias reduction, and overfitting sensitivity.
**Answer:**
- **Bagging vs. Boosting:** Random Forest uses **Bagging** (bootstrap aggregating) to build independent, deep trees in parallel. GBDT uses **Boosting** to build shallow trees sequentially, where each tree corrects the errors (residuals) of the prior ensemble.
- **Bias-Variance Profile:** 
  - Random Forest reduces **variance** by averaging low-bias, high-variance deep trees. 
  - GBDTs reduce **bias** by iteratively summing low-variance, high-bias shallow trees.
- **Overfitting Sensitivity:** Random Forest is robust to overfitting; adding more trees ($B \to \infty$) never increases overfitting; it only stabilizes the ensemble variance. GBDTs are highly sensitive to overfitting; adding too many estimators ($N$) will cause the model to fit noise, requiring early stopping.

---

### Q2 [Scenario - The Fraud Model's Gini ID Trap]: We deployed a Random Forest fraud detection model. To explain it to business stakeholders, we used Gini Importance (MDI) to identify the top features. One of our features is `transaction_id` (a continuous, unique transaction index). The model flagged it as the single most important predictor, but in production, our accuracy collapsed. Why did this happen, how does the Gini math cause this, and how would you fix it?
**Answer:**
- **Why it happened:** `transaction_id` is a unique continuous column with no predictive value. The model overfitted by creating custom splits on this feature to isolate individual fraud samples in the training set.
- **The Gini MDI Bias:** Gini MDI measures the sum of all impurity drops averaged across all tree splits on feature $j$. Because `transaction_id` has high cardinality, the greedy tree search frequently splits on it to fit training noise, artificially inflating its MDI score.
- **The Fix:** 
  1. Drop `transaction_id` from the model entirely since it is a random database index.
  2. For feature selection on remaining columns, use **Permutation Feature Importance** on a validation set. Shuffling values of a continuous noise feature will not affect validation accuracy, correctly flagging its importance as $0.0$.

---

### Q3 [Direct - Bagging & Variance Reduction]: Explain how averaging $B$ trees reduces variance in a Random Forest, and show how tree-to-tree correlation $\rho$ sets the variance floor.
**Answer:**
If we model each tree in a Random Forest as a random variable $T_i(x)$ with variance $\sigma^2$ and positive pairwise correlation $\rho$, the variance of the average prediction is:
$$\text{Var}\left(\frac{1}{B}\sum_{i=1}^B T_i(x)\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
- **The Decay:** As the number of trees $B$ increases, the second term $\frac{1-\rho}{B}\sigma^2$ decays to $0$. This represents the random noise that is averaged out by the ensemble.
- **The Variance Floor:** The remaining term $\rho \sigma^2$ is the variance floor. No matter how many trees you add ($B \to \infty$), you cannot reduce the variance below this floor.
- **How to lower the floor:** You must decrease the tree-to-tree correlation $\rho$. This is achieved via **feature bagging** (restricting each split search to a random subset of features like $\sqrt{n}$), ensuring trees are highly diverse and decorrelated.

---

### Q4 [Scenario - Missing Data in Real-Time Scoring]: We serve a credit default model in production using XGBoost. The feature `monthly_recurring_debt` is missing for 30% of new users. During training, we didn't impute this feature. How does XGBoost handle these missing values natively during inference, and what operational risk does this introduce if a silent data pipeline bug starts writing zeros instead of nulls?
**Answer:**
- **Native XGBoost Handling:** During training, at each node split, XGBoost evaluates sending all missing values to the left child and then to the right child. It learns a **default direction** (left or right) that maximizes the split Gain. During inference, any missing (`NaN` or `None`) values are automatically routed to this default direction.
- **Operational Risk:** If a silent upstream pipeline bug begins converting missing values to `0` or `0.0` (instead of passing `NaN`/`None`), the XGBoost model will no longer route them to the learned default direction. It will instead evaluate them as the numerical value `0` and route them through the standard numeric split paths, silently degrading prediction accuracy in production.
- **The Solution:** Implement ingestion-layer schema validation to ensure missing values are consistently passed as `NaN` to the model, and monitor input feature distributions for unexpected shifts (like spikes in `0.0` values).

---

### Q5 [Direct - Feature Scale Invariance]: Why do tree-based models and ensembles not require normalization or standardization of continuous input features? How does this contrast with linear models?
**Answer:**
- **Rank-Order Splitting:** Decision trees split feature space using orthogonal cuts (e.g., $x_j > \theta$). The split point $\theta$ is determined by sorting the values of feature $x_j$ and evaluating split thresholds.
- **Monotonic Transformation Invariance:** Any monotonic transformation of a feature (e.g., multiplying by 100, taking logarithms, or scaling to Z-scores) preserves the relative rank order of the samples. The split search algorithm will find the exact same sample partition, yielding identical trees.
- **Contrast with Distance Models:** Linear models predict using the weighted dot product $w^T x$. If you scale a feature by 100 without adjusting its weight, you warp its contribution to the log-odds, completely breaking the model. Tree models are unaffected by scale disparities.

---

### Q6 [Scenario - The Noisy Label Divergence]: We are training both a Random Forest and an XGBoost model on a tabular click-through rate dataset. The dataset contains high label noise (about 15% of positive labels are incorrect). During training, we notice that one model's validation loss remains stable, while the other's validation loss climbs rapidly. Which model is degrading, why is it sensitive to noise, and how would you tune its regularization to stabilize it?
**Answer:**
- **Which model is degrading:** **XGBoost** is degrading.
- **Why GBDTs are sensitive to noise:** GBDTs build trees sequentially to minimize the residuals of the prior ensemble. A mislabeled sample (noise) has a massive residual error. In subsequent iterations, XGBoost is forced to focus heavily on these noise samples, warping its decision boundaries to fit them. Random Forest averages independent trees, allowing noise to cancel out.
- **How to tune XGBoost to stabilize it:**
  1. **Shrink step size:** Lower the `learning_rate` (eta) to $0.01 - 0.05$ and use early stopping to halt training before the model fits the noise.
  2. **Add complexity regularization:** Increase `reg_lambda` ($L_2$ regularization) or `min_child_weight` (which prevents leaf splits on small, noisy sample groups).
  3. **Row/Feature sampling:** Lower `subsample` and `colsample_bytree` to $0.6 - 0.7$ to add bagging-like robustness.

---

### Q7 [Direct - Out-of-Bag (OOB) Validation]: What is Out-of-Bag (OOB) validation in a Random Forest? Is OOB error an unbiased estimator of generalization error, and what is its advantage over K-Fold Cross-Validation?
**Answer:**
- **How it works:** Because bootstrapping samples with replacement, approximately $36.8\%$ of the training samples are omitted from any individual tree's training split. To compute the OOB prediction for sample $i$, we pass it through *only* the subset of trees that did not use sample $i$ during their training step.
- **Unbiased Estimate:** Yes, OOB error is an unbiased estimator of the generalization (test) error because each sample is evaluated only on trees that never saw it during training.
- **Production Advantage:** K-Fold Cross-Validation requires training $K$ separate models (e.g., training 5 models for 5-fold CV). OOB validation provides a rigorous validation metric on the fly in a single model training run, saving $5\text{x}$ compute time and resources on large datasets.

---

### Q8 [Scenario - Custom Classification Tree Initialization]: You are implementing a custom GBDT solver from scratch for a binary classification task. If the starting training dataset has 80 positive samples and 20 negative samples, what is the exact value of the base prediction $F_0$ in log-odds, and what is the target value (residual) that the first tree is trained on for a positive sample?
**Answer:**
- **Base Prediction $F_0$:** GBDTs predict log-odds. The base prediction is initialized to the log-odds of the positive class:
  $$F_0 = \ln\left( \frac{\text{Pos}}{\text{Neg}} \right) = \ln\left( \frac{80}{20} \right) = \ln(4) \approx 1.386$$
- **Residual calculation ($r_{i1}$):** Under log-loss, the negative gradient of the loss function is:
  $$r_{i1} = y_i - p_i$$
  Where $p_i$ is the probability prediction:
  $$p_i = \sigma(F_0) = \frac{e^{\ln(4)}}{1 + e^{\ln(4)}} = \frac{4}{5} = 0.80$$
- **Target for Positive Sample ($y_i = 1$):**
  $$r_{11} = 1 - 0.80 = 0.20$$
  The first regression tree will fit these residual errors ($0.20$ for positive samples, and $-0.80$ for negative samples).

---

### Q9 [Scenario - The FWER False Alarm in Multi-variant Ads]: Our growth team ran a layout experiment testing 15 different landing page creatives against a control variant in parallel. They identified 3 significant winners at $\alpha=0.05$ and launched them. In post-launch monitoring, conversion rates did not change. What statistical mistake was made, and how would you redesign the experiment to prevent this?
**Answer:**
- **The Mistake:** The team fell into the **Multiple comparisons trap**. Testing 15 independent variants against a single control inflates the Family-Wise Error Rate (FWER):
  $$FWER = 1 - (1 - 0.05)^{15} \approx 53.6\%$$
  There was a $53.6\%$ chance of declaring at least one false positive winner by pure random chance. The 3 "winners" were statistical noise.
- **The Redesign:**
  1. **Bonferroni Correction:** Set the significance threshold for each variant test to $\alpha_{\text{adj}} = 0.05 / 15 = 0.0033$.
  2. **False Discovery Rate (FDR) / Benjamini-Hochberg:** If 15 is too large and Bonferroni reduces statistical power too severely, use the Benjamini-Hochberg procedure to control the false discovery rate instead of the family-wise error rate.

---

### Q10 [Direct - Sample Size Scaling]: How do you determine the sample size for an A/B test? Explain the relationship between required sample size $m$, metric variance $\sigma^2$, and Minimum Detectable Effect (MDE $\Delta$).
**Answer:**
- **Determination:** Sample size is computed using a power analysis, which relates the significance level ($\alpha$, Type I error rate), statistical power ($1-\beta$, target probability of detecting a true effect), baseline rate, and Minimum Detectable Effect (MDE, $\Delta$).
- **Scaling Relationship:** The required sample size per variant $m$ scales as:
  $$m \propto \frac{\sigma^2}{\Delta^2}$$
- **Implications:**
  - **Quadratic scaling with MDE ($\Delta^2$):** If you halve the MDE (e.g., trying to detect a $0.5\%$ conversion lift instead of a $1.0\%$ lift), you need **four times ($4\text{x}$)** the sample size.
  - **Linear scaling with variance ($\sigma^2$):** Continuous metrics with high variance (e.g., purchase cart values or latency spikes) require larger sample sizes to separate the true campaign lift from statistical noise.
