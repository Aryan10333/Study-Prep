# Interview Flashcards: Tree-Based Models & Probability

This document contains 10 core, production-focused interview questions and answers designed to prepare you for senior machine learning engineering and systems design rounds.

---

### Q1: Contrast latency, cost, and throughput metrics when using a binary XGBoost model vs. a lightweight LLM (e.g., Llama-3-8B) as an upstream intent router.
**Answer:**
- **Latency:** An XGBoost model (trained on TF-IDF or lightweight embeddings) runs in **sub-milliseconds** ($<1\text{ms}$) on a standard CPU. An LLM, even a quantized Llama-3-8B on GPU, has a time-to-first-token (TTFT) latency of **50–150ms** and scales with output token counts, making XGBoost vastly superior for tight SLA gateways.
- **Throughput & Cost:** Storing and running XGBoost consumes negligible RAM ($<10\text{MB}$) and can handle tens of thousands of requests per second on a single $15 CPU instance. A Llama-3-8B model requires at least $16\text{GB}$ of VRAM (requiring expensive GPU instances like an A10G costing $\approx \$1/\text{hr}$), limiting throughput to a few dozen concurrent requests before queue delays accumulate.
- **Trade-off:** Use XGBoost for high-throughput, low-latency, low-cost intent routing. Use the LLM only when intent mapping requires complex zero-shot semantic understanding that cannot be captured by structured text features.

---

### Q2: How is Gradient Boosting mathematically set up for document re-ranking in a RAG search retrieval pipeline (Learning to Rank)?
**Answer:**
Instead of using standard classification or regression, GBDTs are optimized using **Learning to Rank (LTR)** paradigms:
- **Pairwise Loss (e.g., LambdaMART):** The model focuses on the *relative order* of pairs of documents. For a query, the loss function penalizes cases where an irrelevant document is ranked higher than a relevant document.
- **Listwise Loss (NDCG Optimization):** The loss function directly optimizes list-level sorting metrics like Normalized Discounted Cumulative Gain (NDCG), which weights document relevance logarithmically based on their rank position.
- **Implementation:** Modern frameworks like XGBoost (`objective='rank:pairwise'`) or LightGBM (`objective='lambdarank'`) take query group IDs as inputs, grouping documents by search context so trees are trained to sort lists rather than predict independent labels.

---

### Q3: Explain why Gini-importance (Mean Decrease in Impurity) is biased toward continuous high-cardinality features, and how this affects production debugging.
**Answer:**
- **Why the bias occurs:** Gini-importance (MDI) calculates the sum of all impurity decreases across all tree splits on feature $j$. Because a continuous high-cardinality feature (e.g., user transaction timestamp) has thousands of unique values, it offers thousands of candidate split boundaries. During training, the greedy tree algorithm frequently splits on this feature to fit noise, inflating its apparent importance.
- **Impact on production:** If you rely on MDI to debug your pipeline, you will conclude that continuous noise features are your most critical predictors.
- **The Fix:** Use **Permutation Feature Importance** on a validation set. Shuffling a noise feature will not affect validation accuracy, returning an importance score of $0.0$, regardless of its cardinality.

---

### Q4: Explain how averaging $B$ trees reduces variance in a Random Forest, and show how tree-to-tree correlation $\rho$ sets the variance floor.
**Answer:**
Averaging predictions across $B$ bootstrap trees reduces overall variance by smoothing out individual tree errors. Statistically, the variance of the averaged ensemble is bounded by:
- **The Decaying Variance:** Adding more trees ($B \to \infty$) drives the independent variance component of the trees down to zero.
- **The Variance Floor:** The remaining variance is bounded by the product of the average tree correlation and individual tree variance ($\rho \cdot \sigma^2$). Adding more trees cannot reduce variance below this floor.
- **Actionable Interview Insight:** To build a more stable forest, you must lower the tree-to-tree correlation $\rho$. This is achieved via **feature bagging** (restricting each split search to a random subset of features like $\sqrt{n}$), ensuring trees are highly diverse and decorrelated.

---

### Q5: In AdaBoost, if a base estimator achieves exactly $50\%$ error on a binary task, what is its stage weight $\alpha_t$, and what happens to the sample weights in the next step?
**Answer:**
- **Stage Weight:** If a weak learner achieves exactly $50\%$ error (equivalent to random guessing on a binary task), its stage weight $\alpha_t$ becomes exactly **$0.0$** (representing zero voting power in the final model).
- **Sample Weight Update:** Because the stage weight is zero, the exponential weight multiplier for both correct and incorrect samples evaluates to $e^0 = 1$.
- **Outcome:** The sample weights remain completely unchanged for the next iteration, and the weak learner's predictions are ignored by the ensemble.

---

### Q6: Why does XGBoost use a second-order Taylor expansion approximation of the loss function? How does having explicit gradients and Hessians speed up distributed training?
**Answer:**
- **Decoupling Custom Losses:** The second-order Taylor expansion approximates the objective using first-order gradients ($g_i$) and second-order Hessians ($h_i$). This decouples the custom loss function from the tree search solver. You can train XGBoost on any custom loss (e.g., Huber loss for outlier handling) by simply passing its gradients and Hessians, without modifying the core tree-growing code.
- **Distributed Speedup:** For distributed split finding, workers do not need to share raw datasets. They only need to communicate aggregated sums of gradients and Hessians for histogram candidate bins across nodes, minimizing network serialization overhead.

---

### Q7: Detail how the complexity penalty $\gamma$ and regularization parameter $\lambda$ act in the XGBoost node split gain equation.
**Answer:**
- **$\lambda$ ($L_2$ Regularization):** Added to the denominator of the node score. If a split isolates a very small group of samples (low cover/Hessians), $\lambda$ dominates the denominator, shrinking the split score toward zero. This prevents the model from splitting on noise or outlier categories.
- **$\gamma$ (Complexity Threshold):** Represents the minimum Gain required to keep a split. If the score reduction from splitting a parent node into left and right children is less than $\gamma$, the net Gain becomes negative. During bottom-up pruning, XGBoost will remove any split with a negative Gain.

---

### Q8: You are running an A/B test comparing a new Multi-Agent Planning pipeline (Treatment) against a baseline (Control) on average task resolution latency. Formulate the null hypothesis, test statistic, and Type I/II errors in this business context.
**Answer:**
- **Null Hypothesis ($H_0$):** The average latency of the Multi-Agent Planner is equal to or greater than the baseline: $\mu_{\text{treatment}} \ge \mu_{\text{control}}$.
- **Alternative Hypothesis ($H_1$):** The average latency of the treatment is lower: $\mu_{\text{treatment}} < \mu_{\text{control}}$ (one-tailed test).
- **Test Statistic:** A two-sample t-test because latency is a continuous variable and population variance is unknown.
- **Type I Error ($\alpha$):** Concluding the Multi-Agent Planner is faster when it is actually just random latency fluctuations, leading to a costly production rollout of a complex architecture with no real benefit.
- **Type II Error ($\beta$):** Failing to reject the null hypothesis when the Multi-Agent Planner is genuinely faster, missing out on a latency optimization.

---

### Q9: You A/B test 10 different prompt templates or agent planner configurations simultaneously against a control using a significance level of $\alpha = 0.05$. What is the probability of a false positive, and how does Bonferroni correction fix it?
**Answer:**
- **The Trap:** When running $k = 10$ independent tests, the probability of declaring at least one false positive winner by random chance is:
  $$P(\text{At least one FP}) = 1 - (1 - 0.05)^{10} \approx 40.1\%$$
- **The Bonferroni Fix:** Adjust the significance threshold for each of the $10$ individual tests:
  $$\alpha_{\text{adjusted}} = \frac{\alpha}{k} = \frac{0.05}{10} = 0.005$$
  You only reject the null hypothesis for a prompt variant if its individual p-value is less than $0.005$, maintaining the overall experiment-wide false positive rate at $\le 5\%$.

---

### Q10: How does the sample size $m$ required for your A/B test scale with the Minimum Detectable Effect (MDE) and metric variance?
**Answer:**
- **Scaling Relationship:** The required sample size $m$ scales **linearly with the variance** of the metric ($\sigma^2$) and **quadratically inversely with the MDE** ($\Delta^2$).
- **Implications:**
  - *Quadratic scaling with MDE:* If you halve the MDE (trying to detect a $1\%$ change instead of a $2\%$ change), you need **four times ($4\text{x}$)** the sample size.
  - *Linear scaling with variance:* If your metric has high variance (e.g., latency spikes caused by database cold starts), you need a larger sample size to filter out the noise and confirm significance.
