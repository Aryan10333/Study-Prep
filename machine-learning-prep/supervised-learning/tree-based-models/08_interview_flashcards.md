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
If we model each tree in a Random Forest as a random variable $T_i(x)$ with variance $\sigma^2$ and positive pairwise correlation $\rho$, the variance of the average prediction is:
$$\text{Var}\left(\frac{1}{B}\sum_{i=1}^B T_i(x)\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
- **The Decay:** As the number of trees $B$ increases, the second term $\frac{1-\rho}{B}\sigma^2 \to 0$.
- **The Floor:** The remaining term $\rho \sigma^2$ is the variance floor. No matter how many trees you add ($B \to 10,000$), you cannot reduce the variance below this floor.
- **Actionable Insight:** To build a more stable model, you must decrease the correlation $\rho$ between trees. This is achieved in Random Forests by **feature bagging** (restricting each split search to a random subset of $k = \sqrt{n}$ features).

---

### Q5: In AdaBoost, if a base estimator achieves exactly $50\%$ error on a binary task, what is its stage weight $\alpha_t$, and what happens to the sample weights in the next step?
**Answer:**
- **Stage Weight Calculation:** The stage weight $\alpha_t$ is computed as:
  $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
  If error $\epsilon_t = 0.5$ (equivalent to random guessing):
  $$\alpha_t = \frac{1}{2} \ln\left(\frac{0.5}{0.5}\right) = \frac{1}{2} \ln(1) = 0$$
- **Sample Weight Update:** Because $\alpha_t = 0$, the sample weight update multiplier:
  $$\exp\left(-\alpha_t y_i G_t(x_i)\right) = e^0 = 1$$
- **Outcome:** The estimator has zero voting power in the final model ($\alpha_t = 0$), and all sample weights remain completely unchanged for the next iteration.

---

### Q6: Why does XGBoost use a second-order Taylor expansion approximation of the loss function? How does having explicit gradients and Hessians speed up distributed training?
**Answer:**
- **Mathematical Decoupling:** The second-order Taylor expansion approximates the objective as:
  $$\tilde{\text{Obj}}^{(t)} \approx \sum_{i=1}^m \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)$$
  This decouples the training algorithm from the loss function. The split search algorithm only needs to know the array of first-order gradients ($g_i$) and second-order Hessians ($h_i$) for each sample, allowing the core solver to support custom loss functions (e.g., huber loss, quantile loss) without rewriting the tree search engine.
- **Distributed Speedup:** For split finding, XGBoost does not need to transfer raw training data across worker nodes. It only needs to communicate aggregated sums of $g_i$ and $h_i$ for candidates in histograms, reducing network serialization bottlenecks.

---

### Q7: Detail how the complexity penalty $\gamma$ and regularization parameter $\lambda$ act in the XGBoost node split gain equation.
**Answer:**
The Gain for a split is calculated as:
$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum g_L)^2}{\sum h_L + \lambda} + \frac{(\sum g_R)^2}{\sum h_R + \lambda} - \frac{(\sum g_P)^2}{\sum h_P + \lambda} \right] - \gamma$$
- **Role of $\lambda$ ($L_2$ Regularization):** $\lambda$ is added to the denominator of the child scores. If a leaf node has very few samples (small sum of Hessians $h_j$), $\lambda$ dominates the denominator, shrinking the leaf score toward $0.0$. This prevents splits that isolate small sample counts (outliers).
- **Role of $\gamma$ (Split Penalty):** $\gamma$ acts as a hard threshold. If the raw score reduction of a split (the bracketed term) is less than $2\gamma$, the overall Gain becomes negative. During bottom-up pruning, XGBoost will remove any split with a negative Gain.

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
The required sample size per variant scales as:
$$m \propto \frac{\sigma^2}{\Delta^2}$$
Where:
- $\sigma^2$ is the variance of the metric (e.g., latency standard deviation).
- $\Delta$ is the Minimum Detectable Effect (MDE) (the minimum difference you want to detect).
- **Implications:**
  - **Quadratic scaling with MDE:** If you want to detect a difference that is half as small (e.g., detecting a $1\%$ change instead of a $2\%$ change), you need **four times ($4\text{x}$)** the sample size.
  - **Linear scaling with variance:** If your metric has high variance (e.g., latency spikes caused by database cold starts), you need a larger sample size to separate the true effect from the noise.
