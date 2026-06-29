# Interview Flashcards: Logistic Regression

This document contains 10 core, production-focused interview questions and answers designed to prepare you for classification and ML design interviews by referencing our real-world scenarios.

---

### Q1: Why can't we use Mean Squared Error (MSE) as the loss function for Logistic Regression? What happens to the math/loss landscape?
**Answer:**
If we plug the sigmoid prediction $f_{w,b}(x) = g(w \cdot x + b)$ into the MSE cost function:
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( g(w \cdot x^{(i)} + b) - y^{(i)} \right)^2$$
The resulting optimization landscape is **non-convex**.
- **The Math Issue:** Because the sigmoid function is non-linear and flat at its tails, the second derivative (Hessian matrix) of this cost function is not positive semi-definite. Geometrically, this introduces multiple local minima, saddle points, and flat plateaus.
- **The Optimization Failure:** If we run gradient descent, the optimizer can get trapped in a suboptimal local minimum depending on weight initialization. 
- **The Fix:** Using Log-Loss mathematically guarantees that the cost function is convex, ensuring gradient descent converges to the single global minimum.

---

### Q2: What is the relationship between Logistic Regression and the concept of "Log-Odds"?
**Answer:**
Logistic Regression is fundamentally a linear model that predicts **Log-Odds** (also called logits).
- **The Connection:** We define the Odds of an event as the probability it happens divided by the probability it doesn't: $\frac{p}{1-p}$. Taking the natural log gives the Log-Odds: $\log\left(\frac{p}{1-p}\right)$.
- **The Math:** If we set the log-odds equal to a linear model:
  $$\log\left(\frac{f_{w,b}(x)}{1 - f_{w,b}(x)}\right) = w \cdot x + b$$
  And solve for $f_{w,b}(x)$, we mathematically derive the sigmoid activation function:
  $$f_{w,b}(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$
- **The Takeaway:** The model assumes the relationship between features $x$ and the log-odds of class 1 is linear. The non-linear sigmoid is just a wrapper to squeeze these log-odds back into a probability interval $[0, 1]$.

---

### Q3: If your model has a high ROC-AUC but poor accuracy at a 0.5 threshold, what does that tell you, and how do you fix it?
**Answer:**
- **What it tells you:** A high ROC-AUC means the model is highly effective at **ranking** samples (it correctly assigns higher probability scores to positive samples than negative samples). However, poor accuracy at $0.5$ indicates that the decision threshold is poorly calibrated relative to the class distribution. For example, if you have a class imbalance, the model might assign positive samples a probability of $0.12$ and negative samples $0.02$. While the ranking is perfect (yielding high ROC-AUC), thresholding at $0.5$ classifies everything as $0$, yielding poor accuracy.
- **How to fix it:** You need to tune the decision threshold. Plot a Precision-Recall Curve or ROC Curve and identify the optimal threshold (e.g., $0.08$) that maximizes your target metric (such as F1-score or balanced accuracy) rather than blindly using $0.5$.

---

### Q4: What is "Perfect Separation" in Logistic Regression, how does it destroy gradient stability, and how does regularization fix it?
**Answer:**
- **What it is:** Perfect separation occurs when one or more features (or a combination of them) perfectly splits the positive and negative targets in your training set (e.g., feature $x_1 > 5.0$ always leads to $y=1$, and $x_1 \le 5.0$ leads to $y=0$).
- **The Optimization Bug:** To drive the Log-Loss to absolute zero, the model tries to output predictions of exactly $1.0$ and $0.0$. Since the sigmoid function $g(z)$ only reaches these boundaries at $z \rightarrow \infty$ and $z \rightarrow -\infty$, the optimizer will drive the weight $w_1$ toward infinity. This causes gradient updates to oscillate, prevents convergence, and inflates the standard errors to infinity.
- **How Regularization Fixes It:** $L_2$ Regularization (Ridge) adds a penalty term $\frac{\lambda}{2m} \|w\|_2^2$ to the loss. This penalty dominates as weights grow large, capping the weights at a stable, finite value.

---

### Q5: In a highly imbalanced dataset (e.g., 1% fraud, 99% non-fraud), why is optimizing for raw "Accuracy" dangerous, and what metric should you track instead?
**Answer:**
- **Why Accuracy is Dangerous:** On a $99\%$ negative dataset, a dummy model that predicts $0$ for every single input will achieve $99\%$ accuracy. It is a completely useless model for the business, but looks mathematically perfect under the accuracy metric.
- **What to track instead:**
  - **Precision:** To measure how many flagged cases are actually fraud (minimizes false alarms).
  - **Recall:** To measure what percentage of actual fraud was detected (minimizes missed fraud).
  - **F1-Score:** The harmonic mean of precision and recall, balancing the two.
  - **Precision-Recall AUC (PR-AUC):** Evaluating the area under the Precision-Recall curve is highly recommended for imbalanced datasets, as it ignores True Negatives and focuses entirely on the minority class.

---

### Q6: How does L1 ($L_1$) vs L2 ($L_2$) regularization alter the weights vector in Logistic Regression, and when would you use one over the other for a deployment payload?
**Answer:**
- **L1 Regularization (Lasso):** Adds a penalty proportional to the absolute values of the weights ($\sum |w_j|$). The constant gradient force pulls weaker weights all the way to **exactly zero**, creating a sparse model.
- **L2 Regularization (Ridge):** Adds a penalty proportional to the squared values of the weights ($\sum w_j^2$). It shrinks weights close to zero but **never forces them to exactly zero**.
- **Deployment Choices:**
  - **Use L1 (Lasso) when:** You have a massive number of features, limited storage/RAM, or strict latency constraints. Lasso will prune redundant features, allowing you to completely drop those inputs from the feature-engineering pipeline, resulting in a smaller, faster inference payload.
  - **Use L2 (Ridge) when:** Features are highly correlated, or you suspect most features contain a small amount of predictive signal. Ridge handles multicollinearity much more stably than Lasso.

---

### Q7: How does Logistic Regression handle non-linear decision boundaries? Give an engineering example.
**Answer:**
Logistic Regression is natively a linear classifier. However, we can handle non-linear boundaries by applying **polynomial feature engineering** at the data pipeline layer before feeding features to the model.
- **Engineering Example:**
  Suppose you have two features, $x_1$ and $x_2$, and the classes are separated by a circle centered at the origin. A linear model $w_1 x_1 + w_2 x_2 + b = 0$ will fail completely.
  To solve this, we generate crossed features: $x_3 = x_1^2$ and $x_4 = x_2^2$.
  The model's linear output becomes:
  $$z = w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2 + b$$
  During training, the OLS optimizer will learn weights $w_1=0, w_2=0, w_3=1, w_4=1, b=-9$.
  The decision boundary ($z=0$) becomes:
  $$x_1^2 + x_2^2 - 9 = 0 \implies x_1^2 + x_2^2 = 9$$
  This represents a circular decision boundary of radius 3. The pipeline handles the non-linearity while keeping the training algorithm convex and fast.

---

### Q8: If an interviewer asks you to build a multi-class classifier using native binary Logistic Regression blocks, how would you architect the system (One-vs-Rest)?
**Answer:**
I would implement a **One-vs-All (OvA)** architecture:
1. **Training Stage:** For $K$ target classes, I would train $K$ separate binary logistic regression models.
   - For classifier $k$, we label all training samples of class $k$ as $1$, and all other classes as $0$.
   - Each model learns its own weights $w^{(k)}$ and bias $b^{(k)}$.
2. **Inference Stage:** When a new sample $x$ arrives:
   - Feed it to all $K$ models to compute $K$ class probabilities: $f_{w^{(k)},b^{(k)}}(x) = P(y=k | x)$.
   - Outputs will not sum to 1. We choose the class that returns the highest probability:
     $$\hat{y} = \arg\max_{k} f_{w^{(k)},b^{(k)}}(x)$$

---

### Q9: Why is Logistic Regression highly favored in click-through rate (CTR) prediction engines and banking risk pipelines over complex tree-based ensembles? (Focus on inference speed and probability calibration).
**Answer:**
1. **Microsecond Inference Speed:** 
   Evaluating a tree ensemble requires traversing hundreds of decision trees, which involves conditional branching and memory jumps. For a logistic regression model, predicting a probability is a single vectorized dot product ($w \cdot x + b$) followed by a sigmoid lookup. This executes in microseconds, which is crucial for high-throughput systems (like ad bidding).
2. **Natural Probability Calibration:**
   The log-loss objective function directly optimizes for maximum likelihood, making the output probability $f_{w,b}(x)$ mathematically calibrated. If the model predicts a click probability of $0.15$ on 1,000 transactions, approximately 150 of them will actually result in a click. Tree-based ensembles (like Random Forests) output averaged voting fractions, which are notoriously uncalibrated and require post-hoc calibration (e.g., Platt scaling) before they can be trusted for risk calculations.

---

### Q10: If your production input features drift significantly (e.g., user income distribution changes), how does it affect the sigmoid output, and how do you re-calibrate your prediction probabilities?
**Answer:**
- **How it affects the output:** If user income increases over time, the feature $x_{\text{income}}$ will drift right. This increases the dot product $w \cdot x + b$, pushing $z$ into the positive tail of the sigmoid function. The model will start outputting high probability predictions ($\approx 1.0$), even if the underlying relationship between income and the target label hasn't changed. The probabilities are no longer calibrated.
- **How to re-calibrate:**
  1. **Platt Scaling:** Train a simple 1D logistic regression model on top of your model's outputs using a new validation set: $P(y=1) = g(A \cdot f_{w,b}(x) + B)$.
  2. **Isotonic Regression:** Fit a non-parametric monotonic function to map raw model outputs to calibrated probabilities. This is highly effective if you have enough calibration data.
  3. **Rolling Standardization:** Scale incoming features using a rolling Z-score window so the input distribution stays constant.
  4. **Retraining:** Trigger an automated retraining pipeline on the newly shifted dataset to adjust the weights and bias.
