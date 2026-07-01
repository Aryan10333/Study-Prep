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
- **What it tells you:** ROC-AUC is a **threshold-invariant** metric. It measures the quality of the model's ranking (the probability that a random positive instance is scored higher than a random negative instance). However, Accuracy is a **threshold-dependent** metric. If your dataset has a strong class imbalance (e.g., 1% positive), the model's calibrated probabilities might all lie between $0.00$ and $0.15$. The ranking could be perfect (yielding an ROC-AUC of 0.99), but thresholding at $0.5$ classifies every sample as class $0$. This yields a raw accuracy of $99\%$ (equivalent to a dummy model) but catches $0$ true positive cases.
- **How to fix it (Threshold Tuning via Cost Matrix):**
  Instead of using $0.5$, we select an optimal threshold $\tau^*$ by defining a business cost matrix:
  $$\tau^* = \frac{\text{Cost}(\text{FP})}{\text{Cost}(\text{FP}) + \text{Cost}(\text{FN})}$$
  - *Example:* If a False Negative (missing a fraud event) costs \$1,000, and a False Positive (auditing a genuine transaction) costs \$50:
    $$\tau^* = \frac{50}{50 + 1000} \approx 0.0476$$
    Flagging cases with a predicted probability $\ge 0.0476$ mathematically minimizes the total business loss. We can plot a Precision-Recall or cost curve and select this optimal threshold.

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
- **What to track instead (The Imbalanced Metric Suite):**
  1. **Precision:** Measures how many flagged cases are actually fraud ($TP / (TP + FP)$) to minimize support/false alarm overhead.
  2. **Recall (Sensitivity):** Measures what percentage of actual fraud was detected ($TP / (TP + FN)$) to minimize chargeback fees.
  3. **F1-Score:** The harmonic mean of precision and recall:
     $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
     - *Why Harmonic Mean?* If a model has 100% precision and 0% recall, the arithmetic mean is 50%, which masks the model's failure. The harmonic mean resolves this by evaluating to $0\%$, correctly penalizing the model.
  4. **Precision-Recall AUC (PR-AUC):**
     - *Why PR-AUC is superior to ROC-AUC here:* ROC-AUC uses the False Positive Rate ($FP / (FP + TN)$) on the x-axis. In a 10,000-sample dataset where 9,900 are negative, making 100 False Positives yields a tiny FPR:
       $$\text{FPR} = \frac{100}{100 + 9800} \approx 1\%$$
       This causes the ROC curve to look optimistic. However, if there are only 100 actual positives, those 100 False Positives crush the Precision to only 50%.
     - **PR-AUC ignores True Negatives ($TN$) in its calculations.** It focus entirely on the minority class, making it much more sensitive and robust for evaluating imbalanced distributions.

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
- **How it affects the output:** If user incomes double due to inflation, the input vector $x_{\text{income}}$ shifts to larger values. The linear combination $z = w \cdot x + b$ shifts right. Because the sigmoid function:
  $$g(z) = \frac{1}{1 + e^{-z}}$$
  asymptotes to $1.0$ as $z \rightarrow \infty$, the output probabilities will cluster near $1.0$. The model becomes extremely confident in predicting class 1, even if the underlying probability of class 1 has not changed. The probabilities are no longer calibrated.
- **How to re-calibrate:**
  1. **Platt Scaling (Logistic Calibration):** Train a simple 1D logistic regression model on top of your model's raw output logits using a new, non-overlapping validation set:
     $$P(y=1) = \frac{1}{1 + e^{-(A \cdot \text{logit} + B)}}$$
     Where the parameters $A$ and $B$ are learned via maximum likelihood estimation. This scales and shifts the logits to match the new population prior.
  2. **Isotonic Regression:** Fit a non-parametric, piecewise constant monotonic function to map raw model outputs to calibrated probabilities.
     - *Design Trade-off:* Use Isotonic Regression when you have a large calibration dataset ($>1000$ samples) because it is flexible and makes no assumptions about the calibration shape. Use Platt Scaling when you have small calibration splits ($<1000$ samples) to prevent overfitting.
  3. **Recalibration via Downsampling Offset:** If the calibration shift is due to downsampling the majority class in training, adjust the logits directly:
     $$\text{logit}_{\text{calibrated}} = \text{logit}_{\text{model}} + \log(p_{\text{down}})$$
  4. **Retraining:** Trigger an automated retraining pipeline on the newly shifted dataset to adjust the weights and bias.
