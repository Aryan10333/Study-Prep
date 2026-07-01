# AdaBoost: Adaptive Boosting Mathematical Mechanics

This document details the mathematical framework, sample weight update rules, and decision stump aggregation mechanics of the AdaBoost algorithm.

---

## 1. Boosting vs. Bagging

While Bagging trains independent estimators in parallel to reduce variance, Boosting trains sequential estimators to reduce bias.

| Dimension | Bagging (e.g., Random Forest) | Boosting (e.g., AdaBoost) |
|:---:|:---:|:---:|
| **Tree Training** | Parallel and independent. | Sequential; each tree depends on prior errors. |
| **Data Sample** | Independent bootstrap splits. | Re-weighted versions of the training set. |
| **Aggregator** | Simple majority voting or averaging. | Weighted voting based on estimator accuracy. |
| **Core Target** | Minimizes Variance (keeps Bias stable). | Minimizes Bias (can increase Variance). |

---

## 2. The AdaBoost Algorithm Steps

Given a training set of $m$ samples $(x^{(i)}, y^{(i)})$ where $y^{(i)} \in \{-1, 1\}$:

### Step 1: Initialize Sample Weights
Initialize the sample weights vector $w$ uniformly:
$$w_i^{(1)} = \frac{1}{m}, \quad i = 1, 2, \dots, m$$

---

### Step 2: Sequential Training Loop
For $t = 1, 2, \dots, T$:

1. **Fit a Weak Learner:** Train a classifier $G_t(x) \in \{-1, 1\}$ (typically a single-split decision stump) on the training data using the current sample weights $w^{(t)}$.
2. **Compute Weighted Error Rate ($\epsilon_t$):**
   $$\epsilon_t = \frac{\sum_{i=1}^m w_i^{(t)} I(y^{(i)} \ne G_t(x^{(i)}))}{\sum_{i=1}^m w_i^{(t)}}$$
3. **Compute Stage Weight ($\alpha_t$):**
   This represents the classifier's voting power in the final ensemble:
   $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
   - *Intuition:* If error $\epsilon_t \to 0$ (perfect learner), $\alpha_t \to \infty$. If error $\epsilon_t = 0.5$ (random guess), $\alpha_t = 0$.
4. **Update Sample Weights:**
   $$w_i^{(t+1)} = w_i^{(t)} \exp\left( -\alpha_t y^{(i)} G_t(x^{(i)}) \right)$$
   - If correct prediction ($y^{(i)} G_t = 1$): $w_i \rightarrow w_i e^{-\alpha_t}$ (weight decreases).
   - If incorrect prediction ($y^{(i)} G_t = -1$): $w_i \rightarrow w_i e^{\alpha_t}$ (weight increases).
5. **Renormalize Weights:**
   Ensure the weights sum to 1:
   $$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^m w_j^{(t+1)}}$$

---

### Step 3: Final Ensemble Consensus
The final model prediction is the sign of the weighted sum of all base estimators:
$$G(x) = \text{sign}\left( \sum_{t=1}^T \alpha_t G_t(x) \right)$$

---

## 3. Real-World Scenario: Healthcare Patient Disease Diagnosis

### The Problem
We want to classify patients as diabetic (`1`) or non-diabetic (`-1`) based on BMI ($x_1$) and Age ($x_2$).
Let's trace **Iteration 1** of AdaBoost on $m = 4$ patients.

---

### Step-by-Step Hand Calculations

#### 1. Initialize Sample Weights
Since $m = 4$, initial weights are:
$$w_i^{(1)} = 0.25 \quad \text{for all } i \in \{1, 2, 3, 4\}$$

Our starting dataset is:

| Patient ($i$) | BMI ($x_1$) | Age ($x_2$) | True Label ($y$) | Weight ($w^{(1)}$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 32 | 45 | 1 | 0.25 |
| 2 | 20 | 25 | -1 | 0.25 |
| 3 | 28 | 35 | 1 | 0.25 |
| 4 | 30 | 55 | -1 | 0.25 |

---

#### 2. Fit Weak Learner ($G_1$)
A decision stump splits the data on the threshold **BMI $\le 25$**:
- Predicts $G_1(x) = -1$ if BMI $\le 25$
- Predicts $G_1(x) = 1$ if BMI $> 25$

Let's evaluate the predictions of $G_1$:
- **Patient 1 (BMI 32):** Predicts $1$. True $y = 1$. (Correct)
- **Patient 2 (BMI 20):** Predicts $-1$. True $y = -1$. (Correct)
- **Patient 3 (BMI 28):** Predicts $1$. True $y = 1$. (Correct)
- **Patient 4 (BMI 30):** Predicts $1$. True $y = -1$. (**Incorrect**)

---

#### 3. Compute Weighted Error ($\epsilon_1$)
The only misclassified sample is Patient 4:
$$\epsilon_1 = \frac{\sum w_i^{(1)} I(y^{(i)} \ne G_1)}{\sum w_i^{(1)}} = \frac{0.25 \times 1}{0.25 + 0.25 + 0.25 + 0.25} = 0.25$$

---

#### 4. Compute Stage Weight ($\alpha_1$)
$$\alpha_1 = \frac{1}{2} \ln\left( \frac{1 - 0.25}{0.25} \right) = \frac{1}{2} \ln(3) \approx 0.5 \times 1.0986 \approx 0.5493$$

Stump $G_1$ has a voting power of $\approx 0.549$ in the final ensemble.

---

#### 5. Update Sample Weights (Before Normalization)
Using the update rule $w_i^{(2)} = w_i^{(1)} e^{-\alpha_t y_i G_1(x_i)}$:
- **Patients 1, 2, 3 (Correct):**
  $$w_i^{(2)} = 0.25 \times e^{-0.5493} = 0.25 \times 0.5774 \approx 0.1444$$
- **Patient 4 (Incorrect):**
  $$w_4^{(2)} = 0.25 \times e^{0.5493} = 0.25 \times 1.7320 \approx 0.4330$$

- **Sum of Weights:**
  $$\sum_{j=1}^4 w_j^{(2)} = 3(0.1444) + 0.4330 = 0.4332 + 0.4330 = 0.8662$$

---

#### 6. Normalize Weights
Ensure the new weights sum to $1.0$:
- **Patients 1, 2, 3 (Correct):**
  $$w_i^{(2)} = \frac{0.1444}{0.8662} \approx 0.1667$$
- **Patient 4 (Incorrect):**
  $$w_4^{(2)} = \frac{0.4330}{0.8662} \approx 0.5000$$

### Interpretation
After Iteration 1, the sample weights are:
`[Patient 1: 0.167, Patient 2: 0.167, Patient 3: 0.167, Patient 4: 0.500]`

Because Patient 4 was misclassified, their weight has doubled from $0.25$ to $0.50$, representing half of the total weight in the entire dataset. In the next iteration ($t=2$), the base learning algorithm will prioritize minimizing classification error on Patient 4 above all others.

---

## 4. Production Realities: Outlier Sensitivity

AdaBoost is highly sensitive to noisy training data and outliers. 

### Why it breaks in production:
If your dataset contains mislabeled inputs (e.g., a non-diabetic patient labeled as diabetic), AdaBoost will fail to classify them correctly in every iteration. The algorithm will sequentially multiply the weights of these noise samples, forcing subsequent weak learners to warp their decision boundaries to fit the noise, leading to overfitting and poor validation generalization.
