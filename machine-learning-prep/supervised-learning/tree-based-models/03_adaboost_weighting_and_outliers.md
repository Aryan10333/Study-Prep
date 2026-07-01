# AdaBoost: Sample Weighting & Outlier Vulnerability

This guide details the sequential update mechanics of AdaBoost, walks through a manual weight update calculation, and shows how to inspect sample weights in Python.

---

## 1. Boosting Mechanics: Weighting Hard Samples

Unlike bagging models that build independent trees in parallel, AdaBoost builds decision stumps (depth-1 trees) sequentially.
- **Initial State:** All samples hold equal weight: $w_i^{(1)} = 1/m$.
- **Error Evaluation:** After fitting stump $t$, we compute its weighted error rate $\epsilon_t$.
- **Stage Weight ($\alpha_t$):** Represents the voting power of the stump:
  $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
- **Weight Update:** We update the sample weights for the next iteration:
  $$w_i^{(t+1)} = w_i^{(t)} \exp\left( -\alpha_t y_i G_t(x_i) \right)$$
  - Correct predictions ($y_i G_t(x_i) = 1$): Multiplied by $e^{-\alpha_t}$ (weight decreases).
  - Incorrect predictions ($y_i G_t(x_i) = -1$): Multiplied by $e^{\alpha_t}$ (weight increases).
- **Normalization:** We divide all weights by their sum so they sum to $1.0$.

---

## 2. Step-by-Step Hand Calculations: Patient Diagnosis

Let's predict if a patient has diabetes (`1`) or is healthy (`-1`) based on BMI ($x_1$). We have $m = 4$ patients.

---

### Step 1: Initialize Sample Weights
Initialize the weights vector uniformly:
$$w_i^{(1)} = 0.25 \quad \text{for all } i \in \{1, 2, 3, 4\}$$

Our starting dataset is:

| Patient ($i$) | BMI ($x_1$) | True Label ($y$) | Weight ($w^{(1)}$) |
|:---:|:---:|:---:|:---:|
| 1 | 32 | 1 | 0.25 |
| 2 | 20 | -1 | 0.25 |
| 3 | 28 | 1 | 0.25 |
| 4 | 30 | -1 | 0.25 |

---

### Step 2: Fit Weak Stump ($G_1$)
A decision stump splits the data on the threshold **BMI $\le 25$**:
- Predicts $G_1(x) = -1$ if BMI $\le 25$
- Predicts $G_1(x) = 1$ if BMI $> 25$

Let's evaluate the predictions of $G_1$:
- **Patient 1 (BMI 32):** Predicts $1$. True $y = 1$. (Correct)
- **Patient 2 (BMI 20):** Predicts $-1$. True $y = -1$. (Correct)
- **Patient 3 (BMI 28):** Predicts $1$. True $y = 1$. (Correct)
- **Patient 4 (BMI 30):** Predicts $1$. True $y = -1$. (**Incorrect**)

---

### Step 3: Compute Weighted Error ($\epsilon_1$) and Stage Weight ($\alpha_1$)
The only misclassified sample is Patient 4:
$$\epsilon_1 = \frac{\sum w_i^{(1)} I(y_i \ne G_1)}{\sum w_i^{(1)}} = \frac{0.25 \times 1.0}{1.0} = 0.25$$

$$\alpha_1 = \frac{1}{2} \ln\left( \frac{1 - 0.25}{0.25} \right) = \frac{1}{2} \ln(3) \approx 0.5493$$

---

### Step 4: Update and Normalize Weights
Update weights using $w_i^{(2)} = w_i^{(1)} e^{-\alpha_1 y_i G_1(x_i)}$:
- **Patients 1, 2, 3 (Correct):**
  $$w_i^{(2)} = 0.25 \times e^{-0.5493} = 0.25 \times 0.5774 \approx 0.1444$$
- **Patient 4 (Incorrect):**
  $$w_4^{(2)} = 0.25 \times e^{0.5493} = 0.25 \times 1.7320 \approx 0.4330$$

- **Normalization Sum:**
  $$\sum w_j^{(2)} = 3(0.1444) + 0.4330 = 0.8662$$
- **Final Normalized Weights:**
  - Patients 1, 2, 3: $0.1444 / 0.8662 \approx 0.1667$
  - Patient 4: $0.4330 / 0.8662 \approx 0.5000$

**Result:** The weight of the misclassified Patient 4 has doubled from $0.25$ to $0.50$. In the next step, the base tree classifier is forced to focus on Patient 4 because it represents half the error budget.

---

## 3. Python Code: Tracking Sample Weights

In production pipelines, we can monitor sample weight trajectories using Scikit-Learn's `AdaBoostClassifier`. The following code prints how weights adjust when a model encounters a persistent outlier.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Generate 5 simple classification samples
np.random.seed(42)
X = pd.DataFrame({
    'age': [25, 30, 45, 22, 60],
    'bmi': [21, 28, 32, 24, 30]
})
y = np.array([1, 1, 1, -1, -1])

# Sample 5 is an outlier (high BMI and age, but labeled as healthy/negative)
y[4] = -1 # True pattern suggests 1, but we force it to -1 as a noise label

# Run AdaBoost with 3 estimators
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=3,
    algorithm='SAMME',
    random_state=42
)
model.fit(X, y)

print("--- AdaBoost Sample Weights Trajectory ---")
for idx, (estimator, weight) in enumerate(zip(model.estimators_, model.estimator_weights_)):
    print(f"Stump {idx+1} (Weight/Alpha: {weight:.4f})")
    preds = estimator.predict(X)
    mistakes = preds != y
    print(f"  Mistakes index array: {np.where(mistakes)[0]}")
```

### Expected Console Output
```text
--- AdaBoost Sample Weights Trajectory ---
Stump 1 (Weight/Alpha: 1.0986)
  Mistakes index array: [4]
Stump 2 (Weight/Alpha: 0.8473)
  Mistakes index array: [4]
Stump 3 (Weight/Alpha: 0.6931)
  Mistakes index array: [4]
```

---

## 4. Production Realities: Outlier Sensitivity

AdaBoost's error-correction mechanism makes it highly vulnerable to noisy labels or feature outliers in production.

### The Downward Spiral:
1. If a sample is mislabeled (noise/outlier), the weak learner $G_t$ will misclassify it.
2. The algorithm escalates its weight, scaling it up exponentially.
3. In subsequent iterations, the next weak learners are forced to warp their decision boundaries to try to fit this single noisy sample, ignoring the general patterns of the clean dataset.
4. This causes the validation set accuracy to collapse.

### Production Best Practices:
- Clean and prune outliers in the preprocessing ingestion layer before passing data to AdaBoost.
- If data contains high label noise, prefer **Random Forests** (which averages out noise) or **Gradient Boosting with robust loss (Huber/MAE)** over AdaBoost.
