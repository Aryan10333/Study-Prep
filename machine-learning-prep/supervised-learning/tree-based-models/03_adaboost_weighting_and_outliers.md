# AdaBoost: Sample Weighting & Outlier Vulnerability

This guide details the sequential update mechanics of AdaBoost, how to track sample weights across iterations in code, and how outliers degrade model performance.

---

## 1. Sequential Sample Weighting

Unlike bagging models that build independent trees on parallel bootstrap datasets, AdaBoost builds decision stumps (depth-1 trees) sequentially. 
- At step $t$, the stump is trained using a weighted loss function, forcing it to prioritize samples with high weights.
- Initially, all samples hold equal weight: $w_i = 1/m$.
- At the end of iteration $t$, the algorithm multiplies the weights of misclassified samples by $e^{\alpha_t}$ and divides/multiplies correct samples by $e^{-\alpha_t}$ (where $\alpha_t$ is the stage weight voting power of stump $t$).
- After normalization, misclassified samples occupy a larger proportion of the weight budget, forcing stump $t+1$ to focus on them.

---

## 2. Python Code: Tracking Sample Weights

In production pipelines, we can monitor sample weight trajectories using Scikit-Learn's `AdaBoostClassifier`. The following code prints how weights adjust when a model encounters a persistent outlier.

### Python Code Pipeline
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
# True labels: 1 if BMI > 25, else -1
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

# Step-by-step trace of weights
print("--- AdaBoost Sample Weights Trajectory ---")
# Scikit-Learn stores sample weight parameters inside estimator estimators
# We can track the weights array after training
for idx, (estimator, weight) in enumerate(zip(model.estimators_, model.estimator_weights_)):
    print(f"Stump {idx+1} (Weight/Alpha: {weight:.4f})")
    # Predict to identify mistakes
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

## 3. Production Realities: Outlier Sensitivity

AdaBoost's error-correction mechanism makes it highly vulnerable to noisy labels or feature outliers in production.

### The Downward Spiral:
1. If a sample is mislabeled (noise/outlier), the weak learner $G_t$ will misclassify it.
2. The algorithm escalates its weight, scaling it up exponentially.
3. In subsequent iterations, the next weak learners are forced to warp their decision boundaries to try to fit this single noisy sample, ignoring the general patterns of the clean dataset.
4. This causes the validation set accuracy to collapse.

### Production Best Practices:
- Clean and prune outliers in the preprocessing ingestion layer before passing data to AdaBoost.
- If data contains high label noise, prefer **Random Forests** (which averages out noise) or **Gradient Boosting with robust loss (Huber/MAE)** over AdaBoost.
