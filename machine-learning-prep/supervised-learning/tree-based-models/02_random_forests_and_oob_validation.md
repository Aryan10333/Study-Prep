# Random Forests: OOB Validation & Unbiased Feature Importance

This guide details Random Forest bagging mechanics, Out-of-Bag (OOB) score tracking, and how to identify and resolve high-cardinality feature importance bias.

---

## 1. Out-of-Bag (OOB) Validation

Random Forests train multiple decision trees in parallel using bootstrap samples. Because bootstrapping samples with replacement, approximately **$36.8\%$** of the unique training instances are completely left out of any individual tree's training split. These are the **Out-of-Bag (OOB)** samples.

We use OOB samples to validate the model's accuracy on the fly, eliminating the need to set aside a separate validation split during model construction.

### Python Code: OOB Tracking
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Enable oob_score during instantiation
model_rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

print(f"Out-of-Bag Validation Accuracy: {model_rf.oob_score_:.4f}")
print(f"Generalization Test Accuracy:  {model_rf.score(X_test, y_test):.4f}")
```

### Expected Console Output
```text
Out-of-Bag Validation Accuracy: 0.8843
Generalization Test Accuracy:  0.8900
```

---

## 2. Feature Importance: The Gini MDI Bias Trap

In production pipelines, identifying feature drive is essential. However, the default feature importance metric in Random Forests (Mean Decrease in Impurity, or MDI) contains a severe mathematical bias.

### The Bug
MDI measures the total Gini impurity drop averaged across all tree splits on feature $j$. Continuous or high-cardinality categorical features (e.g., unique customer IDs, timestamps) have many unique values. The greedy split search checks every split midpoint. As a result, the model splits on these high-cardinality features frequently to fit training split noise, inflating their apparent Gini importance.

---

## 3. The Fix: Permutation Feature Importance

Permutation Importance is model-agnostic and unbiased:
1. It measures the validation performance of the model (e.g., F1-score).
2. It shuffles the values of feature $j$ across samples, breaking its connection with target $y$, and re-evaluates the score.
3. If shuffling has no impact on accuracy, the importance is $0.0$, regardless of the feature's cardinality.

### Python Code: MDI vs. Permutation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Generate synthetic dataset with 2 true features and 1 high-cardinality noise column
np.random.seed(42)
m = 200
income = np.random.uniform(20, 150, m)
savings = np.random.uniform(5, 50, m)
# High-cardinality noise: 100% random ID column
random_id = np.arange(1000, 1000 + m)

y = np.where((income + 2 * savings) < 100, 1, 0)
# Add label noise
y[::10] = 1 - y[::10] 

X = pd.DataFrame({'income': income, 'savings': savings, 'random_id': random_id})

model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
model.fit(X, y)

# 1. Gini MDI
gini_imp = model.feature_importances_

# 2. Permutation
perm_imp = permutation_importance(model, X, y, n_repeats=5, random_state=42).importances_mean

df_imp = pd.DataFrame({
    'Feature': X.columns,
    'Gini_MDI': gini_imp,
    'Permutation': perm_imp
})
print(df_imp.to_string(index=False))
```

### Expected Console Output
```text
    Feature  Gini_MDI  Permutation
     income  0.224135     0.184000
    savings  0.245129     0.282000
  random_id  0.530736     0.000000
```

### Diagnostic Visual (MDI vs. Permutation)
The comparison bar chart clearly illustrates how Gini MDI flags the random noise column (`random_id`) as the most important feature, whereas Permutation Importance correctly identifies it as having **$0.0$** predictive signal:

![MDI vs Permutation Importance](images/feature_importance_comparison.png)

---

## 4. Interactive Practice Notebook
To sweep estimator convergence curves and run the permutation benchmark from scratch, open the interactive notebook:
- [02_random_forests_oob_and_feature_importance.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/02_random_forests_oob_and_feature_importance.ipynb)
