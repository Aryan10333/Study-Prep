# Classification Metrics and Thresholds

Once a logistic regression model is trained, we must decide how to evaluate it. This requires selecting the correct prediction thresholds and evaluating the model using robust metrics.

---

## 1. Threshold Tuning in Production

The standard $0.5$ threshold is mathematically convenient but rarely optimal for production business logic. Depending on the cost of false positives vs. false negatives, we adjust this threshold.

```
       Maximize Recall (Lower Threshold)      Maximize Precision (Higher Threshold)
       [0.0]-------(0.15)-----------------------------------------[1.0]
                   ^
                   More positive predictions (More False Positives, Fewer False Negatives)

       [0.0]-----------------------------------------(0.85)-------[1.0]
                                                     ^
                                                     More conservative (Fewer False Positives, More False Negatives)
```

### Scenario A: Maximizing Recall (Lowering the Threshold, e.g., $0.15$)
- **Use Case:** Credit Card Fraud Detection or Cancer Diagnosis.
- **Business Logic:** Missing a fraudulent transaction of $\$10,000$ or missing a tumor (False Negative) is catastrophic. However, a false alarm (False Positive) merely results in a minor inconvenience (sending a text verification or running a follow-up test).
- **Action:** We slide the threshold down. We flag a transaction as fraud if $f_{w,b}(x) \ge 0.15$. This increases **Recall** at the cost of **Precision**.

### Scenario B: Maximizing Precision (Raising the Threshold, e.g., $0.85$)
- **Use Case:** Spam Filters or Automatic Push Notifications.
- **Business Logic:** If the system sends a push notification to a user (positive prediction), it must be relevant. Sending irrelevant notifications (False Positives) annoys users and leads to app uninstalls. Missing a notification (False Negative) is acceptable.
- **Action:** We slide the threshold up. We only classify an email as spam if $f_{w,b}(x) \ge 0.85$. This increases **Precision** at the cost of **Recall**.

---

## 2. The Evaluation Toolkit

To measure classification performance, we use several metrics derived from the **Confusion Matrix**:

| | Predicted: $0$ | Predicted: $1$ |
| --- | --- | --- |
| **Actual: $0$** | True Negative (TN) | False Positive (FP) (Type I Error) |
| **Actual: $1$** | False Negative (FN) (Type II Error) | True Positive (TP) |

### 1. Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Intuition:** "Out of all examples the model predicted as positive, what fraction were actually positive?"
- **Usage:** Focus on this when the cost of a False Positive is high.

### 2. Recall (Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Intuition:** "Out of all actual positive examples, what fraction did the model successfully identify?"
- **Usage:** Focus on this when the cost of a False Negative is high.

### 3. F1-Score
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Intuition:** The harmonic mean of precision and recall. A model must perform well on both metrics to get a high F1-score.
- **Usage:** Useful when you want to balance precision and recall on imbalanced datasets.

---

## 3. ROC-AUC: Receiver Operating Characteristic - Area Under the Curve

The ROC curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at every possible threshold from $0.0$ to $1.0$.

- **TPR (Recall):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **FPR:** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ (Out of all actual negatives, what fraction did we flag as positive?).

```
   TPR
    ^       .---''''
    |     ./         
    |    /           
    |   /   <-- Model ROC Curve (AUC = 0.85)
    |  /             
    | /  _ _ _ _ _ _  (Random Guessing, AUC = 0.5)
    |/_______________> FPR
   0.0             1.0
```

### What does ROC-AUC actually measure?
Mathematically, the Area Under the ROC Curve (ROC-AUC) represents:
> **"The probability that the model will rank a randomly chosen positive example higher than a randomly chosen negative example."**

- An $\text{AUC} = 0.5$ means the model performs no better than random guessing.
- An $\text{AUC} = 1.0$ means the model has perfect separation.

### Why is ROC-AUC robust against Class Imbalance?
In production, you often encounter highly imbalanced datasets (e.g., $99\%$ negative, $1\%$ positive). In this scenario, raw **Accuracy** is a deceptive metric (a dummy model that always predicts $0$ gets $99\%$ accuracy). 

ROC-AUC is robust because:
- **TPR** is calculated using only actual positive examples ($\text{TP} + \text{FN}$).
- **FPR** is calculated using only actual negative examples ($\text{FP} + \text{TN}$).

Because these ratios are computed within their respective actual classes, the ratio of positive to negative samples in the dataset does not warp the metric. It measures the quality of the model's **ranking capability**, independent of the threshold.
