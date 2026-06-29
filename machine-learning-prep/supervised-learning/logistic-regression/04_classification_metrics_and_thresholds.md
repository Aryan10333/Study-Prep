# Classification Metrics and Thresholds

Once a model is trained, we must select prediction thresholds and evaluate performance. This guide compares evaluation metrics and details threshold tuning using a concrete transaction fraud screening scenario.

---

## 1. The Evaluation Toolkit

To measure classification performance, we use metrics derived from the **Confusion Matrix**:

|                 | Predicted: $0$                      | Predicted: $1$                     |
| -----------------| -------------------------------------| ------------------------------------|
| **Actual: $0$** | True Negative (TN)                  | False Positive (FP) (Type I Error) |
| **Actual: $1$** | False Negative (FN) (Type II Error) | True Positive (TP)                 |

### 1. Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Intuition:** "Out of all positive predictions, what fraction were actually positive?"

### 2. Recall (Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Intuition:** "Out of all actual positive examples, what fraction did the model identify?"

### 3. F1-Score
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Intuition:** The harmonic mean of precision and recall. A model must perform well on both metrics to get a high F1-score.

---

## 2. Scenario: Transaction Fraud Screening

Suppose you run the risk engine for an e-commerce platform processing $10,000$ transactions daily. 
- **The Imbalance:** Out of 10,000 transactions, **$100$ are actual fraud** ($y=1$), and $9,900$ are genuine ($y=0$).
- **The Business Costs:**
  - **False Negative (FN) Cost:** Missing fraud costs **\$1,000** (chargeback fee + lost inventory).
  - **False Positive (FP) Cost:** Blocking a genuine user costs **\$50** (support overhead + lost customer LTV).

---

## 3. Threshold Tuning: The Financial Balance Sheet

Let's compare the default threshold of $0.5$ against an optimized threshold of $0.08$.

### Threshold A: Standard Threshold ($0.50$)
At this threshold, the model is conservative about blocking transactions, yielding high Precision but poor Recall.

*   **Confusion Matrix:**
    - $\text{TP} = 40$, $\text{FN} = 60$, $\text{FP} = 5$, $\text{TN} = 9895$
*   **Performance Metrics:**
    - $\text{Precision} = \frac{40}{40 + 5} = 88.9\%$
    - $\text{Recall} = \frac{40}{40 + 60} = 40.0\%$
*   **Financial Cost:**
    $$\text{Cost}_{0.50} = (\text{FN} \times \$1000) + (\text{FP} \times \$50) = (60 \times \$1000) + (5 \times \$50) = \mathbf{\$60,250}$$

### Threshold B: Tuned Threshold ($0.08$)
We slide the threshold down. We flag transactions as fraud if the predicted probability $f_{w,b}(x) \ge 0.08$. This increases Recall at the expense of Precision.

*   **Confusion Matrix:**
    - $\text{TP} = 95$, $\text{FN} = 5$, $\text{FP} = 200$, $\text{TN} = 9700$
*   **Performance Metrics:**
    - $\text{Precision} = \frac{95}{95 + 200} = 32.2\%$
    - $\text{Recall} = \frac{95}{95 + 5} = 95.0\%$
*   **Financial Cost:**
    $$\text{Cost}_{0.08} = (5 \times \$1000) + (200 \times \$50) = \$5,000 + \$10,000 = \mathbf{\$15,000}$$

---

## 4. Business Metrics Outcome Comparison

| Metric / Cost | Default Threshold ($0.50$) | Tuned Threshold ($0.08$) |
| :--- | :--- | :--- |
| **Precision** | **88.9%** (Very high) | **32.2%** (Low) |
| **Recall** | **40.0%** (Poor) | **95.0%** (Very high) |
| **Chargeback Cost (FN)** | \$60,000 | \$5,000 |
| **Support Overhead (FP)** | \$250 | \$10,000 |
| **Total Cost to Bank** | **\$60,250** | **\$15,000** |
| **Net Savings** | - | **\$45,250 Daily Savings** |

**The Takeaway:** Even though precision dropped from 88.9% to 32.2%, the tuned threshold of 0.08 saved the company **\$45,250 per day** because the cost of missing fraud is 20 times higher than blocking a genuine user.

---

## 5. ROC-AUC: Ranking Probability

The ROC curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at every threshold:
- $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$

### What ROC-AUC Actually Measures
If you draw one random positive sample (fraud) and one random negative sample (genuine) from the database:
> **"ROC-AUC represents the probability that the model will assign a higher predicted score to the positive instance than the negative instance."**

### Robustness against Class Imbalance
ROC-AUC is robust against class imbalance because:
- **TPR** is calculated strictly within actual positive examples ($\text{TP} + \text{FN}$).
- **FPR** is calculated strictly within actual negative examples ($\text{FP} + \text{TN}$).

Because these ratios are computed within their respective actual classes, the ratio of positive to negative samples in the dataset does not warp the metric. This makes it a pure measure of ranking quality, unlike raw accuracy.
