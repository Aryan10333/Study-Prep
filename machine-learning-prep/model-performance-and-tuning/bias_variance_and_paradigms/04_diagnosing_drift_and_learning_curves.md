# Diagnosing Bottlenecks with Learning Curves

In production ML pipelines, when a model fails to meet performance SLAs, teams often waste time collecting more data or tuning parameters without diagnostics. **Learning Curves** are the primary tool used to determine if a model suffers from high bias or high variance. This guide demonstrates how to analyze these curves using a concrete vehicle collision prediction scenario.

---

## 1. Under the Hood of a Learning Curve

A learning curve plots performance error against the **number of training examples ($m$)** used to train the model.

- **Training Error Behavior:** 
  - When $m$ is small, training error is extremely low. The model has sufficient capacity to interpolate the small dataset perfectly.
  - As $m$ increases, it becomes harder for the model to fit all points. Training error increases, eventually plateauing as the dataset size grows beyond the model's capacity to memorize.
- **Validation Error Behavior:**
  - When $m$ is small, validation error is high. The model overfits the small training sample, failing on unseen data.
  - As $m$ increases, the model generalizes better, and validation error decreases.

---

## 2. Scenario: Vehicle Collision Prediction

You are predicting the probability of a vehicle collision ($y$, binary classification) based on telemetry data:
- $x_1$: Distance to the car ahead (in meters).
- $x_2$: Current vehicle speed (in km/h).

Your production target SLA is a Log-Loss of **$< 0.15$**.

---

## 3. Diagnostics Case A: High Bias (Underfitting)

You train a simple, linear Logistic Regression model: $f_{w,b}(x) = g(w \cdot x + b)$.

### The Experimental Logs

| Training Samples ($m$) | Training Log-Loss | Validation Log-Loss | Visual Trend |
| :--- | :--- | :--- | :--- |
| 100 | 0.05 | 0.45 | Wide gap, low train loss |
| 500 | 0.12 | 0.32 | Gap closing |
| 1,000 | 0.22 | 0.26 | Curves converging |
| 5,000 | **0.24** | **0.25** | Plateau reached (Above SLA) |
| 10,000 | **0.24** | **0.25** | No change (Plateau continues) |

### Analyzing the High-Bias Regime

```text
   Log-Loss
    ^
    |      \
    |       \__ . . . . . . . . . .  Validation Loss (0.25)
    |          \___________________  Training Loss (0.24)
    |                                
    | - - - - - - - - - - - - - - -  Target SLA (0.15)
    +------------------------------------> Training Samples (m)
```

- **The Diagnosis:** The training and validation errors converge quickly to a high plateau ($0.25$) that is far above the target SLA ($0.15$). The gap between the two curves is tiny ($0.01$).
- **The Data Collection Fallacy:** Looking at the transition from $5,000$ to $10,000$ samples, the validation loss did not budge. **Collecting more data in this scenario is an engineering waste**. The model has already reached its maximum capacity.
- **The Remedy:** Increase complexity. Add polynomial features (e.g., $x_2^2$, velocity squared, which is physically proportional to braking distance) or decrease regularization strength.

---

## 4. Diagnostics Case B: High Variance (Overfitting)

You train a highly complex, deep neural network on the same features without regularization.

### The Experimental Logs

| Training Samples ($m$) | Training Log-Loss | Validation Log-Loss | Visual Trend |
| :--- | :--- | :--- | :--- |
| 100 | 0.01 | 0.65 | Massive gap |
| 500 | 0.02 | 0.45 | Large gap |
| 1,000 | 0.03 | 0.35 | Gap narrowing slowly |
| 5,000 | 0.05 | 0.22 | Trend looks promising |
| 10,000 | **0.06** | **0.14** | Gap closed, SLA met! |

### Analyzing the High-Variance Regime

```text
   Log-Loss
    ^
    |      \
    |       \  .  .  .  .  .  .  .   Validation Loss (0.14)
    |        \
    |         \                      <-- Persistent Gap Narrowing
    |          \___________________
    | - - - - - - - - - - - - - - -  Target SLA (0.15)
    |______________________________  Training Loss (0.06)
    +------------------------------------> Training Samples (m)
```

- **The Diagnosis:** The training error remains very low ($0.06$), while the validation error decreases steadily as we add more samples. There is a persistent gap between the curves, but the trajectory is moving downward.
- **The Data Collection Value:** Increasing training samples from $5,000$ to $10,000$ successfully pulled the validation loss down from $0.22$ to $0.14$, breaching the target SLA. In a high-variance regime, **collecting more data is a highly effective way to stabilize performance**.
- **Alternative Remedies:** If data collection is expensive, you can achieve the same curve adjustment by increasing L2 regularization, adding dropout, or simplifying model capacity.
