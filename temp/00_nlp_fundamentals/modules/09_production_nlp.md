# Module 09: Production NLP & Model Maintenance

Moving NLP models from research environments to production requires managing deployment constraints and monitoring data drift. This module details ingestion pipelines, explains data and concept drift, and maps out Population Stability Index (PSI) equations and diagnostics.

---

## 1. Ingestion Patterns: Batch vs. Real-Time Inference

Production systems deploy models using one of two ingestion patterns:
- **Batch Inference**: Processes large collections of text offline (e.g. running sentiment analysis over nightly logs).
  - *Optimization*: High throughput; optimized via parallel worker nodes and large batch sizes to maximize GPU utilization.
- **Real-Time Inference**: Processes streaming user queries with strict latency limits (e.g. search query auto-completion).
  - *Optimization*: Low latency; optimized via model quantization, single-sample processing, and caching.

---

## 2. Monitoring Production Drift: Data vs. Concept Drift

Once deployed, models experience performance decay due to environmental changes:

| Drift Type | Definition | Mathematical Concept | Concrete Example |
| :--- | :--- | :--- | :--- |
| **Data Drift** (Covariate Shift) | The distribution of input features changes, but input-to-label relationships remain the same. | $P(X_{\text{prod}}) \neq P(X_{\text{train}})$ | A customer service classifier trained on formal email text starts processing informal chat logs containing slang and emojis. |
| **Concept Drift** | The mapping relationship between inputs and labels shifts over time. | $P(Y \mid X_{\text{prod}}) \neq P(Y \mid X_{\text{train}})$ | The word `"viral"` shifts from representing a negative healthcare quarantine tag (2019) to a positive marketing campaign indicator (2021). |

---

## 3. Population Stability Index (PSI)

The Population Stability Index (PSI) is a metric used to measure how much a target distribution has shifted away from a reference distribution. It is widely used to monitor covariate shift in production models.

### Mathematical Formulation:
$$\text{PSI} = \sum_{i=1}^B \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$

Where:
- $B$ is the total number of bins or categories.
- $Q_i$ is the expected proportion of samples in bucket $i$ (from the training/reference baseline).
- $P_i$ is the actual proportion of samples in bucket $i$ (from the production window).

### Stability Thresholds (Drift Levels):
- **$\text{PSI} < 0.10$**: **No significant change.** The production distribution matches the baseline; no action is required.
- **$0.10 \le \text{PSI} \le 0.25$**: **Moderate shift.** The distribution has begun to drift; monitor the inputs closely.
- **$\text{PSI} > 0.25$**: **Significant shift.** The distribution has drifted severely. This requires immediate action, such as triggering an automated retraining loop with recent production samples or re-tuning classification thresholds.

---

## 4. Step-by-Step Hand-Calculation of PSI:
Let's calculate the Population Stability Index for a 3-category classifier output distribution (e.g. token classifications: positive, negative, neutral) comparing the production window $P$ against the training baseline $Q$:
- Expected baseline proportions: $\mathbf{q} = [0.60, \ 0.30, \ 0.10]^T$
- Actual production proportions: $\mathbf{p} = [0.50, \ 0.30, \ 0.20]^T$

#### 1. Compute Bucket 1 (Positive):
- Expected proportion ($Q_1$): $0.60$
- Actual proportion ($P_1$): $0.50$
- Difference ($P_1 - Q_1$): $0.50 - 0.60 = -0.10$
- Ratio ($P_1 / Q_1$): $0.50 / 0.60 \approx 0.8333$
- Log ratio ($\ln(0.8333)$): $-0.1823$
- Bucket 1 contribution:
  $$\text{PSI}_1 = (-0.10) \times (-0.1823) = 0.0182$$

#### 2. Compute Bucket 2 (Negative):
- Expected proportion ($Q_2$): $0.30$
- Actual proportion ($P_2$): $0.30$
- Difference ($P_2 - Q_2$): $0.30 - 0.30 = 0.00$
- Ratio ($P_2 / Q_2$): $1.0000$
- Log ratio ($\ln(1.0000)$): $0.0000$
- Bucket 2 contribution:
  $$\text{PSI}_2 = 0.00 \times 0.0000 = 0.0000$$

#### 3. Compute Bucket 3 (Neutral):
- Expected proportion ($Q_3$): $0.10$
- Actual proportion ($P_3$): $0.20$
- Difference ($P_3 - Q_3$): $0.20 - 0.10 = 0.10$
- Ratio ($P_3 / Q_3$): $0.20 / 0.10 = 2.0000$
- Log ratio ($\ln(2.0000)$): $0.6931$
- Bucket 3 contribution:
  $$\text{PSI}_3 = 0.10 \times 0.6931 = 0.0693$$

#### 4. Total PSI score:
$$\text{PSI} = \text{PSI}_1 + \text{PSI}_2 + \text{PSI}_3 = 0.0182 + 0.0000 + 0.0693 = 0.0875$$

##### Findings & Interpretation:
The computed PSI score is $0.0875$. Since $\text{PSI} < 0.10$, the production distribution is considered stable and has not undergone significant drift. The model remains calibrated, and no retraining or diagnostic action is required.

---

## 5. The Production Debugging & Retraining Feedback Loop

Production maintenance requires a continuous monitoring and updating cycle to identify and resolve model decay:

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; flex-direction: column; align-items: center; gap: 10px;">
  <div style="font-weight: bold; font-size: 14px; color: #1e3a8a; text-transform: uppercase;">Production Debugging & Retraining Feedback Loop</div>
  
  <svg width="600" height="150" viewBox="0 0 600 150" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%;">
    <defs>
      <marker id="arrow-prod" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 7 5 L 0 8.5 z" fill="#64748b"/>
      </marker>
    </defs>
    
    <!-- ROW 1 Boxes (y=20) -->
    <!-- Box 1: Inference -->
    <rect x="20" y="20" width="120" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="80" y="44" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">1. Inference</text>
    
    <!-- Box 2: Metrics Log -->
    <rect x="230" y="20" width="120" height="40" rx="6" fill="#f0fdf4" stroke="#10b981" stroke-width="1.5"/>
    <text x="290" y="44" font-family="sans-serif" font-size="11" font-weight="bold" fill="#065f46" text-anchor="middle">2. Metrics Log</text>
    
    <!-- Box 3: Error Analysis -->
    <rect x="440" y="20" width="120" height="40" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
    <text x="500" y="44" font-family="sans-serif" font-size="11" font-weight="bold" fill="#7c2d12" text-anchor="middle">3. Error Analysis</text>
    
    <!-- ROW 2 Boxes (y=90) -->
    <!-- Box 6: Deployment -->
    <rect x="20" y="90" width="120" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="80" y="114" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">6. Deployment</text>
    
    <!-- Box 5: Model Update -->
    <rect x="230" y="90" width="120" height="40" rx="6" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="290" y="114" font-family="sans-serif" font-size="11" font-weight="bold" fill="#5b21b6" text-anchor="middle">5. Model Update</text>
    
    <!-- Box 4: Diagnostics -->
    <rect x="440" y="90" width="120" height="40" rx="6" fill="#fff1f2" stroke="#f43f5e" stroke-width="1.5"/>
    <text x="500" y="114" font-family="sans-serif" font-size="11" font-weight="bold" fill="#9f1239" text-anchor="middle">4. Diagnostics</text>
    
    <!-- Connectors -->
    <!-- Row 1 arrows -->
    <line x1="140" y1="40" x2="223" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
    <line x1="350" y1="40" x2="433" y2="40" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
    
    <!-- Vertical right arrow (3 -> 4) -->
    <line x1="500" y1="60" x2="500" y2="83" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
    
    <!-- Row 2 arrows (4 -> 5 -> 6) -->
    <line x1="440" y1="110" x2="357" y2="110" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
    <line x1="230" y1="110" x2="147" y2="110" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
    
    <!-- Vertical left arrow (6 -> 1) -->
    <line x1="80" y1="90" x2="80" y2="67" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-prod)"/>
  </svg>
</div>

1. **Inference**: Models process incoming user tokens and record output classifications and confidence scores.
2. **Metrics Log**: Tracks performance metrics (e.g. drop in classification confidence, spike in user fallbacks) and checks for data drift.
3. **Error Analysis**: Automatically flags low-confidence or high-loss predictions. Engineers review these flagged logs, categorizing errors into issues like subword tokenization anomalies, Out-of-Vocabulary (OOV) tokens, or class imbalances.
4. **Diagnostic Actions & Model Update**: Adjust the vocabulary mapping tables, tune classification thresholds, or retrain the model on newly drifted samples. The updated model is then validated against a golden test set and re-deployed.

---

### Python Code Integration

The following Python snippet calculates PSI and Wasserstein Distance on our distributions, verifying the hand-calculations:

```python
import numpy as np
from scipy.stats import wasserstein_distance

# Proportions matching hand calculation
expected = np.array([0.6, 0.3, 0.1])
actual = np.array([0.5, 0.3, 0.2])

# 1. Calculate Population Stability Index (PSI)
def calculate_psi(p, q):
    # Avoid zero division or log(0) errors by adding a tiny epsilon
    p = np.clip(p, 1e-15, 1.0)
    q = np.clip(q, 1e-15, 1.0)
    return np.sum((p - q) * np.log(p / q))

psi_val = calculate_psi(actual, expected)

# 2. Calculate Wasserstein Distance (Earth Mover's Distance)
# Represents the cost of transforming actual distribution to expected
w_dist = wasserstein_distance([0, 1, 2], [0, 1, 2], u_weights=actual, v_weights=expected)

print(f"Calculated PSI:                 {psi_val:.4f}")
print(f"Calculated Wasserstein Distance: {w_dist:.4f}")

# Verify exact match with hand calculation (0.0875)
assert abs(psi_val - 0.0875) < 1e-4
```

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Maintains model accuracy, latency limits, and resource efficiency when running in production.
- **Why was it introduced?**
  Introduced to prevent performance decay due to domain shifts and data drift in live environments.
- **What are its limitations?**
  Pruning and quantization can lead to degraded accuracy on edge cases.
- **Computational Complexity (Time & Memory)**
  - **Data Drift Detection Time**: $O(N \cdot |V|)$ where $N$ is sample size and $|V|$ is vocabulary size.
  - **Memory Footprint (int8 Quantized)**: $25\%$ of the original float32 model footprint.
- **Component Variable Denotation Legend**
  - $X$: Input text features.
  - $Y$: Output label targets.
  - $N$: Audit sample window count.
  - $|V|$: Vocabulary size.
- **Production Use Cases**
  - Monitoring customer service classifiers for vocabulary shift.
  - Compressing models to run on edge devices.
- **Follow-up questions interviewers ask**
  - *How do you identify Data Drift in production NLP?* (By comparing the token frequency distribution of live production data against the training dataset using statistical tests like Population Stability Index (PSI) or Wasserstein Distance).
  - *Why do vocabulary differences between training and serving environments crash tokenizers?* (If the serving environment maps token indices using a different dictionary than the training environment, the token indices will map to incorrect word vectors, leading to model degradation).
