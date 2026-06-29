# Production Scenarios and Pipeline Engineering

Deploying linear regression models to production environments requires moving past academic assumptions and addressing data layout and pre-processing engineering. This guide covers industry use cases, data engineering principles, and a complete post-mortem case study.

---

## 1. Industry Use Cases: Why OLS is Still King

1. **Sub-Millisecond Ad-Tech Inferences:**
   In programmatic ad bidding, models must predict click-through rates (CTR) within 5 milliseconds. Running a neural network or tree ensemble is too computationally expensive. A vectorized dot product ($w \cdot x + b$) compiles to raw CPU register instructions, running in single-digit microseconds.
2. **Fair Lending Act Compliance:**
   Under fair lending regulations, if an automated system rejects a loan applicant, the system must generate legally auditable "Adverse Action Notices." A linear model's coefficients are completely deterministic, making compliance audits trivial compared to deep learning models.

---

## 2. Production Preprocessing Pipelines

A machine learning pipeline in production must handle noisy data streams without crashing or leaking information.

### Avoiding Data Leakage
Data leakage occurs when information from the validation or test set accidentally leaks into the training pipeline.
- **The Golden Rule:** Split your dataset before applying any preprocessing transformer (e.g., standardizing features, target encoding). Fit the preprocessors **only** on the training split, and save the parameters (e.g., $\mu, \sigma$, medians) to transform validation/test splits.

### Mitigating High-Cardinality Feature Explosion
Encoding features with thousands of categories (e.g., `user_zipcode` or `merchant_id`) using One-Hot encoding is an engineering hazard. It creates an extremely wide, sparse matrix that bloats RAM and destabilizes OLS.

1. **Target Encoding (with Smoothing):**
   Replace each category with the average target value for that category in the training set. Add a smoothing parameter to prevent overfitting on rare categories:
   $$S_i = \alpha \bar{y} + (1 - \alpha) y_i$$
   Where $\bar{y}$ is the global target mean and $y_i$ is the category mean.
2. **The Hashing Trick (Feature Hashing):**
   Apply a hash function to map categorical features to a fixed-size index (e.g., $1024$ columns), bounding dimensionality and handling new categories out-of-the-box.

---

## 3. Production Post-Mortem: "The Incident of the Shifting Medians"

### The Incident
An e-commerce company deployed a linear regression model to predict customer lifetime value (LTV). During sales events, customers complained that identical checkout baskets were receiving vastly different discounts and LTV scores. 

### The Root Cause: Dynamic Batch Imputation
The engineering team built a preprocessing step that imputed missing data (e.g., `days_since_last_login`) on-the-fly using the **median of the incoming inference batch**. 
- On normal days, the median was $15$ days.
- During a major promotion, a massive wave of new users logged in (with missing values or 0 days). The batch median shifted to $2$ days.
- Because the imputation value shifted dynamically, a customer's LTV score would change depending on *which other customers* happened to be in the same inference batch, creating batch-dependency and prediction drift.

### Before-and-After Implementation

#### The Bad Code (Dynamic Batch Imputation)

```python
import numpy as np
import pandas as pd

# Simulating incoming production inference batches
batch_normal = pd.DataFrame({'days_since_login': [12.0, 15.0, np.nan, 18.0]})
batch_promo = pd.DataFrame({'days_since_login': [0.0, 1.0, np.nan, 2.0, np.nan]})

# --- ANTI-PATTERN: Imputing using the runtime batch median ---
def preprocess_bad(df):
    batch_median = df['days_since_login'].median()
    df_imputed = df.copy()
    df_imputed['days_since_login'] = df_imputed['days_since_login'].fillna(batch_median)
    print(f"DEBUG: Imputed using batch median = {batch_median}")
    return df_imputed

# An identical NaN profile is filled with different values depending on the batch context
res_normal = preprocess_bad(batch_normal)  # Imputes NaN with 15.0
res_promo = preprocess_bad(batch_promo)    # Imputes NaN with 1.0
```

#### The Good Code (Anchored Imputation)
To fix this, we must calculate the median *only* on the training dataset, serialize this static value as metadata, and lock it in the inference pipeline.

```python
import json
import numpy as np
import pandas as pd

# Training phase: Calculate and anchor the median
train_df = pd.DataFrame({'days_since_login': [10.0, 15.0, 12.0, 20.0, 16.0]})
anchored_median = train_df['days_since_login'].median()  # Locked value = 15.0

# Save to metadata JSON config
metadata = {"days_since_login_median": anchored_median}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

# --- PRODUCTION-SAFE PATTERN: Ingestion pipeline load anchored metadata ---
def preprocess_safe(df, metadata_path="model_metadata.json"):
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    impute_value = meta["days_since_login_median"]
    df_imputed = df.copy()
    df_imputed['days_since_login'] = df_imputed['days_since_login'].fillna(impute_value)
    print(f"DEBUG: Imputed using anchored median = {impute_value}")
    return df_imputed

# Both batches now receive consistent imputations (15.0) regardless of the batch composition
res_normal_safe = preprocess_safe(batch_normal)
res_promo_safe = preprocess_safe(batch_promo)
```
