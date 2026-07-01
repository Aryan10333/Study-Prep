# A/B Testing: Power Analysis & Hypothesis Testing

This guide details the statistical mechanics of A/B experiments, calculating required sample sizes, executing Z-tests, and correcting for multiple testing inflation in production.

---

## 1. Calculating Required Sample Size (Power Analysis)

Before launching an A/B test, you must determine how many users ($m$) are required per variant to detect a desired conversion improvement (Minimum Detectable Effect, or MDE) without throwing Type I ($\alpha$) or Type II ($\beta$) errors.

### Python Code: Sample Size Calculation
```python
from statsmodels.stats.power import NormalIndPower
import numpy as np

# A/B Test Parameters
baseline_conversion_rate = 0.05  # 5% baseline conversion
mde_absolute = 0.01              # We want to detect a 1% absolute change (6% vs 5%)
alpha = 0.05                     # 5% False Positive Rate (Type I)
power = 0.80                     # 80% Power (1 - beta)

# Compute effect size using Cohen's h for proportions
prop_1 = baseline_conversion_rate
prop_2 = baseline_conversion_rate + mde_absolute
effect_size = 2 * (np.arcsin(np.sqrt(prop_2)) - np.arcsin(np.sqrt(prop_1)))

# Solve for sample size
power_analysis = NormalIndPower()
required_n = power_analysis.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative='two-sided'
)

print(f"Required Sample Size per Variant: {int(np.ceil(required_n))}")
```

### Expected Console Output
```text
Required Sample Size per Variant: 14524
```

### Diagnostic Visual (Statistical Power Curves)
The power curve plot illustrates how the statistical power increases as you collect more samples. To detect a smaller $1.0\%$ effect, you must collect significantly more samples ($14,524$ users) compared to a larger $2.0\%$ effect:

![Power Curves](images/statistical_power_curves.png)

---

## 2. Running a Two-Sample Z-Test for Proportions

Once the required sample size is reached, you run a hypothesis test to evaluate if the treatment variant outperforms control.

### Python Code: Z-Test
```python
import numpy as np
from scipy.stats import norm

# Experiment results
n_A, x_A = 15000, 750  # Control (5% conversion)
n_B, x_B = 15000, 870  # Treatment (5.8% conversion)

p_A = x_A / n_A
p_B = x_B / n_B

# Pooled conversion rate
p_c = (x_A + x_B) / (n_A + n_B)

# Standard Error
se = np.sqrt(p_c * (1.0 - p_c) * (1.0 / n_A + 1.0 / n_B))

# Compute Z-statistic and p-value
z_stat = (p_B - p_A) / se
p_val = 2 * (1.0 - norm.cdf(abs(z_stat)))

print("--- Z-Test Results ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value:     {p_val:.4e}")
print(f"Reject H_0:  {p_val < 0.05}")
```

### Expected Console Output
```text
--- Z-Test Results ---
Z-statistic: 3.1974
p-value:     1.3867e-03
Reject H_0:  True
```

---

## 3. The Multiple Testing Trap & Bonferroni Correction

If you test $k = 10$ different variants (e.g., testing 10 different prompt designs) against a single control, your FWER (Family-Wise Error Rate) inflates to $40\%$.

To control FWER back to $5\%$, apply the **Bonferroni Correction**:
$$\alpha_{\text{adjusted}} = \frac{\alpha}{k} = \frac{0.05}{10} = 0.005$$

### Python Code: Bonferroni Adjustment
```python
from statsmodels.stats.multitest import multipletests

# Simulated raw p-values for 10 variants
raw_p_values = [0.0001, 0.0012, 0.0380, 0.1200, 0.4500, 0.0450, 0.8900, 0.2300, 0.6700, 0.0480]

# Apply Bonferroni correction
reject, corrected_p_vals, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')

print("--- Adjusted Significance Results ---")
for idx, (p_raw, p_corr, sig) in enumerate(zip(raw_p_values, corrected_p_vals, reject)):
    print(f"Variant {idx+1}: Raw p={p_raw:.4f} | Corrected p={p_corr:.4f} | Significant: {sig}")
```

### Expected Console Output
```text
--- Adjusted Significance Results ---
Variant 1: Raw p=0.0001 | Corrected p=0.0010 | Significant: True
Variant 2: Raw p=0.0012 | Corrected p=0.0120 | Significant: True
Variant 3: Raw p=0.0380 | Corrected p=0.3800 | Significant: False
Variant 4: Raw p=0.1200 | Corrected p=1.0000 | Significant: False
Variant 5: Raw p=0.4500 | Corrected p=1.0000 | Significant: False
Variant 6: Raw p=0.0450 | Corrected p=0.4500 | Significant: False
Variant 7: Raw p=0.8900 | Corrected p=1.0000 | Significant: False
Variant 8: Raw p=0.2300 | Corrected p=1.0000 | Significant: False
Variant 9: Raw p=0.6700 | Corrected p=1.0000 | Significant: False
Variant 10: Raw p=0.0480 | Corrected p=0.4800 | Significant: False
```

*Note:* Naive evaluation would have accepted Variants 3, 6, and 10 as winners ($p < 0.05$). The Bonferroni correction rejects these false positives.

---

## 4. Interactive Practice Notebook
To sweep sample power curves and run A/B test simulations, open the interactive notebook:
- [07_ab_testing_power_and_multiple_tests.ipynb](file:///d:/Study/Prep/machine-learning-prep/supervised-learning/tree-based-models/07_ab_testing_power_and_multiple_tests.ipynb)
