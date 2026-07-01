# Probability & Hypothesis Testing: A/B Testing in Production

This document details probability axioms, Bayes Theorem application, Central Limit Theorem guarantees, statistical error formulations, and the multi-hypothesis correction math used in production A/B tests.

---

## 1. Probability Foundations

### Bayes Theorem
Bayes Theorem updates the probability of a hypothesis ($H$) based on new empirical evidence ($E$):
$$P(H|E) = \frac{P(E|H)P(H)}{P(E)}$$

#### Real-World Interview Case: Disease Screening
Suppose a rare disease affects $1\%$ of the population:
- **Prior Probability ($P(D)$):** $0.01$ (probability of having the disease).
- **Sensitivity ($P(T^+|D)$):** $0.99$ (test correctly flags positive for a diseased patient).
- **False Positive Rate ($P(T^+|D^C)$):** $0.05$ (test incorrectly flags positive for a healthy patient).

If a random patient tests positive, what is the probability that they actually have the disease ($P(D|T^+)$)?

##### Step 1: Calculate Total Probability of Testing Positive ($P(T^+)$)
Using the Law of Total Probability:
$$P(T^+) = P(T^+|D)P(D) + P(T^+|D^C)P(D^C)$$
$$P(T^+) = (0.99 \times 0.01) + (0.05 \times 0.99) = 0.0099 + 0.0495 = 0.0594$$

##### Step 2: Apply Bayes Theorem
$$P(D|T^+) = \frac{P(T^+|D)P(D)}{P(T^+)} = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.1667 \text{ or } 16.67\%$$

- **Intuition:** Even though the test is $99\%$ sensitive, a patient testing positive has only a $16.67\%$ chance of having the disease. This is because the disease is highly rare, and the absolute count of false positives from the healthy majority population dwarfs the count of true positives from the diseased minority.

---

### Central Limit Theorem (CLT)
The Central Limit Theorem states that if you take sufficiently large random samples of size $m$ from any population distribution with mean $\mu$ and variance $\sigma^2$, the distribution of the sample means will be approximately normally distributed, regardless of the shape of the parent population:

$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{m}\right) \quad \text{as } m \to \infty$$

- **Why it matters for A/B testing:** Most production metrics (like Click-Through Rate or average session time) do not follow a normal distribution (e.g., clicks are binary Bernoulli; session times are skewed log-normal). CLT allows us to use parametric normal-distribution tests (like Z-tests or t-tests) to compare the means of these metrics, provided our sample size is large (typically $m > 30$).

---

## 2. Hypothesis Testing Mechanics

Hypothesis testing is used to evaluate whether an observed difference between a Control group ($A$) and a Treatment group ($B$) is due to a genuine improvement or random noise.

### Setup
- **Null Hypothesis ($H_0$):** There is no difference between the two variants ($\mu_A = \mu_B$).
- **Alternative Hypothesis ($H_1$):** A difference exists ($\mu_A \neq \mu_B$).

---

### Statistical Errors and Power
When interpreting test results, we face two types of errors:

```text
               True State of the World
                  H0 is True      |      H1 is True
               +------------------+------------------+
Accept H0      |  Correct Decision|  Type II Error   |
               |   (Prob = 1-α)   |   (Beta: β)      |
               +------------------+------------------+
Reject H0      |  Type I Error    |  Correct Decision|
               |  (Alpha: α)      |  Power (1-β)     |
               +------------------+------------------+
```

1. **Type I Error ($\alpha$, Significance Level):** Rejecting the null hypothesis when it is actually true (False Positive). Typically set to $\alpha = 0.05$ (accepting a $5\%$ risk of declaring a winner by chance).
2. **Type II Error ($\beta$):** Failing to reject the null hypothesis when it is actually false (False Negative). Typically set to $\beta = 0.20$.
3. **Statistical Power ($1 - \beta$):** The probability of correctly rejecting the null hypothesis when a true difference exists. Typically targeted at $80\%$.

---

## 3. Z-Test for Proportions (A/B Test Conversion Hand Calculations)

### The Problem
We are running a web page conversion A/B test. We split users evenly:
- **Control Group (A):** Old design.
- **Treatment Group (B):** New design.

Our experiment yields:
- **Group A:** $n_A = 1000$ visitors, $x_A = 50$ conversions.
- **Group B:** $n_B = 1000$ visitors, $x_B = 75$ conversions.

Let's test if the treatment conversion rate ($\hat{p}_B = 0.075$) is statistically higher than control ($\hat{p}_A = 0.050$) at a significance level of $\alpha = 0.05$.

---

### Step-by-Step Calculations

#### 1. Calculate Pooled Conversion Rate ($\hat{p}_c$)
$$\hat{p}_c = \frac{x_A + x_B}{n_A + n_B} = \frac{50 + 75}{1000 + 1000} = \frac{125}{2000} = 0.0625$$

#### 2. Calculate Standard Error ($SE$)
$$SE = \sqrt{\hat{p}_c (1 - \hat{p}_c) \left( \frac{1}{n_A} + \frac{1}{n_B} \right)}$$
$$SE = \sqrt{0.0625 \times (1 - 0.0625) \left( \frac{1}{1000} + \frac{1}{1000} \right)} = \sqrt{0.0625 \times 0.9375 \times 0.002}$$
$$SE = \sqrt{0.05859375 \times 0.002} = \sqrt{0.0001171875} \approx 0.010825$$

#### 3. Compute Z-Statistic
$$Z = \frac{\hat{p}_B - \hat{p}_A}{SE} = \frac{0.075 - 0.050}{0.010825} = \frac{0.025}{0.010825} \approx 2.3095$$

---

### Decision Rule
For a two-tailed test at $\alpha = 0.05$, the critical Z-values are $\pm 1.96$.
- **Result:** Since our calculated $Z \approx 2.31$ exceeds the critical value of $1.96$, we reject the null hypothesis $H_0$.
- **Conclusion:** The $2.5\%$ absolute increase in conversion rate is statistically significant and not due to random noise.

---

## 4. The Multiple Testing Trap & Bonferroni Correction

In production systems, engineering teams often test multiple variations simultaneously (e.g., testing 10 different discount rates or prompt configurations against a control).

### The Mathematical Trap
If you run $k$ independent hypothesis tests, each at a significance level of $\alpha = 0.05$, the probability of committing at least one Type I error (declaring a false positive winner by random chance) across the entire experiment is:

$$P(\text{At least one False Positive}) = 1 - (1 - \alpha)^k$$

If you test $k = 10$ variations simultaneously:
$$P(\text{At least one FP}) = 1 - (1 - 0.05)^{10} = 1 - (0.95)^{10} \approx 1 - 0.5987 = 0.4013 \text{ or } 40.13\%$$

By running 10 parallel tests, your false positive rate inflates from a safe $5\%$ to a risky $40.1\%$.

### The Bonferroni Correction Fix
To control the Family-Wise Error Rate (FWER) back to your target $\alpha$ (e.g., $0.05$), adjust the significance threshold for each of the $k$ individual tests:

$$\alpha_{\text{adjusted}} = \frac{\alpha}{k}$$

- **For $k = 10$ tests:**
  $$\alpha_{\text{adjusted}} = \frac{0.05}{10} = 0.005$$
  You only reject the null hypothesis for a individual variant if its individual p-value is less than $0.005$ (requiring a much higher Z-score), shielding your pipeline from false positive code deployments.
- **Drawback:** The Bonferroni correction is conservative and can reduce statistical power ($1-\beta$), requiring longer run times to identify true winners.
