# Decision Boundaries and Assumptions

Unlike linear regression models that output continuous targets directly, logistic regression outputs a probability. To make a hard class prediction, we must project these probabilities onto a decision boundary.

---

## 1. Decision Boundaries

A decision boundary is the geometric region in feature space that separates class $0$ predictions from class $1$ predictions.

### The Standard Threshold
By default, we map the model's output probability $f_{w,b}(x)$ to binary classes using a threshold of $0.5$:
- Predict $\hat{y} = 1$ if $f_{w,b}(x) \ge 0.5$
- Predict $\hat{y} = 0$ if $f_{w,b}(x) < 0.5$

Looking at the sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$:
- $g(z) \ge 0.5 \iff z \ge 0$
- $g(z) < 0.5 \iff z < 0$

Since $z = w \cdot x + b$, this means:
- We predict $1$ when $w \cdot x + b \ge 0$
- We predict $0$ when $w \cdot x + b < 0$

The boundary where the model is completely uncertain ($P=0.5$) is defined by the linear equation:
$$w \cdot x + b = 0$$

### Linear Decision Boundaries
For a two-dimensional feature space ($x \in \mathbb{R}^2$), the decision boundary is a straight line:
$$w_1 x_1 + w_2 x_2 + b = 0 \implies x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

```
        x2
         ^      Class 1 (w.x + b > 0)
         |     .  .  .
         |    .  .  /
         |   .  .  /
         |  .  .  /  <-- Decision Boundary Line (w.x + b = 0)
         | .  .  /
         |  .   /  .  .
         |     /  .  .  Class 0 (w.x + b < 0)
         +--------------------> x1
```

### Non-Linear Decision Boundaries (Feature Crossing)
Standard logistic regression can only draw a straight decision line or plane. If your classes are separated by a circle or a complex shape, a raw linear boundary will yield terrible accuracy.
- **The Solution:** Engineer non-linear features by introducing polynomial crosses or interactions.
- **Example (Circular Boundary):**
  Suppose the true boundary is a circle. We can add squared features $x_1^2$ and $x_2^2$ to our model:
  $$z = w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2 + b$$
  
  During training, the model might learn weights: $w_1=0, w_2=0, w_3=1, w_4=1, b=-1$.
  The decision boundary becomes:
  $$x_1^2 + x_2^2 - 1 = 0 \implies x_1^2 + x_2^2 = 1 \quad (\text{A circle of radius } 1)$$

---

## 2. Core Assumptions: Logistic vs. Linear Regression

Many interview candidates mistakenly apply linear regression (OLS) assumptions to logistic regression. It is crucial to highlight what is **not** assumed.

| Linear Regression Assumption (OLS) | Required in Logistic Regression? | Why / Why Not? |
| :--- | :--- | :--- |
| **Normality of Residuals** | **No** | Residuals in classification are binary ($1 - \hat{y}$ or $0 - \hat{y}$), which follow a binomial distribution, not a normal distribution. |
| **Homoscedasticity** | **No** | The variance of a binary target variable $y$ is $P(y)(1 - P(y))$, which changes depending on the prediction probability. Variance is naturally heteroscedastic. |
| **Linear relation between $x$ and $y$** | **No** | The relationship between features $x$ and target label $y$ is non-linear (sigmoidal). |

### What *Does* Matter: The Actual Assumptions
For logistic regression to be stable and mathematically valid, the following must hold:

1. **Independence of Observations:**
   The training examples must not be linked or grouped. For example, if multiple rows in a user database belong to the same household, they violate independence, which artificially deflates standard errors.
2. **Linearity of Independent Variables and Log-Odds:**
   While the features do not have a linear relationship with target $y$, they **must** have a linear relationship with the log-odds (logits):
   $$\log\left(\frac{P(y=1)}{1-P(y=1)}\right) = w \cdot x + b$$
   If this relationship is non-linear, you must apply feature transformations (e.g., binning, logging, or polynomial expansion).
3. **No Severe Multicollinearity:**
   Highly correlated features (multicollinearity) will inflate the standard errors of the coefficients, making it impossible to determine feature importance and causing weights to fluctuate wildly.
4. **Binary Target:**
   The dependent variable $y^{(i)}$ must be binary ($0$ or $1$) for binary logistic regression.
