# Notation and Sigmoid Intuition

In classification tasks, we want to predict a discrete class label (e.g., $y \in \{0, 1\}$) rather than a continuous value. Logistic regression maps linear predictions to probability values using the Sigmoid function. This guide explains these concepts through a concrete SaaS churn prediction scenario.

---

## 1. Applied Mathematical Notation

To represent features and parameters, we define:
- Let $x$ be the input features vector of dimension $n$:
  $$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$
- Let $w$ be the weights vector of dimension $n$:
  $$w = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix} \in \mathbb{R}^n$$
- Let $b$ be the bias scalar (intercept):
  $$b \in \mathbb{R}$$

### The Sigmoid Function
We define the linear output $z$ as:
$$z = w \cdot x + b$$

The sigmoid activation function $g(z)$ is defined as:
$$g(z) = \frac{1}{1 + e^{-z}}$$

The final model prediction $f_{w,b}(x)$ is the output of the sigmoid function, representing the probability of class 1:
$$f_{w,b}(x) = g(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}} = P(y=1 | x; w,b)$$

---

## 2. Flaws of Linear Regression in Classification

If we try to use a standard linear regression model ($w \cdot x + b$) for binary classification, we run into two critical engineering flaws:

1. **Unbounded Output Range:**
   A linear model outputs values in the range $(-\infty, \infty)$. However, classification requires predicting probabilities, which must reside strictly in the interval $[0, 1]$. Using a linear model means we might output predictions of $-3.2$ or $1.7$, which have no meaning as probabilities.
2. **Extreme Sensitivity to Outliers:**
   Fitting a straight OLS line on binary data points ($0$ and $1$) shifts the decision boundary if we add a correct but extreme outlier. Wrapping the linear model inside a sigmoid function resolves this.

---

## 3. Scenario: Subscription Churn Prediction

You are building a model to predict user churn ($y=1$ if a user cancels, $y=0$ if they renew) for a streaming service based on:
- $x_1$: Average daily active streaming time (in hours).
- $x_2$: Number of customer service complaints filed in the last 30 days.

### Model Parameters
Suppose our model has learned the following weights and bias:
- $w_1 = -1.2$ (streaming time decreases churn risk).
- $w_2 = 1.8$ (filing complaints increases churn risk).
- $b = 0.5$ (baseline intercept).

Let's calculate the predicted churn probabilities for two different users.

### User A: Highly Engaged User ($x^{(1)} = [3.0, 0.0]^T$)
1. **Compute linear output $z$:**
   $$z = w \cdot x^{(1)} + b = (-1.2 \times 3.0) + (1.8 \times 0.0) + 0.5 = -3.6 + 0.5 = -3.1$$
2. **Apply Sigmoid function:**
   $$f_{w,b}(x^{(1)}) = g(-3.1) = \frac{1}{1 + e^{-(-3.1)}} = \frac{1}{1 + e^{3.1}} \approx \frac{1}{1 + 22.20} \approx 0.043 \quad (4.3\% \text{ Churn Probability})$$

### User B: Frustrated User ($x^{(2)} = [0.5, 3.0]^T$)
1. **Compute linear output $z$:**
   $$z = w \cdot x^{(2)} + b = (-1.2 \times 0.5) + (1.8 \times 3.0) + 0.5 = -0.6 + 5.4 + 0.5 = 5.3$$
2. **Apply Sigmoid function:**
   $$f_{w,b}(x^{(2)}) = g(5.3) = \frac{1}{1 + e^{-(5.3)}} \approx \frac{1}{1 + 0.005} \approx 0.995 \quad (99.5\% \text{ Churn Probability})$$

---

## 4. Logit Link: The Mathematics of Log-Odds

To show that logistic regression is a linear model at heart, we define the **Odds** and **Log-Odds** of our event:

### The Concept of "Odds"
Odds measure the ratio of the probability of churn ($p$) to the probability of renewal ($1-p$):
$$\text{Odds} = \frac{f_{w,b}(x)}{1 - f_{w,b}(x)}$$

- **User A (Engaged):** $p = 0.043 \implies \text{Odds} = \frac{0.043}{0.957} \approx 0.045$.
- **User B (Frustrated):** $p = 0.995 \implies \text{Odds} = \frac{0.995}{0.005} = 199.0$.

### Mathematical Derivation of Sigmoid from Log-Odds
If we take the natural logarithm of the Odds, we get the **Log-Odds** or **Logit** function. Suppose we assume the log-odds are linear with respect to parameters:

$$\log\left(\frac{f_{w,b}(x)}{1 - f_{w,b}(x)}\right) = w \cdot x + b$$

Let's solve for the prediction function $f_{w,b}(x)$:

$$\frac{f_{w,b}(x)}{1 - f_{w,b}(x)} = e^{w \cdot x + b}$$
$$f_{w,b}(x) = (1 - f_{w,b}(x)) e^{w \cdot x + b}$$
$$f_{w,b}(x) = e^{w \cdot x + b} - f_{w,b}(x) e^{w \cdot x + b}$$
$$f_{w,b}(x) (1 + e^{w \cdot x + b}) = e^{w \cdot x + b}$$
$$f_{w,b}(x) = \frac{e^{w \cdot x + b}}{1 + e^{w \cdot x + b}}$$

Divide the numerator and denominator by $e^{w \cdot x + b}$:

$$f_{w,b}(x) = \frac{1}{\frac{1}{e^{w \cdot x + b}} + 1} = \frac{1}{1 + e^{-(w \cdot x + b)}} = g(w \cdot x + b)$$

### The Engineering Takeaway
This mathematical derivation shows that **Logistic Regression is a linear model predicting Log-Odds**.
- The inputs ($x$) have a linear relationship with the log-odds of the target.
- The non-linear Sigmoid activation is the link function that maps these linear log-odds back into a bounded probability interval $[0, 1]$.
