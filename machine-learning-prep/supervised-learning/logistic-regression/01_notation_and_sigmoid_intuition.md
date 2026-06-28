# Notation and Sigmoid Intuition

In classification tasks, we want to predict a discrete class label (e.g., $y \in \{0, 1\}$) rather than a continuous value. Logistic regression solves this by mapping linear predictions to probability values using the Sigmoid function.

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

The final model prediction $f_{w,b}(x)$ is the output of the sigmoid function:
$$f_{w,b}(x) = g(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

---

## 2. The Sigmoid Map: Why Wrap the Linear Model?

If we try to use a standard linear regression model ($w \cdot x + b$) for binary classification, we run into two critical engineering flaws:

1. **Unbounded Output Range:**
   A linear model outputs values in the range $(-\infty, \infty)$. However, classification requires predicting probabilities, which must reside strictly in the interval $[0, 1]$. Using a linear model means we might output predictions of $-3.2$ or $1.7$, which have no meaning as probabilities.
2. **Extreme Sensitivity to Outliers:**
   Fitting a straight OLS line on binary data points ($0$ and $1$) shifts the decision boundary if we add a correct but extreme outlier. Wrapping the linear model inside a sigmoid function resolves this.

### Sigmoid Mechanics
The Sigmoid function acts as a mathematical squashing function that maps any real-valued number $z \in (-\infty, \infty)$ to the bounded probability interval $[0, 1]$.

```
          g(z)
           ^
       1.0 |         .---''''
           |       .
       0.5 |------/------  (z = 0, g(z) = 0.5)
           |    .
       0.0 |___/____________> z
          -inf              +inf
```

- As $z \rightarrow \infty$, $e^{-z} \rightarrow 0$, so $g(z) \rightarrow \frac{1}{1+0} = 1$.
- As $z \rightarrow -\infty$, $e^{-z} \rightarrow \infty$, so $g(z) \rightarrow \frac{1}{\infty} = 0$.
- When $z = 0$, $e^{0} = 1$, so $g(z) = \frac{1}{1+1} = 0.5$.

---

## 3. Probability Interpretation & The Logit Link

In logistic regression, the model's prediction output is interpreted directly as the conditional probability that the positive class ($y = 1$) occurs, given the inputs $x$:

$$f_{w,b}(x) = P(y=1 | x; w,b)$$

Consequently, the probability of the negative class ($y = 0$) is:
$$P(y=0 | x; w,b) = 1 - f_{w,b}(x)$$

### The Concept of "Odds"
In probability theory, the **Odds** of an event is the ratio of the probability of its occurrence to the probability of its non-occurrence:

$$\text{Odds} = \frac{P(y=1|x)}{1 - P(y=1|x)} = \frac{f_{w,b}(x)}{1 - f_{w,b}(x)}$$

- If an event has a $75\%$ probability, the odds are $\frac{0.75}{0.25} = 3$ (often written as 3-to-1).
- If an event has a $20\%$ probability, the odds are $\frac{0.20}{0.80} = 0.25$.

### Log-Odds (Logits)
If we take the natural logarithm of the Odds, we get the **Log-Odds** or **Logit** function:

$$\text{Log-Odds} = \log\left(\frac{f_{w,b}(x)}{1 - f_{w,b}(x)}\right)$$

If we substitute $f_{w,b}(x) = \frac{1}{1 + e^{-z}}$ into the Log-Odds formula:

$$\frac{f_{w,b}(x)}{1 - f_{w,b}(x)} = \frac{\frac{1}{1 + e^{-z}}}{1 - \frac{1}{1 + e^{-z}}} = \frac{\frac{1}{1 + e^{-z}}}{\frac{e^{-z}}{1 + e^{-z}}} = \frac{1}{e^{-z}} = e^z$$

Taking the natural log of both sides:

$$\log\left(\frac{f_{w,b}(x)}{1 - f_{w,b}(x)}\right) = \log(e^z) = z = w \cdot x + b$$

### The Engineering Takeaway
This shows that **Logistic Regression is fundamentally a linear model predicting Log-Odds**. 
- The input features have a linear relationship with the log-odds of the target.
- The non-linear Sigmoid activation is simply the mathematical tool we use to map these linear log-odds back into a bounded probability interval $[0, 1]$.
