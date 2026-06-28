# Optimization and Trade-offs

Once we have defined the prediction model $f_{w,b}(x)$ and cost function $J(w,b)$, we must find the parameters $w$ and $b$ that minimize $J(w,b)$. There are two primary optimization strategies: **Gradient Descent** (an iterative method) and the **Normal Equation** (an analytical closed-form solution).

---

## 1. Gradient Descent Optimization

Gradient Descent is an iterative optimization algorithm that updates parameters by moving in the direction of steepest descent (opposite to the gradient of the cost function).

### Update Rules
For each iteration, the parameters $w$ and $b$ are updated simultaneously:

$$w_j = w_j - \alpha \frac{\partial J(w,b)}{\partial w_j} \quad \text{for } j = 1, \dots, n$$
$$b = b - \alpha \frac{\partial J(w,b)}{\partial b}$$

Plugging in the partial derivatives of the MSE cost function:

$$w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \quad \text{for } j = 1, \dots, n$$
$$b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$$

### Vectorized Update Rule
In practice, we update the entire weights vector $w$ at once using matrix operations:

$$w = w - \alpha \frac{1}{m} X^T \left( f_{w,b}(X) - y \right)$$

Where:
- $X$ is the $m \times n$ design matrix of features.
- $f_{w,b}(X)$ is the $m \times 1$ vector of predictions.
- $y$ is the $m \times 1$ vector of ground truth labels.

---

## 2. Debugging the Learning Rate $\alpha$ in Production

The learning rate $\alpha$ controls the step size taken towards the minimum. Setting $\alpha$ incorrectly is a common training bug. You can diagnose this by plotting the **Learning Curve** (Training Loss vs. Iterations).

```
Loss J(w,b)
 ^
 |  \
 |   \__
 |      \___             <-- Healthy Convergence
 |          \___________
 +------------------------> Iterations
```

### Case A: Learning Rate $\alpha$ is Too High (Divergence / Oscillation)
- **Visual Symptom:** The training loss curve oscillates violently or steadily increases (diverges to infinity).
- **Mathematical Intuition:** The step size is so large that it overshoots the minimum and lands on the opposite side of the "bowl" at a higher point.
- **Production Fix:** Log-scale search (e.g., divide $\alpha$ by 3 or 10: $0.1 \rightarrow 0.03 \rightarrow 0.01$) or implement an adaptive learning rate scheduler (e.g., Cosine Decay) or switch to optimizers with adaptive momentum (like Adam).

```
Loss J(w,b)
 ^      /\
 |     /  \   /\         <-- Oscillating / Exploding
 |  /\/    \/   \
 | /             \/
 +------------------------> Iterations
```

### Case B: Learning Rate $\alpha$ is Too Low (Stagnant / Flatline)
- **Visual Symptom:** The training loss decreases in a straight line or barely drops over thousands of steps.
- **Mathematical Intuition:** The steps are infinitesimally small, requiring excessive computational time to reach the minimum.
- **Production Fix:** Increase the learning rate (e.g., multiply by 3 or 10: $0.0001 \rightarrow 0.001 \rightarrow 0.01$).

```
Loss J(w,b)
 ^
 |--------------------   <-- Stagnant Flatline
 |
 |
 +------------------------> Iterations
```

---

## 3. The Normal Equation (Analytical Closed-Form Solution)

Instead of iteratively stepping toward the minimum, the Normal Equation solves for the optimal parameters analytically in a single step.

### Notation Adjustment for the Design Matrix
To express the Normal Equation cleanly, we combine the bias $b$ and weights $w$ into a single parameter vector $\theta$, and prepend a column of ones to our design matrix $X$:

- Let $\theta = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_n \end{bmatrix} \in \mathbb{R}^{n+1}$
- Let $X$ be the $m \times (n+1)$ design matrix where the first column is all $1$s:
  $$X = \begin{bmatrix} 1 & x_1^{(1)} & \dots & x_n^{(1)} \\ 1 & x_1^{(2)} & \dots & x_n^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(m)} & \dots & x_n^{(m)} \end{bmatrix}$$

The optimal parameter vector $\theta$ is solved via:
$$\theta = (X^T X)^{-1} X^T y$$

---

## 4. Production Trade-offs: Gradient Descent vs. Normal Equation

| Feature / Scenario | Gradient Descent | Normal Equation |
| :--- | :--- | :--- |
| **Iterative / Closed-Form** | Iterative (requires multiple steps and convergence checks) | Closed-Form (computed in a single analytical step) |
| **Hyperparameter Tuning** | Requires choosing learning rate $\alpha$, batch size, and iterations | No hyperparameters to tune |
| **Computational Complexity** | $O(k \cdot m \cdot n)$ (where $k$ is iterations) | $O(n^3)$ due to calculating the matrix inverse $(X^T X)^{-1}$ |
| **Scalability (Features $n$)** | Scales exceptionally well to large $n$ ($n > 100,000$) | Fails at large $n$ because inverting a large matrix is slow and memory-intensive |
| **Scalability (Samples $m$)** | Out-of-core learning is possible via Mini-batch GD | Requires loading all $m$ examples into memory |

### When to Use Which?
1. **Use Gradient Descent when:**
   - The feature size $n$ is large (e.g., $n > 10,000$).
   - The sample size $m$ is massive (e.g., millions of rows) and does not fit into RAM.
2. **Use Normal Equation when:**
   - The feature size $n$ is small (e.g., $n < 1,000$) and you want a deterministic, tuning-free solution.

---

## 5. Non-Invertibility of $X^T X$: What Causes It?

In production code, trying to solve the Normal Equation can crash your pipeline with a `Singular Matrix Error` (i.e., $X^T X$ is not invertible). 

### The 2 Main Causes
1. **Multicollinearity (Redundant Features):**
   If two or more features are perfectly linearly dependent (e.g., one feature is in square feet and another is in square meters, or a One-Hot encoded variable suffers from the "Dummy Variable Trap"), the columns of $X^T X$ are not linearly independent.
   - *Math implication:* The determinant of $X^T X$ is $0$, meaning its inverse does not exist.
2. **Too Few Samples ($m \le n$):**
   If you have fewer training examples than features (e.g., $10$ rows of data but $100$ features), the matrix $X^T X$ is underdetermined and rank-deficient.

### Production Fixes
- **Drop Redundant Features:** Remove highly correlated columns using a correlation matrix or Variance Inflation Factor (VIF) filter.
- **Regularization:** Use Ridge Regression ($L_2$). Adding a regularization term transforms $(X^T X)^{-1}$ into $(X^T X + \lambda I)^{-1}$. The identity matrix $I$ adds values along the diagonal, guaranteeing invertibility.
- **Use Pseudo-inverse:** Calculate the Moore-Penrose pseudo-inverse (`np.linalg.pinv` in NumPy) which uses Singular Value Decomposition (SVD) to find a solution even when the matrix is singular.
