# The Bias-Variance Decomposition

To debug the generalization error of a model, we must decompose its expected test Mean Squared Error (MSE). This guide details the mathematical formulation of **Bias** and **Variance** and walks through their calculation using a concrete housing price estimation scenario.

---

## 1. The Decomposition Formula

For a test observation $x_0$ with target $y_0 = f(x_0) + \epsilon$, the expected test Mean Squared Error (MSE) of our estimator $\hat{f}(x_0)$ is decomposed as:

$$E\left[ (y_0 - \hat{f}(x_0))^2 \right] = \left[ \text{Bias}(\hat{f}(x_0)) \right]^2 + \text{Var}(\hat{f}(x_0)) + \text{Var}(\epsilon)$$

Which is commonly written as:

$$\text{Expected Test MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

---

## 2. Mathematical Definitions of Bias and Variance

In statistical learning, bias and variance have exact mathematical formulations:

### 1. Structural Bias
**Bias** measures the systematic error introduced by approximating a complex real-world relation $f(x)$ with a simpler model class:

$$\text{Bias}(\hat{f}(x)) = E\left[ \hat{f}(x) \right] - f(x)$$

### 2. Model Variance
**Variance** measures the amount by which our model prediction $\hat{f}(x)$ would change if we trained it on a different dataset:

$$\text{Var}(\hat{f}(x)) = E\left[ \hat{f}(x)^2 \right] - \left( E\left[ \hat{f}(x) \right] \right)^2$$

### Understanding the Expectations
The expectations ($E[\cdot]$) in the bias and variance equations are calculated **over all possible realizations of the training dataset** of a fixed size $m$.

---

## 3. Scenario: Real Estate Valuation Engine

You are predicting **house prices (in thousands of USD)** based on square footage ($x$).
- The true systematic relation is non-linear: $f(x_0) = 400$ (\$400k) for a house size of $2,000$ sq ft.
- We have access to different local historical sales datasets ($D_1 \dots D_5$), representing different neighborhoods.

### Empirical Simulation
Suppose we train five identical models on these five neighborhoods. We evaluate the predicted price of a test house of $2,000$ sq ft ($x_0$):

| Model ($k$) | Training Dataset ($D_k$) | Predicted Price ($\hat{f}_{D_k}(x_0)$) |
| :--- | :--- | :--- |
| 1 | Neighborhood A | \$380k |
| 2 | Neighborhood B | \$420k |
| 3 | Neighborhood C | \$390k |
| 4 | Neighborhood D | \$410k |
| 5 | Neighborhood E | \$400k |

### Step-by-Step Empirical Calculations

#### Step 1: Calculate the Expected Prediction $E[\hat{f}(x_0)]$
The expected prediction is the average prediction outputted by our model class across all training datasets:

$$E\left[ \hat{f}(x_0) \right] \approx \frac{1}{5} \sum_{k=1}^{5} \hat{f}_{D_k}(x_0) = \frac{380 + 420 + 390 + 410 + 400}{5} = 400 \text{ (\$400k)}$$

#### Step 2: Calculate the Structural Bias
Bias measures the distance between this average prediction and the absolute truth $f(x_0) = 400$:

$$\text{Bias}(\hat{f}(x_0)) = E\left[ \hat{f}(x_0) \right] - f(x_0) = 400 - 400 = 0$$

- **Interpretation:** Our model class has **zero bias** at this point because its average prediction matches the true physical value.

#### Step 3: Calculate the Model Variance
Variance measures the spread of the individual model predictions around their own expected average value ($400$):

$$\text{Var}(\hat{f}(x_0)) = E\left[ \left(\hat{f}(x_0) - E[\hat{f}(x_0)]\right)^2 \right]$$
$$\text{Var}(\hat{f}(x_0)) \approx \frac{(380-400)^2 + (420-400)^2 + (390-400)^2 + (410-400)^2 + (400-400)^2}{5}$$
$$\text{Var}(\hat{f}(x_0)) \approx \frac{(-20)^2 + (20)^2 + (-10)^2 + (10)^2 + 0^2}{5} = \frac{400 + 400 + 100 + 100 + 0}{5} = 200 \text{ (\$k)}^2$$

- **Interpretation:** The model variance is **200**.

---

## 4. Calculating Expected Test MSE

If the ambient noise in house pricing (due to negotiation, styling, etc.) is $\text{Var}(\epsilon) = \sigma^2 = 50$:

$$\text{Expected Test MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
$$\text{Expected Test MSE} = 0^2 + 200 + 50 = 250\text{ (\$k)}^2 \quad (\text{RMSE} \approx \$15.8\text{k})$$

Even with a perfectly unbiased model, our predictions will deviate by an average of \$15.8k due to model variance across datasets and irreducible transaction noise.
