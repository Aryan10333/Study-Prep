# Study Report: 1 — Linear Regression & Array Basics

## 🎯 Overview & Daily Focus
- **Objectives**: Master the foundations of Linear Regression (Ordinary Least Squares (OLS) estimation, $R^2$, multiple regression, and model significance) and review foundational array/hash table DSA techniques by solving initial Warm-Up Easy problems.
- **Connection to previous days**: This is Day 1 of the study roadmap. It serves as the baseline for machine learning theory (StatQuest series) and data structures and algorithms (NeetCode arrays/hashing track).
- **Duration**: 2.5 hours total (1.25 hours Morning Session, 1.25 hours Evening Session).
- **Targeted Deliverables**:
  - Watch StatQuest's Linear Regression series (Parts 1-4).
  - Solve three foundational array problems on LeetCode: Two Sum, Contains Duplicate, and Best Time to Buy and Sell Stock.
  - Complete the Day 1 Knowledge Check & Active Recall exercises.

---

## 🌅 Morning Session: Linear Regression & Least Squares (Duration: 1.25 hours)
- **Continuity Note**: Day 1 kickoff of the ML track. Starting the StatQuest machine learning foundations series from scratch.
- **Curated Free Resources**:
  - **StatQuest: Fitting a Line (Ordinary Least Squares)** | An intuitive introduction to how OLS fits a line to quantitative data by minimizing the sum of squared residuals. | [Watch Video](https://www.youtube.com/watch?v=PaFPbb66DxQ)
  - **StatQuest: Linear Regression, Clearly Explained!!!** | Explains the concepts of simple linear regression, residuals, and coefficient interpretation. | [Watch Video](https://www.youtube.com/watch?v=nk2CQITm_eo)
  - **StatQuest: R-squared, Clearly Explained!!!** | Covers how the $R^2$ coefficient measures the proportion of variance in the target variable explained by the regression model. | [Watch Video](https://www.youtube.com/watch?v=2AQKmw14mHM)
  - **StatQuest: Multiple Regression, Clearly Explained!!!** | Extends simple linear regression to multiple independent features and explains the core concepts behind assessing model fit. | [Watch Video](https://www.youtube.com/watch?v=zITIFTsivN8)

---

## 🌃 Evening Session: DSA: Big O & Arrays (Duration: 1.25 hours)
- **Target Problems**:
  - **Two Sum (LeetCode ID: 1 - Easy)** | Technique: Hash Map / Single Pass | [LeetCode Link](https://leetcode.com/problems/two-sum/) | [Video Explanation Link](https://www.youtube.com/watch?v=KLlXCFG5TnA)
    - *Hint 1*: For each number, we need to find its complement (i.e., `target - num`).
    - *Hint 2*: Brute force uses nested loops ($O(n^2)$ time). We can reduce search time to $O(1)$ by storing seen values and their indices.
    - *Hint 3*: Traverse the array once. For each element, check if its complement is already in the hash map. If yes, return the current index and the complement's index; if not, insert the current element and its index into the map.
    - *Optimal Complexity*: Time: $O(n)$, Space: $O(n)$
  - **Contains Duplicate (LeetCode ID: 217 - Easy)** | Technique: Hash Set | [LeetCode Link](https://leetcode.com/problems/contains-duplicate/) | [Video Explanation Link](https://www.youtube.com/watch?v=377y8C-p2Rk)
    - *Hint 1*: We want to determine if any element appears at least twice in the array.
    - *Hint 2*: If we sort the array, duplicates will be adjacent, resulting in $O(n \log n)$ time and $O(1)$ auxiliary space. We can trade space to achieve linear time complexity.
    - *Hint 3*: Initialize an empty hash set. Iterate through the array and check if the current element is in the set. If it is, return `True`. Otherwise, add it to the set. If the loop finishes, return `False`.
    - *Optimal Complexity*: Time: $O(n)$, Space: $O(n)$
  - **Best Time to Buy and Sell Stock (LeetCode ID: 121 - Easy)** | Technique: Sliding Window / Two Pointers | [LeetCode Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | [Video Explanation Link](https://www.youtube.com/watch?v=1pkZkG_edc0)
    - *Hint 1*: We want to find the maximum positive difference between a future price and a past price.
    - *Hint 2*: A brute force comparison of all pairs takes $O(n^2)$ time. We can achieve $O(n)$ time by processing the array in a single pass.
    - *Hint 3*: Maintain a running minimum price (`min_price`) and a running maximum profit (`max_profit`). As you iterate, update `min_price` to be the lowest seen so far, and calculate the potential profit at the current step. Update `max_profit` if the current profit exceeds the historical max.
    - *Optimal Complexity*: Time: $O(n)$, Space: $O(1)$

---

## 🛠️ Weekend Deep Dive & Project Implementation (Duration: 0 hours)
- **Block 1 & 2 (DSA)**: Not scheduled for weekdays. Detailed weekly reviews and assessments are scheduled on weekends (Days 6 & 7).
- **Block 3 (Project)**: **AI Router Gateway Project (Phase 0)** is scheduled for Day 6 (Saturday). No project setup or code implementation is scheduled for today.

---

## ⚠️ Common Pitfalls
- **Overlooking Linear Regression Assumptions**: Students often apply linear regression without verifying the five classical OLS assumptions: linearity of the relationship, homoscedasticity (constant variance of residuals), independence of errors, normality of error distribution, and lack of multicollinearity.
- **R-squared Overestimation (Inflation)**: Assuming a higher $R^2$ always means a better fit. Adding *any* variable to a regression model (even random noise) will increase or maintain the $R^2$ score because it gives the model more degrees of freedom. Always check the **Adjusted $R^2$** or F-statistic when evaluating multi-feature models.
- **Two Sum Sorting Trap**: Attempting to sort the input array in Two Sum to use a two-pointer approach. While sorting works, it changes the original indices of the elements. Tracking the original indices requires storing index-value pairs, which adds memory overhead and increases complexity to $O(n \log n)$ due to the sorting step, which is suboptimal compared to the $O(n)$ single-pass hash map.
- **Array Indexing in Profit Calculation**: In "Best Time to Buy and Sell Stock," updating the maximum profit before updating the minimum price can lead to incorrect logic. Maintain the strict order: calculate potential profit relative to the current minimum, update maximum profit, and then update the minimum price.

---

## ✅ Daily Action Checklist
- [ ] Watch StatQuest: Fitting a Line (OLS) and understand residuals.
- [ ] Watch StatQuest: Linear Regression, Clearly Explained.
- [ ] Watch StatQuest: R-squared, Clearly Explained.
- [ ] Watch StatQuest: Multiple Regression, Clearly Explained.
- [ ] Study the basic theory of Big O notation (time and space analysis).
- [ ] Solve LeetCode 1: Two Sum.
- [ ] Solve LeetCode 217: Contains Duplicate.
- [ ] Solve LeetCode 121: Best Time to Buy and Sell Stock.

---

## 📝 Knowledge Check & Active Recall

### 1. Concept Check
* **Question 1**: Derive the Ordinary Least Squares (OLS) closed-form solutions for the parameter weight $w$ and bias $b$ in a simple linear regression model $f_{w,b}(x^{(i)}) = w x^{(i)} + b$.
  <details>
  <summary>Show Answer</summary>

  Let $m$ be the number of training examples. The goal of Ordinary Least Squares is to minimize the sum of squared residuals (which is proportional to the mean squared error). The cost function $J(w, b)$ is defined as:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2m} \sum_{i=1}^m \left( w x^{(i)} + b - y^{(i)} \right)^2$$

  To find the values of $w$ and $b$ that minimize $J(w, b)$, we take the partial derivatives with respect to each parameter and set them to zero.

  **First: Solving for the bias term $b$**
  $$\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( w x^{(i)} + b - y^{(i)} \right) = 0$$
  $$\sum_{i=1}^m w x^{(i)} + \sum_{i=1}^m b - \sum_{i=1}^m y^{(i)} = 0$$
  Since $\sum_{i=1}^m b = m \cdot b$, we rewrite this as:
  $$w \sum_{i=1}^m x^{(i)} + m \cdot b - \sum_{i=1}^m y^{(i)} = 0$$
  Divide the entire equation by $m$:
  $$w \bar{x} + b - \bar{y} = 0 \implies b = \bar{y} - w \bar{x}$$
  where $\bar{x} = \frac{1}{m} \sum_{i=1}^m x^{(i)}$ and $\bar{y} = \frac{1}{m} \sum_{i=1}^m y^{(i)}$ represent the sample means of the features and targets respectively.

  **Second: Solving for the weight term $w$**
  $$\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum_{i=1}^m \left( w x^{(i)} + b - y^{(i)} \right) x^{(i)} = 0$$
  Substitute the expression for $b$ ($b = \bar{y} - w \bar{x}$) into this derivative equation:
  $$\sum_{i=1}^m \left( w x^{(i)} + (\bar{y} - w \bar{x}) - y^{(i)} \right) x^{(i)} = 0$$
  $$\sum_{i=1}^m \left( w (x^{(i)} - \bar{x}) - (y^{(i)} - \bar{y}) \right) x^{(i)} = 0$$
  $$\sum_{i=1}^m w (x^{(i)} - \bar{x}) x^{(i)} = \sum_{i=1}^m (y^{(i)} - \bar{y}) x^{(i)}$$
  Since $\sum_{i=1}^m (x^{(i)} - \bar{x}) = 0$, we can subtract $\bar{x}$ from the multiplying $x^{(i)}$ term on both sides without changing the sum value:
  $$w \sum_{i=1}^m (x^{(i)} - \bar{x})(x^{(i)} - \bar{x}) = \sum_{i=1}^m (y^{(i)} - \bar{y})(x^{(i)} - \bar{x})$$
  $$w \sum_{i=1}^m (x^{(i)} - \bar{x})^2 = \sum_{i=1}^m (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})$$
  $$w = \frac{\sum_{i=1}^m (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^m (x^{(i)} - \bar{x})^2}$$

  This gives the analytical closed-form solution for the OLS regression parameters:
  - $w = \frac{\text{Covariance}(x, y)}{\text{Variance}(x)}$
  - $b = \bar{y} - w \bar{x}$
  </details>

* **Question 2**: Explain why $R^2$ always increases or remains constant when new features are added to a linear regression model, and state how Adjusted $R^2$ addresses this issue.
  <details>
  <summary>Show Answer</summary>

  The coefficient of determination $R^2$ is defined as:
  $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
  where:
  - $SS_{res} = \sum_{i=1}^m \left( y^{(i)} - f_{w,b}(x^{(i)}) \right)^2$ is the sum of squared residuals.
  - $SS_{tot} = \sum_{i=1}^m \left( y^{(i)} - \bar{y} \right)^2$ is the total sum of squares (total variance of the target).

  When you add a new feature to the model, the optimization algorithm (OLS) has an additional parameter weight to optimize. Even if the feature is completely random noise, the model can assign it a weight of $0$ to maintain the exact same fit, or assign it a non-zero weight to capture random sample-specific correlations. Consequently, the sum of squared residuals $SS_{res}$ must either decrease or stay the same:
  $$SS_{res, \text{ new}} \le SS_{res, \text{ old}}$$
  Because $SS_{tot}$ depends only on the targets $y^{(i)}$ and does not change, the ratio $\frac{SS_{res}}{SS_{tot}}$ will decrease or remain constant, meaning $R^2$ will always increase or stay constant.

  To correct for this inflation, **Adjusted $R^2$** incorporates the degrees of freedom:
  $$R^2_{adj} = 1 - \left[ \frac{(1 - R^2)(m - 1)}{m - n - 1} \right]$$
  where:
  - $m$ is the number of training examples.
  - $n$ is the number of features.

  If a newly added feature does not significantly reduce the residual variance ($SS_{res}$), the denominator $(m - n - 1)$ decreases, increasing the multiplier term $\frac{m - 1}{m - n - 1}$. This penalty offsets any marginal reduction in $(1 - R^2)$, causing $R^2_{adj}$ to decrease. Thus, Adjusted $R^2$ will only increase if the new feature improves the model fit more than what would be expected by pure chance.
  </details>

---

### 2. Active Recall
* **Question 1**: What are the five classical assumptions of Ordinary Least Squares (OLS) linear regression?
  <details>
  <summary>Show Answer</summary>

  The five classical assumptions required for OLS to be the Best Linear Unbiased Estimator (BLUE) are:
  1. **Linearity**: The relationship between the independent variables $x$ and the dependent variable $y$ is linear in the parameters.
  2. **Independence of Errors (No Autocorrelation)**: The residuals (errors) are uncorrelated with one another.
  3. **Homoscedasticity**: The variance of the error terms is constant across all values of the independent variables.
  4. **Exogeneity (No Endogeneity)**: The independent variables are uncorrelated with the error term, i.e., $E[\epsilon | x] = 0$.
  5. **No Multicollinearity**: In multiple linear regression, none of the independent features are perfectly linearly correlated with each other.
  
  *(Note: Normality of error terms is not strictly required for OLS to be unbiased, but it is necessary for performing hypothesis tests, constructing confidence intervals, and calculating p-values).*
  </details>

* **Question 2**: Write down the formula for the $R^2$ coefficient of determination in terms of the sum of squared residuals ($SS_{res}$) and the total sum of squares ($SS_{tot}$), defining the terms mathematically using Andrew Ng notation style.
  <details>
  <summary>Show Answer</summary>

  The formula for $R^2$ is:
  $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
  where:
  $$SS_{res} = \sum_{i=1}^m \left( y^{(i)} - f_{w,b}(x^{(i)}) \right)^2$$
  $$SS_{tot} = \sum_{i=1}^m \left( y^{(i)} - \bar{y} \right)^2$$
  And:
  - $y^{(i)}$ is the actual target value of the $i$-th training example.
  - $f_{w,b}(x^{(i)}) = w^T x^{(i)} + b$ is the model's prediction for the $i$-th training example.
  - $\bar{y} = \frac{1}{m} \sum_{i=1}^m y^{(i)}$ is the mean target value across all $m$ training examples.
  </details>

* **Question 3**: Define the following Big O time complexities and provide a typical array/hashing example for each: $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, $O(n^2)$.
  <details>
  <summary>Show Answer</summary>

  1. **$O(1)$ - Constant Time**: The runtime does not depend on the input size $n$.
     - *Example*: Accessing an array element by index (`arr[i]`) or checking if a key exists in a hash map.
  2. **$O(\log n)$ - Logarithmic Time**: The problem size is reduced by a constant fraction (usually half) at each step.
     - *Example*: Binary Search on a sorted array.
  3. **$O(n)$ - Linear Time**: The runtime grows in direct proportion to the size of the input $n$.
     - *Example*: Single pass traversal of an array to find the maximum value, or constructing a hash set from an unsorted array.
  4. **$O(n \log n)$ - Linearithmic Time**: Performing a logarithmic operation $n$ times.
     - *Example*: Standard comparison-based sorting algorithms like Merge Sort, Quick Sort, or Heap Sort.
  5. **$O(n^2)$ - Quadratic Time**: The runtime is proportional to the square of the input size $n$.
     - *Example*: Finding pairs of elements in an array using nested loops (such as brute force Two Sum).
  </details>

---

### 3. Interview Challenge
* **Challenge 1**: Under what conditions is a linear regression model solved via the closed-form Normal Equation rather than gradient descent? Compare their computational complexity and memory tradeoffs.
  <details>
  <summary>Show Answer</summary>

  The closed-form solution to linear regression (the Normal Equation) is written as:
  $$\theta = (X^T X)^{-1} X^T y$$
  where $X$ is the $m \times (n+1)$ design matrix and $y$ is the $m$-dimensional target vector.

  ### Conditions for choosing the Normal Equation:
  - The number of features $n$ is small to moderate (typically $n < 10,000$).
  - The design matrix $X$ fits entirely in memory.
  - The matrix $X^T X$ is invertible (non-singular). If features are redundant (multicollinearity) or $m < n$, regularized forms like Ridge/Lasso are used to guarantee invertibility.

  ### Tradeoffs & Complexity Analysis:
  | Metric | Normal Equation | Gradient Descent |
  | :--- | :--- | :--- |
  | **Time Complexity** | $O(n^3)$ (due to matrix inversion of $X^T X$) | $O(k \cdot m \cdot n)$ (where $k$ is the number of iterations) |
  | **Space Complexity** | $O(n^2)$ to store $X^T X$ | $O(n)$ (stores weights vector $\theta$) |
  | **Hyperparameters** | None (solves analytically in one step) | Requires tuning learning rate $\alpha$, epochs, and batch size |
  | **Scaling with $n$** | Scales poorly; inversion becomes extremely slow for large $n$ | Scales linearly with feature size $n$; ideal for huge feature spaces |
  | **Scaling with $m$** | Scales linearly with sample count $m$ in terms of matrix multiplication | Fits well with Stochastic or Mini-batch GD for out-of-memory datasets |

  **Summary**: Choose the Normal Equation for datasets with a small number of features ($n < 10,000$) to get exact mathematical values with zero hyperparameter tuning. For larger scale datasets or online learning settings, gradient descent is preferred.
  </details>

---

### 4. Reflection
* **Question**: In a scenario where the number of training examples is small ($m = 100$) but the number of features is large ($n = 200$), what mathematical issues does the Ordinary Least Squares (OLS) Normal Equation encounter, and what is the intuitive way to solve this?
  *No answer key provided. Self-assess your understanding of matrix ranks, collinearity, and regularization concepts.*
