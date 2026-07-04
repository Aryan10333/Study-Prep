# Deep Learning: Neural Network Foundations & Forward Propagation

This guide details the structural and mathematical mechanics of Multi-Layer Perceptrons (MLPs), activation function gradients, and walks through a manual forward propagation pass step-by-step.

---

## 1. Multi-Layer Perceptron (MLP) Architecture

A Multi-Layer Perceptron (MLP) consists of an input layer, one or more hidden layers, and an output layer. We use **vectorized representation** where each column represents a training example, following Andrew Ng's dimension standards.

For a network of $L$ layers:
- **Input Matrix ($X$):** $A^{[0]} = X \in \mathbb{R}^{n_0 \times m}$, where $n_0$ is the input feature count and $m$ is the number of samples.
- **Weight Matrix ($W^{[l]}$):** $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$, where $n_l$ is the number of neurons in layer $l$.
- **Bias Vector ($b^{[l]}$):** $b^{[l]} \in \mathbb{R}^{n_l \times 1}$ (broadcasted across columns during forward prop).

---

## 2. Activation Functions and Gradient Analysis

Activation functions introduce non-linearity, allowing neural networks to learn complex non-linear decision boundaries.

```text
Activation       Formula                        Range          Derivative / Gradient              Failure Mode
----------------------------------------------------------------------------------------------------------------------
Sigmoid          σ(z) = 1 / (1 + e^-z)          (0, 1)         σ'(z) = σ(z)(1 - σ(z))             Vanishing Gradient
Tanh             tanh(z) = (e^z - e^-z)/...     (-1, 1)        tanh'(z) = 1 - tanh²(z)            Vanishing Gradient
ReLU             g(z) = max(0, z)               [0, inf)       g'(z) = 1 (z > 0), 0 (z < 0)       Dead ReLU (z < 0)
Leaky ReLU       g(z) = max(αz, z)              (-inf, inf)    g'(z) = 1 (z > 0), α (z < 0)       None
ELU              g(z) = z (z > 0), α(e^z - 1)   [-α, inf)      g'(z) = 1 (z > 0), g(z) + α (z≤0)  Slow exponent calculation
```

### The Gradient Volatility Trap
- **Vanishing Gradients:** For Sigmoid and Tanh, when inputs are large positive or negative values ($z > 4$ or $z < -4$), the curve flattens. The derivative drops close to $0$. During backpropagation, this small derivative is multiplied layer-by-layer, causing weight updates in early layers to vanish, stalling training.
- **Dead ReLU Problem:** For ReLU, if a neuron's pre-activation $z$ is negative, both its output and its gradient become exactly $0$. Once a neuron is pushed into the negative zone (e.g., due to a large negative gradient step), it will never activate again. **Leaky ReLU** solves this by maintaining a small slope $\alpha$ (typically $0.01 - 0.1$) for negative inputs.
- **ELU (Exponential Linear Unit):** Because ELU can return negative values (down to $-\alpha$), it pushes the average output of activations closer to zero, which speeds up convergence by reducing the bias shift in downstream layers. It also has a continuous derivative at $z=0$, ensuring smoother gradients.

### Activation Functions: Production Use Cases & Selection Rules
- **Sigmoid:** 
  - *Where it is helpful:* Only in the output layers of binary classifiers (to produce a probability $p \in [0, 1]$) or inside gating mechanisms (e.g., LSTM forget gates).
  - *Why:* Never use in hidden layers; its derivative saturates at $0$ for values outside $[-4, 4]$, causing the vanishing gradient problem.
- **Tanh (Hyperbolic Tangent):**
  - *Where it is helpful:* Hidden layers of recurrent sequence networks (like LSTMs) or output layers when the targets are bound within $[-1, 1]$.
  - *Why:* Zero-centered output (mean is close to $0$) makes the gradient updates for downstream layers more stable than Sigmoid, though it still suffers from vanishing gradients.
- **ReLU (Rectified Linear Unit):**
  - *Where it is helpful:* The default choice for hidden layers in deep feed-forward neural networks and CNNs.
  - *Why:* Its gradient is constant ($1.0$) for all positive inputs, which speeds up convergence and enables training very deep models.
- **LeakyReLU:**
  - *Where it is helpful:* Used when training deep MLPs or GANs where neurons are systematically dying (producing flat zero outputs).
  - *Why:* Maintains a small slope $\alpha = 0.01$ for negative inputs, keeping gradients flowing back even when neurons are inactive.
- **ELU (Exponential Linear Unit):**
  - *Where it is helpful:* Deep neural networks operating on highly continuous features where normalization layers (like Batch Norm) are too expensive or unstable.
  - *Why:* Negative values allow the activation mean to remain close to zero (reducing internal covariate shift) while maintaining a smooth gradient at zero.

### Diagnostic Visual (Activations & Derivatives)
The plot below illustrates the activations (blue) and their corresponding gradient curves (red dashed):

![Activations](images/activation_functions_and_derivatives.png)

---

## 3. Loss Functions in Deep Learning

Selecting the correct loss function determines how backpropagation calculates target errors.

### A. Classification Losses
- **Binary Cross-Entropy (BCE) Loss:** Used for binary classification (single output neuron predicting sigmoid probability $a \in [0, 1]$):
  $$L_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \ln(a_i) + (1 - y_i) \ln(1 - a_i) \right]$$
  - *Production Utility:* Essential for **multi-label classification** (e.g., tagging a movie with multiple categories like "Action" and "Sci-Fi" simultaneously). Each output node computes an independent BCE loss.
- **Categorical Cross-Entropy (CCE) Loss:** Used for multi-class classification (softmax output vector $a \in [0, 1]^K$ representing class probabilities):
  $$L_{\text{CCE}} = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{i,k} \ln(a_{i,k})$$
  - *Production Utility:* Standard for **single-label multi-class classification** (e.g., routing support tickets to exactly one department). It forces output probabilities to sum to 1.0.

### B. Regression Losses
- **Mean Squared Error (MSE) Loss ($L_2$):** Penalizes outliers quadratically:
  $$L_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$
  - *Production Utility:* Best for **averages-focused tasks** (e.g., inventory forecasting) where a large error is exponentially worse than small errors. It pulls the model predictions toward the mean.
- **Mean Absolute Error (MAE) Loss ($L_1$):** Penalizes errors linearly:
  $$L_{\text{MAE}} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|$$
  - *Production Utility:* Best for **outlier-heavy data** (e.g., real estate pricing). It is robust to extreme noise and pulls predictions toward the median.

### C. Focal Loss (Handling Class Imbalance)
Modified BCE loss to combat extreme class imbalance (e.g., fraud or click prediction). It introduces a focusing parameter $\gamma \ge 0$ to downweight the loss of easy-to-classify samples, forcing the network to focus on hard, misclassified examples:
$$L_{\text{Focal}} = -\frac{1}{m} \sum_{i=1}^m \alpha_t (1 - p_{t,i})^\gamma \ln(p_{t,i})$$
Where:
- $p_{t} = a$ if $y = 1$, else $1 - a$.
- $\alpha_t$ handles class imbalance weights.
- $\gamma$ (focusing parameter, typically set to $2.0$) controls the rate at which easy examples are downweighted.
- *Production Utility:* Vital for **anomaly detection and CTR prediction** where minority class signals are $<1\%$. It avoids needing to artificially downsample negative datasets.

### D. Metric Learning & Embedding Losses
Used to train networks to project inputs into low-dimensional vector spaces where similar items are close together and dissimilar items are far apart (e.g., face verification, vector search).
- **Contrastive Loss:** Given a distance $d = \|a_1 - a_2\|_2$ between two sample embeddings and similarity label $y$ ($y=0$ if similar, $y=1$ if dissimilar):
  $$L_{\text{Contrastive}} = \frac{1}{2} (1 - y) d^2 + \frac{1}{2} y \max(0, m - d)^2 \quad \text{(where } m \text{ is the margin)}$$
  - *Production Utility:* Used in **Siamese networks** to train text/image match systems (e.g., query-to-product relevance search).
- **Triplet Loss:** Given an anchor sample $a$, a positive sample $p$ (same class), and a negative sample $n$ (different class):
  $$L_{\text{Triplet}} = \max\left(0, \|a - p\|_2^2 - \|a - n\|_2^2 + \alpha\right) \quad \text{(where } \alpha \text{ is the margin gap)}$$
  - *Production Utility:* Standard for **facial recognition, biometrics, and semantic vector database embeddings**, allowing models to map inputs into cluster groupings without explicit class bounds.

---

## 4. Step-by-Step Hand Calculations (Andrew Ng Style)

### A. Forward Propagation (MLP Forward Pass)
- **Input:** $X = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix}$ (2 features, $m=1$ sample)
- **Layer 1 (Hidden, 2 neurons, ReLU activation):**
  $$W^{[1]} = \begin{bmatrix} 0.2 & -0.3 \\ 0.4 & 0.1 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0.05 \\ -0.1 \end{bmatrix}$$
- **Layer 2 (Output, 1 neuron, Sigmoid activation):**
  $$W^{[2]} = \begin{bmatrix} 0.5 & 0.8 \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} -0.2 \end{bmatrix}$$

1. **Pre-Activation Layer 1 ($Z^{[1]}$):**
   $$Z_1^{[1]} = (0.2 \times 0.5) + (-0.3 \times -0.2) + 0.05 = 0.10 + 0.06 + 0.05 = 0.21$$
   $$Z_2^{[1]} = (0.4 \times 0.5) + (0.1 \times -0.2) - 0.10 = 0.20 - 0.02 - 0.10 = 0.08$$
   $$Z^{[1]} = \begin{bmatrix} 0.21 \\ 0.08 \end{bmatrix}$$
2. **Post-Activation Layer 1 ($A^{[1]}$):**
   $$A_1^{[1]} = \max(0, 0.21) = 0.21, \quad A_2^{[1]} = \max(0, 0.08) = 0.08 \implies A^{[1]} = \begin{bmatrix} 0.21 \\ 0.08 \end{bmatrix}$$
3. **Pre-Activation Layer 2 ($Z^{[2]}$):**
   $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} = \begin{bmatrix} 0.5 & 0.8 \end{bmatrix} \begin{bmatrix} 0.21 \\ 0.08 \end{bmatrix} + \begin{bmatrix} -0.2 \end{bmatrix} = 0.105 + 0.064 - 0.20 = -0.031$$
4. **Post-Activation Layer 2 ($A^{[2]}$ / Prediction $\hat{y}$):**
   $$\hat{y} = A^{[2]} = \sigma(-0.031) = \frac{1}{1 + e^{0.031}} \approx \frac{1}{1 + 1.03148} \approx 0.4922$$

---

### B. Focal Loss (Easy vs. Hard Examples)
Let's calculate the loss contribution of two samples under standard **BCE Loss** vs. **Focal Loss** ($\gamma=2.0$, $\alpha=1.0$):
- **Sample 1 (Easy Correct, $y_1=1$):** Prediction probability $p_1 = 0.90$.
  - **BCE Loss:** $L_{\text{BCE}} = -\ln(0.90) \approx 0.1054$
  - **Focal Loss:** $L_{\text{Focal}} = -(1 - 0.90)^2 \ln(0.90) = -(0.10)^2 \times (-0.1054) = 0.01 \times 0.1054 \approx \mathbf{0.00105}$
  - *Result:* The loss is suppressed by **$100\text{x}$** because the model is already confident.
- **Sample 2 (Hard Misclassified, $y_2=1$):** Prediction probability $p_2 = 0.20$.
  - **BCE Loss:** $L_{\text{BCE}} = -\ln(0.20) \approx 1.6094$
  - **Focal Loss:** $L_{\text{Focal}} = -(1 - 0.20)^2 \ln(0.20) = -(0.80)^2 \times (-1.6094) = 0.64 \times 1.6094 \approx \mathbf{1.0300}$
  - *Result:* The loss is only suppressed by **$1.56\text{x}$**.
- **Intuition:** Under BCE, the hard sample contributed **$15.2\text{x}$** more loss than the easy sample. Under Focal Loss, the hard sample contributes **$980.9\text{x}$** more loss than the easy sample, forcing backpropagation to focus updates exclusively on fixing hard/rare errors.

---

### C. Triplet Loss (Embedding Distance Gap)
Calculate the loss for a face embedding model given:
- **Anchor ($a$):** $\begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}$ | **Positive ($p$):** $\begin{bmatrix} 0.15 \\ 0.25 \end{bmatrix}$ | **Negative ($n$):** $\begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix}$ | **Margin ($\alpha$):** $0.5$

1. **Square Euclidean Distance Anchor-to-Positive ($d(a,p)^2$):**
   $$d(a,p)^2 = (0.1 - 0.15)^2 + (0.2 - 0.25)^2 = (-0.05)^2 + (-0.05)^2 = 0.0025 + 0.0025 = 0.0050$$
2. **Square Euclidean Distance Anchor-to-Negative ($d(a,n)^2$):**
   $$d(a,n)^2 = (0.1 - 0.5)^2 + (0.2 - 0.6)^2 = (-0.4)^2 + (-0.4)^2 = 0.1600 + 0.1600 = 0.3200$$
3. **Triplet Loss ($L$):**
   $$L = \max\left(0, d(a,p)^2 - d(a,n)^2 + \alpha\right) = \max\left(0, 0.0050 - 0.3200 + 0.50\right) = \max\left(0, 0.1850\right) = \mathbf{0.1850}$$
- **Intuition:** Since the negative sample was not pushed outside the margin gap ($0.3200 - 0.0050 = 0.3150 < 0.50$), the triplet produces a positive loss of $0.1850$, triggering backpropagation to adjust parameters and push the negative face farther away.

---

## 5. Production Scenario & Example

### Scenario: Ad-Click Conversion Tracking (CTR) under Heavy Imbalance
You are designing an upstream deep intent classification model for an ad auction engine. In production, the click-through conversion rate is extremely sparse—only $0.05\%$ of impressions lead to active conversions ($y=1$).
- **The Failure Mode:** When trained using a standard Binary Cross-Entropy loss, the model gets overwhelmed by the $99.95\%$ majority negative samples. It converges on a trivial solution of predicting $0.0$ for all users, achieving a misleadingly high $99.95\%$ accuracy but failing to identify any active ad conversions.
- **The Solution:** We replace BCE with a custom **Focal Loss** layer. The focusing parameters are set to $\gamma=2.0$ and $\alpha=0.25$. As demonstrated in our hand-calculations, Focal Loss suppresses the gradients of the massive pool of easy negatives (background impressions), allowing the model to optimize its parameters on the minority converted users, recovering positive prediction capacity without needing artificial downsampling.

---

## 6. Interactive Practice Notebook
To see this vectorized forward pass run inside a training loop, check out:
- [01_mlp_from_scratch_numpy.ipynb](file:///d:/Study/Prep/machine-learning-prep/deep-learning-foundations/01_mlp_from_scratch_numpy.ipynb)
