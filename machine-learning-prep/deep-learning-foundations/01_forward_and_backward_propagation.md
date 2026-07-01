# Deep Learning: Forward & Backward Propagation

This guide details the matrix calculus behind forward and backward propagation in multi-layer neural networks, utilizing **Andrew Ng's vectorized notation**, and walks through a manual hand-calculation step-by-step.

---

## 1. Vectorized Propagation Equations

For a deep network containing $L$ layers, let $m$ represent the number of training examples, $n_l$ represent the number of units in layer $l$, and $g^{[l]}$ represent the activation function of layer $l$.

### Forward Propagation
For layer $l = 1 \dots L$:
- **Pre-Activation:**
  $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
- **Post-Activation:**
  $$A^{[l]} = g^{[l]}\left(Z^{[l]}\right)$$

*Dimensions:*
- Input $X = A^{[0]} \in \mathbb{R}^{n_0 \times m}$ (where each column is a training sample).
- Weight matrix $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$.
- Bias vector $b^{[l]} \in \mathbb{R}^{n_l \times 1}$ (broadcasted across all $m$ columns).
- Pre-activation $Z^{[l]} \in \mathbb{R}^{n_l \times m}$, Activation $A^{[l]} \in \mathbb{R}^{n_l \times m}$.

---

### Backward Propagation (Vectorized Gradients)

Backward propagation computes the derivatives of the cost function $J$ with respect to parameters $W$ and $b$ using the chain rule.

For the output layer $L$ using Softmax activation and Categorical Cross-Entropy Loss:
- **Output Error Gradient:**
  $$dZ^{[L]} = A^{[L]} - Y$$
  *Where $Y \in \mathbb{R}^{n_L \times m}$ is the one-hot encoded target matrix.*

For hidden layers $l = L-1 \dots 1$:
- **Activation Gradient:**
  $$dA^{[l]} = W^{[l+1]T} dZ^{[l+1]}$$
- **Pre-Activation Gradient:**
  $$dZ^{[l]} = dA^{[l]} \odot g^{[l]\prime}\left(Z^{[l]}\right)$$
  *Where $\odot$ represents the element-wise (Hadamard) product, and $g^{[l]\prime}$ is the element-wise derivative of activation function $g^{[l]}$.*

To update parameters across the ensemble:
- **Weight Gradient:**
  $$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$$
- **Bias Gradient:**
  $$db^{[l]} = \frac{1}{m} \text{np.sum}\left(dZ^{[l]}, \text{axis}=1, \text{keepdims}=\text{True}\right)$$

---

## 2. Parameter Space vs. Functional Space Optimization

- **Parameter Space (e.g., Linear/Logistic Regression):** We optimize a fixed set of model parameters (coefficients/weights $w$) by taking gradient steps to adjust their values: $w \leftarrow w - \alpha \nabla L(w)$.
- **Functional Space (Gradient Boosting):** Since we cannot easily define a simple parameter vector for an ensemble of trees, we optimize the predictions $F(x)$ directly in **functional space**. Each new tree $h_t(x)$ represents a new function that acts as a step vector (negative gradient) to adjust the ensemble: $F_t(x) \leftarrow F_{t-1}(x) + \nu h_t(x)$.

---

## 3. Step-by-Step Hand Calculations: 3-Layer MLP

Let's trace the forward and backward pass for a tiny classification network:
- **Architecture:** 2 inputs ($n_0 = 2$), 3 hidden units ($n_1 = 3$, ReLU activation), and 2 output classes ($n_2 = 2$, Softmax activation).
- **Batch Size ($m$):** 1 training sample.
- **Inputs ($X$):**
  $$X = A^{[0]} = \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}$$
- **True Label ($Y$):** Class 1 (one-hot vector):
  $$Y = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$

---

### Step 1: Initialize Weights & Biases
Let's specify simple starting parameters:
- **Layer 1:**
  $$W^{[1]} = \begin{bmatrix} 0.5 & 0.2 \\ -0.1 & 0.4 \\ 0.3 & -0.2 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}$$
- **Layer 2:**
  $$W^{[2]} = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.5 \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

---

### Step 2: Forward Propagation

#### 1. Calculate $Z^{[1]}$ and $A^{[1]}$ (Hidden ReLU Layer):
$$Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]} = \begin{bmatrix} 0.5 & 0.2 \\ -0.1 & 0.4 \\ 0.3 & -0.2 \end{bmatrix} \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.3 \\ -0.5 \\ 0.5 \end{bmatrix}$$

Apply ReLU activation ($g^{[1]}(z) = \max(0, z)$):
$$A^{[1]} = \text{ReLU}\left(Z^{[1]}\right) = \begin{bmatrix} 0.3 \\ 0.0 \\ 0.5 \end{bmatrix}$$

#### 2. Calculate $Z^{[2]}$ and $A^{[2]}$ (Output Softmax Layer):
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.5 \end{bmatrix} \begin{bmatrix} 0.3 \\ 0.0 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
$$Z^{[2]} = \begin{bmatrix} (0.2 \times 0.3) + 0 + (0.1 \times 0.5) \\ (0.4 \times 0.3) + 0 + (-0.5 \times 0.5) \end{bmatrix} = \begin{bmatrix} 0.06 + 0.05 \\ 0.12 - 0.25 \end{bmatrix} = \begin{bmatrix} 0.11 \\ -0.13 \end{bmatrix}$$

Apply Softmax activation ($A^{[2]} = e^{Z^{[2]}} / \sum e^{z_j}$):
- $e^{0.11} \approx 1.1163$
- $e^{-0.13} \approx 0.8781$
- Sum of exponentials $= 1.1163 + 0.8781 = 1.9944$
$$A^{[2]} = \begin{bmatrix} 1.1163 / 1.9944 \\ 0.8781 / 1.9944 \end{bmatrix} \approx \begin{bmatrix} 0.5597 \\ 0.4403 \end{bmatrix}$$

---

### Step 3: Compute Loss
$$L = -\sum_{k=1}^2 Y_k \ln(A_k^{[2]}) = -\left( 1.0 \times \ln(0.5597) + 0.0 \right) \approx -\ln(0.5597) \approx 0.5804$$

---

### Step 4: Backward Propagation

#### 1. Output Layer Gradient ($dZ^{[2]}$):
$$dZ^{[2]} = A^{[2]} - Y = \begin{bmatrix} 0.5597 \\ 0.4403 \end{bmatrix} - \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} -0.4403 \\ 0.4403 \end{bmatrix}$$

#### 2. Parameter Gradients for Layer 2 ($dW^{[2]}$, $db^{[2]}$):
Since $m = 1$:
$$dW^{[2]} = dZ^{[2]} A^{[1]T} = \begin{bmatrix} -0.4403 \\ 0.4403 \end{bmatrix} \begin{bmatrix} 0.3 & 0.0 & 0.5 \end{bmatrix} = \begin{bmatrix} -0.1321 & 0.0 & -0.2202 \\ 0.1321 & 0.0 & 0.2202 \end{bmatrix}$$

$$db^{[2]} = dZ^{[2]} = \begin{bmatrix} -0.4403 \\ 0.4403 \end{bmatrix}$$

#### 3. Backpropagate to Hidden Layer ($dA^{[1]}$, $dZ^{[1]}$):
$$dA^{[1]} = W^{[2]T} dZ^{[2]} = \begin{bmatrix} 0.2 & 0.4 \\ -0.3 & 0.1 \\ 0.1 & -0.5 \end{bmatrix} \begin{bmatrix} -0.4403 \\ 0.4403 \end{bmatrix} = \begin{bmatrix} (0.2 \times -0.4403) + (0.4 \times 0.4403) \\ (-0.3 \times -0.4403) + (0.1 \times 0.4403) \\ (0.1 \times -0.4403) + (-0.5 \times 0.4403) \end{bmatrix} = \begin{bmatrix} 0.0881 \\ 0.1761 \\ -0.2642 \end{bmatrix}$$

Now multiply element-wise by the derivative of the ReLU activation ($g^{[1]\prime}(Z^{[1]}) = [1.0, 0.0, 1.0]^T$ since $Z^{[1]} = [0.3, -0.5, 0.5]^T$):
$$dZ^{[1]} = dA^{[1]} \odot \text{ReLU}^\prime\left(Z^{[1]}\right) = \begin{bmatrix} 0.0881 \\ 0.1761 \\ -0.2642 \end{bmatrix} \odot \begin{bmatrix} 1.0 \\ 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.0881 \\ 0.0 \\ -0.2642 \end{bmatrix}$$
*Note: The gradient at index 1 is zeroed out because that hidden unit was inactive (died) during the forward pass.*

#### 4. Parameter Gradients for Layer 1 ($dW^{[1]}$, $db^{[1]}$):
$$dW^{[1]} = dZ^{[1]} A^{[0]T} = \begin{bmatrix} 0.0881 \\ 0.0 \\ -0.2642 \end{bmatrix} \begin{bmatrix} 1.0 & -1.0 \end{bmatrix} = \begin{bmatrix} 0.0881 & -0.0881 \\ 0.0 & 0.0 \\ -0.2642 & 0.2642 \end{bmatrix}$$

$$db^{[1]} = dZ^{[1]} = \begin{bmatrix} 0.0881 \\ 0.0 \\ -0.2642 \end{bmatrix}$$

---

## 4. Interactive Practice Notebook
To run backpropagation from scratch in NumPy, train on non-linear datasets, and check numerical gradients, open the interactive notebook:
- [01_mlp_backpropagation_from_scratch.ipynb](file:///d:/Study/Prep/machine-learning-prep/deep-learning-foundations/01_mlp_backpropagation_from_scratch.ipynb)
