# Deep Learning: Regularization & Gradient Control

This guide details Dropout regularization scaling, L1/L2 weight decay equations, and gradient clipping methods.

---

## 1. Dropout & Inverted Dropout

Dropout prevents co-adaptation of features by randomly setting a fraction $p$ of hidden unit activations to $0.0$ at each step of training.

### Inverted Dropout Mechanics (Standard in PyTorch)
During training, if we drop nodes with probability $p$, the expected value of the activations in the next layer is scaled down by a factor of $(1 - p)$. 

To prevent having to modify activations during evaluation, **Inverted Dropout** scales the remaining active nodes *during training* by $\frac{1}{1 - p}$:

```python
# NumPy Simulation of Inverted Dropout for Layer l
p = 0.2  # Keep probability = 0.8
mask = (np.random.rand(*A_l.shape) >= p) / (1.0 - p)
A_l_regularized = A_l * mask
```

- **Training:** Active nodes are scaled up by $1.25\text{x}$ (for $p=0.2$) to compensate for dropped nodes.
- **Evaluation:** Dropout is disabled. The test pass is identical to a standard feedforward pass without scaling adjustments, keeping prediction latency low.

---

## 2. L1/L2 Regularization (Weight Decay)

Adding regularization penalties to the cost function forces weight values to remain small, preventing the model from fitting high-frequency noise.

### L2 Regularization (Weight Decay)
The regularized cost function is:
$$J_{\text{regularized}} = J + \frac{\lambda}{2m} \sum_{l=1}^L \|W^{[l]}\|_F^2$$
*Where $\|W\|_F^2 = \sum \sum w_{ij}^2$ is the squared Frobenius norm.*

During backpropagation, the gradient of the regularization term is added to the parameter update:
$$dW_{\text{regularized}}^{[l]} = dW^{[l]} + \frac{\lambda}{m} W^{[l]}$$

Substituting this into gradient descent:
$$W^{[l]} \leftarrow W^{[l]} - \alpha \left( dW^{[l]} + \frac{\lambda}{m} W^{[l]} \right)$$
$$W^{[l]} \leftarrow \left(1 - \frac{\alpha\lambda}{m}\right) W^{[l]} - \alpha \cdot dW^{[l]}$$

*Intuition:* In each step, the weight is multiplied by a shrinkage factor $\left(1 - \frac{\alpha\lambda}{m}\right)$ before subtracting the gradient, driving weights toward zero.

---

## 3. Gradient Clipping

When training deep networks (especially Recurrent Neural Networks), gradients can explode, causing weight adjustments to diverge. Gradient clipping restricts updates to a stable range.

### Value Clipping
Clips each element of the gradient vector independently to a range $[-c, c]$:
$$g_i \leftarrow \max\left(-c, \, \min(g_i, c)\right)$$

- **Drawback:** Altering individual elements changes the direction of the overall gradient vector in parameter space.

### Norm Clipping
Rescales the entire gradient vector if its $L_2$ norm exceeds a threshold $c$:
$$g \leftarrow g \cdot \frac{c}{\max(c, \, \|g\|_2)}$$

- **Advantage:** Preserves the direction of the gradient vector, scaling down only the magnitude of the step size to prevent unstable jumps.
