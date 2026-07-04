# Deep Learning: Weight Initialization & Regularization

This guide details the mathematical foundations of weight initialization (Xavier/He), $L_2$ regularization (Weight Decay), Inverted Dropout, and Batch Normalization.

---

## 1. Weight Initialization and Variance Scaling

Improper weight initialization leads to vanishing or exploding gradients in deep networks. 

- If weights are initialized too small (e.g., standard deviation $0.01$ in a 10-layer net), activation variance decays to exactly $0.0$ at the output layer, causing gradients to vanish.
- If weights are initialized too large, activation variance explodes to infinity, causing gradients to overflow.

### Xavier (Glorot) Initialization
Designed for **Sigmoid** and **Tanh** activation functions. It scales the weight variance to match input and output distributions:
$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \quad \text{or} \quad W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)$$
- **Production Utility / Selection Rule:** Standard initialization for recurrent sequence layers (like LSTMs/GRUs) and autoencoders using tanh. Using Xavier with ReLU activations causes the signal variance to drop by $50\%$ at each layer, resulting in vanishing gradients in deep networks.

### He (Kaiming) Initialization
Designed for **ReLU** and **LeakyReLU** activations. Since ReLU discards half of its negative input space, He doubles the variance multiplier to maintain signal flow:
$$\text{Var}(W) = \frac{2}{n_{\text{in}}} \quad \text{or} \quad W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$
- **Production Utility / Selection Rule:** The default choice for all CNNs, MLPs, and deep vision encoders. Selecting He instead of Xavier is critical when using ReLU to prevent the model from failing to learn.

### Diagnostic Visual (Gradient Flow & Variance)
The plot below illustrates how standard small initialization causes variance to vanish immediately, while He initialization maintains a stable variance of $1.0$ across 10 deep layers:

![Initialization Variance](images/weight_gradient_flow_variance.png)

---

## 2. Regularization Math

### $L_2$ Regularization (Weight Decay)
$L_2$ regularization adds a quadratic penalty to the loss function to shrink weights toward zero, reducing model complexity:
$$L_{\text{reg}} = L + \frac{\lambda}{2m} \sum_{l=1}^L \sum_{i} \sum_{j} \left( W_{i,j}^{[l]} \right)^2$$

When taking a gradient step, this penalty alters the weight update formula:
$$\frac{\partial L_{\text{reg}}}{\partial W^{[l]}} = \frac{\partial L}{\partial W^{[l]}} + \frac{\lambda}{m} W^{[l]}$$
$$W^{[l]} \leftarrow W^{[l]} - \alpha \left( dW^{[l]} + \frac{\lambda}{m} W^{[l]} \right) = W^{[l]} \left( 1 - \frac{\alpha \lambda}{m} \right) - \alpha \cdot dW^{[l]}$$
- **Intuition:** Before applying the gradient update, the weights are multiplied by a decay factor $\left(1 - \frac{\alpha\lambda}{m}\right)$ which is slightly less than $1.0$, giving rise to the term **Weight Decay**.
- **Production Utility / AdamW Rule:** When using Adam, weight decay and gradient updates interact improperly, causing weight updates to be scaled incorrectly. In production, always use **`AdamW`** instead of standard `Adam` with `weight_decay`, as `AdamW` decouples the weight decay calculation from the moving averages of gradients.

---

### Inverted Dropout
Dropout randomly deactivates a fraction $1-p$ of activations during training to prevent neuron co-dependency. 

In production environments, we use **Inverted Dropout** to keep the forward pass scaling identical during training and evaluation:
1. **Mask Generation:** Generate a binary mask $D^{[l]}$ of shape $A^{[l]}$ where elements are $1$ with probability $p$, and $0$ otherwise.
2. **Deactivation:** Apply the mask to layer activations: $A^{[l]} = A^{[l]} \odot D^{[l]}$.
3. **Inverted Scaling:** Divide activations by $p$:
   $$A^{[l]} \leftarrow \frac{A^{[l]}}{p}$$
- **Intuition:** Dividing by $p$ scales the remaining active neurons back up to preserve the expected activation value. This eliminates the need to scale weights or activations during inference/evaluation, making evaluation deterministic.
- **Production Integration Rules:** 
  1. Only apply Dropout in fully connected dense layers. 
  2. **Never** place Dropout immediately before a Batch Normalization layer. Since Dropout randomly shifts the activation distribution, it warps BatchNorm's calculation of mean and variance, creating a massive training-serving skew that degrades validation performance.

---

## 3. Normalization Techniques

Normalization layers stabilize the distribution of activations (internal covariate shift) throughout training.

```text
Norm Type       Normalization Dimension                       Batch Size Sensitivity   Ideal Use Case
----------------------------------------------------------------------------------------------------------------------
BatchNorm       Across Batch (N) for each feature/channel     High (unstable for N < 4) Tabular, CNNs
LayerNorm       Across features/channels (C, H, W) per sample None                     Transformers, NLP, RNNs
InstanceNorm    Across spatial dimensions (H, W) per channel   None                     Style Transfer, GANs
GroupNorm       Across groups of channels per sample          None                     ConvNets with small batches
```

### A. Batch Normalization (BatchNorm)
Normalizes a single feature across all samples in a mini-batch. For a mini-batch $\mathcal{B}$:
1. **Compute Batch Mean:** $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$
2. **Compute Batch Variance:** $\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$
3. **Normalize:** $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$
4. **Scale and Shift:** $y_i = \gamma \hat{x}_i + \beta \quad \text{(where } \gamma, \beta \text{ are learnable parameters)}$

- **Training vs. Evaluation shifts:**
  - *Training:* Mean and variance are computed on the current mini-batch. Running averages ($\mu_{\text{running}}$, $\sigma^2_{\text{running}}$) are updated using exponentially weighted moving averages.
  - *Evaluation (`model.eval()`):* Mini-batch statistics are frozen. The model normalizes using the fixed running statistics, making predictions deterministic.
- **Production Selection Rules:** Use primarily in **tabular networks and CNNs** with large, stable batch sizes ($B \ge 16$). **Avoid** using in RNNs or Transformers because sequence lengths vary dynamically, causing batch statistics to be highly noisy. Also avoid in real-time streaming APIs where batch size is typically 1.

---

### B. Layer Normalization (LayerNorm)
Normalizes all features of a single sample independently. For a sample $x$ with $D$ features:
1. **Compute Sample Mean:** $\mu = \frac{1}{D} \sum_{j=1}^D x_j$
2. **Compute Sample Variance:** $\sigma^2 = \frac{1}{D} \sum_{j=1}^D (x_j - \mu)^2$
3. **Normalize:** $\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- **Serving Behavior:** Because LayerNorm does not rely on batch statistics, its behavior is identical during training and evaluation, making it robust for small batch sizes and dynamic sequence lengths.
- **Production Selection Rules:** The standard for **Transformers (LLMs) and RNNs**. Since LayerNorm computes stats per sample, it is fully compatible with batch sizes of 1 and sequence padding masks without causing training-serving skew.

---

### C. Group Normalization (GroupNorm) & Instance Normalization (InstanceNorm)
- **Group Normalization:** Divides features/channels into groups and normalizes within each group.
  - *Where it is helpful:* Used in **3D image segmentation, object detection, or high-resolution vision systems** where large image sizes force a batch size of $1$ or $2$ per GPU. It achieves the performance of BatchNorm without relying on batch dimensions.
- **Instance Normalization:** Normalizes each channel of each image sample independently.
  - *Where it is helpful:* Used in **style transfer and GANs**. It discards style contrast variances (brightness/contrast styles) while keeping content structures intact.

---

## 4. Gradient Clipping: Value vs. Norm Clipping

When gradients grow exponentially (exploding gradients), we constrain them before running `optimizer.step()`.

- **Gradient Value Clipping:** Truncates each gradient element independently to a range $[-c, c]$:
  $$g_i \leftarrow \max(\min(g_i, c), -c)$$
  - *Drawback:* Alters the direction of the gradient vector, potentially warping optimization paths.
- **Gradient Norm Clipping:** Scales the entire gradient vector $g$ if its $L_2$ norm exceeds a threshold $c$:
  $$g \leftarrow c \cdot \frac{g}{\max(\|g\|_2, c)}$$
  - *Advantage:* Preserves the direction of the gradient vector, only scaling down its magnitude.

---

## 5. Step-by-Step Hand Calculations (Andrew Ng Style)

### A. LayerNorm vs. BatchNorm
Given a tiny batch of $m = 2$ samples, each with $D = 3$ features:
$$X = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & 4.0 \\ 3.0 & 6.0 \end{bmatrix} \in \mathbb{R}^{3 \times 2} \quad \left( \text{where column } 1 \text{ is Sample 1, column } 2 \text{ is Sample 2} \right)$$

#### 1. Batch Normalization (Feature-wise, down columns):
- **For Feature 1 (Row 1: $[1.0, 2.0]$):**
  - $\mu_1 = \frac{1.0 + 2.0}{2} = 1.5$
  - $\sigma_1^2 = \frac{(1.0 - 1.5)^2 + (2.0 - 1.5)^2}{2} = 0.25 \implies \sigma_1 = 0.5$
  - $\hat{x}_{1,1} = \frac{1.0 - 1.5}{0.5} = -1.0, \quad \hat{x}_{1,2} = \frac{2.0 - 1.5}{0.5} = 1.0$
- **For Feature 2 (Row 2: $[2.0, 4.0]$):**
  - $\mu_2 = 3.0, \quad \sigma_2^2 = 1.0 \implies \sigma_2 = 1.0$
  - $\hat{x}_{2,1} = \frac{2.0 - 3.0}{1.0} = -1.0, \quad \hat{x}_{2,2} = \frac{4.0 - 3.0}{1.0} = 1.0$
- **For Feature 3 (Row 3: $[3.0, 6.0]$):**
  - $\mu_3 = 4.5, \quad \sigma_3^2 = 2.25 \implies \sigma_3 = 1.5$
  - $\hat{x}_{3,1} = \frac{3.0 - 4.5}{1.5} = -1.0, \quad \hat{x}_{3,2} = \frac{6.0 - 4.5}{1.5} = 1.0$

$$\hat{X}_{\text{BatchNorm}} = \begin{bmatrix} -1.0 & 1.0 \\ -1.0 & 1.0 \\ -1.0 & 1.0 \end{bmatrix}$$

#### 2. Layer Normalization (Sample-wise, down rows per column):
- **For Sample 1 (Column 1: $[1.0, 2.0, 3.0]^T$):**
  - $\mu^{(1)} = \frac{1.0 + 2.0 + 3.0}{3} = 2.0$
  - $\sigma^{(1)2} = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{2}{3} \approx 0.667 \implies \sigma^{(1)} \approx 0.8165$
  - $\hat{x}_{1,1} = \frac{1.0 - 2.0}{0.8165} \approx -1.2247, \quad \hat{x}_{2,1} = 0.0, \quad \hat{x}_{3,1} = \frac{3.0 - 2.0}{0.8165} \approx 1.2247$
- **For Sample 2 (Column 2: $[2.0, 4.0, 6.0]^T$):**
  - $\mu^{(2)} = \frac{2.0 + 4.0 + 6.0}{3} = 4.0$
  - $\sigma^{(2)2} = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{8}{3} \approx 2.667 \implies \sigma^{(2)} \approx 1.633$
  - $\hat{x}_{1,2} = \frac{2.0 - 4.0}{1.633} \approx -1.2247, \quad \hat{x}_{2,2} = 0.0, \quad \hat{x}_{3,2} = \frac{6.0 - 4.0}{1.633} \approx 1.2247$

$$\hat{X}_{\text{LayerNorm}} = \begin{bmatrix} -1.2247 & -1.2247 \\ 0.0 & 0.0 \\ 1.2247 & 1.2247 \end{bmatrix}$$

---

### B. Gradient Norm Clipping
Let's clip gradient vector $g = \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix}$ with max norm threshold $c = 2.5$:
1. **Calculate L2 Norm of Gradient ($L_2$):**
   $$\|g\|_2 = \sqrt{3.0^2 + 4.0^2} = \sqrt{9 + 16} = 5.0$$
2. **Apply Norm Clipping Scale:**
   Since $\|g\|_2 = 5.0 > c = 2.5$:
   $$g_{\text{clipped}} = c \cdot \frac{g}{\|g\|_2} = 2.5 \cdot \frac{\begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix}}{5.0} = 0.5 \cdot \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 2.0 \end{bmatrix}$$
3. **Compare Direction (Angle $\theta$):**
   - *Original Direction:* $\theta = \arctan(4.0 / 3.0) \approx 53.13^\circ$
   - *Clipped Direction:* $\theta = \arctan(2.0 / 1.5) \approx 53.13^\circ$
   - *Value Clipping Alternative* (Clipping elements individually to $[-2.5, 2.5]$):
     $$g_{\text{value\_clipped}} = \begin{bmatrix} 2.5 \\ 2.5 \end{bmatrix} \implies \theta = \arctan(2.5/2.5) = 45.00^\circ \quad \mathbf{\text{(Direction is altered!)}}$$

---

## 6. Production Scenario & Example

### Scenario: FastAPI Serving Crash due to BatchNorm Mode
You deploy a PyTorch Intent Routing model containing `nn.BatchNorm1d` layers inside a containerized FastAPI endpoint.
- **The Failure Mode:** During testing, you send single JSON requests (batch size of 1). The API server crashes, throwing runtime variance errors, or silently returns `NaN` intent classifications.
  - *Why?* By default, the model is in training mode (`model.train()`). When a batch size of $m=1$ is passed, BatchNorm's variance calculation evaluates to $\sigma_{\mathcal{B}}^2 = (x_i - x_i)^2 = 0.0$. Dividing by $\sqrt{0.0 + \epsilon}$ causes extreme numeric instability or triggers a runtime crash in PyTorch.
- **The Solution:** Inside the FastAPI request handler, you wrap model execution with `model.eval()`:
  ```python
  @app.post("/predict")
  def predict(request: UserRequest):
      # Convert input features to tensor of batch size 1
      x = torch.tensor(request.features).unsqueeze(0) 
      
      model.eval()  # Freeze BatchNorm, force it to use training's running statistics
      with torch.no_grad():
          logits = model(x)
      return {"intent": logits.argmax(dim=-1).item()}
  ```
  This ensures stable, deterministic inference independent of batch size.

