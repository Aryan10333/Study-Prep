# Interview Flashcards: Deep Learning Foundations

This document contains 15 core interview questions and answers, blending direct conceptual questions with scenario-based diagnostic questions to prepare you for senior deep learning rounds.

---

### Q1 [Direct - Vanishing Gradients]: How does the vanishing gradient problem manifest mathematically during backpropagation? Contrast how Sigmoid, Tanh, and ReLU activations influence gradient flow.
**Answer:**
- **Mathematical Manifestation:** The gradient of the cost function with respect to weights in early layers is computed using the chain rule:
  $$\frac{\partial J}{\partial W^{[1]}} = dZ^{[L]} \prod_{l=1}^{L-1} \left( W^{[l+1]T} \odot g^{[l]\prime}(Z^{[l]}) \right) X^T$$
  If the activation derivatives $g^{[l]\prime}$ are consistently less than $1.0$, multiplying these terms repeatedly across deep layers causes the gradient to decay exponentially ($dZ^{[1]} \to 0$), halting parameter updates in early layers.
- **Activation Influences:**
  - **Sigmoid:** Maximum derivative value is $0.25$ (at $z=0$). Saturates quickly as $|z| > 3$, driving $g^\prime(z) \to 0$ and accelerating vanishing gradients.
  - **Tanh:** Maximum derivative value is $1.0$ (at $z=0$). Also saturates at boundaries ($z > 3$ or $z < -3$), driving $g^\prime(z) \to 0$.
  - **ReLU:** Derivative is exactly $1.0$ for all positive inputs ($z > 0$). This constant gradient prevents vanishing gradients.

---

### Q2 [Scenario - The Saturation Cliff]: We deployed a deep feedforward network with Sigmoid activations to classify documents. During training, the loss stalled at a high value and the gradients of the first two layers dropped to exactly $10^{-7}$. Diagnose the root cause and propose a solution.
**Answer:**
- **Diagnosis:** The network has entered **activation saturation**. In early epochs, large weight initializations or rapid gradient steps pushed the pre-activations $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$ into extreme positive or negative ranges ($> 4.0$ or $< -4.0$). At these plateaus, the Sigmoid derivative $g^\prime(z) = g(z)(1 - g(z))$ becomes near-zero, freezing the weights.
- **Solution:** 
  1. Replace Sigmoid hidden activations with **ReLU** or **Leaky ReLU** which do not saturate for positive inputs.
  2. Implement proper **Xavier Initialization** to keep activation variances constant across layers, preventing early activations from exploding into saturation zones.
  3. Introduce **Batch Normalization** to center and scale activations before the Sigmoid functions, keeping them in the active, non-saturated gradient region.

---

### Q3 [Direct - Xavier vs. He Initialization]: Explain the difference between Xavier (Glorot) and He (Kaiming) initialization. What is the mathematical reason for doubling the weight variance for ReLU?
**Answer:**
- **Xavier Initialization:** Designed for linear or symmetric activations (Tanh/Sigmoid). To keep activation variance constant across layers ($\text{Var}(z^{[l]}) = \text{Var}(a^{[l-1]})$, it sets:
  $$\text{Var}(w) = \frac{1}{n_{\text{in}}} \quad \left(\text{or } \frac{2}{n_{\text{in}} + n_{\text{out}}} \right)$$
- **He Initialization:** Designed for non-symmetric ReLU activations ($\max(0, z)$). 
- **Mathematical Reason for Doubling:** ReLU zeroes out all negative inputs. For a symmetric distribution centered at zero, this cuts the variance of the activation signal in half: $\text{Var}(a^{[l-1]}) = \frac{1}{2} \text{Var}(z^{[l-1]})$. To compensate for this loss of signal variance, we must double the weight variance:
  $$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$

---

### Q4 [Scenario - The Dying ReLU Diagnosis]: After training a deep network with ReLU, you inspect the activations and find that 40% of the hidden units in layer 3 output exactly $0.0$ for all samples in the validation set. Explain this anomaly, how it occurred, and how you would modify the architecture to fix it.
**Answer:**
- **Anomaly (Dying ReLU):** 40% of the neurons in layer 3 are "dead".
- **How it occurred:** During training, large gradient updates adjusted the weights and biases of these units such that the pre-activation $z = Wx + b$ became negative for all samples in the dataset. Because the derivative of ReLU is exactly $0.0$ for $z < 0$, the gradient backpropagated through these units is exactly $0.0$, preventing them from updating and locking them in a dead state.
- **The Fix:**
  1. Change the activation function of hidden layers to **Leaky ReLU** ($\max(0.01z, z)$) or **ELU**. These functions have a non-zero derivative for negative inputs, allowing dead units to recover.
  2. Decrease the training **learning rate** to prevent large, destructive weight updates.
  3. Apply **Batch Normalization** to prevent pre-activations from shifting entirely into the negative region.

---

### Q5 [Direct - Adam Optimization]: How does the Adam optimizer combine Momentum and RMSprop? Why are first and second moment bias corrections mathematically necessary, especially in the first few epochs?
**Answer:**
- **Momentum + RMSprop:** Adam tracks the Exponential Moving Average (EMA) of gradients (first moment $v_t$, like Momentum) and the EMA of squared gradients (second moment $s_t$, like RMSprop):
  $$v_t = \beta_1 v_{t-1} + (1 - \beta_1) dw, \quad s_t = \beta_2 s_{t-1} + (1 - \beta_2) dw^2$$
- **Bias Correction Necessity:** In the first step ($t=1$), $v_0$ and $s_0$ are initialized to $0.0$. Thus:
  $$v_1 = (1 - \beta_1) dw$$
  If $\beta_1 = 0.9$, then $v_1 = 0.1 \cdot dw$. The update is artificially scaled down by **$10\text{x}$** simply because of the initialization choice.
- **The Correction:**
  $$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}$$
  At step 1, $\hat{v}_1 = v_1 / (1 - 0.9) = v_1 / 0.1 = dw$, removing the initialization bias. As $t \to \infty$, $\beta_1^t \to 0$, rendering the correction inactive.

---

### Q6 [Scenario - The Learning Rate Oscillation]: During training, your loss curve oscillates wildly. What parameter changes would you make if you are using Momentum vs. Adam to stabilize convergence?
**Answer:**
- **If using Momentum:**
  1. **Decrease the learning rate ($\alpha$):** The steps are too large, causing the parameters to overshoot the local minimum.
  2. **Increase momentum ($\beta_1$):** Increasing $\beta_1$ (e.g., from $0.9$ to $0.95$) averages gradients over a wider window, smoothing out alternating oscillatory gradients.
- **If using Adam:**
  1. **Decrease the learning rate ($\alpha$):** Since Adam scales step sizes automatically, oscillations are typically due to $\alpha$ being set too high.
  2. **Adjust $\beta_1$ and $\beta_2$:** Ensure $\beta_2$ (default $0.999$) is high enough to track gradient variance stably.

---

### Q7 [Direct - BatchNorm Dynamics]: Explain the mathematical formulas and training/inference dynamics of Batch Normalization. Why does behavior differ between `model.train()` and `model.eval()`?
**Answer:**
- **Dynamics:** BatchNorm normalizes activations across the mini-batch:
  $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
- **Why behavior differs:**
  - **`model.train()`:** BatchNorm uses the current mini-batch mean $\mu_{\mathcal{B}}$ and variance $\sigma_{\mathcal{B}}^2$. It also updates global running averages using an exponential moving average (EMA) scheme.
  - **`model.eval()`:** During evaluation, calculating batch-level statistics is impossible (e.g., during single-sample inference). BatchNorm freezes the scale ($\gamma$) and shift ($\beta$) parameters, and normalizes activations using the global running mean and variance accumulated during training. This ensures predictions are deterministic.

---

### Q8 [Scenario - The Batch Size Validation Drift]: You deploy a computer vision model with BatchNorm layers. In production, requests are processed individually (batch size = 1). The model predicts poorly on single samples compared to the validation set. What went wrong and how do you fix it?
**Answer:**
- **What went wrong:** During inference, the model was left in **training mode** (e.g., forgetting to call `model.eval()`). Consequently, the model attempted to calculate mean and variance on a batch size of 1. A single sample has a variance of $0.0$, making normalization evaluate to $0.0$ or divide by $\epsilon$, completely warping activation signals.
- **The Fix:** Ensure the model is set to evaluation mode (`model.eval()` in PyTorch) prior to inference. This forces the model to use the stable, global running statistics accumulated during training.

---

### Q9 [Direct - Inverted Dropout]: How does Inverted Dropout scale activations during training, and why is this preferred over standard scaling during inference?
**Answer:**
- **Inverted Dropout Scaling:** If nodes are dropped with probability $p$, the remaining active activations are scaled up by dividing by $(1-p)$ *during the forward pass of training*:
  $$A^{[l]} = \frac{A^{[l]} \odot M}{1 - p}$$
  *Where $M$ is a binary dropout mask.*
- **Why preferred:** Standard dropout requires no scaling during training, but requires multiplying all activations by $(1-p)$ during inference to match expected signal scales. Inverted dropout performs all scaling during training. This leaves the inference pass unmodified, avoiding extra arithmetic operations and minimizing deployment latency.

---

### Q10 [Scenario - The Dropout Evaluation Leak]: You noticed your model's validation loss fluctuates wildly, but training loss is stable. You realize you forgot to call `model.eval()` during validation. Explain how this affects prediction scaling and validation metrics.
**Answer:**
- **Effect on Scaling:** Without `model.eval()`, dropout remains active during validation. The network randomly zeroes out $p\%$ of features for validation samples. 
- **Effect on Metrics:** Because the validation pass is evaluated with missing feature signals, the predictions are noisy and inconsistent, inflating validation loss. 

---

### Q11 [Direct - BatchNorm vs. LayerNorm]: Contrast Batch Normalization and Layer Normalization. Why is Layer Normalization the standard for Recurrent Networks and Transformers?
**Answer:**
- **BatchNorm:** Normalizes activations down the mini-batch dimension (for each feature, across all samples).
- **LayerNorm:** Normalizes activations across features for each individual sample.
- **Why standard for RNNs/Transformers:**
  1. **Batch Size Independence:** RNN/Transformer batch sizes are often small or vary dynamically. LayerNorm does not rely on batch statistics, maintaining stability.
  2. **Sequence Length Robustness:** Sequence lengths vary from batch to batch. Calculating stable BatchNorm statistics across time steps with varying padding is difficult, whereas LayerNorm normalizes each token position independently.

---

### Q12 [Scenario - The Anomaly Focal Shift]: You are training a fraud detector on a dataset where only 0.05% of transactions are fraudulent. The binary cross-entropy loss is low, but the model fails to detect any fraud. Explain how Focal Loss shifts the optimization focus and write down its parameterized formula.
**Answer:**
- **How it shifts focus:** Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to BCE:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  For easy-to-classify samples (e.g., the $99.95\%$ normal transactions where predicted probability $p_t$ is high), $(1 - p_t)^\gamma$ decays to near-zero, scaling down their loss contribution. The model's loss gradient is dominated by hard, misclassified samples (the rare fraud cases).

---

### Q13 [Direct - Triplet Loss]: What is Triplet Loss? Explain the relationship between anchor, positive, and negative samples, and the role of the margin parameter.
**Answer:**
- **Triplet Loss:** A loss function used in metric learning to train deep networks to output semantic embeddings:
  $$L(a, p, n) = \max\left( 0, \, d(a, p) - d(a, n) + \text{margin} \right)$$
- **Components:**
  - **Anchor ($a$):** A baseline reference sample.
  - **Positive ($p$):** A sample belonging to the same class as the anchor.
  - **Negative ($n$):** A sample belonging to a different class.
- **Margin Parameter:** Sets the minimum required distance barrier. The negative embedding must be pushed farther away from the anchor than the positive embedding by at least this margin.

---

### Q14 [Direct - L2 Regularization vs. Weight Decay]: What is the mathematical difference between $L_2$ regularization and Weight Decay? Are they always equivalent?
**Answer:**
- **The difference:**
  - **$L_2$ Regularization:** Adds a penalty $\frac{\lambda}{2} \|W\|^2$ to the loss function. The gradient of this penalty is added to the gradient of the loss during optimization.
  - **Weight Decay:** Directly shrinks the weight parameter values by a factor in each step: $W \leftarrow (1 - \alpha\lambda)W$.
- **Equivalency:** They are mathematically equivalent *only* when using vanilla Stochastic Gradient Descent (SGD). For adaptive optimizers like Adam, they diverge. Under Adam, the $L_2$ gradient penalty is scaled by the running second moment $\sqrt{s_t}$, meaning weights with large gradients receive less regularization. To resolve this, **AdamW** decouples weight decay from the gradient steps, applying shrinkage directly.

---

### Q15 [Scenario - The Exploding RNN Gradient]: You are training an LSTM on long sequences. The training loss suddenly evaluates to `NaN` after epoch 4. Explain how gradient norm clipping prevents this divergence.
**Answer:**
- **Why it occurred:** During backpropagation through time (BPTT), long sequence dependencies multiply weight matrices repeatedly. If the eigenvalues of the weight matrices are $> 1.0$, the gradients explode, causing a massive parameter update step that overflows to `NaN`.
- **How Norm Clipping prevents it:** Norm clipping measures the $L_2$ norm of the joint gradient vector $\|g\|_2$. If this norm exceeds a threshold $c$, it rescales the entire gradient:
  $$g \leftarrow g \cdot \frac{c}{\|g\|_2}$$
  This preserves the direction of the gradient step but limits the step size, preventing the optimizer from making unstable updates.
