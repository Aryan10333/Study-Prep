# Week 3 Applied Assignment: Custom NumPy MLP, Adaptive Optimizers & PyTorch Productionization

## Business Context & Task Overview
Your objective is to build, regularize, and productionize a deep image classification system to recognize apparel categories from the **Kaggle Fashion MNIST Dataset**. The dataset consists of 60,000 training and 10,000 testing grayscale images of shape $28 \times 28$ across 10 apparel classes.

To build deep systems intuition, you will write a complete 3-layer MLP and a custom Adam optimizer **from scratch in NumPy** on Saturday. On Sunday, you will port the network to **PyTorch**, implement regularization, compile to ONNX runtime, and troubleshoot real-world production failures.

```text
Expected Workspace Layout:
d:/Study/Prep/
├── assignments/
│   └── week_3.md                  <-- This assignment
└── machine-learning-prep/
    └── deep-learning-foundations/
        ├── 01_forward_and_backward_propagation.md
        ├── ...
        ├── 01_mlp_backpropagation_from_scratch.ipynb
        ├── 04_optimizers_momentum_and_adam.ipynb
        └── 05_initialization_and_normalization.ipynb
```

---

## Saturday Sprint: Custom NumPy MLP & Adam from Scratch

### Objectives:
1. **Load and Format Fashion MNIST:** Flat-vectorize the $28 \times 28$ images to feature vectors of size $784$. Scale inputs to $[0.0, 1.0]$. Transpose data to shape **$(784, m)$** to strictly match **Andrew Ng's vectorized notation** ($n_x = 784$, columns = examples).
2. **Build the Custom Neural Network:** Implement a 3-layer MLP containing:
   - Input layer: $n_x = 784$ features.
   - Hidden layer 1: $n_1 = 128$ units, ReLU activation.
   - Hidden layer 2: $n_2 = 64$ units, ReLU activation.
   - Output layer: $n_3 = 10$ classes, Softmax activation.
3. **Weight Initialization:** Implement **He (Kaiming) Initialization** for weight matrices:
   $$W^{[l]} \sim \mathcal{N}\left( 0, \, \sqrt{\frac{2}{n_{l-1}}} \right)$$
4. **Implement Custom Adam Optimizer:** Implement the full Adam update formulas from scratch:
   - Track first moments $v$ and second moments $s$ for all weights and biases.
   - Apply bias corrections at step $t$:
     $$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$
   - Update parameter arrays:
     $$\theta \leftarrow \theta - \alpha \cdot \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$
5. **Run the training loop:** Train the NumPy model for 5 epochs using mini-batches of size $128$. Log training loss and accuracy at the end of each epoch.

---

## Sunday Sprint: PyTorch Refactor, Regularization Sweeps & Production serving

### Objectives:
1. **Port to PyTorch:** Recreate the 3-layer MLP using `torch.nn.Module`.
2. **Run Initialization Sweeps:**
   - Train the network using **Zero Initialization** and observe the gradient metrics.
   - Train using **He Initialization** and plot training loss convergence.
3. **Implement Regularization & Normalization:**
   - Add **Batch Normalization** (`nn.BatchNorm1d`) after linear layers and before ReLU activations.
   - Add **Dropout** (`nn.Dropout(p=0.3)`) to the hidden layers.
   - Sweep Dropout keep probabilities ($p = 0.1, 0.3, 0.5$) and compile a table showing validation accuracy tradeoffs.
4. **Production Deployment Compilation:**
   - Freeze the trained network parameters.
   - Compile the model to **ONNX runtime** format (`.onnx`).
   - Write a micro-benchmark comparing single-image inference latency (in microseconds) between the native PyTorch model and the compiled ONNX model.

---

## Deliverables & Submission Checklist
Provide a clean directory structure containing:
1. **`fashion_mnist_numpy_mlp.py`**: The NumPy MLP and custom Adam optimizer implementation.
2. **`fashion_mnist_pytorch_pipeline.py`**: The PyTorch model, normalization/dropout sweeps, and ONNX serialization script.
3. **`ONNX_benchmark_results.json`**: JSON log containing the latency comparison results.
4. **Written Responses:** Detailed answers to the 5 production interview scenarios below.

---

## Written Production Scenarios

### Scenario 1: The Zero Gradient Mystery
During the Sunday sprint, you initialize all MLP weights to exactly $0.0$ and train with SGD. The training loss does not change at all. Explain the mathematical reason behind this failure. What is the difference in behavior between the hidden layers and the output layer?

### Scenario 2: BatchNorm Batch Drift
Your deployed PyTorch classification model performs with $92\%$ accuracy on the validation split. However, in production, single incoming user images (batch size = 1) yield completely random class predictions. What coding step was missed before serving inputs to the model, and why did this break the normalization mathematics?

### Scenario 3: Adam vs. AdamW
You add $L_2$ weight regularization directly to the Adam optimizer's weight decay parameter (`weight_decay=0.01`). The validation loss climbs, and weights are not regularized correctly. Explain why standard Adam with $L_2$ weight decay diverges from standard SGD with $L_2$, and how **AdamW** resolves this issue.

### Scenario 4: RNN Normalization Decisions
Your team is expanding the apparel classifier to handle sequential clickstreams using an LSTM. The lead engineer suggests using BatchNorm. Why is **Layer Normalization** preferred over **Batch Normalization** for Recurrent Neural Networks (RNNs) and sequence Transformers?

### Scenario 5: Focal Loss Imbalance Adjustment
You notice the Fashion MNIST classes are heavily imbalanced: Class 0 (T-shirt/top) makes up only $0.1\%$ of the dataset, while other classes are balanced. Standard Cross-Entropy loss leads to a model that ignores Class 0. Write down the **Focal Loss** formula, explain how the parameters $\alpha$ and $\gamma$ shift the gradient updates, and how they prevent easy background classes from dominating optimization.
