# PyTorch: Production Data Pipelines & Troubleshooting

This guide details how to structure PyTorch Dataset and DataLoader pipelines for high-throughput GPU training, and covers diagnostics for common deep learning training bottlenecks.

---

## 1. PyTorch Dataset & DataLoader Architecture

For production pipelines, we separate data storage from training iteration logic using PyTorch's `Dataset` and `DataLoader` APIs:

- **`Dataset`:** Stores samples and labels, defining how a single sample is loaded and processed via the `__getitem__` method.
- **`DataLoader`:** Wraps the dataset and provides batching, shuffling, multi-process loading, and memory pinning.

### High-Throughput DataLoader Hyperparameters
- **`num_workers`:** Specifies the number of child processes to use for data loading.
  - *Production Utility / Bottleneck:* If `num_workers = 0`, data is loaded in the main execution thread, blocking training step calculations while waiting on disk reads or data preprocessing. Spawning $4 \times \text{num\_gpus}$ processes pre-fetches batches into a queue, keeping the GPU constantly supplied with training data.
- **`pin_memory` (Set `True`):** Puts data tensors in page-locked host memory.
  - *Production Utility:* Enables fast, direct memory access (DMA) transfers from CPU host memory to GPU device memory, bypassing the need for the CPU to copy the data from virtual memory pages, maximizing PCIe bandwidth usage.
- **`non_blocking=True` (Used during `.to(device)`):**
  - *Production Utility:* Works alongside `pin_memory=True` to run the CPU-to-GPU copy asynchronously, allowing CPU execution (such as launching CUDA kernels) to continue without blocking, overlapping host-device transfers with active GPU tensor computations.

---

## 2. PyTorch Pipeline: Code Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Custom Dataset Definition
class UserEngagementDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 2. DataLoader Pipeline Instantiation
# Set pin_memory=True to lock tensors in page-locked CPU memory for fast GPU transfer
train_dataset = UserEngagementDataset(features_train, targets_train)
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

# 3. Model Training Pass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for batch_x, batch_y in train_loader:
    # Send tensors to GPU. non_blocking=True speeds up pin_memory transfers
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
```

---

## 3. Production Diagnostics & Troubleshooting

### Problem A: CPU-GPU Data Transfer Bottlenecks
- **Symptom:** GPU utilization fluctuates wildly and drops to $0\%$ periodically while CPU utilization is pinned at $100\%$. The GPU is waiting for the CPU to load and process the next batch.
- **Remedy:** Set `pin_memory=True`, increase `num_workers` to match CPU cores, and set `non_blocking=True` during `.to(device)` transfers to overlapping batch loading with GPU computation.

### Problem B: Exploding Gradients & Loss
- **Symptom:** Loss evaluates to `NaN` or gradients grow exponentially, causing numerical overflow.
- **Remedy:** 
  1. Reduce the `learning_rate` (which controls update step sizes).
  2. Implement **Gradient Clipping** to truncate gradients whose norm exceeds a maximum threshold before running `optimizer.step()`:
     ```python
     loss.backward()
     # Truncate gradient norms at 1.0 to prevent exploding steps
     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     optimizer.step()
     ```

### Problem C: Learning Rate Decay Schedules
To help the optimizer settle into the global minimum during late-stage training, decrease the learning rate over time using schedulers:
- **`StepLR`:** Scales learning rate by a decay factor $\gamma$ after a fixed number of epochs:
  ```python
  from torch.optim.lr_scheduler import StepLR
  scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # Decays LR by 10x every 10 epochs
  ```
  - *Production Utility:* Best for stable training scenarios (e.g., computer vision model pipelines) where the convergence rate is predictable, and you want a simple, deterministic decay schedule.
- **`ReduceLROnPlateau`:** Dynamically drops the learning rate when validation loss stops improving:
  ```python
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
  # Run scheduler.step(val_loss) in evaluation loops
  ```
  - *Production Utility:* Standard for LLMs, custom transformers, or multi-modal models where training trajectories are erratic and hard to predict. It dynamically adapts the learning rate to the model's actual progress, preventing training from stalling.

---

## 4. Interactive Practice Notebook
To run a complete PyTorch pipeline, configure datasets, and benchmark device timing profiles, check out:
- [02_pytorch_training_pipeline_and_profiler.ipynb](file:///d:/Study/Prep/machine-learning-prep/deep-learning-foundations/02_pytorch_training_pipeline_and_profiler.ipynb)
