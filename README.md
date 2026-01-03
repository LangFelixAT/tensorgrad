# Tensorgrad - A Minimal Deep Learning Framework from Scratch (NumPy / CuPy)

This repository contains a **fully functional deep learning framework implemented from scratch**, inspired by PyTorch but deliberately minimal and explicit.  
The goal of the project is **not performance**, but **clarity, correctness, and understanding** of how modern deep learning systems actually work under the hood.

The framework includes:

- a custom **autograd engine**
- a **Tensor** class with backward graph construction
- neural network **modules and layers**
- **optimizers**, **learning rate schedulers**
- **data loading**, **training loops**
- **checkpointing**
- optional **GPU acceleration via CuPy**

All core functionality is implemented manually — without relying on PyTorch, TensorFlow, or JAX.

---

## Motivation

Modern deep learning frameworks abstract away a lot of complexity. While this is great for productivity, it can obscure:

- how automatic differentiation really works,
- how gradients flow through broadcasting, reductions, and indexing,
- how optimizers and schedulers interact with parameter state,
- how training pipelines are structured end-to-end.

This project was built to:

- deeply understand **autograd mechanics**
- design **clean, PyTorch-like APIs**
- explore **numerical stability and correctness**
- serve as a **learning and reference implementation**
- dive deeper beyond “using a framework”

---

## Inspiration

This project was strongly inspired by **Andrej Karpathy’s** [*micrograd*](https://github.com/karpathy/micrograd) project, which provides a beautifully minimal implementation of reverse-mode automatic differentiation.

While [*micrograd*](https://github.com/karpathy/micrograd) focuses on scalar-valued computation and pedagogical clarity, this framework extends similar core ideas to:
- tensor-valued operations,
- broadcasting and reductions,
- modular neural network layers,
- optimizers, schedulers, and training utilities,
- optional GPU acceleration via CuPy.

The goal here is not to reimplement [*micrograd*](https://github.com/karpathy/micrograd), but to build a more complete, tensor-based framework while preserving the same level of explicitness and conceptual transparency.

---

## Key Features

### Custom Autograd Engine
- Reverse-mode automatic differentiation
- Dynamic computation graphs
- Correct gradient handling for:
  - broadcasting
  - reductions (`sum`, `mean`, `max`, `logsumexp`)
  - indexing and slicing
  - matrix multiplication
- Explicit gradient accumulation semantics

### Tensor Class
- NumPy / CuPy backed
- Device-aware (`cpu` / `cuda`)
- Supports most common tensor operations
- PyTorch-inspired API (`backward`, `zero_grad`, `no_grad`, etc.)
- Clear and extensively documented internals

### Neural Network Modules (`nn.py`)
- `Module` base class (PyTorch-style)
- Layers:
  - `Linear`
  - `Conv2d`
  - `Embedding`
  - `BatchNorm1d`
  - `LayerNorm`
  - `Dropout`
- Activations:
  - `ReLU`
  - `Tanh`
- Containers:
  - `Sequential`
- Losses:
  - `MSELoss`
  - `CrossEntropyLoss`

### Optimizers
- `SGD` (momentum, dampening, Nesterov, weight decay)
- `Adam`
- `AdamW`
- Fully stateful and serializable

### Learning Rate Schedulers
- `StepLR`
- `MultiStepLR`
- `ExponentialLR`
- `CosineAnnealingLR`
- `ReduceLROnPlateau`
- `WarmupCosineLR`

### Data Utilities
- `Dataset` base class
- `TensorDataset`
- `DataLoader` with batching, shuffling, and backend-aware stacking

### Training Utilities
- `train_one_epoch`
- `evaluate`
- `fit`
- Proper handling of training / evaluation modes
- `no_grad` context for evaluation

### Checkpointing
- Save / load:
  - model parameters
  - optimizer state
  - scheduler state
  - epoch counter
- Pickle-based, simple and explicit

---

## Project Structure

    .
    ├── tensorgrad/
    │   ├── tensor.py           # Core Tensor + autograd engine
    │   ├── nn.py               # Neural network modules and layers
    │   ├── optim.py            # Optimizers (SGD, Adam, AdamW)
    │   ├── lr_scheduler.py     # Learning rate schedulers
    │   ├── data.py             # Dataset and DataLoader
    │   ├── training.py         # Training and evaluation loops
    │   └── checkpoint.py       # Save/load checkpoints
    └── README.md

---

## Example Usage

```python
from tensorgrad.tensor import Tensor
from tensorgrad.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from tensorgrad.optim import Adam
from tensorgrad.data import TensorDataset, DataLoader
from tensorgrad.training import fit

# Dummy data
x = Tensor.randn(1000, 20)
y = Tensor.randint(0, 5, (1000,))

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = Sequential(
    Linear(20, 64),
    ReLU(),
    Linear(64, 5),
)

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

fit(model, loader, loss_fn, optimizer, num_epochs=10)
```

---

## Design Philosophy

This framework is guided by a small set of explicit design constraints:

- **Explicit over implicit**  
  All operations and gradient flows are implemented directly, without hidden abstractions.

- **Correctness over speed**  
  Numerical correctness and well-defined semantics take precedence over performance optimizations.

- **Minimal but complete**  
  The framework aims to be small, yet sufficiently complete to train real neural networks end-to-end.

- **PyTorch-inspired, not PyTorch-dependent**  
  Familiar APIs are used where helpful, while all core functionality is implemented independently.

---

## Scope and Limitations

This project is intended as an educational and experimental framework rather than a
drop-in replacement for production-grade deep learning libraries. Performance,
distributed training, and large-scale deployment are explicitly out of scope.

---

## Possible Extensions

- CUDA kernel fusion  
- Autograd graph visualization  
- Mixed precision training  
- Additional layers (RNNs, Transformers)  
- JIT compilation  
- Distributed training  

---

## License

MIT License — feel free to explore, learn from, and build upon this project.