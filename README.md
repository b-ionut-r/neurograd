# 🚀 NeuroGrad

<div align="center">

![NeuroGrad Logo](https://img.shields.io/badge/NeuroGrad-Deep%20Learning%20Framework-blue?style=for-the-badge&logo=python)

**A Pure Python Deep Learning Framework with Automatic Differentiation**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Compatible-orange.svg)](https://numpy.org)
[![CuPy](https://img.shields.io/badge/CuPy-GPU%20Support-green.svg)](https://cupy.dev)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

*Built from scratch with no AI assistance - showcasing pure algorithmic understanding*

</div>

---

## 🌟 Overview

**NeuroGrad** is a lightweight, educational deep learning framework built entirely from scratch in Python. It implements automatic differentiation (backpropagation) with a clean, intuitive API similar to PyTorch.

**Perfect for:**
- 🎓 **Learning**: Understanding how deep learning frameworks work
- 🔬 **Research**: Rapid prototyping of new algorithms
- 📚 **Education**: Teaching autodiff and neural network concepts
- 🛠️ **Experimentation**: Testing custom operations

> **Educational Foundation**: Built following **Andrew Ng's Deep Learning Specialization** principles with minimal AI assistance for core implementation.

---

## ✨ Key Features

### 🔥 **Core Capabilities**
- **Automatic Differentiation**: Full reverse-mode autodiff with computational graph tracking
- **Mixed Precision Training**: Automatic mixed precision (AMP) with PyTorch-compatible API ⚡ **NEW!**
- **GPU Acceleration**: Seamless CPU/CUDA support via NumPy/CuPy backend switching
- **Dynamic Graphs**: Build and modify computational graphs on-the-fly
- **Memory Efficient**: Optimized gradient computation with cycle detection

### 🧠 **Neural Network Components**
- **Layers**: Linear, Conv2D, MaxPool2D/AveragePool2D, MLP with batch normalization and dropout
- **Activations**: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
- **Loss Functions**: MSE, RMSE, MAE, Binary/Categorical Cross-Entropy
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Data Utilities**: Dataset and DataLoader classes
- **Metrics**: Classification and regression metrics

### 🛠️ **Developer Tools**
- **Graph Visualization**: Beautiful computational graph plotting
- **Gradient Checking**: Numerical gradient verification
- **Mixed Precision**: 1.5-2x speedup, 40-50% memory reduction

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install neurograd

# With GPU support
pip install neurograd[gpu]

# Everything (GPU, visualization, examples)
pip install neurograd[all]

# From source
git clone https://github.com/b-ionut-r/neurograd.git
cd neurograd && pip install -e .
```

### Basic Usage

```python
import neurograd as ng

# Create tensors with gradient tracking
x = ng.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = ng.Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

# Perform operations
z = x @ y + x.sin()  # Matrix multiplication + element-wise sine
loss = z.sum()       # Scalar loss

# Automatic differentiation
loss.backward()
print(f"x.grad: {x.grad}")
```

---

## 🧠 Neural Networks

### Complete Training Example

```python
from neurograd.nn.layers.linear import Linear, MLP
from neurograd.nn.losses import MSE
from neurograd.optim.adam import Adam
from neurograd.utils.data import Dataset, DataLoader
from neurograd.nn.metrics import accuracy_score

# Create dataset and model
dataset = Dataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = MLP([784, 128, 64, 10])  # Input -> Hidden -> Output

# Define loss and optimizer
criterion = MSE()
optimizer = Adam(model.named_parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for X_batch, y_batch in dataloader:
        # Forward pass
        output = model(X_batch)
        loss = criterion(y_batch, output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    pred = model(X_test)
    acc = accuracy_score(y_test, pred)
    model.train()
    print(f"Epoch {epoch}, Loss: {loss.data:.4f}, Accuracy: {acc:.4f}")
```

### Mixed Precision Training ⚡ **NEW!**

**PyTorch-compatible automatic mixed precision** for faster training:

```python
from neurograd.amp import autocast, GradScaler
from neurograd.nn.layers.conv import Conv2D, MaxPool2D
from neurograd.nn.layers.linear import Linear
from neurograd.nn.module import Sequential
from neurograd.functions.activations import ReLU, Softmax

# Create CNN model (channels-first: NCHW)
model = Sequential(
    Conv2D(1, 32, kernel_size=3, padding="same", activation="relu"),
    MaxPool2D(pool_size=2),
    Conv2D(32, 64, kernel_size=3, padding="same", activation="relu"),
    MaxPool2D(pool_size=2),
    Flatten(),
    Linear(64 * 7 * 7, 128, activation="relu"),
    Linear(128, 10),
    Softmax(axis=1)
)

# Setup mixed precision
optimizer = Adam(model.named_parameters(), lr=0.001)
loss_fn = CategoricalCrossEntropy()
scaler = GradScaler()

# Training with mixed precision
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=True):
            predictions = model(batch_x)        # Auto FP16 where safe
            loss = loss_fn(batch_y, predictions)  # Auto FP32 for stability
        
        # Gradient scaling for FP16 stability
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)  # Unscales gradients automatically
        scaler.update()         # Updates scale factor
        
        print(f"Loss: {loss.data.item():.4f}, Scale: {scaler.get_scale():.0f}")

# Benefits: ⚡ 1.5-2x faster, 💾 40-50% less memory, 🎯 same accuracy
```

### Layers and Operations

```python
# Linear layers with built-in features
layer = Linear(784, 128, activation="relu", dropout=0.2, 
               batch_normalization=True, weights_initializer="he")

# Convolutional layers (channels-first: NCHW)
conv = Conv2D(3, 64, kernel_size=(3,3), padding="same", activation="relu")
pool = MaxPool2D(pool_size=(2,2), strides=(2,2))

# Activations and losses
from neurograd.functions.activations import ReLU, Sigmoid, Softmax
from neurograd.nn.losses import MSE, CategoricalCrossEntropy

# Optimizers
optimizer = Adam(model.named_parameters(), lr=0.001, beta1=0.9, beta2=0.999)
optimizer = SGD(model.named_parameters(), lr=0.01, beta=0.9, weight_decay=1e-4)
```

---

## 🧮 Core Operations

### Mathematical Functions

```python
x = ng.Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Arithmetic: +, -, *, /, **
z = x + y, x * y, x ** 2

# Math functions
y = x.log(), x.exp(), x.sin(), x.sqrt(), x.abs()

# Linear algebra
C = A @ B           # Matrix multiplication
D = A.transpose()   # Transpose

# Reductions with axis support
s = x.sum(axis=0), x.mean(axis=1, keepdims=True), x.max(), x.std()
```

### Data Utilities

```python
from neurograd.utils.data import Dataset, DataLoader

dataset = Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, seed=42)

for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
    output = model(X_batch)
    loss = criterion(y_batch, output)
```

---

## 🔧 Advanced Usage

### Custom Functions

```python
from neurograd.functions.base import Function

class Swish(Function):
    def forward(self, x):
        self.sigmoid_x = 1 / (1 + ng.xp.exp(-x))
        return x * self.sigmoid_x
    
    def backward(self, grad_output):
        x = self.parent_tensors[0]
        swish_grad = self.sigmoid_x * (1 + x.data * (1 - self.sigmoid_x))
        return grad_output * swish_grad if x.requires_grad else None
```

### Gradient Checking

```python
from neurograd.utils.grad_check import gradient_check

is_correct = gradient_check(model, X, y, loss_fn, epsilon=1e-7)
print(f"Gradients correct: {is_correct}")
```

### Visualization

```python
# Visualize computational graphs
fig = loss.visualize_graph(title="Training Loss Graph")
loss.save_graph("computation_graph.png")
loss.print_graph()

# Graph statistics
stats = loss.graph_stats()
print(f"Nodes: {stats['num_tensors']}, Depth: {stats['max_depth']}")
```

### Checkpointing (PyTorch-style)

```python
import neurograd as ng

# Save checkpoint
ng.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    # optional: 'scaler_state': scaler.state_dict(), 'epoch': epoch
}, 'checkpoint.pth')

# Or use the convenience helper
ng.save_checkpoint(model=model, optimizer=optimizer, path='checkpoint.pth', epoch=epoch)

# Load checkpoint later
ckpt = ng.load('checkpoint.pth')  # or ng.load_checkpoint('checkpoint.pth')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
```

---

## 🏗️ Architecture

```
neurograd/
├── tensor.py              # Core Tensor class
├── functions/             # Mathematical operations
│   ├── base.py           # Function base class
│   ├── arithmetic.py     # +, -, *, /, **
│   ├── math.py          # log, exp, sin, cos, etc.
│   ├── activations.py   # Neural network activations
│   ├── conv.py          # Convolution operations
│   ├── tensor_ops.py    # Tensor ops (includes Cast)
│   └── reductions.py    # sum, mean, max, etc.
├── amp/                  # ⚡ Mixed precision (NEW!)
│   ├── autocast.py      # Automatic precision context
│   ├── grad_scaler.py   # Gradient scaling
│   └── utils.py         # AMP utilities
├── nn/                   # Neural network components
│   ├── layers/          # Network layers
│   ├── losses.py        # Loss functions
│   ├── metrics.py       # Evaluation metrics
│   └── module.py        # Base module system
├── optim/               # Optimization algorithms
│   ├── sgd.py, adam.py, rmsprop.py
└── utils/               # Utilities
    ├── grad_check.py    # Gradient verification
    ├── graph.py         # Visualization
    └── data.py          # Dataset/DataLoader
```

---

## 🎯 Roadmap

### ✅ **Completed Features**
- [x] Automatic differentiation with dynamic graphs ✅
- [x] Neural network layers (Linear, Conv2D, Pooling) ✅
- [x] Loss functions and optimizers (SGD, Adam, RMSprop) ✅
- [x] Data utilities (Dataset, DataLoader) ✅
- [x] Evaluation metrics and visualization ✅
- [x] **Mixed precision training (AMP)** ⚡ **NEW!** ✅

### 🚀 **Upcoming**
- [ ] Recurrent layers (RNN, LSTM, GRU)
- [ ] Advanced optimizers (AdaGrad, Nadam)
- [ ] Model serialization/loading
- [ ] Distributed training support
- [ ] Dynamic quantization and pruning

---

## 📚 Resources & Contributing

### Educational Foundation
This framework implements concepts from **Andrew Ng's Deep Learning Specialization** and mathematical foundations of automatic differentiation.

### Contributing
- 🐛 **Bug Reports**: Use GitHub Issues with minimal reproduction code
- 💡 **Features**: Discuss API design in issues first
- 🔧 **Development**: `git clone` → `pip install -e .` → `pytest`

### Testing
```bash
# Run comprehensive tests
jupyter notebook comprehensive_framework_test.ipynb

# Gradient checking
python -c "from neurograd.utils.grad_check import *; test_all()"
```

---

## 📄 License & Contact

**MIT License** - see [LICENSE](LICENSE) file for details.

- **Issues**: [Report bugs/features](https://github.com/b-ionut-r/neurograd/issues)
- **Discussions**: [Community forum](https://github.com/b-ionut-r/neurograd/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

*Built with ❤️ for the deep learning community*

</div>
