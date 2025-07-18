# 🚀 NeuroGrad

<div align="center">

![NeuroGrad Logo](https://img.shields.io/badge/NeuroGrad-Deep%20Learning%20Framework-blue?style=for-the-badge&logo=python)

**A Pure Python Deep Learning Framework with Automatic Differentiation**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Compatible-orange.svg)](https://numpy.org)
[![CuPy](https://img.shields.io/badge/CuPy-GPU%20Support-green.svg)](https://cupy.dev)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

*Built from scratch with minimal AI assistance - showcasing pure algorithmic understanding*

</div>

---

## 📖 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📚 Core Components](#-core-components)
- [🧮 Mathematical Operations](#-mathematical-operations)
- [🧠 Neural Networks](#-neural-networks)
- [📊 Visualization](#-visualization)
- [🔧 Advanced Usage](#-advanced-usage)
- [📈 Examples](#-examples)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Overview

**NeuroGrad** is a lightweight, educational deep learning framework built entirely from scratch in Python. It implements automatic differentiation (backpropagation) with a clean, intuitive API similar to PyTorch, making it perfect for:

- 🎓 **Learning**: Understanding how deep learning frameworks work under the hood
- 🔬 **Research**: Rapid prototyping of new algorithms and architectures  
- 📚 **Education**: Teaching automatic differentiation and neural network concepts
- 🛠️ **Experimentation**: Testing custom operations and gradient computations

> **Educational Foundation**: This framework was built following the principles and methodologies taught in **Andrew Ng's Deep Learning Specialization on Coursera**. The implementation closely follows the mathematical foundations and algorithmic approaches presented in those courses, providing a practical implementation of the theoretical concepts.

> **Development Note**: This framework was developed with minimal to no AI assistance for the core implementation, demonstrating pure algorithmic understanding and mathematical foundations. This README was created using Claude AI to ensure comprehensive documentation.

---

## ✨ Key Features

### 🔥 **Core Capabilities**

- **Automatic Differentiation**: Full reverse-mode autodiff with computational graph tracking
- **GPU Acceleration**: Seamless CPU/CUDA support via NumPy/CuPy backend switching
- **Dynamic Graphs**: Build and modify computational graphs on-the-fly
- **Broadcasting**: Full NumPy-compatible broadcasting for tensor operations
- **Memory Efficient**: Optimized gradient computation with cycle detection

### 🧮 **Rich Operation Set**

- **Arithmetic**: `+`, `-`, `*`, `/`, `**` with full gradient support
- **Mathematical**: `log`, `exp`, `sin`, `cos`, `tan`, `sqrt`, `abs`, and more
- **Linear Algebra**: Matrix multiplication, transpose, dot products
- **Reductions**: `sum`, `mean`, `max`, `min`, `std` with axis support
- **Activations**: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax

### 🧠 **Neural Network Components**

- **Layers**: Linear (Dense), MLP, with batch normalization and dropout
- **Activations**: Comprehensive activation function library
- **Loss Functions**: MSE, RMSE, MAE, Binary/Categorical Cross-Entropy
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Initializers**: Xavier/Glorot, He, Normal, Zero initialization

### 🛠️ **Developer Tools**

- **Graph Visualization**: Beautiful computational graph plotting
- **Gradient Checking**: Numerical gradient verification utilities
- **Debugging**: Comprehensive error messages and graph inspection
- **Modular Design**: Clean, extensible architecture

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/b-ionut-r/neurograd.git
cd neurograd

# Install dependencies
pip install numpy matplotlib cupy-cuda11x  # For GPU support
# OR
pip install numpy matplotlib  # CPU only
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
print(f"y.grad: {y.grad}")
```

### Neural Network Example

```python
from neurograd.nn.layers.linear import Linear, MLP
from neurograd.nn.losses import MSE
from neurograd.optim.adam import Adam

# Create a simple neural network
model = MLP([784, 128, 64, 10])  # Input -> Hidden -> Hidden -> Output

# Define loss and optimizer
criterion = MSE()
optimizer = Adam(model.named_parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(X_train)
    loss = criterion(y_train, output)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

---

## 📚 Core Components

### 🎯 **Tensor Class**

The `Tensor` class is the fundamental building block, wrapping NumPy/CuPy arrays with gradient tracking:

```python
# Creation
x = ng.Tensor([1, 2, 3], requires_grad=True)
y = ng.zeros((3, 3))

# Properties
print(x.shape)      # (3,)
print(x.device)     # 'cpu' or 'cuda'
print(x.requires_grad)  # True

# Operations (all differentiable)
z = x.exp().sum()
z.backward()
print(x.grad)       # Gradients computed automatically
```

### ⚡ **Function System**

All operations inherit from the `Function` base class:

```python
from neurograd.functions.base import Function

class CustomOperation(Function):
    def forward(self, x):
        return x ** 3
    
    def backward(self, grad_output):
        x = self.parent_tensors[0]
        return 3 * x.data ** 2 * grad_output
```

### 🏗️ **Module System**

Neural network components use the `ModuleMixin` for parameter management:

```python
from neurograd.nn.module import ModuleMixin
import numpy as np

class CustomLayer(ModuleMixin):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = ng.Tensor(np.random.randn(out_features, in_features), requires_grad=True)
        self.add_parameter("weight", self.weight)
    
    def forward(self, x):
        return self.weight @ x
```

---

## 🧮 Mathematical Operations

### Arithmetic Operations

```python
x = ng.Tensor([1, 2, 3], requires_grad=True)
y = ng.Tensor([4, 5, 6], requires_grad=True)

# Element-wise operations
z1 = x + y      # Addition
z2 = x - y      # Subtraction  
z3 = x * y      # Multiplication
z4 = x / y      # Division
z5 = x ** 2     # Power
```

### Mathematical Functions

```python
x = ng.Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Transcendental functions
y1 = x.log()    # Natural logarithm
y2 = x.exp()    # Exponential
y3 = x.sin()    # Sine
y4 = x.cos()    # Cosine
y5 = x.sqrt()   # Square root
y6 = x.abs()    # Absolute value
```

### Linear Algebra

```python
A = ng.Tensor([[1, 2], [3, 4]], requires_grad=True)
B = ng.Tensor([[5, 6], [7, 8]], requires_grad=True)

# Matrix operations
C = A @ B           # Matrix multiplication
D = A.transpose()   # Transpose
E = A.T             # Transpose (property)
```

### Reduction Operations

```python
x = ng.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

# Reductions with axis support
y1 = x.sum()                    # Sum all elements
y2 = x.sum(axis=0)             # Sum along axis 0
y3 = x.mean(axis=1, keepdims=True)  # Mean with dimension preservation
y4 = x.max()                   # Maximum value
y5 = x.std(ddof=1)            # Standard deviation
```

---

## 🧠 Neural Networks

### Layers

#### Linear Layer

```python
from neurograd.nn.layers.linear import Linear

# Dense/Fully-connected layer
layer = Linear(
    in_features=784, 
    out_features=128,
    activation="relu",           # Built-in activation
    dropout=0.2,                # Dropout rate
    batch_normalization=True,   # Batch normalization
    weights_initializer="he"    # He initialization
)

output = layer(input_tensor)
```

#### Multi-Layer Perceptron

```python
from neurograd.nn.layers.linear import MLP

# Quick MLP creation
model = MLP([784, 512, 256, 10])  # 4-layer network
output = model(input_tensor)
```

#### Sequential Container

```python
from neurograd.nn.module import Sequential
from neurograd.functions.activations import ReLU, Sigmoid

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128), 
    ReLU(),
    Linear(128, 10),
    Sigmoid()
)
```

### Activation Functions

```python
from neurograd.functions.activations import ReLU, Sigmoid, Tanh, LeakyReLU, Softmax

x = ng.Tensor([-2, -1, 0, 1, 2], requires_grad=True)

# Available activations
relu_out = ReLU()(x)                    # ReLU
sigmoid_out = Sigmoid()(x)              # Sigmoid  
tanh_out = Tanh()(x)                   # Hyperbolic tangent
leaky_relu_out = LeakyReLU(0.01)(x)    # Leaky ReLU
softmax_out = Softmax(axis=0)(x)       # Softmax
```

### Loss Functions

```python
from neurograd.nn.losses import MSE, RMSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy

y_true = ng.Tensor([[1, 0, 0], [0, 1, 0]])
y_pred = ng.Tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], requires_grad=True)

# Regression losses
mse_loss = MSE()(y_true, y_pred)
rmse_loss = RMSE()(y_true, y_pred)  
mae_loss = MAE()(y_true, y_pred)

# Classification losses
bce_loss = BinaryCrossEntropy()(y_true, y_pred)
cce_loss = CategoricalCrossEntropy()(y_true, y_pred)
```

### Optimizers

```python
from neurograd.optim import SGD, Adam, RMSprop

# SGD with momentum
optimizer = SGD(
    model.named_parameters(), 
    lr=0.01, 
    beta=0.9,           # Momentum
    weight_decay=1e-4   # L2 regularization
)

# Adam optimizer
optimizer = Adam(
    model.named_parameters(),
    lr=0.001,
    beta1=0.9,          # First moment decay
    beta2=0.999,        # Second moment decay
    epsilon=1e-8        # Numerical stability
)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 📊 Visualization

### Computational Graph Visualization

```python
# Create a computation
x = ng.Tensor([1, 2], requires_grad=True, name="input")
y = x.exp().sum()

# Visualize the computational graph
fig = y.visualize_graph(title="Exponential Sum Graph")
y.save_graph("computation_graph.png")

# Print text representation
y.print_graph()
```

### Graph Statistics

```python
stats = y.graph_stats()
print(f"Nodes: {stats['num_tensors']}")
print(f"Operations: {stats['num_functions']}")
print(f"Depth: {stats['max_depth']}")
```

---

## 🔧 Advanced Usage

### Custom Functions

```python
from neurograd.functions.base import Function

class Swish(Function):
    """Swish activation: x * sigmoid(x)"""
    
    def forward(self, x):
        self.sigmoid_x = 1 / (1 + ng.xp.exp(-x))
        return x * self.sigmoid_x
    
    def backward(self, grad_output):
        x = self.parent_tensors[0]
        swish_grad = self.sigmoid_x * (1 + x.data * (1 - self.sigmoid_x))
        return grad_output * swish_grad if x.requires_grad else None

# Usage
swish = Swish()
output = swish(input_tensor)
```

### Gradient Checking

```python
from neurograd.utils.grad_check import gradient_check
from neurograd.nn.losses import MSE

# Verify gradients numerically
model = Linear(10, 5)
X = ng.Tensor(ng.randn(10, 32), requires_grad=True)
y = ng.Tensor(ng.randn(5, 32))
loss_fn = MSE()

is_correct = gradient_check(model, X, y, loss_fn, epsilon=1e-7)
print(f"Gradients correct: {is_correct}")
```

### Device Management

```python
# Automatic device detection
print(f"Using device: {ng.DEVICE}")

# Manual tensor device checking
x = ng.Tensor([1, 2, 3])
print(f"Tensor device: {x.device}")

# Type casting
x = x.cast(ng.float64)
```

---

## 📈 Examples

### Complete Training Example

```python
import neurograd as ng
from neurograd.nn.layers.linear import MLP
from neurograd.nn.losses import MSE
from neurograd.optim.adam import Adam
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensors (note: shape is [features, samples])
X_tensor = ng.Tensor(X.T, requires_grad=True)
y_tensor = ng.Tensor(y.reshape(1, -1))

# Create model
model = MLP([20, 64, 32, 1])
criterion = MSE()
optimizer = Adam(model.named_parameters(), lr=0.001)

# Training loop
losses = []
for epoch in range(200):
    # Forward pass
    output = model(X_tensor)
    loss = criterion(y_tensor, output)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.data.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.6f}")

print("Training completed!")
```

### Custom Layer Example

```python
class BatchNormLinear(ng.nn.ModuleMixin):
    """Linear layer with built-in batch normalization"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features, use_bias=False)
        self.gamma = ng.Tensor(ng.ones(out_features, 1), requires_grad=True)
        self.beta = ng.Tensor(ng.zeros(out_features, 1), requires_grad=True)
        
        self.add_parameter("gamma", self.gamma)
        self.add_parameter("beta", self.beta)
        self.add_module("linear", self.linear)
    
    def forward(self, x):
        # Linear transformation
        out = self.linear(x)
        
        # Batch normalization
        mean = out.mean(axis=1, keepdims=True)
        var = ((out - mean) ** 2).mean(axis=1, keepdims=True)
        out_norm = (out - mean) / (var + 1e-8).sqrt()
        
        # Scale and shift
        return self.gamma * out_norm + self.beta
```

---

## 🧪 Testing and Validation

The framework includes comprehensive tests to ensure correctness:

```python
# Run gradient checking on your models
from neurograd.utils.grad_check import GradientChecker
from neurograd.nn.losses import MSE

checker = GradientChecker(epsilon=1e-7)
loss_fn = MSE()
is_valid = checker.check(your_model, X_test, y_test, loss_fn)

# Visualize gradients
loss.backward()
for name, param in model.named_parameters():
    print(f"{name}: grad_norm = {param.grad.std():.6f}")
```

---

## 🏗️ Architecture

### Framework Structure

```
neurograd/
├── tensor.py              # Core Tensor class
├── functions/             # Mathematical operations
│   ├── base.py           # Function base class
│   ├── arithmetic.py     # +, -, *, /, **
│   ├── math.py          # log, exp, sin, cos, etc.
│   ├── linalg.py        # Matrix operations
│   ├── activations.py   # Neural network activations
│   └── reductions.py    # sum, mean, max, etc.
├── nn/                   # Neural network components
│   ├── module.py        # Base module system
│   ├── layers/          # Network layers
│   ├── losses.py        # Loss functions
│   ├── activations.py   # Activation layers
│   └── initializers.py  # Weight initialization
├── optim/               # Optimization algorithms
│   ├── optimizer.py     # Base optimizer
│   ├── sgd.py          # SGD with momentum
│   ├── adam.py         # Adam optimizer
│   └── rmsprop.py      # RMSprop optimizer
└── utils/               # Utilities
    ├── device.py        # Device management
    ├── grad_check.py    # Gradient verification
    └── graph.py         # Graph visualization
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 **Bug Reports**

- Use GitHub Issues to report bugs
- Include minimal reproduction code
- Specify your environment (Python version, OS, etc.)

### 💡 **Feature Requests**

- Propose new operations or layers
- Discuss API design in issues first
- Consider backward compatibility

### 🔧 **Development Setup**

```bash
git clone https://github.com/b-ionut-r/neurograd.git
cd neurograd

# Install development dependencies
pip install -e .
pip install pytest matplotlib

# Run the comprehensive test notebook
jupyter notebook comprehensive_framework_test.ipynb
```

### 📝 **Code Style**

- Follow PEP 8 guidelines
- Add docstrings to all public functions
- Include type hints where appropriate
- Write tests for new functionality

---

## 🎯 Roadmap

### 🚀 **Upcoming Features**

- [ ] Convolutional layers (Conv1D, Conv2D)
- [ ] Recurrent layers (RNN, LSTM, GRU)
- [ ] Advanced optimizers (AdaGrad, Nadam)
- [ ] Model serialization/loading
- [ ] Mixed precision training
- [ ] Distributed training support

### 🔬 **Research Features**

- [ ] Automatic mixed precision
- [ ] Dynamic quantization
- [ ] Pruning utilities
- [ ] Neural architecture search tools

---

## 📚 Educational Resources

### 🎓 **Learning Materials**

This framework implements concepts from:
- **Andrew Ng's Deep Learning Specialization** on Coursera
- Mathematical foundations of automatic differentiation
- Neural network architectures and training algorithms

### 📖 **Tutorials**

Check the `comprehensive_framework_test.ipynb` notebook for extensive examples and testing of all framework components.

---

## ❓ FAQ

**Q: How does NeuroGrad compare to PyTorch/TensorFlow?**
A: NeuroGrad is educational-focused and much smaller. It's perfect for learning but not production-ready for large-scale applications.

**Q: Can I use this for research?**
A: Absolutely! It's great for prototyping new ideas and understanding algorithmic details.

**Q: Is GPU support required?**
A: No, NeuroGrad works perfectly on CPU-only systems using NumPy.

**Q: How accurate are the gradients?**
A: Very accurate! All gradients are mathematically verified and include numerical gradient checking utilities.

---

## 🙏 Acknowledgments

- **Andrew Ng**: For the exceptional Deep Learning Specialization on Coursera that provided the theoretical foundation for this implementation
- **NumPy/CuPy**: For providing the computational backend
- **Matplotlib**: For graph visualization capabilities  
- **PyTorch**: For API design inspiration
- **Deep Learning Community**: For the wealth of educational resources

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/b-ionut-r/neurograd/issues)
- **Discussions**: [Join the community discussion](https://github.com/b-ionut-r/neurograd/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

*Built with ❤️ for the deep learning community*

</div>
