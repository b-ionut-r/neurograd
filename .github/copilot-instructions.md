# NeuroGrad Copilot Instructions

## Project Overview
NeuroGrad is a pure Python deep learning framework with automatic differentiation, built from scratch for educational purposes. It provides a PyTorch-like API with dual CPU/GPU backend support.

## Core Architecture

### Backend Abstraction Pattern
- **Critical**: `xp` alias in `__init__.py` points to NumPy (CPU) or CuPy (GPU)
- Device auto-detection sets global `ng.DEVICE` ('cpu' or 'cuda')
- All operations must use `xp` not `numpy` directly for cross-backend compatibility
- Example: `output_data = xp.matmul(a, b)` works on both CPU/GPU

### Autograd System (`tensor.py`)
- `Tensor` class wraps data with gradient tracking (`requires_grad=True`)
- Dynamic computational graph built during forward pass
- Backward propagation via `Function.backward()` chain
- Each operation creates new tensor with `grad_fn` pointing to operation

### Function-Based Operations (`functions/`)
All operations inherit from `Function` base class:
```python
class CustomOp(Function):
    def forward(self, x):
        self.saved_value = x  # Store for backward
        return result
    
    def backward(self, grad_output):
        return grad_output * derivative
```

### Module System (`nn/module.py`)
- PyTorch-like inheritance from `Module` base class
- Use `self.add_parameter(name, tensor)` for learnable parameters
- `parameters()` and `named_parameters()` for optimization
- Training/eval modes with `module.train()` and `module.eval()`

## Development Workflows

### Environment Setup
```bash
# Core development
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .[dev]

# GPU support (requires CUDA 12.x)
pip install -e .[gpu]

# Everything (GPU, visualization, examples)
pip install -e .[all]
```

### Testing Patterns
```bash
# Run all tests
pytest -q

# With coverage
pytest --cov=neurograd

# Gradient checking for new operations
python -c "from neurograd.utils.grad_check import gradient_check; assert gradient_check(model, X, y, loss_fn='MSE')"
```

### Code Quality Pipeline
```bash
# Format, lint, type-check in sequence
black . && flake8 neurograd && mypy neurograd
```

## Key Implementation Patterns

### Adding New Operations
1. Create class in appropriate `functions/` subdirectory
2. Inherit from `Function`, implement `forward()` and `backward()`
3. Add gradient checking in tests: `gradient_check(op, inputs, loss_fn)`
4. Import in `functions/__init__.py` for public API

### Mixed Precision Support (AMP)
- All operations automatically support AMP via `maybe_cast_tensor`
- FP16-unsafe operations (losses, normalizations) stay in FP32
- Use `with autocast():` context for automatic casting
- GradScaler prevents underflow: `scaler.scale(loss).backward()`

### Memory Management
- `flush()` function clears GPU memory pools
- Operations can implement `_memsave()` for memory-efficient backward pass
- Use small, deterministic examples in tests (fixed seeds)

### Device Handling
- Never hardcode 'cuda' checks - use `ng.DEVICE == 'cuda'`
- Operations work transparently on both backends via `xp` alias
- Move tensors between devices using `.to('cpu')` or `.to('cuda')`

## File Organization

### Core Implementation
- `tensor.py`: Main Tensor class and autograd engine
- `functions/base.py`: Abstract Function class for all operations
- `nn/module.py`: Base Module class with parameter management
- `amp/`: Mixed precision training (autocast, GradScaler)

### Operation Categories
- `functions/arithmetic.py`: Basic math (+, -, *, /, **)
- `functions/math.py`: Mathematical functions (log, exp, sin, sqrt)
- `functions/linalg.py`: Linear algebra (matmul, dot, transpose)
- `functions/conv.py`: Convolution and pooling operations
- `functions/reductions.py`: Aggregation (sum, mean, max) with axis support

### Utilities
- `utils/grad_check.py`: Numerical gradient verification
- `utils/device.py`: CPU/GPU device detection
- `utils/graph.py`: Computational graph visualization
- `utils/data.py`: Dataset and DataLoader classes

## Testing Guidelines

### Gradient Verification
Always test new operations with numerical gradient checking:
```python
def test_new_operation():
    model = SimpleModel()
    X = ng.Tensor(test_data, requires_grad=True)
    y = ng.Tensor(targets)
    assert gradient_check(model, X, y, loss_fn="MSE")
```

### Cross-Backend Testing
Test both CPU and GPU when applicable:
```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_operation(device):
    with ng.device(device):
        # Test implementation
```

### Memory and Performance
- Use small tensors (e.g., 2x3x4) for unit tests
- Set fixed random seeds for reproducible tests
- Profile memory usage in notebooks, not unit tests

## Common Patterns

### Standard Training Loop
```python
from neurograd.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # Automatic mixed precision
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
```

### Custom Layer Implementation
```python
class CustomLayer(ng.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_parameter('weight', ng.Tensor(ng.xp.random.randn(in_features, out_features), requires_grad=True))
        self.add_parameter('bias', ng.zeros(out_features, requires_grad=True))
    
    def forward(self, x):
        return ng.linear(x, self.weight, self.bias)
```

## Build and Distribution

### Package Building
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
.\upload_to_pypi.bat  # Windows
./upload_to_pypi.sh   # Linux/macOS
```

### Extras Configuration
- `[dev]`: Testing and development tools
- `[gpu]`: CuPy for CUDA acceleration
- `[visualization]`: Matplotlib for plotting
- `[all]`: Everything above