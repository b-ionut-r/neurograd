# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test & Development Commands

### Environment Setup
```bash
# Create virtual environment and install in development mode
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]

# For GPU support (optional)
pip install -e .[gpu]      # Requires CUDA 12.x

# For all features (GPU, visualization, notebooks, dev tools)
pip install -e .[all]
```

### Testing
```bash
# Run all tests
pytest -q

# Run tests with coverage
pytest --cov=neurograd

# Run specific test file
pytest tests/test_tensor.py -v

# Gradient checking utility for new operations
python -c "from neurograd.utils.grad_check import gradient_check; print('Testing gradients...')"
```

### Code Quality
```bash
# Format code
black .

# Lint code  
flake8 neurograd

# Type checking
mypy neurograd

# Run all quality checks
black . && flake8 neurograd && mypy neurograd
```

### Building & Distribution
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (uses provided scripts)
./upload_to_pypi.sh        # Linux/macOS
./upload_to_pypi.bat       # Windows
```

## High-Level Architecture

### Core Components

**Tensor & Autograd System (`tensor.py`)**
- `Tensor`: Core data structure with automatic differentiation
- Supports CPU (NumPy) and GPU (CuPy) backends via `xp` alias
- Dynamic computational graph construction for backpropagation
- Device auto-detection: `ng.DEVICE` shows 'cpu' or 'cuda'

**Functions Layer (`functions/`)**
- `base.py`: Abstract `Function` class for all operations
- `arithmetic.py`: Basic ops (+, -, *, /, **)
- `math.py`: Mathematical functions (log, exp, sin, sqrt, etc.)
- `linalg.py`: Linear algebra (matmul, transpose, dot)
- `reductions.py`: Aggregation ops (sum, mean, max, std) with axis support
- `conv.py`: Convolution and pooling operations
- `tensor_ops.py`: Shape manipulation (reshape, flatten, cast, pad)

**Neural Networks (`nn/`)**
- `module.py`: Base `Module` class with parameter management
- `layers/`: Network layers (Linear, Conv2D, BatchNorm, Dropout)
- `losses.py`: Loss functions (MSE, Cross-entropy)
- `metrics.py`: Evaluation metrics (accuracy, F1, etc.)

**Optimization (`optim/`)**
- `optimizer.py`: Abstract base optimizer
- Concrete optimizers: SGD, Adam, RMSprop
- Support for weight decay and momentum

**Automatic Mixed Precision (`amp/`)**
- `autocast.py`: Context manager for automatic FP16/FP32 casting
- `grad_scaler.py`: Gradient scaling for FP16 training stability
- PyTorch-compatible API for easy migration

**Utilities (`utils/`)**
- `device.py`: CPU/GPU device detection and management
- `data.py`: Dataset and DataLoader classes
- `grad_check.py`: Numerical gradient verification
- `graph.py`: Computational graph visualization

### Key Design Patterns

**Backend Abstraction**
- Uses `xp` alias that points to either NumPy (CPU) or CuPy (GPU)
- Automatic device detection on import: `ng.DEVICE`
- All operations work seamlessly on both backends

**Function-Based Operations**
- All operations inherit from `Function` base class
- Automatic mixed precision integration via `maybe_cast_tensor`
- Forward/backward methods for custom operation definition

**Module System**
- PyTorch-like module hierarchy with `parameters()` and `named_parameters()`
- Automatic parameter registration and gradient tracking
- Training/evaluation mode switching

## Development Guidelines

### Adding New Operations
1. Inherit from `Function` in appropriate `functions/` file
2. Implement `forward()` and `backward()` methods
3. Add gradient checking in tests: `gradient_check(op, inputs, loss_fn)`
4. Update `__init__.py` imports if adding public API

### Adding New Layers
1. Inherit from `Module` in `nn/layers/`
2. Use `self.add_parameter()` for learnable parameters
3. Implement `forward()` method
4. Add to `nn/layers/__init__.py` for public access

### Testing New Features
- Place tests in `tests/test_<component>.py`
- Use small, deterministic examples with fixed seeds
- Include both forward pass and gradient checking
- Test both CPU and GPU backends when applicable

### Mixed Precision Support
- New operations automatically support AMP via `maybe_cast_tensor`
- FP16-unsafe operations (losses, normalizations) stay in FP32
- Use `should_cast_to_fp16()` to check operation compatibility

## Important Files

### Core Implementation
- `neurograd/tensor.py`: Main Tensor class and autograd engine
- `neurograd/__init__.py`: Package initialization and backend setup
- `neurograd/functions/base.py`: Base class for all operations

### Configuration & Build
- `setup.py`: Package configuration with extras for GPU, dev tools
- `requirements.txt`: Minimal dependencies (numpy>=1.19.0)
- `upload_to_pypi.{sh,bat}`: Distribution scripts

### Examples & Testing
- `*.ipynb`: Jupyter notebooks with usage examples and tests
- `test_intermediate_grads.py`: Gradient checking utilities

## Common Patterns

### Creating Custom Operations
```python
from neurograd.functions.base import Function

class CustomOp(Function):
    def forward(self, x):
        # Store values needed for backward pass
        self.saved_value = x.data
        return some_operation(x.data)
    
    def backward(self, grad_output):
        # Return gradient w.r.t. input
        grad_input = grad_output * derivative_calculation(self.saved_value)
        return grad_input if self.parent_tensors[0].requires_grad else None
```

### Model Training Loop
```python
# Standard training pattern with AMP support
from neurograd.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
```