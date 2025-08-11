from neurograd import xp
import numpy as np
from typing import TYPE_CHECKING, Generator, Tuple

if TYPE_CHECKING:
    from neurograd.tensor import Tensor


class Module:
    """Base class providing Module functionality for neural network components."""
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError("Forward method must be implemented in the subclass.")

    def add_parameter(self, name: str, param: 'Tensor'):
        self._parameters[name] = param
        super().__setattr__(name, param)

    def add_module(self, name: str, module: "Module"):
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self) -> Generator['Tensor', None, None]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def modules(self) -> Generator['Module', None, None]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_parameters(self) -> Generator[Tuple[str, 'Tensor'], None, None]:
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for subname, param in module.named_parameters():
                yield f"{name}.{subname}", param

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode=True):
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(mode=False)

    def state_dict(self):
        """Flat dict of parameter name -> NumPy array."""
        to_np = (lambda a: xp.asnumpy(a)) if hasattr(xp, "asnumpy") else (lambda a: np.array(a, copy=True))
        return {name: to_np(p.data) for name, p in self.named_parameters()}

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load parameters from a state dict."""
        missing, unexpected, seen = [], [], set()
        for name, p in self.named_parameters():
            if name not in state_dict:
                missing.append(name)
                continue
            seen.add(name)
            arr = xp.array(state_dict[name], dtype=p.data.dtype)
            if arr.shape != p.data.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': {arr.shape} vs {p.data.shape}")
                if arr.size != p.data.size:
                    continue
                arr = arr.reshape(p.data.shape)
            p.data[...] = arr
        unexpected = [k for k in state_dict.keys() if k not in seen]
        if strict and (missing or unexpected):
            raise KeyError(f"missing={missing}, unexpected={unexpected}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}


    def __setattr__(self, name, value):
        from neurograd.tensor import Tensor  # prevent circular import
        # Always call super first to set the attribute
        super().__setattr__(name, value)
        # Then register in appropriate dictionary if initialized
        if isinstance(value, Module) and hasattr(self, '_modules'):
            self._modules[name] = value
        elif isinstance(value, Tensor) and hasattr(self, '_parameters'):
            self._parameters[name] = value
            

class Sequential(Module):
    """A container for a sequence of modules."""
    def __init__(self, *modules: Module):
        super().__init__()
        self._sequential_modules = modules
        for i, module in enumerate(modules):
            self.add_module(f"layer_{i}", module)

    def forward(self, X):
        for module in self._sequential_modules:
            X = module(X)
        return X
