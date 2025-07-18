from neurograd import xp
from typing import TYPE_CHECKING, Generator, Tuple

if TYPE_CHECKING:
    from neurograd.tensor import Tensor


class ModuleMixin:
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

    def add_module(self, name: str, module: "ModuleMixin"):
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self) -> Generator['Tensor', None, None]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def modules(self) -> Generator['ModuleMixin', None, None]:
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

    def __setattr__(self, name, value):
        from neurograd.tensor import Tensor  # prevent circular import
        
        # Always call super first to set the attribute
        super().__setattr__(name, value)
        
        # Then register in appropriate dictionary if initialized
        if isinstance(value, ModuleMixin) and hasattr(self, '_modules'):
            self._modules[name] = value
        elif isinstance(value, Tensor) and hasattr(self, '_parameters'):
            self._parameters[name] = value


class Sequential(ModuleMixin):
    """A container for a sequence of modules."""
    def __init__(self, *modules: ModuleMixin):
        super().__init__()
        self._sequential_modules = modules
        for i, module in enumerate(modules):
            self.add_module(f"layer_{i}", module)

    def forward(self, *inputs):
        output = inputs[0] if len(inputs) == 1 else inputs
        for module in self._sequential_modules:
            output = module(output)
        return output
