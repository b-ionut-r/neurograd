from neurograd import xp, Tensor

class Normal:
    def __init__(self, scale=0.01, dtype = xp.float32):
        self.scale = scale
        self.dtype = dtype

    def generate(self, *inputs):
        shape = inputs[0] if isinstance(inputs[0], tuple) else tuple(inputs)
        data = xp.random.randn(*shape) * self.scale
        return Tensor(data = data.astype(self.dtype), dtype = self.dtype, requires_grad=True)
    

class Xavier:
    """Xavier / Glorot initialization. Recommended for layers with Tanh activation."""
    def __init__(self, n_in, n_out=None, dtype = xp.float32):
        self.n_in = n_in
        self.n_out = n_out or n_in
        self.dtype = dtype

    def generate(self, *inputs):
        shape = inputs[0] if isinstance(inputs[0], tuple) else tuple(inputs)
        data = xp.random.randn(*shape) * xp.sqrt(2.0 / (self.n_in + self.n_out))
        return Tensor(data = data.astype(self.dtype), dtype = self.dtype, requires_grad=True)

class He: 
    """He initialization. Recommended for layers with ReLU activation."""
    def __init__(self, n_in, dtype = xp.float32):
        self.n_in = n_in
        self.dtype = dtype

    def generate(self, *inputs):
        shape = inputs[0] if isinstance(inputs[0], tuple) else tuple(inputs)
        data = xp.random.randn(*shape) * xp.sqrt(2.0 / self.n_in)
        return Tensor(data = data.astype(self.dtype), dtype = self.dtype, requires_grad=True)
    


class Zeros:
    def __init__(self, dtype = xp.float32):
        self.dtype = dtype

    def generate(self, *inputs):
        shape = inputs[0] if isinstance(inputs[0], tuple) else tuple(inputs)
        data = xp.zeros(shape)
        return Tensor(data = data, dtype = self.dtype, requires_grad=True)