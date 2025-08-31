import neurograd as ng
from neurograd import xp
from neurograd.nn.module import Module
from neurograd.functions.normalize import BatchNormalizer

@ng.fuse
def exp_mov_avg(old, new, momentum):
    # new_running = momentum*old + (1-momentum)*new
    return momentum * old + (1.0 - momentum) * new


class BatchNorm(Module):
    """
    BN for inputs shaped (N, C)
    """
    def __init__(self, num_features: int, batch_momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.batch_momentum = float(batch_momentum)
        self.epsilon = float(epsilon)
        # Learnable scale/shift
        self.add_parameter("mean_scaler", ng.zeros((1, num_features), dtype=ng.float32, requires_grad=True)) # beta
        self.add_parameter("var_scaler",  ng.ones((1, num_features), dtype=ng.float32, requires_grad=True))  # gamma
        # Running stats (buffers, no grad)
        self.add_buffer("running_mean", ng.zeros((1, num_features), dtype=ng.float32, requires_grad=False))
        self.add_buffer("running_var",  ng.ones((1, num_features), dtype=ng.float32, requires_grad=False))
        # Pre-create the op with the right axes
        self._bn_op = BatchNormalizer(axes=(0,), epsilon=self.epsilon)

    def forward(self, X):
        if self.training:
            # batch stats (keepdims so shapes broadcast to X)
            batch_mean = X.mean(axis=0, keepdims=True)
            batch_var  = X.var(axis=0, keepdims=True)
            # Update running stats (on .data to avoid autograd tracking)
            self.running_mean.data[:] = exp_mov_avg(self.running_mean.data, batch_mean.data, self.batch_momentum)
            self.running_var.data[:]  = exp_mov_avg(self.running_var.data,  batch_var.data,  self.batch_momentum)
            # Call the fused BN op (training path uses batch stats)
            return self._bn_op(X, batch_mean, batch_var, self.mean_scaler, self.var_scaler)
        else:
            # Eval path: use running stats (no grad through them)
            return self._bn_op(X, self.running_mean, self.running_var, self.mean_scaler, self.var_scaler)


class BatchNorm2D(Module):
    """
    BN for NCHW inputs (N, C, H, W)
    """
    def __init__(self, num_features: int, batch_momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.batch_momentum = float(batch_momentum)
        self.epsilon = float(epsilon)
        shape = (1, num_features, 1, 1)
        # Learnable scale/shift
        self.add_parameter("mean_scaler", ng.zeros(shape, dtype=ng.float32, requires_grad=True))  # beta
        self.add_parameter("var_scaler",  ng.ones(shape, dtype=ng.float32, requires_grad=True))   # gamma
        # Running stats
        self.add_buffer("running_mean", ng.zeros(shape, dtype=ng.float32, requires_grad=False))
        self.add_buffer("running_var",  ng.ones(shape,  dtype=ng.float32, requires_grad=False))
        # Axes over which per-channel stats are computed: (N, H, W)
        self._bn_op = BatchNormalizer(axes=(0, 2, 3), epsilon=self.epsilon)

    def forward(self, X):
        if self.training:
            batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
            batch_var  = X.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean.data[:] = exp_mov_avg(self.running_mean.data, batch_mean.data, self.batch_momentum)
            self.running_var.data[:]  = exp_mov_avg(self.running_var.data,  batch_var.data,  self.batch_momentum)
            return self._bn_op(X, batch_mean, batch_var, self.mean_scaler, self.var_scaler)
        else:
            return self._bn_op(X, self.running_mean, self.running_var, self.mean_scaler, self.var_scaler)