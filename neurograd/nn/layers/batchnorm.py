import neurograd as ng
from ..module import Module

@ng.fuse
def exp_mov_avg(old, new, alpha):
    return alpha * old + (1 - alpha) * new



class BatchNorm(Module):
    def __init__(self, num_features: int, batch_momentum: float = 0.9, epsilon: float = 1e-5):
        import neurograd as ng
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        super().__init__()
        self.add_parameter("mean_scaler", ng.zeros((1, num_features), dtype=ng.float32, requires_grad=True))
        self.add_parameter("std_scaler", ng.ones((1, num_features), dtype=ng.float32, requires_grad=True))
        self.add_buffer("running_mean", ng.zeros((1, num_features), dtype=ng.float32, requires_grad=False))
        self.add_buffer("running_var", ng.ones((1, num_features), dtype=ng.float32, requires_grad=False))

    def forward(self, X):
        if self.training:
            batch_mean = X.mean(axis=0, keepdims=True)
            # Use built-in variance reduction to avoid materializing squared residuals
            batch_var = X.var(axis=0, keepdims=True)
            # Update running stats
            self.running_mean.data = exp_mov_avg(self.running_mean.data, batch_mean.data, self.batch_momentum)
            self.running_var.data = exp_mov_avg(self.running_var.data, batch_var.data, self.batch_momentum)
            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt() 
        return self.std_scaler * X_norm + self.mean_scaler


class BatchNorm2D(Module):
    def __init__(self, num_features: int, batch_momentum: float = 0.9, epsilon: float = 1e-5):
        import neurograd as ng
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        super().__init__()
        self.add_parameter("mean_scaler", ng.zeros((1, num_features, 1, 1), dtype=ng.float32, requires_grad=True))
        self.add_parameter("std_scaler", ng.ones((1, num_features, 1, 1), dtype=ng.float32, requires_grad=True))
        self.add_buffer("running_mean", ng.zeros((1, num_features, 1, 1), dtype=ng.float32, requires_grad=False))
        self.add_buffer("running_var", ng.ones((1, num_features, 1, 1), dtype=ng.float32, requires_grad=False))

    def forward(self, X):
        if self.training:
            batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
            # Use built-in variance reduction to avoid materializing squared residuals
            batch_var = X.var(axis=(0, 2, 3), keepdims=True)
            # Update running stats
            self.running_mean.data = exp_mov_avg(self.running_mean.data, batch_mean.data, self.batch_momentum)
            self.running_var.data = exp_mov_avg(self.running_var.data, batch_var.data, self.batch_momentum)
            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt()
        return self.std_scaler * X_norm + self.mean_scaler
