from ..module import Module


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
            batch_var = ((X - batch_mean) ** 2).mean(axis=0, keepdims=True)
            
            # Update running stats
            self.running_mean.data = self.batch_momentum * self.running_mean.data + (1 - self.batch_momentum) * batch_mean.data
            self.running_var.data = self.batch_momentum * self.running_var.data + (1 - self.batch_momentum) * batch_var.data
            
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
            batch_var = ((X - batch_mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            
            # Update running stats
            self.running_mean.data = self.batch_momentum * self.running_mean.data + (1 - self.batch_momentum) * batch_mean.data
            self.running_var.data = self.batch_momentum * self.running_var.data + (1 - self.batch_momentum) * batch_var.data
            
            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt()
            
        return self.std_scaler * X_norm + self.mean_scaler