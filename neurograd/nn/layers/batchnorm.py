from ..module import Module


class BatchNorm(Module):

    def __init__(self, num_features: int, batch_momentum: float = 0.9, epsilon: float = 1e-5):
        import neurograd as ng
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        super().__init__()
        # Register scalers params in init
        self.add_parameter(name="mean_scaler", param=ng.zeros((1, num_features), dtype=ng.float32, requires_grad=True))  # beta
        self.add_parameter(name="std_scaler", param=ng.ones((1, num_features), dtype=ng.float32, requires_grad=True))   # gamma
        # Running stats as buffers (Tensors) to avoid optimizer state
        self.add_buffer(name="running_mean", buffer=ng.zeros((1, num_features), dtype=ng.float32, requires_grad=False))
        self.add_buffer(name="running_var", buffer=ng.ones((1, num_features), dtype=ng.float32, requires_grad=False))

    def forward(self, X):
        from neurograd import xp
        # Training mode: compute and update running statistics
        if self.training:
            batch_mean = X.mean(axis=0, keepdims=True)
            batch_var = ((X - batch_mean) ** 2).mean(axis=0, keepdims=True)

            # Use float32 for stability and detach from autograd
            batch_mean_f32 = batch_mean.data.astype(xp.float32) if batch_mean.data.dtype != xp.float32 else batch_mean.data
            batch_var_f32 = batch_var.data.astype(xp.float32) if batch_var.data.dtype != xp.float32 else batch_var.data

            # Validate stats
            batch_mean_finite = xp.isfinite(batch_mean_f32).all()
            batch_var_finite = xp.isfinite(batch_var_f32).all()
            batch_var_nonneg = (batch_var_f32 >= 0).all()

            if batch_mean_finite and batch_var_finite and batch_var_nonneg:
                self.running_mean.data = (self.batch_momentum * self.running_mean.data +
                                          (1 - self.batch_momentum) * batch_mean_f32)
                self.running_var.data = (self.batch_momentum * self.running_var.data +
                                         (1 - self.batch_momentum) * batch_var_f32)
                self.running_var.data = xp.maximum(self.running_var.data, self.epsilon)

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
        # Register scalers params in init
        self.add_parameter(name="mean_scaler", param=ng.zeros((1, num_features, 1, 1), dtype=ng.float32, requires_grad=True))  # beta
        self.add_parameter(name="std_scaler", param=ng.ones((1, num_features, 1, 1), dtype=ng.float32, requires_grad=True))   # gamma
        # Running stats as buffers (Tensors)
        self.add_buffer(name="running_mean", buffer=ng.zeros((1, num_features, 1, 1), dtype=ng.float32, requires_grad=False))
        self.add_buffer(name="running_var", buffer=ng.ones((1, num_features, 1, 1), dtype=ng.float32, requires_grad=False))

    def forward(self, X):
        from neurograd import xp
        # X shape: (N, C, H, W)
        _, C, _, _ = X.shape
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}"

        if self.training:
            batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
            batch_var = ((X - batch_mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

            batch_mean_f32 = batch_mean.data.astype(xp.float32) if batch_mean.data.dtype != xp.float32 else batch_mean.data
            batch_var_f32 = batch_var.data.astype(xp.float32) if batch_var.data.dtype != xp.float32 else batch_var.data

            batch_mean_finite = xp.isfinite(batch_mean_f32).all()
            batch_var_finite = xp.isfinite(batch_var_f32).all()
            batch_var_nonneg = (batch_var_f32 >= 0).all()

            if batch_mean_finite and batch_var_finite and batch_var_nonneg:
                self.running_mean.data = (self.batch_momentum * self.running_mean.data +
                                          (1 - self.batch_momentum) * batch_mean_f32)
                self.running_var.data = (self.batch_momentum * self.running_var.data +
                                         (1 - self.batch_momentum) * batch_var_f32)
                self.running_var.data = xp.maximum(self.running_var.data, self.epsilon)

            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt()

        return self.std_scaler * X_norm + self.mean_scaler
