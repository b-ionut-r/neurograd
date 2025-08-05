from ..module import Module


class BatchNorm(Module):

    def __init__(self, batch_momentum=0.9, epsilon=1e-8):
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        super().__init__()  

    def forward(self, X):
        import neurograd as ng
        from neurograd import xp
        # Register scalers params if not already done
        if not hasattr(self, 'mean_scaler'):
            self.add_parameter(name="mean_scaler", param=ng.zeros((X.shape[1], 1))) # beta
            self.add_parameter(name="std_scaler", param=ng.ones((X.shape[1], 1))) # gamma
            self.running_mean = ng.zeros((X.shape[1], 1))
            self.running_var = ng.ones((X.shape[1], 1))
        # Apply BatchNorm if needed
        if self.training:
            # Training mode: compute batch statistics
            batch_mean = X.mean(axis=1, keepdims=True)
            batch_var = ((X - batch_mean) ** 2).mean(axis=1, keepdims=True)
            
            # Update running statistics (detached from computation graph)
            self.running_mean.data = (self.batch_momentum * self.running_mean.data + 
                                    (1 - self.batch_momentum) * batch_mean.data)
            self.running_var.data = (self.batch_momentum * self.running_var.data + 
                                    (1 - self.batch_momentum) * batch_var.data)
            
            # Normalize using batch statistics
            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            # Inference mode: use running statistics
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt()

        # Scale and shift
        X = self.std_scaler * X_norm + self.mean_scaler
        return X



class BatchNorm2D(Module):

    def __init__(self, num_features, batch_momentum=0.9, epsilon=1e-8):
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        super().__init__()  

    def forward(self, X):
        import neurograd as ng
        from neurograd import xp
        # X shape: (N, C, H, W) - channels first format
        N, C, H, W = X.shape
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}"
        
        # Register scalers params if not already done
        if not hasattr(self, 'mean_scaler'):
            self.add_parameter(name="mean_scaler", param=ng.zeros((1, C, 1, 1))) # beta
            self.add_parameter(name="std_scaler", param=ng.ones((1, C, 1, 1))) # gamma
            self.running_mean = ng.zeros((1, C, 1, 1))
            self.running_var = ng.ones((1, C, 1, 1))
        
        # Apply BatchNorm if needed
        if self.training:
            # Training mode: compute batch statistics across N, H, W dimensions
            batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
            batch_var = ((X - batch_mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics (detached from computation graph)
            self.running_mean.data = (self.batch_momentum * self.running_mean.data + 
                                    (1 - self.batch_momentum) * batch_mean.data)
            self.running_var.data = (self.batch_momentum * self.running_var.data + 
                                    (1 - self.batch_momentum) * batch_var.data)
            
            # Normalize using batch statistics
            X_norm = (X - batch_mean) / (batch_var + self.epsilon).sqrt()
        else:
            # Inference mode: use running statistics
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon).sqrt()

        # Scale and shift
        X = self.std_scaler * X_norm + self.mean_scaler
        return X
