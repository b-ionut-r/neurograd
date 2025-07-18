from neurograd.functions.activations import ReLU
from ..module import ModuleMixin

class Linear(ModuleMixin):

    def __init__(self, in_features: int, out_features: int, activation = "passthrough", 
                 dropout = 0.0, weights_initializer = "normal", bias_initializer = "zeros",
                 batch_normalization = False, batch_momentum = 0.9,
                 use_bias = True, dtype = None):
        from neurograd import xp
        from neurograd.utils.aliases import ACTIVATIONS, INITIALIZERS
        import neurograd as ng
        
        if dtype is None:
            dtype = xp.float32
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        activation_factory = ACTIVATIONS.get(activation, activation)
        self.activation = activation_factory() if callable(activation_factory) else activation_factory
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.batch_momentum = batch_momentum
        self.use_bias = use_bias
        self.dtype = dtype
        
        # Running statistics for batch norm (not trainable parameters)
        if batch_normalization:
            self.running_mean = ng.zeros((out_features, 1)) # Tensor
            self.running_var = ng.ones((out_features, 1)) # Tensor
        
        # Helper function to instantiate initializers
        def get_initializer(init_name, n_in):
            init_class = INITIALIZERS.get(init_name, init_name)
            init_params = {"dtype": dtype}
            
            if init_name == "normal":
                init_params["scale"] = 0.01
            elif init_name in ["xavier", "he"]:
                init_params["n_in"] = n_in
                
            return init_class(**init_params) if init_name in ["normal", "xavier", "he", "zeros"] else init_class

        # Initialize weights and bias
        self.weights_initializer = get_initializer(weights_initializer, in_features)
        self.bias_initializer = get_initializer(bias_initializer, in_features)

        # Add parameters
        self.add_parameter(name="weight", param=self.weights_initializer.generate((out_features, in_features)))
        if batch_normalization:
            self.use_bias = False
            self.add_parameter(name="mean_scaler", param=ng.zeros((out_features, 1))) # beta
            self.add_parameter(name="std_scaler", param=ng.ones((out_features, 1))) # gamma
        if use_bias:
            self.add_parameter(name="bias", param=self.bias_initializer.generate((out_features, 1)))

    def forward(self, X):
        import neurograd as ng
        from neurograd import xp
        # X will be of shape in_features x n_samples 
        X = X.cast(self.dtype)
        Z = ng.dot(self.weight, X)
        if self.use_bias:
            Z += self.bias
            
        # Apply BatchNorm if needed
        if self.batch_normalization:
            if self.training:
                # Training mode: compute batch statistics
                batch_mean = Z.mean(axis=1, keepdims=True)
                batch_var = ((Z - batch_mean) ** 2).mean(axis=1, keepdims=True)
                
                # Update running statistics (detached from computation graph)
                self.running_mean.data = (self.batch_momentum * self.running_mean.data + 
                                        (1 - self.batch_momentum) * batch_mean.data)
                self.running_var.data = (self.batch_momentum * self.running_var.data + 
                                       (1 - self.batch_momentum) * batch_var.data)
                
                # Normalize using batch statistics
                Z_norm = (Z - batch_mean) / (batch_var + 1e-8).sqrt()
            else:
                # Inference mode: use running statistics
                Z_norm = (Z - self.running_mean) / (self.running_var + 1e-8).sqrt()
            
            # Scale and shift
            Z = self.std_scaler * Z_norm + self.mean_scaler
            
        A = self.activation(Z)
        
        # Apply dropout
        if self.dropout > 0.0 and self.training:
            keep_prob = 1.0 - self.dropout
            mask = xp.random.rand(*A.shape) < keep_prob
            A = A * mask / keep_prob
            
        return A


class MLP(ModuleMixin):
    def __init__(self, layers_sizes):
        from neurograd.functions.activations import ReLU
        super().__init__()
        for i in range(len(layers_sizes) - 1):
            self.add_module(f'linear_{i}', 
                Linear(layers_sizes[i], layers_sizes[i+1]))
            if i < len(layers_sizes) - 2:  # No ReLU after last layer
                self.add_module(f'relu_{i}', ReLU())
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x