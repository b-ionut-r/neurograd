from ..module import Module
from .batchnorm import BatchNorm
from .dropout import Dropout

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, activation = "passthrough", 
                 dropout = 0.0, weights_initializer = "he", bias_initializer = "zeros",
                 batch_normalization = False, batch_momentum = 0.9,
                 use_bias = True, dtype = None):
        from neurograd import xp
        from neurograd.utils.aliases import ACTIVATIONS, INITIALIZERS
        import neurograd as ng
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = ACTIVATIONS[activation]() if isinstance(activation, str) else activation
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.batch_momentum = batch_momentum
        self.use_bias = use_bias
        dtype = dtype if dtype is not None else ng.float32
        self.dtype = dtype
        
        # Create BatchNorm layer if needed
        if batch_normalization:
            self.batch_norm = BatchNorm(out_features, batch_momentum=batch_momentum)
            
        # Create Dropout layer if needed
        if dropout > 0.0:
            self.dropout_layer = Dropout(dropout)
        
        # Helper function to instantiate initializers
        def get_initializer(init_name):
            init_class = INITIALIZERS.get(init_name, init_name)
            init_params = {"dtype": dtype}
            if init_name == "normal":
                init_params["scale"] = 0.01  # Improved scale for better gradient flow
            elif init_name == "xavier":
                init_params["n_in"] = in_features
                init_params["n_out"] = out_features
            elif init_name == "he":
                init_params["n_in"] = in_features
            return init_class(**init_params) if init_name in ["normal", "xavier", "he", "zeros"] else init_class
        # Initialize weights and bias
        self.weights_initializer = get_initializer(weights_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        # Add parameters
        self.add_parameter(name="weight", param=self.weights_initializer.generate((in_features, out_features)))
        if batch_normalization:
            self.use_bias = False
        if self.use_bias:
            self.add_parameter(name="bias", param=self.bias_initializer.generate((out_features,)))

    def forward(self, X):
        import neurograd as ng
        from neurograd import xp
        # X will be of shape (batch_size, in_features)
        X = X.cast(self.dtype)
        Z = ng.matmul(X, self.weight)
        if self.use_bias:
            Z += self.bias
            
        # Apply BatchNorm if needed
        if self.batch_normalization:
            Z = self.batch_norm(Z)
            
        A = self.activation(Z)
        
        # Apply dropout
        if self.dropout > 0.0:
            A = self.dropout_layer(A)
            
        return A


class MLP(Module):
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
