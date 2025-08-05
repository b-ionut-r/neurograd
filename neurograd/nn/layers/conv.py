from neurograd.functions.activations import ReLU
from ..module import Module
from .batchnorm import BatchNorm2D
from .dropout import Dropout2D
from typing import TYPE_CHECKING, Union, Tuple, Sequence, Literal
from numpy.typing import ArrayLike

class Conv2D(Module):

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, ...]] = (3, 3),
                strides: Union[int, Tuple[int, ...]] = (1, 1),
                padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
                padding_value: Union[int, float] = 0,
                weights_initializer = "normal", bias_initializer = "zeros",
                activation = "passthrough", dropout = 0.0, 
                batch_normalization = False, batch_momentum = 0.9,
                use_bias = True, dtype = None):
        
        import neurograd as ng     
        from neurograd.utils.aliases import ACTIVATIONS, INITIALIZERS

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        # Handle padding string literals properly
        if isinstance(padding, str):
            self.padding = padding  # Keep as string for conv2d function to handle
        else:
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding_value = padding_value
        
        dtype = dtype if dtype is not None else ng.float32
        def get_initializer(init_name):
            init_class = INITIALIZERS.get(init_name, init_name)
            init_params = {"dtype": dtype}
            if init_name == "normal":
                init_params["scale"] = 0.01
            elif init_name == "xavier":
                init_params["n_in"] = in_channels * self.kernel_size[0] * self.kernel_size[1]
                init_params["n_out"] = out_channels * self.kernel_size[0] * self.kernel_size[1]
            elif init_name == "he":
                init_params["n_in"] = in_channels * self.kernel_size[0] * self.kernel_size[1]
            return init_class(**init_params) if init_name in ["normal", "xavier", "he", "zeros"] else init_class
        self.kernels_initializer = get_initializer(weights_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        self.activation = ACTIVATIONS[activation]() if isinstance(activation, str) else activation
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.batch_momentum = batch_momentum
        self.use_bias = use_bias
        self.dtype = dtype
        
        # Super init before adding parameters
        super().__init__()
        if batch_normalization:
            self.batch_norm = BatchNorm2D(out_channels, batch_momentum=batch_momentum)
        if dropout > 0.0:
            self.dropout_layer = Dropout2D(dropout)
        # Add parameters
        self.add_parameter(name="kernels", param=self.kernels_initializer.generate((out_channels, in_channels, *self.kernel_size)))
        if batch_normalization:
            self.use_bias = False
        if use_bias:
            self.add_parameter(name="bias", param=self.bias_initializer.generate((1, out_channels, 1, 1)))




    def forward(self, X):
        from neurograd import conv2d
        X = X.cast(self.dtype)
        Z = conv2d(X, self.kernels, self.strides, self.padding, self.padding_value)
        if self.use_bias:
            Z += self.bias
        # Apply BatchNorm if needed
        if self.batch_normalization:
            Z = self.batch_norm(Z)  
        A = self.activation(Z)
        # Apply dropout if needed
        if self.dropout > 0.0:
            A = self.dropout_layer(A)
        return A





class MaxPool2D(Module):
    def __init__(self,
                pool_size: Union[int, Tuple[int, ...]] = (2, 2),
                strides: Union[int, Tuple[int, ...]] = (2, 2),
                padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
                padding_value: Union[int, float] = 0,
                dtype = None):
        
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding_value = padding_value
        self.dtype = dtype
        super().__init__()

    def forward(self, X):
        from neurograd import maxpool2d
        X = X.cast(self.dtype)
        Z = maxpool2d(X, self.pool_size, self.strides, self.padding, self.padding_value)  
        return Z



class AveragePool2D(Module):
    def __init__(self,
                pool_size: Union[int, Tuple[int, ...]] = (2, 2),
                strides: Union[int, Tuple[int, ...]] = (1, 1),
                padding: Union[Sequence, ArrayLike, int, Literal["valid", "same"]] = (0, 0),
                padding_value: Union[int, float] = 0,
                dtype = None):
        
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding_value = padding_value
        self.dtype = dtype
        super().__init__()

    def forward(self, X):
        from neurograd import averagepool2d
        X = X.cast(self.dtype)
        Z = averagepool2d(X, self.pool_size, self.strides, self.padding, self.padding_value)  
        return Z

MaxPooling2D = MaxPool2D
AveragePooling2D = AveragePool2D