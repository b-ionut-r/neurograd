from neurograd.functions.activations import ReLU, LeakyReLU, ReLU6, Sigmoid, Softmax, Tanh, Passthrough
from neurograd.nn.losses import MSE, RMSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy
from neurograd.nn.initializers import Xavier, He, Zeros, Normal

ACTIVATIONS = {
    "relu": lambda: ReLU(),
    "leaky_relu": lambda: LeakyReLU(),
    "relu6": lambda: ReLU6(),
    "sigmoid": lambda: Sigmoid(),
    "softmax": lambda: Softmax(),
    "tanh": lambda: Tanh(),
    "passthrough": lambda: Passthrough()
}

LOSSES = {
    "mse": lambda: MSE(),
    "rmse": lambda: RMSE(),
    "mae": lambda: MAE(),
    "binary_crossentropy": lambda: BinaryCrossEntropy(),
    "categorical_crossentropy": lambda: CategoricalCrossEntropy()
}


# Note: Initializer aliases require parameters, so they're defined as dictionaries
INITIALIZERS = {
    "xavier": Xavier,
    "he": He,
    "zeros": Zeros,
    "normal": Normal
}
#fixed
