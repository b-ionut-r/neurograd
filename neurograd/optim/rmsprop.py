from .optimizer import Optimizer
from typing import Generator, Tuple
import neurograd as ng
from neurograd import Tensor, xp
import numpy as real_numpy

if xp is real_numpy:
    conditional_fuse = lambda f: f
else:
    from cupy import fuse
    conditional_fuse = fuse

@conditional_fuse
def fused_rmsprop_step(param, grad, weight_decay, momentum, lr, beta, eps):
    # grad_eff = grad + weight_decay * param
    grad_eff = grad + weight_decay * param
    # momentum_new = beta * momentum + (1 - beta) * grad_eff^2
    momentum_new = beta * momentum + (1.0 - beta) * grad_eff * grad_eff
    # denom = sqrt(momentum_new) + eps
    denom = xp.sqrt(momentum_new) + eps
    # param_new = param - lr * grad_eff / denom
    param_new = param - lr * grad_eff / denom
    return param_new, momentum_new


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    """

    def __init__(self, model_parameters: Generator[Tuple[str, Tensor], None, None], lr: float = 0.01,
                 beta: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        """
        Initializes the RMSprop optimizer.

        Args:
            model_parameters (Generator[Tuple[str, Tensor]]): Named parameters of the model to optimize.
            lr (float): Learning rate for the optimizer.
            beta (float): Smoothing constant for squared gradient moving average.
            eps (float): Small value to prevent division by zero.
            weight_decay (float): Weight decay factor for the optimizer (L2/Ridge).
        """
        super().__init__(model_parameters, lr, weight_decay)
        self.momentum = [(name, xp.zeros_like(param.data)) for name, param in self.params]
        self.beta = beta
        self.eps = eps

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for i, (name, param) in enumerate(self.params):
            if param.requires_grad and param.grad is not None:
                momentum = self.momentum[i][1]
                # Single fused step: weight decay + momentum update + param update
                param.data[:], momentum[:] = fused_rmsprop_step(
                    param.data, param.grad, self.weight_decay, momentum, self.lr, self.beta, self.eps
                )

    
    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "beta": self.beta,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "momentum": self.momentum,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = state_dict["lr"]
        self.beta = state_dict["beta"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.t = state_dict["t"]
        self.momentum = [(n, xp.array(a)) for n, a in state_dict["momentum"]]
    
    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, beta={self.beta}, eps={self.eps}, weight_decay={self.weight_decay})."
    
    def __str__(self) -> str:
        return f"RMSprop with learning rate {self.lr}, beta {self.beta}, eps {self.eps}, and weight decay {self.weight_decay}."
