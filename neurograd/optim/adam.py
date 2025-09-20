from .optimizer import Optimizer
from typing import Generator, Tuple
import neurograd as ng
from neurograd import Tensor, xp
import numpy as real_numpy


@ng.fuse
def fused_adam_step(param, grad, weight_decay, m, v, lr, beta1, beta2, eps, t):
    grad_eff = grad + weight_decay * param
    m_new = beta1 * m + (1.0 - beta1) * grad_eff
    v_new = beta2 * v + (1.0 - beta2) * grad_eff * grad_eff
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)
    param_new = param - lr * (m_hat / (xp.sqrt(v_hat) + eps))
    return param_new, m_new, v_new



class Adam(Optimizer):
    """
    Adam optimizer with momentum and adaptive learning rate.
    This optimizer combines the benefits of AdaGrad and RMSProp, and is well-suited for a wide range of problems.
    """

    def __init__(self, model_parameters: Generator[Tuple[str, Tensor], None, None], lr: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(model_parameters, lr, weight_decay)
        self.first_momentum = [(name, xp.zeros_like(param.data)) for name, param in self.params]
        self.second_momentum = [(name, xp.zeros_like(param.data)) for name, param in self.params]
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0


    def step(self) -> None:
        self.t += 1
        for i, (name, param) in enumerate(self.params):
            if param.requires_grad and param.grad is not None:
                m = self.first_momentum[i][1]
                v = self.second_momentum[i][1]
                # Fused Adam step (weight decay first, momentums updates, bias correction, parameters update)
                param.data[:], m[:], v[:] = fused_adam_step(param.data, param.grad, self.weight_decay, m, v, self.lr,
                                                            self.beta1, self.beta2, self.epsilon, self.t)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "first_momentum": self.first_momentum,
            "second_momentum": self.second_momentum
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.epsilon = state_dict["epsilon"]
        self.weight_decay = state_dict["weight_decay"]
        self.t = state_dict["t"]
        # Rehydrate moment buffers to current backend (NumPy/CuPy)
        self.first_momentum = [(n, xp.array(a)) for n, a in state_dict["first_momentum"]]
        self.second_momentum = [(n, xp.array(a)) for n, a in state_dict["second_momentum"]]

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon}, weight_decay={self.weight_decay})."

    def __str__(self) -> str:
        return f"Adam with learning rate {self.lr}, beta1 {self.beta1}, beta2 {self.beta2}, epsilon {self.epsilon}, and weight decay {self.weight_decay}."
