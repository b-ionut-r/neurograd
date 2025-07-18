from .optimizer import Optimizer
from typing import Generator, Tuple
import neurograd as ng
from neurograd import Tensor, xp


class Adam(Optimizer):
    """
    Adam optimizer with momentum and adaptive learning rate.
    This optimizer combines the benefits of AdaGrad and RMSProp, and is well-suited for a wide range of problems.
    """

    def __init__(self, model_parameters: Generator[Tuple[str, Tensor], None, None], lr: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        """
        Initializes the Adam optimizer.

        Args:
            model_parameters (Generator[Tuple[str, Tensor]]): Named parameters of the model to optimize.
            lr (float): Learning rate for the optimizer.
            beta1 (float): Exponential decay rate for the first moment estimate.
            beta2 (float): Exponential decay rate for the second moment estimate.
            epsilon (float): Small value to prevent division by zero.
            weight_decay(float): Weight decay factor for the optimizer (L2/Ridge).
        """
        super().__init__(model_parameters, lr, weight_decay)
        self.first_momentum = [(name, xp.zeros_like(param.data)) for name, param in self.params]
        self.second_momentum = [(name, xp.zeros_like(param.data)) for name, param in self.params]
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1


    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for i, (name, param) in enumerate(self.params):
            if param.requires_grad and param.grad is not None:
                # Get gradient and apply weight decay
                grad = param.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update first moment (momentum)
                m_name, m_value = self.first_momentum[i]
                m_value = m_value * self.beta1 + grad * (1 - self.beta1)
                self.first_momentum[i] = (m_name, m_value)
                
                # Update second moment (RMSprop)
                v_name, v_value = self.second_momentum[i]
                v_value = v_value * self.beta2 + (grad ** 2) * (1 - self.beta2)
                self.second_momentum[i] = (v_name, v_value)
                
                # Bias correction
                m_corrected = m_value / (1 - self.beta1 ** self.t)
                v_corrected = v_value / (1 - self.beta2 ** self.t)
                
                # Update parameters in-place
                param.data = param.data - self.lr * m_corrected / (xp.sqrt(v_corrected) + self.epsilon)
        self.t += 1

    
    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "params": self.params,
            "first_momentum": self.first_momentum,
            "second_momentum": self.second_momentum
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.epsilon = state_dict["epsilon"]
        self.params = state_dict["params"]
        self.first_momentum = state_dict["first_momentum"]
        self.second_momentum = state_dict["second_momentum"]

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon}, weight_decay={self.weight_decay})."

    def __str__(self) -> str:
        return f"Adam with learning rate {self.lr}, beta1 {self.beta1}, beta2 {self.beta2}, epsilon {self.epsilon}, and weight decay {self.weight_decay}."