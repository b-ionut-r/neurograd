from typing import Dict, Generator, Tuple
from neurograd.tensor import Tensor
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, model_parameters: Generator[Tuple[str, Tensor], None, None], lr: float = 0.01,
                 weight_decay: float = 0):
        """
        Base class for optimizers.
        
        Args:
            model_parameters (Generator[Tuple[str, Tensor]]): Named parameters of the model to optimize.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for the optimizer (L2/Ridge).
        """
        self.params = list(model_parameters) if model_parameters is not None else []
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for _, param in self.params:
            param.zero_grad()

    @abstractmethod
    def state_dict(self):
        pass
        
    @abstractmethod
    def load_state_dict(self, state_dict) -> None:
        pass
            
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} with learning rate {self.lr}"
    
    def __iter__(self):
         return iter(self.params)   
    
    def __len__(self) -> int:
        return len(self.params)