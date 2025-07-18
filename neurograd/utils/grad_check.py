from neurograd import xp, Tensor
from neurograd.nn.module import ModuleMixin
from .aliases import LOSSES
from typing import Union


class GradientChecker:

    def __init__(self, epsilon: float = 1e-4):  # float32 default
        self.epsilon = epsilon

    def check(self, module: ModuleMixin, X: Union[Tensor, xp.ndarray], y: Union[Tensor, xp.ndarray],
              loss_fn: Union[str, ModuleMixin]) -> bool:
        # Convert inputs to Tensors if needed
        if not isinstance(X, Tensor):
            X = Tensor(X)
        if not isinstance(y, Tensor):
            y = Tensor(y)
            
        # Get loss_fn if passed as string
        loss_fn = LOSSES.get(loss_fn, loss_fn)
        
        # One forward pass to compute the output and gradients
        module.zero_grad()
        output = module(X)
        loss = loss_fn(y, output)
        loss.backward()

        # Numerical gradient checking
        ok = True
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Create actual copies, not views
                param_flat = param.data.flatten().copy()
                grad_flat = param.grad.flatten().copy()
                param_shape = param.shape
                
                for i in range(len(param_flat)):
                    original_param_value = param_flat[i].copy()
                    analytical_gradient = grad_flat[i].copy()
                    
                    # Perturb the parameter (+ epsilon)
                    param_flat[i] = original_param_value + self.epsilon
                    param.data = param_flat.reshape(param_shape)
                    module.zero_grad()  # Clear any accumulated gradients
                    output_plus = module(X)
                    loss_plus = loss_fn(y, output_plus)
                    
                    # Perturb the parameter (- epsilon)
                    param_flat[i] = original_param_value - self.epsilon
                    param.data = param_flat.reshape(param_shape)
                    module.zero_grad()  # Clear any accumulated gradients
                    output_minus = module(X)
                    loss_minus = loss_fn(y, output_minus)
                    
                    # Compute numerical gradient
                    numerical_gradient = (loss_plus.data - loss_minus.data) / (2 * self.epsilon)
                    
                    # Reset the parameter to its original value
                    param_flat[i] = original_param_value
                    param.data = param_flat.reshape(param_shape)
                    
                    # Compare gradients using both relative and absolute tolerance
                    if not xp.allclose(numerical_gradient, analytical_gradient, 
                                      atol=self.epsilon):
                        print(f"Gradient check failed for {name} at index {i}")
                        print(f"Numerical gradient: {numerical_gradient}")
                        print(f"Analytical gradient: {analytical_gradient}")
                        
                        # Calculate relative error for debugging
                        abs_diff = abs(numerical_gradient - analytical_gradient)
                        rel_error = abs_diff / (abs(analytical_gradient) + 1e-8)
                        print(f"Absolute difference: {abs_diff}")
                        print(f"Relative error: {rel_error}")
                        ok = False
                        
        # Restore gradients from the initial forward pass
        module.zero_grad()
        output = module(X)
        loss = loss_fn(y, output)
        loss.backward()
        
        return ok


def gradient_check(model: ModuleMixin, X: Union[Tensor, xp.ndarray], y: Union[Tensor, xp.ndarray], 
                  loss_fn: Union[str, ModuleMixin], epsilon: float = 1e-4) -> bool:
    """
    Convenience function for gradient checking.
    
    Args:
        model: The model to check gradients for
        X: Input data tensor or numpy array
        y: Target/label tensor or numpy array
        loss_fn: Loss function instance (e.g., MSE(), CrossEntropy()) or string name
        epsilon: Small perturbation for numerical gradient computation
        
    Returns:
        bool: True if gradients match within tolerance, False otherwise
    """
    checker = GradientChecker(epsilon=epsilon)
    return checker.check(model, X, y, loss_fn)