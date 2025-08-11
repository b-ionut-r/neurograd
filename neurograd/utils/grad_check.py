from neurograd import xp, Tensor
from neurograd.nn.module import Module
from .aliases import LOSSES
from typing import Union


class GradientChecker:

    def __init__(self, epsilon: float = 1e-4):  # float32 default
        self.epsilon = epsilon

    def check(self, module: Module, X: Union[Tensor, xp.ndarray], y: Union[Tensor, xp.ndarray],
              loss_fn: Union[str, Module]) -> bool:
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


def gradient_check(model: Module, X: Union[Tensor, xp.ndarray], y: Union[Tensor, xp.ndarray], 
                  loss_fn: Union[str, Module], epsilon: float = 1e-4) -> bool:
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



def check_vanishing_grads(model: Module, tol: float = 1e-8):
    """
    Inspect existing gradients on `model` and report vanishing/NaN stats.
    Assumes you've already called backward().
    Returns a compact dict; no side effects beyond that.
    """
    n_params_with_grad = 0
    tiny, nan_names = [], []
    total_g2 = 0.0

    for name, p in model.named_parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        n_params_with_grad += 1

        if xp.isnan(g).any():
            nan_names.append(name)
            continue  # skip norm accumulation for NaN grads

        # grad norm
        try:
            gnorm = float(xp.linalg.norm(g))
        except Exception:
            gnorm = float(xp.sqrt((g * g).sum()))

        if gnorm < tol:
            tiny.append(name)

        total_g2 += gnorm * gnorm

    total_grad_norm = total_g2 ** 0.5

    return {
        "n_with_grad": n_params_with_grad,
        "total_grad_norm": total_grad_norm,
        "n_vanishing": len(tiny),
        "vanishing_params": tiny,   # names
        "n_nan": len(nan_names),
        "nan_params": nan_names,    # names
        "threshold": tol,
    }