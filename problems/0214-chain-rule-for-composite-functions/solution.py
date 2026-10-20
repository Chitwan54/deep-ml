import torch
import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Your code here
    forward_pass = [x]
    rev_functions = list(reversed(functions))
    for func in rev_functions:
        callable_func = getattr(np, func)
        input = forward_pass[-1]
        value = callable_func(input)
        forward_pass.append(value)

    # print(forward_pass)

    gradient = 1.0
    for i in range(len(rev_functions)):
        function = getattr(torch, rev_functions[i])
        input = forward_pass[i]

        d_function = torch.func.grad(function)
        input_torch = torch.tensor(input)
        gradient *= d_function(input_torch)
    return gradient
