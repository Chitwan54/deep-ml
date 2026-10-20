import numpy as np

def get_derivative(func_name: str, x: float, h: float = 1e-5) -> float:
    """
    Computes the numerical derivative of a numpy function at x using central difference.
    """
    func = getattr(np, func_name)
    return (func(x + h) - func(x - h)) / (2 * h)

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Computes the gradient of a composed function f(x) = f_1(f_2(...(f_n(x)))) using the chain rule.
    The list `functions` is ordered from outer to inner. For example, ["sin", "square"] represents sin(x^2).
    """
    # 1. Forward Pass: compute and store intermediate values.
    # For a chain [f_1, f_2, ..., f_n], the evaluation starts with f_n(x).
    intermediates = [x]
    current_val = x
    for func_name in reversed(functions):
        func = getattr(np, func_name)
        current_val = func(current_val)
        intermediates.append(current_val)
    
    # 2. Backward Pass: compute derivative step by step.
    # Using the chain rule: d/dx f_1(f_2(x)) = f_1'(f_2(x)) * f_2'(x)
    gradient = 1.0
    current_input = x
    
    # Iterate backwards through functions (from inner to outer to match their evaluation points)
    # i-th function in reversed order is evaluated at the (i)-th intermediate value.
    reversed_funcs = list(reversed(functions))
    for i in range(len(reversed_funcs)):
        func_name = reversed_funcs[i]
        evaluation_point = intermediates[i]
        derivative_val = get_derivative(func_name, evaluation_point)
        gradient *= derivative_val
        
    return float(gradient)
	