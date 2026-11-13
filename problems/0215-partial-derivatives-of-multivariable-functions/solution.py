import math
def compute_partial_derivatives(func_name: str, point: tuple[float, ...], h: float = 1e-5) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions using central finite differences.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate
        h: Step size for numerical differentiation
    
    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    FUNCTIONS = {
        'poly2d':        lambda p: p[0]**2 * p[1] + p[0] * p[1]**2,
        'exp_sum':       lambda p: math.exp(p[0] + p[1]),
        'product_sin':   lambda p: p[0] * math.sin(p[1]),
        'poly3d':        lambda p: p[0]**2 * p[1] + p[1] * p[2]**2,
        'squared_error': lambda p: (p[0] - p[1])**2,
    }

    f = FUNCTIONS[func_name]
    point_lst = list(point)
    grads = []

    # Numerically differentiate with respect to each coordinate using central difference
    for i in range(len(point_lst)):
        p_plus = point_lst.copy()
        p_minus = point_lst.copy()
        
        p_plus[i] += h
        p_minus[i] -= h
        
        partial = (f(p_plus) - f(p_minus)) / (2 * h)
        grads.append(partial)

    return tuple(grads)

    

        


