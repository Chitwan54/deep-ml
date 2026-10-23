import numpy as np
from copy import copy

def find_gradient(f, x, epsilon):
    numerical_gradients = []
    x_copy = copy(x)
    for i in range(len(x_copy)):
        x_copy[i] = x[i] + epsilon
        val1 = f(x_copy)
        x_copy[i] = x[i] - epsilon
        val2 = f(x_copy)
        x_copy[i] = x[i]

        grad = (val1 - val2) / (2 * epsilon)
        numerical_gradients.append(grad)
    return np.array(numerical_gradients)

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    numerical_gradient = find_gradient(f, x, epsilon)
    num = np.linalg.norm(numerical_gradient - analytical_grad)
    denom = np.linalg.norm(numerical_gradient, 2) + np.linalg.norm(analytical_grad, 2)
    relative_error = 0 if denom == 0 else num / denom

    return numerical_gradient, relative_error

