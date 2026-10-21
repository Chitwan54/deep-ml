import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    # 1. Create a mini batch
    X_mini_batch = X[batch_indices]
    y_mini_batch = y[batch_indices]
    m = len(X_mini_batch)

    # 2. Compute the Gradient
    y_hat = np.dot(X_mini_batch, weights) + bias
    gradient_w = -2.0 * (np.dot(X_mini_batch.T, (y_mini_batch - y_hat))) / m
    gradient_b = -2.0 * (np.mean((y_mini_batch - y_hat))) 

    # 3. Update the weight matrix
    weights = weights - lr * gradient_w
    bias = bias - lr * gradient_b

    return np.concatenate([weights, [bias]])




