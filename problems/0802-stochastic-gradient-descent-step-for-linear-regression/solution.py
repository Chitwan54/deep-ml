import numpy as np
def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    num_rows, num_cols = X.shape
    y = y.reshape(-1, 1)

    # Make a copy of weights to avoid modifying original array in-place
    weights = weights.copy().astype(float)

    for i in range(n_iter):
        sample = i % num_rows
        X_input = X[sample]
        y_input = y[sample]

        # Compute the predictions
        y_hat = np.dot(X_input, weights.reshape(-1, 1))

        # Compute the per-sample gradient
        gradient = -2.0 * np.dot(X_input.reshape(-1, 1), (y_input - y_hat))

        # Update the weights by flattening the 2D gradient to 1D
        weights = weights - learning_rate * gradient.flatten()

    return weights.tolist()

    
