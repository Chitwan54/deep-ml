import numpy as np

def compute_loss(X, y, w0):
    y_hat = np.dot(X, w0)
    return np.mean((y_hat - y) ** 2)

def compute_gradient(X, y, w0):
    y_hat = np.dot(X, w0)
    return 2 * np.dot(X.T, (y_hat - y)) / len(X)

def update_weights(weights, gradient, learning_rate):
    weights = weights - learning_rate * gradient
    return weights

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    # learning_rates = [np.power(10, a + ((b-a) * (i) / n_steps-1)) for i in range(n_steps)]
    results = []

    for i in range(n_steps):
        exponent = a + (b - a) * i / (n_steps - 1) if n_steps > 1 else a
        learning_rate = np.power(10.0, exponent)
        gradient = compute_gradient(X, y, w0)
        w0 = update_weights(w0, gradient, learning_rate)
        loss = compute_loss(X, y, w0)

        results.append(float(loss))
    return results
    


