import numpy as np

def compute_gradient(X, y, weights):
  y_hat = np.dot(X, weights.reshape(-1,1))
  if len(X.shape) == 1:
    return -2.0 * (np.dot(X.T.reshape(-1,1), (y.reshape(-1,1) - y_hat)))
  else:
    return -2.0 * (np.dot(X.T, (y.reshape(-1,1) - y_hat))) / X.shape[0]

def weight_update(weights, gradient, learning_rate):
  weights = weights.reshape(-1, 1)
  weights -= learning_rate * gradient
  return weights

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
  """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
  """

  # 1. Batch GD: Take all the data points to make an update to the weights
  if method == "batch":
    for epoch in range(n_epochs):
      # Compute Gradients and the update the weights basis the whole data
      gradient = compute_gradient(X, y, weights)
      weights = weight_update(weights, gradient, learning_rate)
      # print(weights)

  # 2. Mini Batch GD: Create mini batches and then update the weights using the mini batch as the input
  elif method == "mini_batch":
    for epoch in range(n_epochs):
      mini_batches = [(X[i : i + batch_size], y.reshape(-1, 1)[i : i + batch_size]) for i in range(0, len(X), batch_size)]

      for mini_batch in mini_batches:
        X_mini_batch, y_mini_batch = mini_batch
        gradient = compute_gradient(X_mini_batch, y_mini_batch, weights)
        weights = weight_update(weights, gradient, learning_rate)
  # 3. Stochasitic GD: Take each data point to update the weight matrix
  elif method == "stochastic":
    for epoch in range(n_epochs):
      for i in range(len(X)):
        X_sgd = X[i]
        y_sgd = y.reshape(-1, 1)[i]
        gradient = compute_gradient(X_sgd, y_sgd, weights)
        weights = weight_update(weights, gradient, learning_rate)
  
  return weights.reshape(-1,)
