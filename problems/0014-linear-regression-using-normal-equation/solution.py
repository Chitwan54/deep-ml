import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X  = np.array(X)
	y = np.array(y)

	# theta = (XTX)^-1XTy
	a = np.linalg.inv(np.matmul(X.T, X))
	b = np.matmul(X.T, y)

	theta = np.matmul(a, b)
	return theta