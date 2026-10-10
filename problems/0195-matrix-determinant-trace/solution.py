import numpy as np
def calc_det(A: list[list[float]]) -> int:
	"""
	Compute the determinant of a 2X2 matrix.
	"""
	return A[0][0] * A[1][1] - A[0][1] * A[1][0]

def helper(matrix: list[list[float]])-> tuple[float, float]:
	num_rows = num_cols = len(matrix) # N X N matrix
	if num_cols == 2:
		return calc_det(matrix)

	determinant = 0

	# 1. Iterate over the first row
	for i in range(num_cols):
		# 2. If 2 X 2 matrix calculate the determinant else recursively call the function on smaller matrx
		# 2.1. Find the smaller matrix
		small_matrix = [A[ : i] + A[i+1 : ] for A in matrix[1 : ]]
		# 2.2. Recursively call the function on the smaller function
		determinant_small, trace = matrix_determinant_and_trace(small_matrix)
		# 2.3. Update the determinant value
		determinant += (((-1) ** (i)) * matrix[0][i] * determinant_small)

	return determinant

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	determinant = helper(matrix)
	trace = sum([matrix[i][i] for i in range(len(matrix))])

	return determinant, trace
	

		

