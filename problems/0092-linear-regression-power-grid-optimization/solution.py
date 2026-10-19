import math
import numpy as np
from sklearn.linear_model import LinearRegression

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	fluctuation = lambda i : 10 * math.sin(2 * PI * (i+1) / 10)
	detrend_data = [consumption_data[i] - fluctuation(i) for i in range(10)]
	X = np.arange(1, 11).reshape(-1, 1)
	# 2) Perform linear regression on the detrended data.
	lr = LinearRegression()
	lr.fit(X, detrend_data)
	# 3) Predict day 15's base consumption.
	preds = lr.predict([[15]])[0]
	# 4) Add the day 15 fluctuation back.
	preds += fluctuation(14)
	# 5) Round, then add a 5% safety margin (rounded up).
	preds = round(preds)
	preds = math.ceil(1.05 * preds)
	# 6) Return the final integer.
	return preds