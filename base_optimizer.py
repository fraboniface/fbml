import numpy as np

class Optimizer:

	def __init__(self, gamma):
		self.gamma = gamma
		self.h = 1e-8


	def gradient_check(self, func, grad, points):
		"""Checks wether the given gradient is correct."""
		def base_vector(size, index):
			e = np.zeros(size)
			e[index] = 1.0
			return e

		mean, cnt = 0, 0
		dim = points.shape[1]
		for point in points:
			if not np.isnan(func(point)):
				numeric_grad = []
				for i in range(dim):
					e = base_vector(dim, i)
					numeric_partial_derivative = (func(point+self.h*e) - func(point-self.h*e)) / (2*self.h)
					numeric_grad.append(numeric_partial_derivative)

				numeric_grad = np.array(numeric_grad)
				relative_error = np.absolute(grad(point) - numeric_grad)/np.maximum(np.absolute(grad(point)), np.absolute(numeric_grad))
				mean += relative_error.max()
				cnt += 1

		mean /= cnt

		if mean < 1e-6	:		# this threshold is rather arbitrary, see http://cs231n.github.io/neural-networks-3/
			return 1
		else:
			return 0


	def minimize(self, func, grad, theta0, epsilon):
		"""Finds the parameters which minimize the objective function."""
		pass