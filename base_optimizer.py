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
		dim = points.shape[1]-1 # the bias is added at the beginning so the dimension is that of the weight vector plus one
		for point in points:
			b, w = point[0], point[1:]
			if not np.isnan(func(b,w)):
				bias_partial_derivative = (func(b+self.h, w) - func(b-self.h, w)) / (2*self.h)
				numeric_grad = [bias_partial_derivative]
				for i in range(dim):
					e = base_vector(dim, i)
					weight_partial_derivative = (func(b, w+self.h*e) - func(b, w-self.h*e)) / (2*self.h)
					numeric_grad.append(weight_partial_derivative)

				numeric_grad = np.array(numeric_grad)
				relative_error = np.absolute(grad(b,w) - numeric_grad)/np.maximum(np.absolute(grad(b,w)), np.absolute(numeric_grad))
				mean += relative_error.max()
				cnt += 1

		mean /= cnt

		if mean < 1e-4*w.shape[0]	:		# this threshold is rather arbitrary, see http://cs231n.github.io/neural-networks-3/
			return 1
		else:
			return 0

	def minimize(self, func, grad, theta0, epsilon):
		"""Finds the parameters which minimize the objective function."""
		pass