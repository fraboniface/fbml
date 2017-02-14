import numpy as np

class Optimizer:
	"""Optimizer base class. It works but much slower and not as accuate as scikit-learn implementation."""

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

	def minimize(self, func, grad, b0, w0, epsilon):
		"""Finds the parameters which minimize the objective function."""
		points = 10*np.random.random_sample((10, w0.shape[0]+1)) - 5	# we pick 10 points in [-5,5]
		assert self.gradient_check(func, grad, points), "Gradient check has failed"



class VanillaGradientDescent(Optimizer):

	def __init__(self, gamma=1e-3):
		"""Vanilla gradient descent with adaptive learning rate not to be unbearably inefficient."""
		super().__init__(gamma)


	def minimize(self, func, grad, b0, w0, epsilon=1e-4):
		super().minimize(func, grad, b0, w0, epsilon)
		b, w = b0, w0
		gamma = self.gamma
		old = func(b0,w0)
		diff = np.inf
		while abs(diff) > epsilon:
			step = gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]

			diff = func(b,w) - old
			# we use an adaptive learning rate to go fatser: if the loss is decreasing, increase the learning rate by 5%, if it's increasing, halve it and go back to previous point
			if diff < 0:
				gamma *= 1.05
			else:
				b = b + step[0]
				w = w + step[1:]
				gamma /=2

			old = func(b,w)

		return b, w


class Momentum(Optimizer):

	def __init__(self, gamma=1e-4, momentum=0.9):
		super().__init__(gamma)
		self.momentum = momentum

	def minimize(self, func, grad, b0, w0, epsilon=1e-4):
		super().minimize(func, grad, b0, w0, epsilon)
		b,w = b0, w0
		step = 0
		old = func(b0,w0)
		diff = np.inf
		while abs(diff) > epsilon:
			step = self.momentum*step + self.gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]
			diff = func(b,w) - old
			old = func(b,w)

		return b, w


class Nesterov(Optimizer):

	def __init__(self, gamma=1e-4, momentum=0.9):
		super().__init__(gamma)
		self.momentum = momentum

	def minimize(self, func, grad, b0, w0, epsilon=1e-4):
		super().minimize(func, grad, b0, w0, epsilon)
		b,w = b0, w0
		step = np.zeros(w0.shape[0]+1)
		old = func(b0,w0)
		diff = np.inf
		while abs(diff) > epsilon:
			step = self.momentum*step + self.gamma*grad(b - self.momentum*step[0], w - self.momentum*step[1:])
			b = b - step[0]
			w = w - step[1:]
			diff = func(b,w) - old
			old = func(b,w)

		return b, w